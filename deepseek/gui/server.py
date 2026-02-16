from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deepseek_agent import AgentConfig, CodingAgent
from deepseek_agent.tools import ToolExecutor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRequest(BaseModel):
    task: str = Field(min_length=1)
    workspace: str = "."
    image_path: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=200)
    min_new_tokens: int | None = Field(default=None, ge=1, le=16_384)
    max_new_tokens: int | None = Field(default=None, ge=1, le=16_384)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    lazy_load: bool = True
    sparse_load: bool = True
    max_gpu_memory_gib: int | None = Field(default=None, ge=1, le=256)
    no_shell: bool = False
    no_ocr: bool = False
    coder_model: str | None = None
    ocr_model: str | None = None


class UploadFilePayload(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1)


class UploadRequest(BaseModel):
    workspace: str = "."
    destination: str = "."
    files: list[UploadFilePayload] = Field(min_length=1, max_length=64)


class RunManager:
    def __init__(self, base_workspace: Path) -> None:
        self.base_workspace = base_workspace.resolve()
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._session_memory: list[dict[str, Any]] = []
        self._memory_limit = 30

    def resolve_workspace(self, raw_workspace: str) -> Path:
        candidate = Path(raw_workspace)
        if not candidate.is_absolute():
            candidate = self.base_workspace / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"workspace does not exist or is not a directory: {raw_workspace}")
        return candidate

    def _append_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run["events"].append(
                {
                    "index": len(run["events"]),
                    "timestamp": _utc_now(),
                    **event,
                }
            )

    def _build_session_memory_context(self, max_entries: int = 8, max_chars: int = 5_000) -> str:
        with self._lock:
            entries = list(self._session_memory[-max_entries:])

        if not entries:
            return ""

        lines: list[str] = []
        for idx, item in enumerate(entries, start=1):
            patterns = ", ".join(item.get("patterns", [])) or "unknown"
            outcome = str(item.get("outcome", "")).strip().replace("\n", " ")
            if len(outcome) > 300:
                outcome = outcome[:300] + "..."
            task = str(item.get("task", "")).strip().replace("\n", " ")
            if len(task) > 220:
                task = task[:220] + "..."
            lines.append(
                f"[Run {idx}] status={item.get('status')} time={item.get('finished_at')}\n"
                f"task={task}\n"
                f"pattern={patterns}\n"
                f"outcome={outcome}"
            )

        rendered = "\n\n".join(lines)
        if len(rendered) > max_chars:
            return rendered[-max_chars:]
        return rendered

    def _remember_run(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            events = list(run["events"])
            status = str(run["status"])
            task = str(run["task"])
            finished_at = run["finished_at"] or _utc_now()
            outcome = str(run.get("final_answer") or run.get("error") or "")

        patterns: list[str] = []
        for event in events:
            if event.get("type") != "progress_update":
                continue
            pattern = str(event.get("pattern", "")).strip()
            if pattern and pattern not in patterns:
                patterns.append(pattern)

        entry = {
            "run_id": run_id,
            "status": status,
            "task": task,
            "patterns": patterns,
            "outcome": outcome,
            "finished_at": finished_at,
        }

        with self._lock:
            self._session_memory.append(entry)
            if len(self._session_memory) > self._memory_limit:
                self._session_memory = self._session_memory[-self._memory_limit :]

    def _run_agent(self, run_id: str, request: RunRequest, workspace: Path) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = "running"
            run["started_at"] = _utc_now()

        try:
            memory_context = self._build_session_memory_context()
            config = AgentConfig.from_env(workspace=workspace)
            if request.coder_model:
                config.coder_model_name = request.coder_model
            if request.ocr_model:
                config.ocr_model_name = request.ocr_model
            if request.no_shell:
                config.allow_shell = False
            config.lazy_model_load = request.lazy_load
            config.sparse_load = request.sparse_load
            if request.max_gpu_memory_gib is not None:
                config.max_gpu_memory_gib = request.max_gpu_memory_gib
            if request.min_new_tokens is not None:
                config.min_new_tokens = request.min_new_tokens
            if request.max_new_tokens is not None:
                config.max_new_tokens = request.max_new_tokens
            if request.temperature is not None:
                config.temperature = request.temperature
            if request.top_p is not None:
                config.top_p = request.top_p

            image_path: str | None = None
            if request.image_path:
                candidate = Path(request.image_path)
                if not candidate.is_absolute():
                    candidate = workspace / candidate
                candidate = candidate.resolve()
                try:
                    candidate.relative_to(workspace)
                except ValueError as exc:
                    raise ValueError("image_path escapes workspace") from exc
                if not candidate.exists() or not candidate.is_file():
                    raise ValueError(f"image file not found: {request.image_path}")
                image_path = str(candidate)

            agent = CodingAgent(config, enable_ocr=not request.no_ocr)
            if memory_context:
                with self._lock:
                    memory_entries = min(len(self._session_memory), 8)
                self._append_event(
                    run_id,
                    {
                        "type": "memory_context_used",
                        "memory_entries": memory_entries,
                    },
                )
            final_answer = agent.run(
                request.task,
                image_path=image_path,
                max_steps=request.max_steps,
                verbose=False,
                on_event=lambda event: self._append_event(run_id, event),
                session_memory=memory_context or None,
            )
            with self._lock:
                run = self._runs[run_id]
                run["status"] = "completed"
                run["final_answer"] = final_answer
                run["finished_at"] = _utc_now()
            self._remember_run(run_id)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                run = self._runs[run_id]
                run["status"] = "error"
                run["error"] = message
                run["finished_at"] = _utc_now()
            self._append_event(run_id, {"type": "run_error", "message": message})
            self._remember_run(run_id)

    def start_run(self, request: RunRequest) -> str:
        workspace = self.resolve_workspace(request.workspace)
        run_id = uuid4().hex[:12]
        run_data = {
            "id": run_id,
            "status": "queued",
            "task": request.task,
            "workspace": str(workspace),
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "final_answer": None,
            "error": None,
            "events": [],
        }
        with self._lock:
            self._runs[run_id] = run_data

        worker = threading.Thread(
            target=self._run_agent,
            args=(run_id, request, workspace),
            daemon=True,
        )
        worker.start()
        return run_id

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = []
            for run in self._runs.values():
                runs.append(
                    {
                        "id": run["id"],
                        "status": run["status"],
                        "task": run["task"],
                        "workspace": run["workspace"],
                        "created_at": run["created_at"],
                        "finished_at": run["finished_at"],
                    }
                )
        runs.sort(key=lambda row: row["created_at"], reverse=True)
        return runs

    def get_run(self, run_id: str, since: int = 0) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            all_events = list(run["events"])
            total_count = len(all_events)
            if since < 0:
                since = 0
            events = all_events[since:]
            return {
                "id": run["id"],
                "status": run["status"],
                "task": run["task"],
                "workspace": run["workspace"],
                "created_at": run["created_at"],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "final_answer": run["final_answer"],
                "error": run["error"],
                "events": events,
                "next_index": total_count,
            }

    def get_memory(self, max_entries: int = 20) -> dict[str, Any]:
        with self._lock:
            entries = list(self._session_memory[-max_entries:])
        return {
            "entries": entries,
            "context_preview": self._build_session_memory_context(max_entries=6, max_chars=2_500),
        }

    def clear_memory(self) -> None:
        with self._lock:
            self._session_memory = []


def create_app() -> FastAPI:
    here = Path(__file__).resolve().parent
    static_dir = here / "static"
    workspace = Path(os.getenv("DEEPSEEK_GUI_WORKSPACE", Path.cwd())).resolve()

    manager = RunManager(workspace)
    app = FastAPI(title="DeepSeek Agent GUI", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": manager.list_runs()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, since: int = Query(default=0, ge=0)) -> dict[str, Any]:
        try:
            return manager.get_run(run_id, since=since)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.post("/api/runs")
    def create_run(request: RunRequest) -> dict[str, str]:
        try:
            run_id = manager.start_run(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id}

    @app.post("/api/uploads")
    def upload_files(request: UploadRequest) -> dict[str, Any]:
        try:
            workspace_resolved = manager.resolve_workspace(request.workspace)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        destination_path = Path(request.destination)
        if destination_path.is_absolute():
            raise HTTPException(status_code=400, detail="destination must be a relative path")
        save_dir = (workspace_resolved / destination_path).resolve()
        try:
            save_dir.relative_to(workspace_resolved)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="destination escapes workspace") from exc
        save_dir.mkdir(parents=True, exist_ok=True)

        max_single_bytes = 25 * 1024 * 1024
        max_total_bytes = 100 * 1024 * 1024
        total_bytes = 0
        saved: list[dict[str, Any]] = []

        for item in request.files:
            filename = Path(item.name).name.strip()
            if not filename or filename in {".", ".."}:
                raise HTTPException(status_code=400, detail=f"invalid filename: {item.name!r}")
            try:
                content = base64.b64decode(item.content_base64, validate=True)
            except binascii.Error as exc:
                raise HTTPException(status_code=400, detail=f"invalid base64 for file: {filename}") from exc

            if len(content) > max_single_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"file too large: {filename} exceeds {max_single_bytes} bytes",
                )
            total_bytes += len(content)
            if total_bytes > max_total_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"total upload size exceeds {max_total_bytes} bytes",
                )

            target = (save_dir / filename).resolve()
            try:
                target.relative_to(workspace_resolved)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid target path for {filename}") from exc

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            saved.append(
                {
                    "name": filename,
                    "path": str(target.relative_to(workspace_resolved)),
                    "bytes": len(content),
                }
            )

        return {
            "workspace": str(workspace_resolved),
            "destination": str(destination_path),
            "saved": saved,
        }

    @app.get("/api/files")
    def list_files(workspace_path: str = ".", limit: int = Query(default=500, ge=1, le=2_000)) -> dict[str, Any]:
        try:
            workspace_resolved = manager.resolve_workspace(workspace_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        executor = ToolExecutor(workspace_resolved, allow_shell=False, output_char_limit=10_000)
        output = executor.list_files(".", limit=limit)
        if output.startswith("error:"):
            raise HTTPException(status_code=400, detail=output)
        files = [line for line in output.splitlines() if line.strip()]
        return {"workspace": str(workspace_resolved), "files": files}

    @app.get("/api/memory")
    def get_memory(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        return manager.get_memory(max_entries=limit)

    @app.post("/api/memory/clear")
    def clear_memory() -> dict[str, str]:
        manager.clear_memory()
        return {"status": "ok"}

    return app


app = create_app()
