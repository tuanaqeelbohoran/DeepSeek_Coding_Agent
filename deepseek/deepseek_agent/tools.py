from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
from typing import Any


TOOL_DESCRIPTIONS = [
    {
        "name": "list_files",
        "description": "List files under a directory in the workspace.",
        "args": {"path": "relative path, default '.'", "limit": "max entries, default 200"},
    },
    {
        "name": "read_file",
        "description": "Read file content with optional line range.",
        "args": {
            "path": "relative file path",
            "start_line": "1-based start line, default 1",
            "end_line": "1-based end line inclusive, optional",
        },
    },
    {
        "name": "write_file",
        "description": "Write full content to file, creating parent directories as needed.",
        "args": {"path": "relative file path", "content": "new file content"},
    },
    {
        "name": "append_file",
        "description": "Append content to end of file.",
        "args": {"path": "relative file path", "content": "content to append"},
    },
    {
        "name": "run_shell",
        "description": "Run a shell command inside the workspace.",
        "args": {"command": "bash command string", "timeout_sec": "optional timeout in seconds"},
    },
]


_BLOCKLIST_PATTERNS = [
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"rm\s+-rf\s+/",
    r":\(\)\s*{\s*:\|:\s*&\s*};:",
]


def _safe_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


class ToolExecutor:
    def __init__(
        self,
        root: Path,
        *,
        allow_shell: bool = True,
        shell_timeout_sec: int = 45,
        output_char_limit: int = 12_000,
    ) -> None:
        self.root = root.resolve()
        self.allow_shell = allow_shell
        self.shell_timeout_sec = shell_timeout_sec
        self.output_char_limit = output_char_limit

    def _resolve(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes workspace: {raw_path}") from exc
        return candidate

    def _clip(self, text: str) -> str:
        if len(text) <= self.output_char_limit:
            return text
        return f"{text[: self.output_char_limit]}\n\n[output truncated]"

    def list_files(self, path: str = ".", limit: int = 200) -> str:
        base = self._resolve(path)
        if not base.exists():
            return f"error: path does not exist: {path}"
        if not base.is_dir():
            return f"error: path is not a directory: {path}"

        ignored_dirs = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".venv",
            "node_modules",
            "models",
            "unsloth_compiled_cache",
        }
        rows: list[str] = []
        for current, dirs, files in os.walk(base):
            dirs[:] = [d for d in sorted(dirs) if d not in ignored_dirs]
            for file_name in sorted(files):
                full = Path(current) / file_name
                rel = _safe_relative(full, self.root)
                rows.append(rel)
                if len(rows) >= limit:
                    return "\n".join(rows) + "\n[list_files clipped by limit]"
        return "\n".join(rows) if rows else "(no files)"

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        target = self._resolve(path)
        if not target.exists():
            return f"error: file does not exist: {path}"
        if not target.is_file():
            return f"error: not a file: {path}"

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(start_line, 1)
        finish = end_line if end_line is not None else len(lines)
        finish = min(finish, len(lines))

        if start > finish:
            return "error: invalid line range"

        segment = lines[start - 1 : finish]
        rendered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(segment, start=start))
        return self._clip(rendered)

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {path}"

    def append_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"appended {len(content)} bytes to {path}"

    def run_shell(self, command: str, timeout_sec: int | None = None) -> str:
        if not self.allow_shell:
            return "error: shell tool is disabled"

        for pattern in _BLOCKLIST_PATTERNS:
            if re.search(pattern, command):
                return f"error: blocked command pattern matched: {pattern}"

        timeout = timeout_sec or self.shell_timeout_sec
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout}s"

        combined = (
            f"[exit_code={completed.returncode}]\n"
            f"$ {command}\n\n"
            f"{completed.stdout}"
        )
        if completed.stderr:
            combined += f"\n[stderr]\n{completed.stderr}"
        return self._clip(combined)

    def execute(self, tool: str, args: dict[str, Any]) -> str:
        try:
            if tool == "list_files":
                return self.list_files(
                    path=str(args.get("path", ".")),
                    limit=int(args.get("limit", 200)),
                )
            if tool == "read_file":
                if "path" not in args:
                    return "error: read_file requires path"
                return self.read_file(
                    path=str(args["path"]),
                    start_line=int(args.get("start_line", 1)),
                    end_line=int(args["end_line"]) if args.get("end_line") is not None else None,
                )
            if tool == "write_file":
                if "path" not in args or "content" not in args:
                    return "error: write_file requires path and content"
                return self.write_file(path=str(args["path"]), content=str(args["content"]))
            if tool == "append_file":
                if "path" not in args or "content" not in args:
                    return "error: append_file requires path and content"
                return self.append_file(path=str(args["path"]), content=str(args["content"]))
            if tool == "run_shell":
                if "command" not in args:
                    return "error: run_shell requires command"
                timeout = args.get("timeout_sec")
                parsed_timeout = int(timeout) if timeout is not None else None
                return self.run_shell(command=str(args["command"]), timeout_sec=parsed_timeout)
            return f"error: unknown tool {tool}"
        except Exception as exc:  # noqa: BLE001
            return f"error: tool execution failed: {exc}"
