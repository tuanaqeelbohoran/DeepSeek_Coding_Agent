from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import AgentConfig
from .model import DeepSeekCoderModel, DeepSeekOCR2Model
from .tools import TOOL_DESCRIPTIONS, ToolExecutor


SYSTEM_PROMPT = """You are an autonomous agent.
You solve both coding and non-coding tasks by reasoning briefly, then using tools when needed.

Response format rules:
1) Always return valid JSON.
2) JSON schema:
{
  "thought": "short reasoning",
  "actions": [{"tool": "tool_name", "args": {...}}],
  "final_answer": null
}
3) Use either `actions` or `final_answer` each turn. Do not leave both empty.
4) If the user question can be answered directly, return a direct `final_answer` with no tool calls.
5) Use tools only when they improve accuracy or are required for workspace changes.
6) Do not refuse a request only because it is outside coding; give the best direct answer you can.
7) Respect workspace boundaries; do not reference files outside workspace.
8) Do not output chain-of-thought or <think> tags.
9) Do not wrap JSON in markdown fences.
"""

FORMAT_RETRY_PROMPT = """Your previous response was invalid.
Return ONLY one valid JSON object with keys: thought, actions, final_answer.
- If work remains, set final_answer to null and include at least 1 action.
- If done, set actions to [] and provide final_answer.
- Never include <think> tags or markdown code fences.
- Do not repeat your prior response.
"""


class CodingAgent:
    def __init__(self, config: AgentConfig, *, enable_ocr: bool = True) -> None:
        self.config = config
        self.tools = ToolExecutor(
            config.workspace,
            allow_shell=config.allow_shell,
            shell_timeout_sec=config.shell_timeout_sec,
            output_char_limit=config.tool_output_chars,
        )
        self.coder = DeepSeekCoderModel(config)
        self.enable_ocr = enable_ocr
        self._ocr_model: DeepSeekOCR2Model | None = None

    def _strip_think(self, text: str) -> str:
        lower = text.lower()
        start_tag = "<think>"
        end_tag = "</think>"
        while True:
            start = lower.find(start_tag)
            if start == -1:
                break
            end = lower.find(end_tag, start + len(start_tag))
            if end == -1:
                text = text[:start].strip()
                break
            text = (text[:start] + text[end + len(end_tag) :]).strip()
            lower = text.lower()
        return text.strip()

    def _looks_like_decision_payload(self, text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        if candidate.startswith("```"):
            return True
        if candidate.startswith("{") and candidate.endswith("}"):
            return True
        lowered = candidate.lower()
        return '"actions"' in lowered or '"final_answer"' in lowered or '"thought"' in lowered

    def _format_tool_specs(self) -> str:
        rows = []
        for tool in TOOL_DESCRIPTIONS:
            rows.append(
                f"- {tool['name']}: {tool['description']} args={tool['args']}"
            )
        return "\n".join(rows)

    def _workspace_snapshot(self) -> str:
        return self.tools.list_files(".", limit=120)

    def _infer_pattern(self, actions: list[dict[str, Any]], final_answer: str | None, step: int) -> str:
        if final_answer:
            return "wrap_up"
        if not actions:
            return "reasoning"
        tools = {str(action.get("tool", "")) for action in actions}
        if "write_file" in tools or "append_file" in tools:
            return "editing"
        if "run_shell" in tools:
            return "verification"
        if "read_file" in tools or "list_files" in tools:
            if step <= 2:
                return "discovery"
            return "analysis"
        return "execution"

    def _trim_messages(self, messages: list[dict[str, str]], max_chars: int = 24_000) -> list[dict[str, str]]:
        if len(messages) <= 2:
            return messages

        head = messages[:2]
        tail = messages[2:]
        total = sum(len(part["content"]) for part in head)
        kept_tail: list[dict[str, str]] = []
        for part in reversed(tail):
            size = len(part["content"])
            if total + size > max_chars:
                break
            kept_tail.append(part)
            total += size
        kept_tail.reverse()
        return head + kept_tail

    def _load_ocr(self) -> DeepSeekOCR2Model:
        if self._ocr_model is None:
            self._ocr_model = DeepSeekOCR2Model(self.config)
        return self._ocr_model

    def _build_first_user_message(
        self,
        task: str,
        image_path: str | None,
        session_memory: str | None = None,
    ) -> str:
        ocr_context = ""
        if image_path and self.enable_ocr:
            ocr_text = self._load_ocr().ocr(image_path)
            ocr_context = f"\n\nOCR_CONTEXT:\n{ocr_text}\n"
        memory_context = f"\n\nSESSION_MEMORY:\n{session_memory}\n" if session_memory else ""

        return (
            f"TASK:\n{task}\n\n"
            f"WORKSPACE_ROOT:\n{self.config.workspace}\n\n"
            f"AVAILABLE_TOOLS:\n{self._format_tool_specs()}\n\n"
            f"INITIAL_FILE_TREE:\n{self._workspace_snapshot()}"
            f"{memory_context}{ocr_context}\n"
            "The task may be coding or non-coding. "
            "If tools are unnecessary, answer directly via final_answer.\n"
            "Start now."
        )

    def _log_step(self, step: int, payload: dict[str, Any]) -> None:
        print(f"\n--- Step {step} ---")
        thought = payload.get("thought", "")
        if thought:
            print(f"thought: {thought}")
        actions = payload.get("actions", [])
        if actions:
            print(f"actions: {actions}")
        final_answer = payload.get("final_answer")
        if final_answer:
            print("agent proposed final answer")

    def _emit(
        self,
        on_event: Callable[[dict[str, Any]], None] | None,
        event_type: str,
        **payload: Any,
    ) -> None:
        if on_event is None:
            return
        try:
            on_event({"type": event_type, **payload})
        except Exception:  # noqa: BLE001
            return

    def run(
        self,
        task: str,
        *,
        image_path: str | None = None,
        max_steps: int | None = None,
        verbose: bool = True,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        session_memory: str | None = None,
    ) -> str:
        step_limit = max_steps or self.config.max_steps
        tools_executed = 0
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_first_user_message(
                    task,
                    image_path,
                    session_memory=session_memory,
                ),
            },
        ]
        self._emit(
            on_event,
            "run_started",
            task=task,
            step_limit=step_limit,
            workspace=str(self.config.workspace),
            image_path=image_path,
            has_session_memory=bool(session_memory),
            coder_model=self.coder.loaded_model_name or self.config.coder_model_name,
            lazy_load=self.config.lazy_model_load,
            sparse_load=self.config.sparse_load,
            max_gpu_memory_gib=self.config.max_gpu_memory_gib,
        )

        for step in range(1, step_limit + 1):
            messages = self._trim_messages(messages)
            decision = self.coder.decide(messages)
            self._emit(
                on_event,
                "step_decision",
                step=step,
                thought=decision.thought,
                actions=[a.__dict__ for a in decision.actions],
                final_answer=decision.final_answer,
            )

            if verbose:
                self._log_step(
                    step,
                    {
                        "thought": decision.thought,
                        "actions": [a.__dict__ for a in decision.actions],
                        "final_answer": decision.final_answer,
                    },
                )

            progress_pct = int((step / max(step_limit, 1)) * 100)
            pattern = self._infer_pattern(
                [a.__dict__ for a in decision.actions],
                decision.final_answer,
                step,
            )
            self._emit(
                on_event,
                "progress_update",
                step=step,
                step_limit=step_limit,
                progress_pct=progress_pct,
                pattern=pattern,
                tools_executed=tools_executed,
            )

            if decision.final_answer and not decision.actions:
                cleaned = self._strip_think(decision.final_answer)
                if cleaned:
                    self._emit(on_event, "run_completed", final_answer=cleaned, step=step)
                    return cleaned

            if not decision.actions and not decision.final_answer:
                raw_fallback = self._strip_think(decision.raw_text)
                mentions_tool = any(
                    token in raw_fallback for token in ("list_files", "read_file", "write_file", "append_file", "run_shell")
                )
                if (
                    tools_executed > 0
                    and raw_fallback
                    and not mentions_tool
                    and not self._looks_like_decision_payload(raw_fallback)
                ):
                    self._emit(on_event, "run_completed", final_answer=raw_fallback, step=step)
                    return raw_fallback
                if verbose:
                    print("format error: model returned no actions/final answer; retrying with stricter prompt")
                self._emit(on_event, "format_retry", step=step)
                messages.append({"role": "user", "content": FORMAT_RETRY_PROMPT})
                continue

            messages.append({"role": "assistant", "content": decision.raw_text})

            for action in decision.actions:
                self._emit(
                    on_event,
                    "tool_started",
                    step=step,
                    tool=action.tool,
                    args=action.args,
                )
                output = self.tools.execute(action.tool, action.args)
                tools_executed += 1
                self._emit(
                    on_event,
                    "tool_result",
                    step=step,
                    tool=action.tool,
                    args=action.args,
                    output=output,
                )
                tool_feedback = (
                    f"TOOL_RESULT\n"
                    f"tool={action.tool}\n"
                    f"args={action.args}\n"
                    f"output:\n{output}"
                )
                messages.append({"role": "user", "content": tool_feedback})

            if decision.final_answer:
                cleaned = self._strip_think(decision.final_answer)
                if cleaned:
                    self._emit(on_event, "run_completed", final_answer=cleaned, step=step)
                    return cleaned

        timeout_message = (
            "Reached max steps without final answer. "
            "Run again with a higher --max-steps or a narrower task."
        )
        self._emit(on_event, "run_timeout", message=timeout_message)
        return timeout_message
