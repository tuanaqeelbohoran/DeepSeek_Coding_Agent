from __future__ import annotations

from deepseek_agent.agent import CodingAgent
from deepseek_agent.config import AgentConfig
from deepseek_agent.model import AgentDecision, ToolAction


class _FakeCoder:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self._decisions = decisions
        self.calls = 0
        self.loaded_model_name = "fake/model"

    def decide(self, messages: list[dict[str, str]]) -> AgentDecision:  # noqa: ARG002
        decision = self._decisions[min(self.calls, len(self._decisions) - 1)]
        self.calls += 1
        return decision


def test_structured_null_final_answer_does_not_end_run(tmp_path) -> None:
    config = AgentConfig(workspace=tmp_path, allow_shell=False, max_steps=3)
    agent = CodingAgent(config, enable_ocr=False)
    fake = _FakeCoder(
        [
            AgentDecision(
                raw_text='{"thought":"inspect files","actions":[{"tool":"list_files","args":{"path":"."}}],"final_answer":null}',
                thought="inspect files",
                actions=[ToolAction(tool="list_files", args={"path": "."})],
                final_answer=None,
            ),
            AgentDecision(
                raw_text='{"thought":"continue","actions":[],"final_answer":null}',
                thought="continue",
                actions=[],
                final_answer=None,
            ),
            AgentDecision(
                raw_text='{"thought":"done","actions":[],"final_answer":"Completed."}',
                thought="done",
                actions=[],
                final_answer="Completed.",
            ),
        ]
    )
    agent.coder = fake

    answer = agent.run("task", verbose=False, max_steps=3)

    assert answer == "Completed."
    assert fake.calls == 3


def test_plain_text_fallback_still_completes_after_tool_use(tmp_path) -> None:
    config = AgentConfig(workspace=tmp_path, allow_shell=False, max_steps=2)
    agent = CodingAgent(config, enable_ocr=False)
    fallback_answer = "Use a remittance platform with lower spread and fixed fees."
    fake = _FakeCoder(
        [
            AgentDecision(
                raw_text='{"thought":"inspect files","actions":[{"tool":"list_files","args":{"path":"."}}],"final_answer":null}',
                thought="inspect files",
                actions=[ToolAction(tool="list_files", args={"path": "."})],
                final_answer=None,
            ),
            AgentDecision(
                raw_text=fallback_answer,
                thought="",
                actions=[],
                final_answer=None,
            ),
        ]
    )
    agent.coder = fake

    answer = agent.run("task", verbose=False, max_steps=2)

    assert answer == fallback_answer
    assert fake.calls == 2
