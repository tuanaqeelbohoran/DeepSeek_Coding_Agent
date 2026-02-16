from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Tuple


@dataclass
class AgentConfig:
    workspace: Path
    coder_model_name: str = "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"
    fallback_coder_models: Tuple[str, ...] = (
        "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    )
    ocr_model_name: str = "deepseek-ai/DeepSeek-OCR-2"
    load_in_4bit: bool = True
    enable_model_fallback: bool = True
    lazy_model_load: bool = True
    sparse_load: bool = True
    max_gpu_memory_gib: int | None = None
    max_steps: int = 10
    min_new_tokens: int = 32
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.95
    allow_shell: bool = True
    shell_timeout_sec: int = 45
    tool_output_chars: int = 4_000
    model_cache_dir: Path = Path("models")

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "AgentConfig":
        fallback_models_raw = os.getenv(
            "DEEPSEEK_FALLBACK_CODER_MODELS",
            "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit,unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        )
        fallback_models = tuple(
            model.strip() for model in fallback_models_raw.split(",") if model.strip()
        )
        return cls(
            workspace=(workspace or Path.cwd()).resolve(),
            coder_model_name=os.getenv(
                "DEEPSEEK_CODER_MODEL",
                "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit",
            ),
            fallback_coder_models=fallback_models,
            ocr_model_name=os.getenv("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR-2"),
            load_in_4bit=os.getenv("UNSLOTH_LOAD_IN_4BIT", "1") != "0",
            enable_model_fallback=os.getenv("DEEPSEEK_ENABLE_MODEL_FALLBACK", "1") != "0",
            lazy_model_load=os.getenv("DEEPSEEK_LAZY_LOAD", "1") != "0",
            sparse_load=os.getenv("DEEPSEEK_SPARSE_LOAD", "1") != "0",
            max_gpu_memory_gib=(
                int(os.getenv("DEEPSEEK_MAX_GPU_MEMORY_GIB"))
                if os.getenv("DEEPSEEK_MAX_GPU_MEMORY_GIB")
                else None
            ),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "10")),
            min_new_tokens=int(os.getenv("AGENT_MIN_NEW_TOKENS", "32")),
            max_new_tokens=int(os.getenv("AGENT_MAX_NEW_TOKENS", "2048")),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("AGENT_TOP_P", "0.95")),
            allow_shell=os.getenv("AGENT_ALLOW_SHELL", "1") != "0",
            shell_timeout_sec=int(os.getenv("AGENT_SHELL_TIMEOUT_SEC", "45")),
            tool_output_chars=int(os.getenv("AGENT_TOOL_OUTPUT_CHARS", "4000")),
            model_cache_dir=Path(os.getenv("UNSLOTH_MODEL_CACHE_DIR", "models")),
        )
