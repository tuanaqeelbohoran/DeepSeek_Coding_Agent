from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
from typing import Any

from .config import AgentConfig


@dataclass
class ToolAction:
    tool: str
    args: dict[str, Any]


@dataclass
class AgentDecision:
    raw_text: str
    thought: str
    actions: list[ToolAction]
    final_answer: str | None


class DeepSeekCoderModel:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.loaded_model_name: str | None = None
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        if not self.config.lazy_model_load:
            self._ensure_loaded()

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        with self._load_lock:
            if self._model is None or self._tokenizer is None:
                self._model, self._tokenizer = self._load_model()
        return self._model, self._tokenizer

    def _load_model(self):  # type: ignore[no-untyped-def]
        try:
            from unsloth import FastLanguageModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Unsloth is required for coder model. Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        attempts: list[str] = []

        def attempt(model_name: str, **extra_kwargs: Any):  # type: ignore[no-untyped-def]
            kwargs: dict[str, Any] = {
                "model_name": model_name,
                "max_seq_length": 8192,
                "dtype": None,
                "load_in_4bit": self.config.load_in_4bit,
                "trust_remote_code": True,
            }
            kwargs.update(extra_kwargs)
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
                FastLanguageModel.for_inference(model)
                self.loaded_model_name = model_name
                return model, tokenizer
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{model_name} ({extra_kwargs or 'default'}): {type(exc).__name__}: {exc}")
                return None

        def sparse_kwargs() -> dict[str, Any]:
            if not self.config.sparse_load:
                return {}
            offload_dir = (self.config.model_cache_dir / "offload").resolve()
            offload_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "llm_int8_enable_fp32_cpu_offload": True,
                "offload_folder": str(offload_dir),
            }
            if self.config.max_gpu_memory_gib is not None and self.config.max_gpu_memory_gib > 0:
                kwargs["max_memory"] = {
                    0: f"{self.config.max_gpu_memory_gib}GiB",
                    "cpu": "64GiB",
                }
            return kwargs

        primary_name = self.config.coder_model_name
        loaded = attempt(primary_name)
        if loaded is not None:
            return loaded

        if self.config.enable_model_fallback:
            # Retry with sparse loading / CPU offload when VRAM is tight.
            sparse = sparse_kwargs()
            loaded = attempt(primary_name, **sparse) if sparse else None
            if loaded is not None:
                return loaded

            for fallback_name in self.config.fallback_coder_models:
                if fallback_name == primary_name:
                    continue
                loaded = attempt(fallback_name)
                if loaded is not None:
                    return loaded

                loaded = attempt(fallback_name, **sparse) if sparse else None
                if loaded is not None:
                    return loaded

        joined = "\n - ".join(attempts) if attempts else "no attempts recorded"
        raise RuntimeError(
            "Unable to load coder model.\n"
            f"Tried:\n - {joined}\n\n"
            "Try setting DEEPSEEK_CODER_MODEL to a smaller 4-bit model, for example:\n"
            "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
        )

    def _render_messages(self, messages: list[dict[str, str]], tokenizer: Any) -> str:
        active_model_name = (self.loaded_model_name or self.config.coder_model_name).lower()
        uses_r1_qwen3 = "deepseek-r1-0528-qwen3" in active_model_name
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            if uses_r1_qwen3:
                # For DeepSeek-R1-0528-Qwen3 non-thinking mode, append empty think block.
                # Model card guidance:
                # https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit
                rendered += "<think>\n\n</think>\n\n"
            return rendered
        rendered = []
        for msg in messages:
            rendered.append(f"{msg['role'].upper()}:\n{msg['content']}")
        rendered.append("ASSISTANT:")
        prompt = "\n\n".join(rendered)
        if uses_r1_qwen3:
            prompt += "\n<think>\n\n</think>\n\n"
        return prompt

    def decide(self, messages: list[dict[str, str]]) -> AgentDecision:
        model, tokenizer = self._ensure_loaded()
        prompt = self._render_messages(messages, tokenizer)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        generation_kwargs: dict[str, Any] = {
            "min_new_tokens": self.config.min_new_tokens,
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        outputs = model.generate(**inputs, **generation_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0][prompt_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self._parse_decision(raw)

    def _strip_think(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _extract_json_blob(self, raw: str) -> str | None:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fence:
            return fence.group(1)

        decoder = json.JSONDecoder()
        for index, char in enumerate(raw):
            if char != "{":
                continue
            try:
                payload, end = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return raw[index : index + end]
        return None

    def _parse_decision(self, raw: str) -> AgentDecision:
        blob = self._extract_json_blob(raw)
        if not blob:
            return AgentDecision(raw_text=raw, thought="", actions=[], final_answer=None)

        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            return AgentDecision(raw_text=raw, thought="", actions=[], final_answer=None)

        thought = self._strip_think(str(payload.get("thought", "")).strip())
        final_answer_raw = payload.get("final_answer")
        final_answer = (
            self._strip_think(str(final_answer_raw).strip())
            if final_answer_raw not in (None, "")
            else None
        )

        actions: list[ToolAction] = []
        for item in payload.get("actions", []) or []:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            args = item.get("args", {})
            if not tool:
                continue
            actions.append(ToolAction(tool=tool, args=args if isinstance(args, dict) else {}))

        if not actions and final_answer is None and thought:
            lowered = thought.lower()
            done_markers = (
                "task is complete",
                "task was completed",
                "now i should stop",
                "no further actions",
                "all done",
                "done",
                "finished",
            )
            if any(marker in lowered for marker in done_markers):
                final_answer = thought

        return AgentDecision(
            raw_text=self._strip_think(raw),
            thought=thought,
            actions=actions,
            final_answer=final_answer,
        )


class DeepSeekOCR2Model:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):  # type: ignore[no-untyped-def]
        # Follows Unsloth's DeepSeek-OCR-2 loading pattern:
        # https://unsloth.ai/docs/models/deepseek-ocr-2#running-deepseek-ocr-2
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoModel
            from unsloth import FastVisionModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Unsloth OCR dependencies are missing. Install with `pip install -r requirements.txt`."
            ) from exc

        cache_dir = (self.config.model_cache_dir / "deepseek_ocr_2").resolve()
        model_path = snapshot_download(
            self.config.ocr_model_name,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        model, tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=self.config.load_in_4bit,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    def ocr(self, image_path: str | Path, prompt: str = "Convert this image to markdown.") -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image file not found: {path}")

        result = self.model.infer(
            self.tokenizer,
            prompt=f"<image>\n{prompt}",
            image_file=str(path),
            temperature=0.0,
            max_new_tokens=8192,
            max_num_segments=4,
            ngram_size=30,
            window_size=90,
            think=False,
        )
        return self._normalize_output(result)

    def _normalize_output(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("result", "text", "markdown", "output", "content", "prediction"):
                if key in result:
                    return str(result[key]).strip()
            return json.dumps(result, ensure_ascii=True)
        if isinstance(result, (list, tuple)):
            return "\n".join(str(part) for part in result).strip()
        return str(result).strip()
