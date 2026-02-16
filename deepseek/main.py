from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_agent import AgentConfig, CodingAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic coding assistant powered by DeepSeek + Unsloth."
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Task instruction for the coding agent.",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory the agent can access.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image path. If set, DeepSeek-OCR-2 extracts context first.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum agent steps.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help="Minimum generated tokens per model turn.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens per model turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Use 0.0 for most reliable tool JSON output.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling value (used when temperature > 0).",
    )
    parser.add_argument(
        "--no-lazy-load",
        action="store_true",
        help="Disable lazy model loading and load coder model during startup.",
    )
    parser.add_argument(
        "--no-sparse-load",
        action="store_true",
        help="Disable sparse/offloaded model retry path.",
    )
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=int,
        default=None,
        help="Set GPU memory cap (GiB) for sparse offload attempts.",
    )
    parser.add_argument(
        "--coder-model",
        default=None,
        help="Override coder model id (default from config/env).",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help="Override OCR model id (default deepseek-ai/DeepSeek-OCR-2).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR flow even if --image is supplied.",
    )
    parser.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable run_shell tool for stricter safety.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = " ".join(args.task).strip()
    workspace = Path(args.workspace).resolve()

    config = AgentConfig.from_env(workspace=workspace)
    if args.coder_model:
        config.coder_model_name = args.coder_model
    if args.ocr_model:
        config.ocr_model_name = args.ocr_model
    if args.no_shell:
        config.allow_shell = False
    if args.no_lazy_load:
        config.lazy_model_load = False
    if args.no_sparse_load:
        config.sparse_load = False
    if args.max_gpu_memory_gib is not None:
        config.max_gpu_memory_gib = args.max_gpu_memory_gib
    if args.min_new_tokens is not None:
        config.min_new_tokens = args.min_new_tokens
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p

    agent = CodingAgent(config, enable_ocr=not args.no_ocr)
    result = agent.run(
        task,
        image_path=args.image,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    print("\n=== Final Answer ===")
    print(result)


if __name__ == "__main__":
    main()
