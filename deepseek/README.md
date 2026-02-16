# DeepSeek Agentic AI (Unsloth)

This project gives you an autonomous agent that:

- Uses a DeepSeek model through **Unsloth** for planning + tool use.
- Optionally runs **DeepSeek-OCR-2** first (following Unsloth docs) to extract context from screenshots.
- Can read/write files and run shell commands inside a workspace loop.

Reference used for OCR2 runtime pattern:
- https://unsloth.ai/docs/models/deepseek-ocr-2#running-deepseek-ocr-2

## What it does

1. Accepts a user task (coding or non-coding).
2. Builds a workspace snapshot.
3. Asks DeepSeek for JSON actions (`read_file`, `write_file`, `run_shell`, etc.).
4. Executes actions and feeds tool results back to the model.
5. Repeats until a `final_answer` is returned or max steps are reached.

## Requirements

- Linux
- Python 3.10+
- NVIDIA GPU with CUDA (recommended for Unsloth models)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py "Create a FastAPI app with /health and tests" --workspace .
```

## GUI (Claude Code-style workflow)

Run the local web console:

```bash
python run_gui.py
```

Open:

```text
http://127.0.0.1:7860
```

GUI features:

- Task composer with model controls (`max steps`, token limits, temperature, shell/OCR toggles)
- File upload to workspace and image upload that auto-fills OCR `image_path`
- Live run timeline (step decisions, tool calls, tool outputs, final answer)
- Iterative progress and thinking-pattern tracker (`planning`, `discovery`, `editing`, `verification`, `wrap_up`)
- Run history panel to revisit previous runs
- Workspace file tree panel
- Session memory panel (summaries of prior runs reused as context in new runs)

GPU memory notes:

- The loader now auto-retries with CPU offload and then smaller fallback models if the primary model does not fit.
- Default fallback models:
  - `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit`
  - `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
- You can set a smaller model directly in the GUI via `Coder Model`, or by env var:
- You can also toggle `Lazy load` and `Sparse/offload retry` in the GUI per run.

```bash
export DEEPSEEK_CODER_MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
```

Use OCR-assisted context from an image:

```bash
python main.py "Fix the failing tests shown in this screenshot" --workspace . --image ./error.png
```

Disable shell execution tool:

```bash
python main.py "Refactor src/ for readability" --workspace . --no-shell
```

Force deterministic generation for stricter tool-format behavior:

```bash
python main.py "Create a FastAPI app with tests" --workspace . --temperature 0.0 --max-new-tokens 2048
```

## Model configuration

Defaults:

- Coder model: `unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit`
- OCR model: `deepseek-ai/DeepSeek-OCR-2`

Override with flags:

```bash
python main.py "..." --coder-model "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit" --ocr-model "deepseek-ai/DeepSeek-OCR-2"
```

Or env vars:

```bash
export DEEPSEEK_CODER_MODEL="unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"
export DEEPSEEK_OCR_MODEL="deepseek-ai/DeepSeek-OCR-2"
export DEEPSEEK_LAZY_LOAD=1
export DEEPSEEK_SPARSE_LOAD=1
export DEEPSEEK_MAX_GPU_MEMORY_GIB=10
export UNSLOTH_LOAD_IN_4BIT=1
export AGENT_MAX_STEPS=10
export AGENT_TEMPERATURE=0.0
export AGENT_TOOL_OUTPUT_CHARS=4000
export AGENT_MAX_NEW_TOKENS=2048
```

## Safety notes

- Tool access is restricted to the chosen `--workspace`.
- A blocklist prevents obvious dangerous shell commands.
- Use `--no-shell` for stricter operation.

## Files

- `main.py`: CLI entrypoint
- `run_gui.py`: GUI server entrypoint
- `deepseek_agent/config.py`: runtime config
- `deepseek_agent/model.py`: Unsloth DeepSeek model wrappers
- `deepseek_agent/tools.py`: safe tool implementations
- `deepseek_agent/agent.py`: autonomous action loop
- `gui/server.py`: FastAPI backend for GUI
- `gui/static/index.html`: GUI shell
- `gui/static/app.js`: frontend logic
- `gui/static/styles.css`: frontend styling
