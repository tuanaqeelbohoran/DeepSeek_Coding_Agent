# DeepSeek Agentic AI (Unsloth)

Autonomous agent loop powered by DeepSeek + Unsloth, with:

- CLI runner
- Web GUI (Claude Code-style workflow)
- Workspace tools (`list_files`, `read_file`, `write_file`, `append_file`, `run_shell`)
- Optional OCR context via DeepSeek-OCR-2
- Session memory between runs
- File and image upload in GUI
- Lazy loading, sparse/offload retries, and fallback models for lower VRAM GPUs

OCR reference:
- https://unsloth.ai/docs/models/deepseek-ocr-2#running-deepseek-ocr-2

## Requirements

- Linux
- Python 3.10+
- NVIDIA GPU with CUDA recommended

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run from CLI:

```bash
python main.py "Create a FastAPI app with tests" --workspace .
```

Run GUI:

```bash
python run_gui.py
```

Open:

```text
http://127.0.0.1:7860
```

## GUI Features

- Task composer with generation/model controls
- Live timeline (step decisions, tool calls, outputs, final answer)
- Thinking pattern + progress tracker
- Run history
- Workspace file tree
- Session memory viewer and clear action
- File upload to workspace
- Image upload that auto-fills `image_path` for OCR runs

## CLI Usage

```bash
python main.py "your task here" --workspace .
```

Useful flags:

- `--workspace`: root directory agent can access
- `--image`: image file path for OCR context
- `--max-steps`
- `--min-new-tokens`
- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--coder-model`
- `--ocr-model`
- `--no-lazy-load`
- `--no-sparse-load`
- `--max-gpu-memory-gib`
- `--no-shell`
- `--no-ocr`
- `--quiet`

Examples:

```bash
python main.py "Fix failing tests" --workspace . --max-steps 20
python main.py "Summarize this screenshot" --workspace . --image ./error.png
python main.py "Refactor project" --workspace . --no-shell
```

## Model Loading Strategy (Low VRAM)

Default coder model:
- `unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit`

Default fallback coder models:
- `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

Load behavior:

1. Try primary model.
2. If enabled, retry with sparse/offload settings (`device_map=auto`, CPU offload).
3. If enabled, try fallback models.

Use this for 11 GB GPUs:

```bash
export DEEPSEEK_MAX_GPU_MEMORY_GIB=10
export DEEPSEEK_SPARSE_LOAD=1
```

## Environment Variables

Model/runtime:

- `DEEPSEEK_CODER_MODEL`
- `DEEPSEEK_OCR_MODEL`
- `DEEPSEEK_ENABLE_MODEL_FALLBACK`
- `DEEPSEEK_FALLBACK_CODER_MODELS` (comma-separated)
- `DEEPSEEK_LAZY_LOAD`
- `DEEPSEEK_SPARSE_LOAD`
- `DEEPSEEK_MAX_GPU_MEMORY_GIB`
- `UNSLOTH_LOAD_IN_4BIT`
- `UNSLOTH_MODEL_CACHE_DIR`

Agent behavior:

- `AGENT_MAX_STEPS`
- `AGENT_MIN_NEW_TOKENS`
- `AGENT_MAX_NEW_TOKENS`
- `AGENT_TEMPERATURE`
- `AGENT_TOP_P`
- `AGENT_ALLOW_SHELL`
- `AGENT_SHELL_TIMEOUT_SEC`
- `AGENT_TOOL_OUTPUT_CHARS`

GUI server:

- `DEEPSEEK_GUI_HOST` (default `127.0.0.1`)
- `DEEPSEEK_GUI_PORT` (default `7860`)
- `DEEPSEEK_GUI_RELOAD` (`1` to enable uvicorn reload)
- `DEEPSEEK_GUI_WORKSPACE` (base workspace for GUI)

## API Endpoints (GUI Backend)

- `GET /api/runs`
- `POST /api/runs`
- `GET /api/runs/{run_id}?since=0`
- `GET /api/files?workspace_path=.&limit=500`
- `GET /api/memory`
- `POST /api/memory/clear`
- `POST /api/uploads`

Upload payload example:

```json
{
  "workspace": ".",
  "destination": "uploads",
  "files": [
    {
      "name": "notes.txt",
      "content_base64": "SGVsbG8="
    }
  ]
}
```

## Safety Notes

- Paths are constrained to the selected workspace.
- Shell tool can be disabled with `--no-shell`.
- Shell blocklist denies obvious destructive patterns.
- Upload destination/path is validated to prevent workspace escape.

## Development

Run tests:

```bash
pytest -q
```

## Project Structure

- `main.py`: CLI entrypoint
- `run_gui.py`: GUI entrypoint
- `deepseek_agent/config.py`: config + env loading
- `deepseek_agent/model.py`: model wrappers (coder + OCR)
- `deepseek_agent/agent.py`: step loop and event emission
- `deepseek_agent/tools.py`: workspace tool execution
- `gui/server.py`: FastAPI backend
- `gui/static/index.html`: GUI markup
- `gui/static/app.js`: GUI logic
- `gui/static/styles.css`: GUI styles
