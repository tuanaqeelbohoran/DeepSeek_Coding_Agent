from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("DEEPSEEK_GUI_HOST", "127.0.0.1")
    port = int(os.getenv("DEEPSEEK_GUI_PORT", "7860"))
    reload_mode = os.getenv("DEEPSEEK_GUI_RELOAD", "0") == "1"
    uvicorn.run("gui.server:app", host=host, port=port, reload=reload_mode)


if __name__ == "__main__":
    main()
