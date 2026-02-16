from __future__ import annotations

import base64
from fastapi.testclient import TestClient

from gui.server import create_app


def test_gui_index_page_loads() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "DeepSeek Agent Console" in response.text


def test_gui_files_endpoint_returns_listing() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/files", params={"workspace_path": ".", "limit": 50})
    assert response.status_code == 200
    payload = response.json()
    assert "workspace" in payload
    assert isinstance(payload.get("files"), list)


def test_gui_memory_endpoint_shape() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/memory", params={"limit": 10})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("entries"), list)
    assert "context_preview" in payload


def test_gui_memory_clear_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post("/api/memory/clear")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_gui_upload_files_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_GUI_WORKSPACE", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    content = b"hello upload"
    payload = {
        "workspace": ".",
        "destination": "uploads",
        "files": [
            {
                "name": "note.txt",
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        ],
    }
    response = client.post("/api/uploads", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["saved"][0]["path"] == "uploads/note.txt"
    assert (tmp_path / "uploads" / "note.txt").read_bytes() == content


def test_gui_upload_rejects_escape_destination(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_GUI_WORKSPACE", str(tmp_path))
    app = create_app()
    client = TestClient(app)

    payload = {
        "workspace": ".",
        "destination": "../outside",
        "files": [
            {
                "name": "bad.txt",
                "content_base64": base64.b64encode(b"x").decode("ascii"),
            }
        ],
    }
    response = client.post("/api/uploads", json=payload)
    assert response.status_code == 400
