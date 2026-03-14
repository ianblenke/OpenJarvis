"""Tests for PWA static file serving in the SPA catch-all endpoint."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server.app import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Typed fake engine for PWA serving tests."""

    def __init__(self) -> None:
        self.engine_id = "mock"

    def health(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["test-model"]

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return {
            "content": "hello",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
            "finish_reason": "stop",
        }


def _make_engine():
    return _FakeEngine()


def _create_static_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary static directory with index.html and PWA files."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html><body>SPA</body></html>")
    (static / "sw.js").write_text("// service worker")
    (static / "manifest.webmanifest").write_text('{"name":"OpenJarvis"}')
    (static / "pwa-192x192.png").write_bytes(b"\x89PNG placeholder")
    assets = static / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('app')")
    return static


@pytest.fixture()
def client_with_static(tmp_path, monkeypatch):
    """Create a test client with a real temporary static directory."""
    static_dir = _create_static_dir(tmp_path)
    engine = _make_engine()

    # Patch Path(__file__).parent to make static_dir resolve to our tmp dir
    original_truediv = pathlib.Path.__truediv__

    def patched_truediv(self, key):
        result = original_truediv(self, key)
        # Intercept the "static" lookup in app.py
        if key == "static" and str(self).endswith("server"):
            return static_dir
        return result

    monkeypatch.setattr(pathlib.Path, "__truediv__", patched_truediv)
    app = create_app(engine, "test-model")
    monkeypatch.undo()  # Restore immediately after app creation
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPWAServing:
    @pytest.mark.spec("REQ-server.pwa")
    def test_sw_js_served_as_file(self, client_with_static):
        """Service worker file should be served directly, not as index.html."""
        resp = client_with_static.get("/sw.js")
        assert resp.status_code == 200
        assert "// service worker" in resp.text

    @pytest.mark.spec("REQ-server.pwa")
    def test_manifest_served_as_file(self, client_with_static):
        """Web manifest should be served directly."""
        resp = client_with_static.get("/manifest.webmanifest")
        assert resp.status_code == 200
        assert "OpenJarvis" in resp.text

    @pytest.mark.spec("REQ-server.pwa")
    def test_icon_served_as_file(self, client_with_static):
        """PWA icon should be served directly."""
        resp = client_with_static.get("/pwa-192x192.png")
        assert resp.status_code == 200
        assert b"PNG" in resp.content

    @pytest.mark.spec("REQ-server.pwa")
    def test_api_routes_bypass_spa(self, client_with_static):
        """API routes should still work regardless of SPA catch-all."""
        resp = client_with_static.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"

    @pytest.mark.spec("REQ-server.pwa")
    def test_path_traversal_blocked(self, client_with_static):
        """Path traversal attempts should fall back to index.html."""
        resp = client_with_static.get("/../../etc/passwd")
        assert resp.status_code == 200
        # Should get index.html, not the passwd file
        assert "SPA" in resp.text
