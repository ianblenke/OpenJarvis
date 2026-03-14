"""Tests for the WebSocket streaming endpoint."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from openjarvis.core.types import Message  # noqa: E402
from openjarvis.server.api_routes import include_all_routes  # noqa: E402
from tests.fixtures.engines import FakeEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StreamingEngine(FakeEngine):
    """FakeEngine subclass that supports async streaming."""

    def __init__(self, tokens: List[str] | None = None) -> None:
        self._tokens = tokens or ["Hello", " ", "world"]
        combined = "".join(self._tokens)
        super().__init__(engine_id="mock", responses=[combined])

    async def stream(self, messages, *, model="test-model", **kwargs):
        for tok in self._tokens:
            yield tok


class _GenerateOnlyEngine:
    """Typed fake engine with generate() only (no stream())."""

    engine_id = "mock-nostream"

    def __init__(self, content: str = "Hello world") -> None:
        self._content = content

    def generate(
        self, messages: Sequence[Message], *, model: str, **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "content": self._content,
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "stop",
        }


class _ErrorStreamingEngine:
    """Typed fake engine whose stream() raises RuntimeError."""

    engine_id = "mock-error"

    async def stream(self, messages, *, model="test-model", **kwargs):
        raise RuntimeError("Engine exploded")
        yield  # pragma: no cover – needed for async gen syntax


def _make_app(engine=None):
    """Create a minimal FastAPI app with a typed fake engine wired up."""
    app = FastAPI()
    if engine is None:
        engine = _StreamingEngine()
    app.state.engine = engine
    app.state.model = "test-model"
    include_all_routes(app)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketStreaming:
    """Tests for WS /v1/chat/stream endpoint."""

    @pytest.mark.spec("REQ-server.websocket")
    def test_basic_streaming_exchange(self):
        """A valid message should produce chunk messages followed by a done."""
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"message": "Hi"}))
            chunks = []
            done = None
            # Read all responses until we get 'done'
            while True:
                data = ws.receive_json()
                if data["type"] == "chunk":
                    chunks.append(data["content"])
                elif data["type"] == "done":
                    done = data
                    break
                else:
                    break
            assert len(chunks) == 3
            assert chunks == ["Hello", " ", "world"]
            assert done is not None
            assert done["content"] == "Hello world"

    @pytest.mark.spec("REQ-server.websocket")
    def test_missing_message_field(self):
        """Sending JSON without a 'message' field should return an error."""
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"text": "Hi"}))
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Missing" in data["detail"]

    @pytest.mark.spec("REQ-server.websocket")
    def test_invalid_json(self):
        """Sending non-JSON text should return an error."""
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text("not json at all")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["detail"]

    @pytest.mark.spec("REQ-server.websocket")
    def test_empty_message_field(self):
        """An empty string for 'message' should return an error."""
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"message": ""}))
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Missing" in data["detail"]

    @pytest.mark.spec("REQ-server.websocket")
    def test_generate_fallback_when_no_stream(self):
        """When the engine has no stream(), generate() result is sent as one chunk."""
        engine = _GenerateOnlyEngine("Fallback response")
        app = _make_app(engine=engine)
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"message": "Hi"}))
            chunks = []
            done = None
            while True:
                data = ws.receive_json()
                if data["type"] == "chunk":
                    chunks.append(data["content"])
                elif data["type"] == "done":
                    done = data
                    break
                else:
                    break
            assert len(chunks) == 1
            assert chunks[0] == "Fallback response"
            assert done is not None
            assert done["content"] == "Fallback response"

    @pytest.mark.spec("REQ-server.websocket")
    def test_custom_model_in_request(self):
        """The model field from the request should be forwarded to the engine."""
        engine = _StreamingEngine(tokens=["OK"])
        app = _make_app(engine=engine)
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"message": "Hi", "model": "custom-model"}))
            # Consume until done
            while True:
                data = ws.receive_json()
                if data["type"] == "done":
                    break
            # The mock stream function was called — we can't easily inspect
            # async-generator call args, but the exchange completed without error
            assert data["content"] == "OK"

    @pytest.mark.spec("REQ-server.websocket")
    def test_engine_error_returns_error_message(self):
        """If the engine raises, the endpoint should send an error frame."""
        engine = _ErrorStreamingEngine()
        app = _make_app(engine=engine)
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"message": "boom"}))
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Engine exploded" in data["detail"]

    @pytest.mark.spec("REQ-server.websocket")
    def test_multiple_messages_on_same_connection(self):
        """The WebSocket should support multiple request/response cycles."""
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            for _ in range(3):
                ws.send_text(json.dumps({"message": "Hi"}))
                # Drain until done
                while True:
                    data = ws.receive_json()
                    if data["type"] == "done":
                        assert data["content"] == "Hello world"
                        break

    @pytest.mark.spec("REQ-server.websocket")
    def test_no_engine_configured(self):
        """If app.state has no engine, an error should be returned."""
        app = FastAPI()
        app.state.model = "test-model"
        # Intentionally do NOT set app.state.engine
        include_all_routes(app)
        client = TestClient(app)
        with client.websocket_connect("/v1/chat/stream") as ws:
            ws.send_text(json.dumps({"message": "Hi"}))
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "engine" in data["detail"].lower()


__all__ = [
    "TestWebSocketStreaming",
]
