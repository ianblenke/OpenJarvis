"""Tests for the API server routes -- no MagicMock, uses FakeEngine.

Tests route registration, basic request/response for all core endpoints
using Starlette/FastAPI TestClient.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server.app import create_app  # noqa: E402
from tests.fixtures.engines import FakeEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Fake agent (no MagicMock)
# ---------------------------------------------------------------------------


class FakeAgent:
    """Lightweight agent stand-in that implements the protocol used by routes."""

    agent_id = "fake-agent"

    def __init__(self, content: str = "Hello from fake agent") -> None:
        self._content = content
        self._model = "fake-model"

    def run(self, input_text, *, context=None):
        from openjarvis.agents._stubs import AgentResult

        return AgentResult(
            content=self._content,
            turns=1,
            metadata={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return FakeEngine(
        engine_id="test-engine",
        responses=["Hello from server!"],
        models=["test-model"],
    )


@pytest.fixture
def client(engine):
    app = create_app(engine, "test-model")
    return TestClient(app)


@pytest.fixture
def client_with_agent(engine):
    agent = FakeAgent()
    app = create_app(engine, "test-model", agent=agent)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    @pytest.mark.spec("REQ-server.routes-chat-basic")
    def test_basic_completion(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "Hello from" in data["choices"][0]["message"]["content"]

    @pytest.mark.spec("REQ-server.routes-chat-usage")
    def test_completion_has_usage(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert "usage" in data
        assert data["usage"]["total_tokens"] >= 0

    @pytest.mark.spec("REQ-server.routes-chat-id")
    def test_completion_has_id(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")

    @pytest.mark.spec("REQ-server.routes-chat-temperature")
    def test_custom_temperature(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
        })
        assert resp.status_code == 200

    @pytest.mark.spec("REQ-server.routes-chat-system-message")
    def test_with_system_message(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
        })
        assert resp.status_code == 200

    @pytest.mark.spec("REQ-server.routes-chat-tools")
    def test_with_tool_calls(self) -> None:
        engine = FakeEngine(
            responses=[""],
            tool_calls=[[
                {"id": "c1", "name": "calc", "arguments": '{"expr":"2+2"}'},
            ]],
        )
        app = create_app(engine, "test-model")
        c = TestClient(app)
        resp = c.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Calc"}],
            "tools": [{"type": "function", "function": {"name": "calc"}}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["tool_calls"] is not None

    @pytest.mark.spec("REQ-server.routes-chat-finish-reason")
    def test_finish_reason_default(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Agent mode
# ---------------------------------------------------------------------------


class TestAgentMode:
    @pytest.mark.spec("REQ-server.routes-agent-completion")
    def test_agent_mode_returns_agent_content(self, client_with_agent) -> None:
        resp = client_with_agent.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from fake agent"

    @pytest.mark.spec("REQ-server.routes-agent-conversation")
    def test_agent_with_prior_messages(self, client_with_agent) -> None:
        resp = client_with_agent.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    @pytest.mark.spec("REQ-server.routes-streaming-sse")
    def test_streaming_returns_sse(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        lines = resp.text.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("data:")]
        assert len(data_lines) > 0
        # Last data line should be [DONE]
        assert data_lines[-1].strip() == "data: [DONE]"

    @pytest.mark.spec("REQ-server.routes-streaming-content")
    def test_streaming_content_accumulates(self, client) -> None:
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        content = ""
        for line in resp.text.strip().split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                data = json.loads(line[5:].strip())
                choices = data.get("choices", [{}])
                delta_content = choices[0].get("delta", {}).get("content")
                if delta_content:
                    content += delta_content
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.spec("REQ-server.routes-list-models")
    def test_list_models(self, client) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    @pytest.mark.spec("REQ-server.routes-model-object-format")
    def test_model_object_format(self, client) -> None:
        resp = client.get("/v1/models")
        data = resp.json()
        model = data["data"][0]
        assert model["object"] == "model"
        assert "owned_by" in model

    @pytest.mark.spec("REQ-server.routes-multiple-models")
    def test_multiple_models(self) -> None:
        engine = FakeEngine(models=["model-a", "model-b", "model-c"])
        app = create_app(engine, "model-a")
        c = TestClient(app)
        resp = c.get("/v1/models")
        data = resp.json()
        assert len(data["data"]) == 3


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.spec("REQ-server.routes.health")
    @pytest.mark.spec("REQ-server.routes-health-ok")
    def test_healthy(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.spec("REQ-server.routes-health-503")
    def test_unhealthy(self) -> None:
        engine = FakeEngine(healthy=False)
        app = create_app(engine, "test-model")
        c = TestClient(app)
        resp = c.get("/health")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Server info endpoint
# ---------------------------------------------------------------------------


class TestServerInfo:
    @pytest.mark.spec("REQ-server.routes-info")
    def test_info_returns_model_and_engine(self) -> None:
        engine = FakeEngine()
        app = create_app(engine, "my-model", engine_name="ollama")
        c = TestClient(app)
        resp = c.get("/v1/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "my-model"
        assert data["engine"] == "ollama"

    @pytest.mark.spec("REQ-server.routes-info-agent")
    def test_info_includes_agent(self) -> None:
        engine = FakeEngine()
        agent = FakeAgent()
        app = create_app(engine, "my-model", agent=agent)
        c = TestClient(app)
        resp = c.get("/v1/info")
        data = resp.json()
        assert data["agent"] == "fake-agent"


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------


class TestCreateApp:
    @pytest.mark.spec("REQ-server.routes-app-state")
    def test_app_state_engine(self) -> None:
        engine = FakeEngine()
        app = create_app(engine, "test-model")
        assert app.state.engine is engine
        assert app.state.model == "test-model"

    @pytest.mark.spec("REQ-server.routes-app-state-agent")
    def test_app_state_agent(self) -> None:
        engine = FakeEngine()
        agent = FakeAgent()
        app = create_app(engine, "test-model", agent=agent)
        assert app.state.agent is agent

    @pytest.mark.spec("REQ-server.routes-app-state-no-agent")
    def test_app_state_no_agent(self) -> None:
        engine = FakeEngine()
        app = create_app(engine, "test-model")
        assert app.state.agent is None


# ---------------------------------------------------------------------------
# Channel endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Telemetry endpoint
# ---------------------------------------------------------------------------


class TestTelemetryEndpoint:
    @pytest.mark.spec("REQ-server.routes.telemetry")
    def test_telemetry_stats_endpoint_exists(self, client) -> None:
        """Telemetry stats endpoint should exist and return a response."""
        resp = client.get("/v1/telemetry/stats")
        # May return 200 or 500 depending on db availability, but should not 404
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# Channel endpoints
# ---------------------------------------------------------------------------


class TestChannelEndpoints:
    @pytest.mark.spec("REQ-server.routes-channels-no-bridge")
    def test_list_channels_no_bridge(self, client) -> None:
        resp = client.get("/v1/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channels"] == []

    @pytest.mark.spec("REQ-server.routes-channel-status-not-configured")
    def test_channel_status_not_configured(self, client) -> None:
        resp = client.get("/v1/channels/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_configured"

    @pytest.mark.spec("REQ-server.routes-channel-send-no-bridge")
    def test_channel_send_no_bridge(self, client) -> None:
        resp = client.post("/v1/channels/send", json={
            "channel": "test", "content": "hi"
        })
        assert resp.status_code == 503
