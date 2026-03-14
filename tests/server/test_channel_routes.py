"""Tests for /v1/channels endpoints.

Requires the ``[server]`` optional extra (fastapi, uvicorn, pydantic).
Skipped automatically when those packages are not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="openjarvis[server] not installed")

from openjarvis.channels._stubs import ChannelStatus  # noqa: E402
from tests.fixtures.engines import FakeEngine  # noqa: E402


class _FakeChannelBridge:
    """Typed fake channel bridge implementing the protocol used by routes."""

    def __init__(self) -> None:
        self._send_result = True
        self.send_count = 0

    def status(self) -> ChannelStatus:
        return ChannelStatus.CONNECTED

    def list_channels(self) -> list[str]:
        return ["slack", "discord", "telegram"]

    def send(self, channel: str, content: str) -> bool:
        self.send_count += 1
        return self._send_result


@pytest.fixture
def fake_engine():
    """FakeEngine for app creation."""
    return FakeEngine(
        engine_id="mock",
        responses=["Hello!"],
        models=["test-model"],
        healthy=True,
    )


@pytest.fixture
def fake_bridge():
    """Fake channel bridge."""
    return _FakeChannelBridge()


@pytest.fixture
def app_with_bridge(fake_engine, fake_bridge):
    """FastAPI app with channel bridge configured."""
    from openjarvis.server.app import create_app

    return create_app(
        fake_engine, "test-model",
        channel_bridge=fake_bridge,
    )


@pytest.fixture
def app_without_bridge(fake_engine):
    """FastAPI app without channel bridge."""
    from openjarvis.server.app import create_app

    return create_app(fake_engine, "test-model")


@pytest.fixture
def client_with_bridge(app_with_bridge):
    """Test client with channel bridge."""
    from starlette.testclient import TestClient

    return TestClient(app_with_bridge)


@pytest.fixture
def client_without_bridge(app_without_bridge):
    """Test client without channel bridge."""
    from starlette.testclient import TestClient

    return TestClient(app_without_bridge)


class TestListChannels:
    @pytest.mark.spec("REQ-server.routes.channels")
    def test_list_channels_with_bridge(self, client_with_bridge, fake_bridge):
        resp = client_with_bridge.get("/v1/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channels"] == ["slack", "discord", "telegram"]
        assert data["status"] == "connected"

    @pytest.mark.spec("REQ-server.routes.channels")
    def test_list_channels_no_bridge(self, client_without_bridge):
        resp = client_without_bridge.get("/v1/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channels"] == []
        assert "not configured" in data.get("message", "").lower()


class TestChannelSend:
    @pytest.mark.spec("REQ-server.routes.channels")
    def test_send_success(self, client_with_bridge, fake_bridge):
        resp = client_with_bridge.post(
            "/v1/channels/send",
            json={"channel": "slack", "content": "Hello from Jarvis"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "sent"
        assert data["channel"] == "slack"
        assert fake_bridge.send_count == 1

    @pytest.mark.spec("REQ-server.routes.channels")
    def test_send_no_bridge(self, client_without_bridge):
        resp = client_without_bridge.post(
            "/v1/channels/send",
            json={"channel": "slack", "content": "Hello"},
        )
        assert resp.status_code == 503

    @pytest.mark.spec("REQ-server.routes.channels")
    def test_send_missing_channel(self, client_with_bridge):
        resp = client_with_bridge.post(
            "/v1/channels/send",
            json={"content": "Hello"},
        )
        assert resp.status_code == 400

    @pytest.mark.spec("REQ-server.routes.channels")
    def test_send_missing_content(self, client_with_bridge):
        resp = client_with_bridge.post(
            "/v1/channels/send",
            json={"channel": "slack"},
        )
        assert resp.status_code == 400

    @pytest.mark.spec("REQ-server.routes.channels")
    def test_send_failure(self, client_with_bridge, fake_bridge):
        fake_bridge._send_result = False
        resp = client_with_bridge.post(
            "/v1/channels/send",
            json={"channel": "slack", "content": "Hello"},
        )
        assert resp.status_code == 502


class TestChannelStatus:
    @pytest.mark.spec("REQ-server.routes.channels")
    def test_status_with_bridge(self, client_with_bridge):
        resp = client_with_bridge.get("/v1/channels/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "connected"

    @pytest.mark.spec("REQ-server.routes.channels")
    def test_status_no_bridge(self, client_without_bridge):
        resp = client_without_bridge.get("/v1/channels/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_configured"
