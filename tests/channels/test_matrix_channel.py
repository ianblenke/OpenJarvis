"""Tests for the MatrixChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.matrix_channel import MatrixChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

# ---------------------------------------------------------------------------
# Typed fake for httpx.Response
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    """Typed fake for httpx.Response replacing MagicMock."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.text = text


# ---------------------------------------------------------------------------
# Call-tracking helpers
# ---------------------------------------------------------------------------


class _HttpPutTracker:
    """Typed callable that records httpx.put calls for assertion."""

    def __init__(self, response: _FakeHttpResponse) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.call_count = 0

    def __call__(self, url, **kwargs):
        self.call_count += 1
        self.calls.append({"url": url, **kwargs})
        return self._response


@pytest.fixture(autouse=True)
def _register_matrix():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("matrix"):
        ChannelRegistry.register_value("matrix", MatrixChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_registry_key(self):
        assert ChannelRegistry.contains("matrix")

    def test_channel_id(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )
        assert ch.channel_id == "matrix"


class TestInit:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_defaults(self):
        ch = MatrixChannel()
        assert ch._homeserver == ""
        assert ch._access_token == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.matrix")
    def test_constructor_param(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )
        assert ch._homeserver == "https://matrix.example.com"
        assert ch._access_token == "test-token"

    @pytest.mark.spec("REQ-channels.matrix")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://env.example.com")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "env-token")
        ch = MatrixChannel()
        assert ch._homeserver == "https://env.example.com"
        assert ch._access_token == "env-token"

    @pytest.mark.spec("REQ-channels.matrix")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://env.example.com")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "env-token")
        ch = MatrixChannel(
            homeserver="https://explicit.example.com",
            access_token="explicit-token",
        )
        assert ch._homeserver == "https://explicit.example.com"
        assert ch._access_token == "explicit-token"


class TestSend:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_send_success(self, monkeypatch):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPutTracker(fake_response)
        monkeypatch.setattr("httpx.put", tracker)

        result = ch.send("!room123:example.com", "Hello!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert "/_matrix/client/v3/rooms/" in url
        assert "!room123:example.com" in url

    @pytest.mark.spec("REQ-channels.matrix")
    def test_send_failure(self, monkeypatch):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.put", lambda *a, **kw: fake_response)

        result = ch.send("!room123:example.com", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.matrix")
    def test_send_exception(self, monkeypatch):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.put", _raise)

        result = ch.send("!room123:example.com", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.matrix")
    def test_send_no_token(self):
        ch = MatrixChannel()
        result = ch.send("!room123:example.com", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.matrix")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
            bus=bus,
        )

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.put", lambda *a, **kw: fake_response)

        ch.send("!room123:example.com", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_list_channels(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )
        assert ch.list_channels() == ["matrix"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_disconnected_initially(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.matrix")
    def test_no_homeserver_connect_error(self):
        ch = MatrixChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_on_message(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.matrix")
    def test_disconnect(self):
        ch = MatrixChannel(
            homeserver="https://matrix.example.com",
            access_token="test-token",
        )
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
