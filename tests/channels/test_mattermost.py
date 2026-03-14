"""Tests for the MattermostChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.mattermost import MattermostChannel
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


class _HttpPostTracker:
    """Typed callable that records httpx.post calls for assertion."""

    def __init__(self, response: _FakeHttpResponse) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.call_count = 0

    def __call__(self, url, **kwargs):
        self.call_count += 1
        self.calls.append({"url": url, **kwargs})
        return self._response


@pytest.fixture(autouse=True)
def _register_mattermost():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("mattermost"):
        ChannelRegistry.register_value("mattermost", MattermostChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_registry_key(self):
        assert ChannelRegistry.contains("mattermost")

    def test_channel_id(self):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")
        assert ch.channel_id == "mattermost"


class TestInit:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_defaults(self):
        ch = MattermostChannel()
        assert ch._url == ""
        assert ch._token == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_constructor_param(self):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")
        assert ch._url == "https://mattermost.example.com"
        assert ch._token == "test-token"

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_URL", "https://env.example.com")
        monkeypatch.setenv("MATTERMOST_TOKEN", "env-token")
        ch = MattermostChannel()
        assert ch._url == "https://env.example.com"
        assert ch._token == "env-token"

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_URL", "https://env.example.com")
        monkeypatch.setenv("MATTERMOST_TOKEN", "env-token")
        ch = MattermostChannel(
            url="https://explicit.example.com",
            token="explicit-token",
        )
        assert ch._url == "https://explicit.example.com"
        assert ch._token == "explicit-token"


class TestSend:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_send_success(self, monkeypatch):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("channel-id-123", "Hello!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert url.endswith("/api/v4/posts")

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_send_with_conversation_id(self, monkeypatch):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("channel-id-123", "Hello!", conversation_id="root-123")
        assert result is True
        payload = tracker.calls[0]["json"]
        assert "root_id" in payload
        assert payload["root_id"] == "root-123"

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_send_failure(self, monkeypatch):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("channel-id-123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_send_exception(self, monkeypatch):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("channel-id-123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_send_no_token(self):
        ch = MattermostChannel()
        result = ch.send("channel-id-123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = MattermostChannel(
            url="https://mattermost.example.com",
            token="test-token",
            bus=bus,
        )

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("channel-id-123", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_list_channels(self):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")
        assert ch.list_channels() == ["mattermost"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_disconnected_initially(self):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.mattermost")
    def test_no_url_connect_error(self):
        ch = MattermostChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_on_message(self):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.mattermost")
    def test_disconnect(self):
        ch = MattermostChannel(url="https://mattermost.example.com", token="test-token")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
