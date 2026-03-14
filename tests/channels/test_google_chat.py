"""Tests for the GoogleChatChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.google_chat import GoogleChatChannel
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
def _register_google_chat():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("google_chat"):
        ChannelRegistry.register_value("google_chat", GoogleChatChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_registry_key(self):
        assert ChannelRegistry.contains("google_chat")

    def test_channel_id(self):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")
        assert ch.channel_id == "google_chat"


class TestInit:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_defaults(self):
        ch = GoogleChatChannel()
        assert ch._webhook_url == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_constructor_url(self):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")
        assert ch._webhook_url == "https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy"

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv(
            "GOOGLE_CHAT_WEBHOOK_URL",
            "https://chat.googleapis.com/v1/spaces/env/messages?key=env",
        )
        ch = GoogleChatChannel()
        assert ch._webhook_url == "https://chat.googleapis.com/v1/spaces/env/messages?key=env"

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv(
            "GOOGLE_CHAT_WEBHOOK_URL",
            "https://chat.googleapis.com/v1/spaces/env/messages?key=env",
        )
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/explicit/messages?key=explicit")
        assert ch._webhook_url == "https://chat.googleapis.com/v1/spaces/explicit/messages?key=explicit"


class TestSend:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_send_success(self, monkeypatch):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("space", "Hello!")
        assert result is True
        assert tracker.call_count == 1

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_send_failure(self, monkeypatch):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("space", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_send_exception(self, monkeypatch):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("space", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_send_no_url(self):
        ch = GoogleChatChannel()
        result = ch.send("space", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = GoogleChatChannel(
            webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy",
            bus=bus,
        )

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("space", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_list_channels(self):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")
        assert ch.list_channels() == ["google_chat"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_disconnected_initially(self):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.google-chat")
    def test_no_url_connect_error(self):
        ch = GoogleChatChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_on_message(self):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.google-chat")
    def test_disconnect(self):
        ch = GoogleChatChannel(webhook_url="https://chat.googleapis.com/v1/spaces/xxx/messages?key=yyy")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
