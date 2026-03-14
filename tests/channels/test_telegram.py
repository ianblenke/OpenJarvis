"""Tests for the TelegramChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.telegram import TelegramChannel
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
def _register_telegram():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("telegram"):
        ChannelRegistry.register_value("telegram", TelegramChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_registry_key(self):
        assert ChannelRegistry.contains("telegram")

    def test_channel_id(self):
        ch = TelegramChannel(bot_token="test-token")
        assert ch.channel_id == "telegram"


class TestInit:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_defaults(self):
        ch = TelegramChannel()
        assert ch._token == ""
        assert ch._parse_mode == "Markdown"
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.telegram")
    def test_constructor_token(self):
        ch = TelegramChannel(bot_token="my-token")
        assert ch._token == "my-token"

    @pytest.mark.spec("REQ-channels.telegram")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        ch = TelegramChannel()
        assert ch._token == "env-token"

    @pytest.mark.spec("REQ-channels.telegram")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        ch = TelegramChannel(bot_token="explicit-token")
        assert ch._token == "explicit-token"


class TestSend:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_send_success(self, monkeypatch):
        ch = TelegramChannel(bot_token="123:ABC")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("12345678", "Hello!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert "api.telegram.org" in url
        assert "bot123:ABC" in url
        assert "sendMessage" in url
        payload = tracker.calls[0]["json"]
        assert payload["chat_id"] == "12345678"
        assert payload["text"] == "Hello!"
        assert payload["parse_mode"] == "Markdown"

    @pytest.mark.spec("REQ-channels.telegram")
    def test_send_failure(self, monkeypatch):
        ch = TelegramChannel(bot_token="123:ABC")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("12345678", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.telegram")
    def test_send_exception(self, monkeypatch):
        ch = TelegramChannel(bot_token="123:ABC")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("12345678", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.telegram")
    def test_send_no_token(self):
        ch = TelegramChannel()
        result = ch.send("12345678", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.telegram")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = TelegramChannel(bot_token="123:ABC", bus=bus)

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("12345678", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_list_channels(self):
        ch = TelegramChannel(bot_token="123:ABC")
        assert ch.list_channels() == ["telegram"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_disconnected_initially(self):
        ch = TelegramChannel(bot_token="123:ABC")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.telegram")
    def test_no_token_connect_error(self):
        ch = TelegramChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_on_message(self):
        ch = TelegramChannel(bot_token="123:ABC")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.telegram")
    def test_disconnect(self):
        ch = TelegramChannel(bot_token="123:ABC")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
