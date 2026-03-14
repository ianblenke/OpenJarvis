"""Tests for the DiscordChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.discord_channel import DiscordChannel
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
def _register_discord():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("discord"):
        ChannelRegistry.register_value("discord", DiscordChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.discord")
    def test_registry_key(self):
        assert ChannelRegistry.contains("discord")

    def test_channel_id(self):
        ch = DiscordChannel(bot_token="test-token")
        assert ch.channel_id == "discord"


class TestInit:
    @pytest.mark.spec("REQ-channels.discord")
    def test_defaults(self):
        ch = DiscordChannel()
        assert ch._token == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.discord")
    def test_constructor_token(self):
        ch = DiscordChannel(bot_token="my-token")
        assert ch._token == "my-token"

    @pytest.mark.spec("REQ-channels.discord")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "env-token")
        ch = DiscordChannel()
        assert ch._token == "env-token"

    @pytest.mark.spec("REQ-channels.discord")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "env-token")
        ch = DiscordChannel(bot_token="explicit-token")
        assert ch._token == "explicit-token"


class TestSend:
    @pytest.mark.spec("REQ-channels.discord")
    def test_send_success(self, monkeypatch):
        ch = DiscordChannel(bot_token="my-bot-token")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("987654321", "Hello Discord!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert "discord.com/api/v10/channels/987654321/messages" in url
        headers = tracker.calls[0]["headers"]
        assert headers["Authorization"] == "Bot my-bot-token"
        payload = tracker.calls[0]["json"]
        assert payload["content"] == "Hello Discord!"

    @pytest.mark.spec("REQ-channels.discord")
    def test_send_with_conversation_id(self, monkeypatch):
        ch = DiscordChannel(bot_token="my-bot-token")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        ch.send("987654321", "Reply!", conversation_id="msg-123")
        payload = tracker.calls[0]["json"]
        assert payload["message_reference"] == {"message_id": "msg-123"}

    @pytest.mark.spec("REQ-channels.discord")
    def test_send_failure(self, monkeypatch):
        ch = DiscordChannel(bot_token="my-bot-token")

        fake_response = _FakeHttpResponse(status_code=403, text="Missing Permissions")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("987654321", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.discord")
    def test_send_exception(self, monkeypatch):
        ch = DiscordChannel(bot_token="my-bot-token")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("987654321", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.discord")
    def test_send_no_token(self):
        ch = DiscordChannel()
        result = ch.send("987654321", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.discord")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = DiscordChannel(bot_token="my-bot-token", bus=bus)

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("987654321", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.discord")
    def test_list_channels(self):
        ch = DiscordChannel(bot_token="my-bot-token")
        assert ch.list_channels() == ["discord"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.discord")
    def test_disconnected_initially(self):
        ch = DiscordChannel(bot_token="my-bot-token")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.discord")
    def test_no_token_connect_error(self):
        ch = DiscordChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.discord")
    def test_on_message(self):
        ch = DiscordChannel(bot_token="my-bot-token")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.discord")
    def test_disconnect(self):
        ch = DiscordChannel(bot_token="my-bot-token")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
