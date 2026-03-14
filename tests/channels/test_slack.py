"""Tests for the SlackChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.slack import SlackChannel
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
        json_data: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self) -> dict:
        return self._json_data


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
def _register_slack():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("slack"):
        ChannelRegistry.register_value("slack", SlackChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.slack")
    def test_registry_key(self):
        assert ChannelRegistry.contains("slack")

    def test_channel_id(self):
        ch = SlackChannel(bot_token="xoxb-test")
        assert ch.channel_id == "slack"


class TestInit:
    @pytest.mark.spec("REQ-channels.slack")
    def test_defaults(self):
        ch = SlackChannel()
        assert ch._token == ""
        assert ch._app_token == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.slack")
    def test_constructor_token(self):
        ch = SlackChannel(bot_token="xoxb-my-token")
        assert ch._token == "xoxb-my-token"

    @pytest.mark.spec("REQ-channels.slack")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
        ch = SlackChannel()
        assert ch._token == "xoxb-env"

    @pytest.mark.spec("REQ-channels.slack")
    def test_app_token_env_var(self, monkeypatch):
        monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-env")
        ch = SlackChannel()
        assert ch._app_token == "xapp-env"

    @pytest.mark.spec("REQ-channels.slack")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
        ch = SlackChannel(bot_token="xoxb-explicit")
        assert ch._token == "xoxb-explicit"


class TestSend:
    @pytest.mark.spec("REQ-channels.slack")
    def test_send_success(self, monkeypatch):
        ch = SlackChannel(bot_token="xoxb-test")

        fake_response = _FakeHttpResponse(
            status_code=200, json_data={"ok": True},
        )
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("C1234567890", "Hello Slack!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert "slack.com/api/chat.postMessage" in url
        headers = tracker.calls[0]["headers"]
        assert headers["Authorization"] == "Bearer xoxb-test"
        payload = tracker.calls[0]["json"]
        assert payload["channel"] == "C1234567890"
        assert payload["text"] == "Hello Slack!"

    @pytest.mark.spec("REQ-channels.slack")
    def test_send_with_thread(self, monkeypatch):
        ch = SlackChannel(bot_token="xoxb-test")

        fake_response = _FakeHttpResponse(
            status_code=200, json_data={"ok": True},
        )
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        ch.send("C123", "Reply", conversation_id="1234567890.123456")
        payload = tracker.calls[0]["json"]
        assert payload["thread_ts"] == "1234567890.123456"

    @pytest.mark.spec("REQ-channels.slack")
    def test_send_api_error(self, monkeypatch):
        ch = SlackChannel(bot_token="xoxb-test")

        fake_response = _FakeHttpResponse(
            status_code=200,
            json_data={"ok": False, "error": "channel_not_found"},
        )
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("C123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.slack")
    def test_send_http_failure(self, monkeypatch):
        ch = SlackChannel(bot_token="xoxb-test")

        fake_response = _FakeHttpResponse(status_code=500)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("C123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.slack")
    def test_send_exception(self, monkeypatch):
        ch = SlackChannel(bot_token="xoxb-test")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("C123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.slack")
    def test_send_no_token(self):
        ch = SlackChannel()
        result = ch.send("C123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.slack")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = SlackChannel(bot_token="xoxb-test", bus=bus)

        fake_response = _FakeHttpResponse(
            status_code=200, json_data={"ok": True},
        )
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("C123", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.slack")
    def test_list_channels(self):
        ch = SlackChannel(bot_token="xoxb-test")
        assert ch.list_channels() == ["slack"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.slack")
    def test_disconnected_initially(self):
        ch = SlackChannel(bot_token="xoxb-test")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.slack")
    def test_no_token_connect_error(self):
        ch = SlackChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.slack")
    def test_on_message(self):
        ch = SlackChannel(bot_token="xoxb-test")
        received = []
        def handler(msg):
            return received.append(msg)
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.slack")
    def test_disconnect(self):
        ch = SlackChannel(bot_token="xoxb-test")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
