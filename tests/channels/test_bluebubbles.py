"""Tests for the BlueBubblesChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.bluebubbles import BlueBubblesChannel
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
def _register_bluebubbles():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("bluebubbles"):
        ChannelRegistry.register_value("bluebubbles", BlueBubblesChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_registry_key(self):
        assert ChannelRegistry.contains("bluebubbles")

    def test_channel_id(self):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")
        assert ch.channel_id == "bluebubbles"


class TestInit:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_defaults(self):
        ch = BlueBubblesChannel()
        assert ch._url == ""
        assert ch._password == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_constructor_param(self):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")
        assert ch._url == "http://localhost:1234"
        assert ch._password == "test-pass"

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_URL", "http://env:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "env-pass")
        ch = BlueBubblesChannel()
        assert ch._url == "http://env:1234"
        assert ch._password == "env-pass"

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_URL", "http://env:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "env-pass")
        ch = BlueBubblesChannel(
            url="http://explicit:1234",
            password="explicit-pass",
        )
        assert ch._url == "http://explicit:1234"
        assert ch._password == "explicit-pass"


class TestSend:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_send_success(self, monkeypatch):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("iMessage;+;chat123", "Hello!")
        assert result is True
        assert tracker.call_count == 1
        assert tracker.calls[0]["params"] == {"password": "test-pass"}
        payload = tracker.calls[0]["json"]
        assert "chatGuid" in payload
        assert payload["chatGuid"] == "iMessage;+;chat123"
        assert "message" in payload
        assert payload["message"] == "Hello!"

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_send_failure(self, monkeypatch):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("iMessage;+;chat123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_send_exception(self, monkeypatch):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("iMessage;+;chat123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_send_no_token(self):
        ch = BlueBubblesChannel()
        result = ch.send("iMessage;+;chat123", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = BlueBubblesChannel(
            url="http://localhost:1234",
            password="test-pass",
            bus=bus,
        )

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("iMessage;+;chat123", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_list_channels(self):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")
        assert ch.list_channels() == ["bluebubbles"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_disconnected_initially(self):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_no_url_connect_error(self):
        ch = BlueBubblesChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_on_message(self):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.bluebubbles")
    def test_disconnect(self):
        ch = BlueBubblesChannel(url="http://localhost:1234", password="test-pass")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
