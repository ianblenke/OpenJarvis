"""Tests for the WhatsAppChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.whatsapp import WhatsAppChannel
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
def _register_whatsapp():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("whatsapp"):
        ChannelRegistry.register_value("whatsapp", WhatsAppChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_registry_key(self):
        assert ChannelRegistry.contains("whatsapp")

    def test_channel_id(self):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")
        assert ch.channel_id == "whatsapp"


class TestInit:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_defaults(self):
        ch = WhatsAppChannel()
        assert ch._token == ""
        assert ch._phone_number_id == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_constructor_token(self):
        ch = WhatsAppChannel(access_token="my-token", phone_number_id="12345")
        assert ch._token == "my-token"
        assert ch._phone_number_id == "12345"

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "env-token")
        monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "env-id")
        ch = WhatsAppChannel()
        assert ch._token == "env-token"
        assert ch._phone_number_id == "env-id"

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "env-token")
        monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "env-id")
        ch = WhatsAppChannel(
            access_token="explicit-token",
            phone_number_id="explicit-id",
        )
        assert ch._token == "explicit-token"
        assert ch._phone_number_id == "explicit-id"


class TestSend:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_send_success(self, monkeypatch):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("+1234567890", "Hello!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert "graph.facebook.com" in url
        assert "12345" in url

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_send_failure(self, monkeypatch):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("+1234567890", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_send_exception(self, monkeypatch):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("+1234567890", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_send_no_token(self):
        ch = WhatsAppChannel()
        result = ch.send("+1234567890", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = WhatsAppChannel(
            access_token="test-token",
            phone_number_id="12345",
            bus=bus,
        )

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("+1234567890", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_list_channels(self):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")
        assert ch.list_channels() == ["whatsapp"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_disconnected_initially(self):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_no_token_connect_error(self):
        ch = WhatsAppChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_on_message(self):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.whatsapp")
    def test_disconnect(self):
        ch = WhatsAppChannel(access_token="test-token", phone_number_id="12345")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
