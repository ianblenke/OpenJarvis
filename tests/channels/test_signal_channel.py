"""Tests for the SignalChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.signal_channel import SignalChannel
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
def _register_signal():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("signal"):
        ChannelRegistry.register_value("signal", SignalChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.signal")
    def test_registry_key(self):
        assert ChannelRegistry.contains("signal")

    def test_channel_id(self):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")
        assert ch.channel_id == "signal"


class TestInit:
    @pytest.mark.spec("REQ-channels.signal")
    def test_defaults(self):
        ch = SignalChannel()
        assert ch._api_url == ""
        assert ch._phone_number == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.signal")
    def test_constructor_params(self):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")
        assert ch._api_url == "http://localhost:8080"
        assert ch._phone_number == "+1234567890"

    @pytest.mark.spec("REQ-channels.signal")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_API_URL", "http://env-signal:8080")
        monkeypatch.setenv("SIGNAL_PHONE_NUMBER", "+9876543210")
        ch = SignalChannel()
        assert ch._api_url == "http://env-signal:8080"
        assert ch._phone_number == "+9876543210"

    @pytest.mark.spec("REQ-channels.signal")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_API_URL", "http://env-signal:8080")
        monkeypatch.setenv("SIGNAL_PHONE_NUMBER", "+9876543210")
        ch = SignalChannel(
            api_url="http://explicit:8080",
            phone_number="+1111111111",
        )
        assert ch._api_url == "http://explicit:8080"
        assert ch._phone_number == "+1111111111"


class TestSend:
    @pytest.mark.spec("REQ-channels.signal")
    def test_send_success(self, monkeypatch):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("+0987654321", "Hello!")
        assert result is True
        assert tracker.call_count == 1

    @pytest.mark.spec("REQ-channels.signal")
    def test_send_failure(self, monkeypatch):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("+0987654321", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.signal")
    def test_send_exception(self, monkeypatch):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("+0987654321", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.signal")
    def test_send_no_config(self):
        ch = SignalChannel()
        result = ch.send("+0987654321", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.signal")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = SignalChannel(
            api_url="http://localhost:8080",
            phone_number="+1234567890",
            bus=bus,
        )

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("+0987654321", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.signal")
    def test_list_channels(self):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")
        assert ch.list_channels() == ["signal"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.signal")
    def test_disconnected_initially(self):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.signal")
    def test_no_config_connect_error(self):
        ch = SignalChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.signal")
    def test_on_message(self):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.signal")
    def test_disconnect(self):
        ch = SignalChannel(api_url="http://localhost:8080", phone_number="+1234567890")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
