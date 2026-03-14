"""Tests for the MessengerChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.messenger_channel import MessengerChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_messenger():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("messenger"):
        ChannelRegistry.register_value("messenger", MessengerChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("messenger")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = MessengerChannel(access_token="tok")
        assert ch.channel_id == "messenger"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestInit:
    def test_defaults(self):
        ch = MessengerChannel()
        assert ch._access_token == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_token(self):
        ch = MessengerChannel(access_token="my-page-token")
        assert ch._access_token == "my-page-token"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("MESSENGER_ACCESS_TOKEN", "env-tok")
        ch = MessengerChannel()
        assert ch._access_token == "env-tok"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MESSENGER_ACCESS_TOKEN", "env-tok")
        ch = MessengerChannel(access_token="explicit-tok")
        assert ch._access_token == "explicit-tok"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = MessengerChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestConnect:
    def test_connect_no_token_sets_error(self):
        ch = MessengerChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = MessengerChannel(access_token="tok")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pymessenger" or name.startswith("pymessenger."):
                raise ImportError("No module named 'pymessenger'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pymessenger"):
            ch.connect()


@pytest.mark.spec("REQ-channels.messenger")
class TestDisconnect:
    def test_disconnect(self):
        ch = MessengerChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestStatus:
    def test_disconnected_initially(self):
        ch = MessengerChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.messenger")
class TestListChannels:
    def test_list_channels(self):
        ch = MessengerChannel()
        assert ch.list_channels() == ["messenger"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = MessengerChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = MessengerChannel()

        def h1(msg):
            pass

        def h2(msg):
            pass

        ch.on_message(h1)
        ch.on_message(h2)
        assert len(ch._handlers) == 2


# ---------------------------------------------------------------------------
# Send (no credentials)
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestSend:
    def test_send_no_credentials(self):
        ch = MessengerChannel()
        result = ch.send("user-psid", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.messenger")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = MessengerChannel(bus=bus)
        ch._publish_sent("user-psid", "hello world", "conv-123")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "user-psid"
        assert event.data["content"] == "hello world"
        assert event.data["conversation_id"] == "conv-123"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = MessengerChannel()
        ch._publish_sent("user-psid", "hello", "")
