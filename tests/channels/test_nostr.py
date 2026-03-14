"""Tests for the NostrChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.nostr_channel import NostrChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_nostr():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("nostr"):
        ChannelRegistry.register_value("nostr", NostrChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.nostr")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("nostr")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = NostrChannel(private_key="aa" * 32)
        assert ch.channel_id == "nostr"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.nostr")
class TestInit:
    def test_defaults(self):
        ch = NostrChannel()
        assert ch._private_key == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_default_relay(self):
        ch = NostrChannel()
        assert "wss://relay.damus.io" in ch._relays

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = NostrChannel(private_key="bb" * 32, relays="wss://r1.example.com,wss://r2.example.com")
        assert ch._private_key == "bb" * 32
        assert len(ch._relays) == 2
        assert ch._relays[0] == "wss://r1.example.com"
        assert ch._relays[1] == "wss://r2.example.com"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("NOSTR_PRIVATE_KEY", "cc" * 32)
        monkeypatch.setenv("NOSTR_RELAYS", "wss://r1.example.com,wss://r2.example.com")
        ch = NostrChannel()
        assert ch._private_key == "cc" * 32
        assert len(ch._relays) == 2
        assert ch._relays[0] == "wss://r1.example.com"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("NOSTR_PRIVATE_KEY", "cc" * 32)
        monkeypatch.setenv("NOSTR_RELAYS", "wss://env.example.com")
        ch = NostrChannel(private_key="dd" * 32, relays="wss://explicit.example.com")
        assert ch._private_key == "dd" * 32
        assert ch._relays == ["wss://explicit.example.com"]

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_relays_strips_whitespace(self):
        ch = NostrChannel(relays="  wss://r1.example.com , wss://r2.example.com  ")
        assert ch._relays == ["wss://r1.example.com", "wss://r2.example.com"]

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_relays_skips_empty(self):
        ch = NostrChannel(relays="wss://r1.example.com,,wss://r2.example.com,")
        assert ch._relays == ["wss://r1.example.com", "wss://r2.example.com"]

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = NostrChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.nostr")
class TestConnect:
    def test_connect_no_key_sets_error(self):
        ch = NostrChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = NostrChannel(private_key="aa" * 32)
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pynostr" or name.startswith("pynostr."):
                raise ImportError("No module named 'pynostr'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pynostr"):
            ch.connect()


@pytest.mark.spec("REQ-channels.nostr")
class TestDisconnect:
    def test_disconnect(self):
        ch = NostrChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.nostr")
class TestStatus:
    def test_disconnected_initially(self):
        ch = NostrChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.nostr")
class TestListChannels:
    def test_list_channels(self):
        ch = NostrChannel()
        assert ch.list_channels() == ["nostr"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.nostr")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = NostrChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = NostrChannel()

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


@pytest.mark.spec("REQ-channels.nostr")
class TestSend:
    def test_send_no_key(self):
        ch = NostrChannel()
        result = ch.send("", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.nostr")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = NostrChannel(bus=bus)
        ch._publish_sent("recipient-npub", "hello nostr", "conv-456")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "recipient-npub"
        assert event.data["content"] == "hello nostr"
        assert event.data["conversation_id"] == "conv-456"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = NostrChannel()
        ch._publish_sent("", "hello", "")
