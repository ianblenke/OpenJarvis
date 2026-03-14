"""Tests for the XMPPChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.xmpp_channel import XMPPChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_xmpp():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("xmpp"):
        ChannelRegistry.register_value("xmpp", XMPPChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.xmpp")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("xmpp")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = XMPPChannel(jid="bot@example.com", password="pass")
        assert ch.channel_id == "xmpp"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.xmpp")
class TestInit:
    def test_defaults(self):
        ch = XMPPChannel()
        assert ch._jid == ""
        assert ch._password == ""
        assert ch._server == ""
        assert ch._port == 5222
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = XMPPChannel(
            jid="bot@example.com",
            password="pass",
            server="xmpp.example.com",
            port=5223,
        )
        assert ch._jid == "bot@example.com"
        assert ch._password == "pass"
        assert ch._server == "xmpp.example.com"
        assert ch._port == 5223

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "bot@env.com")
        monkeypatch.setenv("XMPP_PASSWORD", "env-pass")
        monkeypatch.setenv("XMPP_SERVER", "xmpp.env.com")
        monkeypatch.setenv("XMPP_PORT", "5223")
        ch = XMPPChannel()
        assert ch._jid == "bot@env.com"
        assert ch._password == "env-pass"
        assert ch._server == "xmpp.env.com"
        assert ch._port == 5223

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("XMPP_JID", "bot@env.com")
        monkeypatch.setenv("XMPP_PASSWORD", "env-pass")
        ch = XMPPChannel(jid="bot@explicit.com", password="explicit-pass")
        assert ch._jid == "bot@explicit.com"
        assert ch._password == "explicit-pass"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_port_from_env_as_int(self, monkeypatch):
        monkeypatch.setenv("XMPP_PORT", "5269")
        ch = XMPPChannel()
        assert ch._port == 5269
        assert isinstance(ch._port, int)

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_default_port(self):
        ch = XMPPChannel()
        assert ch._port == 5222

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = XMPPChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.xmpp")
class TestConnect:
    def test_connect_no_jid_sets_error(self):
        ch = XMPPChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_password_sets_error(self):
        ch = XMPPChannel(jid="bot@example.com")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_jid_only_sets_error(self):
        ch = XMPPChannel(jid="bot@example.com")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = XMPPChannel(jid="bot@example.com", password="pass")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "slixmpp" or name.startswith("slixmpp."):
                raise ImportError("No module named 'slixmpp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="slixmpp"):
            ch.connect()


@pytest.mark.spec("REQ-channels.xmpp")
class TestDisconnect:
    def test_disconnect(self):
        ch = XMPPChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.xmpp")
class TestStatus:
    def test_disconnected_initially(self):
        ch = XMPPChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.xmpp")
class TestListChannels:
    def test_list_channels(self):
        ch = XMPPChannel()
        assert ch.list_channels() == ["xmpp"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.xmpp")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = XMPPChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = XMPPChannel()

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


@pytest.mark.spec("REQ-channels.xmpp")
class TestSend:
    def test_send_no_jid(self):
        ch = XMPPChannel()
        result = ch.send("user@example.com", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_no_password(self):
        ch = XMPPChannel(jid="bot@example.com")
        result = ch.send("user@example.com", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.xmpp")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = XMPPChannel(bus=bus)
        ch._publish_sent("user@example.com", "hello xmpp", "conv-x1")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "user@example.com"
        assert event.data["content"] == "hello xmpp"
        assert event.data["conversation_id"] == "conv-x1"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = XMPPChannel()
        ch._publish_sent("user@example.com", "hello", "")
