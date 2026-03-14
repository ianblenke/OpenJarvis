"""Tests for the ViberChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.viber_channel import ViberChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_viber():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("viber"):
        ChannelRegistry.register_value("viber", ViberChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.viber")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("viber")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = ViberChannel(auth_token="tok")
        assert ch.channel_id == "viber"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.viber")
class TestInit:
    def test_defaults(self):
        ch = ViberChannel()
        assert ch._auth_token == ""
        assert ch._name == "OpenJarvis"
        assert ch._avatar == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = ViberChannel(auth_token="my-tok", name="TestBot", avatar="https://img.example.com/bot.png")
        assert ch._auth_token == "my-tok"
        assert ch._name == "TestBot"
        assert ch._avatar == "https://img.example.com/bot.png"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("VIBER_AUTH_TOKEN", "env-tok")
        monkeypatch.setenv("VIBER_BOT_NAME", "EnvBot")
        monkeypatch.setenv("VIBER_BOT_AVATAR", "https://env.example.com/avatar.png")
        ch = ViberChannel()
        assert ch._auth_token == "env-tok"
        assert ch._name == "EnvBot"
        assert ch._avatar == "https://env.example.com/avatar.png"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("VIBER_AUTH_TOKEN", "env-tok")
        monkeypatch.setenv("VIBER_BOT_NAME", "EnvBot")
        ch = ViberChannel(auth_token="explicit-tok", name="ExplicitBot")
        assert ch._auth_token == "explicit-tok"
        assert ch._name == "ExplicitBot"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = ViberChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.viber")
class TestConnect:
    def test_connect_no_token_sets_error(self):
        ch = ViberChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = ViberChannel(auth_token="tok")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "viberbot" or name.startswith("viberbot."):
                raise ImportError("No module named 'viberbot'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="viberbot"):
            ch.connect()


@pytest.mark.spec("REQ-channels.viber")
class TestDisconnect:
    def test_disconnect(self):
        ch = ViberChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.viber")
class TestStatus:
    def test_disconnected_initially(self):
        ch = ViberChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.viber")
class TestListChannels:
    def test_list_channels(self):
        ch = ViberChannel()
        assert ch.list_channels() == ["viber"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.viber")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = ViberChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = ViberChannel()

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


@pytest.mark.spec("REQ-channels.viber")
class TestSend:
    def test_send_no_token(self):
        ch = ViberChannel()
        result = ch.send("user-id", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.viber")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = ViberChannel(bus=bus)
        ch._publish_sent("user-id", "hello viber", "conv-v1")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "user-id"
        assert event.data["content"] == "hello viber"
        assert event.data["conversation_id"] == "conv-v1"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = ViberChannel()
        ch._publish_sent("user-id", "hello", "")
