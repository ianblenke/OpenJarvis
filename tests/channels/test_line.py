"""Tests for the LineChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.line_channel import LineChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_line():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("line"):
        ChannelRegistry.register_value("line", LineChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.line")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("line")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = LineChannel(channel_access_token="tok", channel_secret="sec")
        assert ch.channel_id == "line"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.line")
class TestInit:
    def test_defaults(self):
        ch = LineChannel()
        assert ch._channel_access_token == ""
        assert ch._channel_secret == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = LineChannel(channel_access_token="my-tok", channel_secret="my-sec")
        assert ch._channel_access_token == "my-tok"
        assert ch._channel_secret == "my-sec"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "env-tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "env-sec")
        ch = LineChannel()
        assert ch._channel_access_token == "env-tok"
        assert ch._channel_secret == "env-sec"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "env-tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "env-sec")
        ch = LineChannel(
            channel_access_token="explicit-tok",
            channel_secret="explicit-sec",
        )
        assert ch._channel_access_token == "explicit-tok"
        assert ch._channel_secret == "explicit-sec"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = LineChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.line")
class TestConnect:
    def test_connect_no_token_sets_error(self):
        ch = LineChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_secret_sets_error(self):
        ch = LineChannel(channel_access_token="tok")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_token_with_secret_sets_error(self):
        ch = LineChannel(channel_secret="sec")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = LineChannel(channel_access_token="tok", channel_secret="sec")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "linebot" or name.startswith("linebot."):
                raise ImportError("No module named 'linebot'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="line-bot-sdk"):
            ch.connect()


@pytest.mark.spec("REQ-channels.line")
class TestDisconnect:
    def test_disconnect(self):
        ch = LineChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.line")
class TestStatus:
    def test_disconnected_initially(self):
        ch = LineChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.line")
class TestListChannels:
    def test_list_channels(self):
        ch = LineChannel()
        assert ch.list_channels() == ["line"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.line")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = LineChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = LineChannel()

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


@pytest.mark.spec("REQ-channels.line")
class TestSend:
    def test_send_no_token(self):
        ch = LineChannel()
        result = ch.send("user-id", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_secret_only_no_token(self):
        ch = LineChannel(channel_secret="sec")
        result = ch.send("user-id", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.line")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = LineChannel(bus=bus)
        ch._publish_sent("user-id", "hello line", "conv-l1")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "user-id"
        assert event.data["content"] == "hello line"
        assert event.data["conversation_id"] == "conv-l1"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = LineChannel()
        ch._publish_sent("user-id", "hello", "")
