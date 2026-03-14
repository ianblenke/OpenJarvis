"""Tests for the ZulipChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.zulip_channel import ZulipChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_zulip():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("zulip"):
        ChannelRegistry.register_value("zulip", ZulipChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("zulip")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = ZulipChannel(email="bot@zulip.com", api_key="key", site="https://z.com")
        assert ch.channel_id == "zulip"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestInit:
    def test_defaults(self):
        ch = ZulipChannel()
        assert ch._email == ""
        assert ch._api_key == ""
        assert ch._site == ""
        assert ch._zuliprc == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_explicit_creds(self):
        ch = ZulipChannel(email="bot@zulip.com", api_key="key123", site="https://myorg.zulipchat.com")
        assert ch._email == "bot@zulip.com"
        assert ch._api_key == "key123"
        assert ch._site == "https://myorg.zulipchat.com"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_zuliprc(self):
        ch = ZulipChannel(zuliprc="/home/bot/.zuliprc")
        assert ch._zuliprc == "/home/bot/.zuliprc"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ZULIP_EMAIL", "bot@zulip.com")
        monkeypatch.setenv("ZULIP_API_KEY", "env-key")
        monkeypatch.setenv("ZULIP_SITE", "https://env.zulipchat.com")
        ch = ZulipChannel()
        assert ch._email == "bot@zulip.com"
        assert ch._api_key == "env-key"
        assert ch._site == "https://env.zulipchat.com"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_zuliprc(self, monkeypatch):
        monkeypatch.setenv("ZULIP_RC", "/path/to/zuliprc")
        ch = ZulipChannel()
        assert ch._zuliprc == "/path/to/zuliprc"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ZULIP_EMAIL", "env@zulip.com")
        monkeypatch.setenv("ZULIP_API_KEY", "env-key")
        monkeypatch.setenv("ZULIP_SITE", "https://env.zulipchat.com")
        ch = ZulipChannel(email="explicit@zulip.com", api_key="explicit-key", site="https://explicit.com")
        assert ch._email == "explicit@zulip.com"
        assert ch._api_key == "explicit-key"
        assert ch._site == "https://explicit.com"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = ZulipChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestConnect:
    def test_connect_no_credentials_sets_error(self):
        ch = ZulipChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_partial_creds_sets_error(self):
        ch = ZulipChannel(email="bot@zulip.com")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_email_and_key_no_site_sets_error(self):
        ch = ZulipChannel(email="bot@zulip.com", api_key="key")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error_explicit_creds(self, monkeypatch):
        ch = ZulipChannel(email="bot@zulip.com", api_key="key", site="https://z.com")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "zulip" or name.startswith("zulip."):
                raise ImportError("No module named 'zulip'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="zulip"):
            ch.connect()

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error_zuliprc(self, monkeypatch):
        ch = ZulipChannel(zuliprc="/path/to/zuliprc")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "zulip" or name.startswith("zulip."):
                raise ImportError("No module named 'zulip'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="zulip"):
            ch.connect()


@pytest.mark.spec("REQ-channels.zulip")
class TestDisconnect:
    def test_disconnect(self):
        ch = ZulipChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestStatus:
    def test_disconnected_initially(self):
        ch = ZulipChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.zulip")
class TestListChannels:
    def test_list_channels(self):
        ch = ZulipChannel()
        assert ch.list_channels() == ["zulip"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = ZulipChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = ZulipChannel()

        def h1(msg):
            pass

        def h2(msg):
            pass

        ch.on_message(h1)
        ch.on_message(h2)
        assert len(ch._handlers) == 2


# ---------------------------------------------------------------------------
# Send (no library — falls through import guard)
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestSend:
    def test_send_no_library_returns_false(self):
        """Zulip send() does not guard on credentials; it tries to import zulip
        directly. Without the library it returns False."""
        ch = ZulipChannel()
        result = ch.send("general", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.zulip")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = ZulipChannel(bus=bus)
        ch._publish_sent("general", "hello zulip", "conv-z1")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "general"
        assert event.data["content"] == "hello zulip"
        assert event.data["conversation_id"] == "conv-z1"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = ZulipChannel()
        ch._publish_sent("general", "hello", "")
