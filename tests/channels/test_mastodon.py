"""Tests for the MastodonChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.mastodon_channel import MastodonChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_mastodon():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("mastodon"):
        ChannelRegistry.register_value("mastodon", MastodonChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("mastodon")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = MastodonChannel(api_base_url="https://mastodon.social", access_token="tok")
        assert ch.channel_id == "mastodon"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestInit:
    def test_defaults(self):
        ch = MastodonChannel()
        assert ch._api_base_url == ""
        assert ch._access_token == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = MastodonChannel(api_base_url="https://m.social", access_token="tok123")
        assert ch._api_base_url == "https://m.social"
        assert ch._access_token == "tok123"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("MASTODON_API_BASE_URL", "https://env.social")
        monkeypatch.setenv("MASTODON_ACCESS_TOKEN", "env-tok")
        ch = MastodonChannel()
        assert ch._api_base_url == "https://env.social"
        assert ch._access_token == "env-tok"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MASTODON_API_BASE_URL", "https://env.social")
        monkeypatch.setenv("MASTODON_ACCESS_TOKEN", "env-tok")
        ch = MastodonChannel(
            api_base_url="https://explicit.social",
            access_token="explicit-tok",
        )
        assert ch._api_base_url == "https://explicit.social"
        assert ch._access_token == "explicit-tok"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = MastodonChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestConnect:
    def test_connect_no_credentials_sets_error(self):
        ch = MastodonChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_url_sets_error(self):
        ch = MastodonChannel(access_token="tok")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_token_sets_error(self):
        ch = MastodonChannel(api_base_url="https://m.social")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = MastodonChannel(api_base_url="https://m.social", access_token="tok")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mastodon" or name.startswith("mastodon."):
                raise ImportError("No module named 'mastodon'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="Mastodon.py"):
            ch.connect()


@pytest.mark.spec("REQ-channels.mastodon")
class TestDisconnect:
    def test_disconnect(self):
        ch = MastodonChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestStatus:
    def test_disconnected_initially(self):
        ch = MastodonChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.mastodon")
class TestListChannels:
    def test_list_channels(self):
        ch = MastodonChannel()
        assert ch.list_channels() == ["mastodon"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = MastodonChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = MastodonChannel()

        def h1(msg):
            pass

        def h2(msg):
            pass

        ch.on_message(h1)
        ch.on_message(h2)
        assert len(ch._handlers) == 2
        assert h1 in ch._handlers
        assert h2 in ch._handlers


# ---------------------------------------------------------------------------
# Send (no credentials)
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestSend:
    def test_send_no_credentials(self):
        ch = MastodonChannel()
        result = ch.send("public", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_no_url_returns_false(self):
        ch = MastodonChannel(access_token="tok")
        result = ch.send("public", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_no_token_returns_false(self):
        ch = MastodonChannel(api_base_url="https://m.social")
        result = ch.send("public", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.mastodon")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = MastodonChannel(bus=bus)
        ch._publish_sent("public", "hello world", "conv-123")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "public"
        assert event.data["content"] == "hello world"
        assert event.data["conversation_id"] == "conv-123"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = MastodonChannel()
        # Should not raise
        ch._publish_sent("public", "hello", "")
