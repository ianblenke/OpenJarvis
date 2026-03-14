"""Tests for the RocketChatChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.rocketchat_channel import RocketChatChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_rocketchat():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("rocketchat"):
        ChannelRegistry.register_value("rocketchat", RocketChatChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.rocketchat")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("rocketchat")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = RocketChatChannel(
            url="https://rc.example.com",
            user="bot",
            password="pass",
        )
        assert ch.channel_id == "rocketchat"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.rocketchat")
class TestInit:
    def test_defaults(self):
        ch = RocketChatChannel()
        assert ch._url == ""
        assert ch._user == ""
        assert ch._password == ""
        assert ch._auth_token == ""
        assert ch._user_id == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_password_auth(self):
        ch = RocketChatChannel(
            url="https://rc.example.com",
            user="bot",
            password="pass",
        )
        assert ch._url == "https://rc.example.com"
        assert ch._user == "bot"
        assert ch._password == "pass"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_token_auth(self):
        ch = RocketChatChannel(
            url="https://rc.example.com",
            auth_token="tok",
            user_id="uid",
        )
        assert ch._url == "https://rc.example.com"
        assert ch._auth_token == "tok"
        assert ch._user_id == "uid"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback_password_auth(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://env.example.com")
        monkeypatch.setenv("ROCKETCHAT_USER", "env-bot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "env-pass")
        ch = RocketChatChannel()
        assert ch._url == "https://env.example.com"
        assert ch._user == "env-bot"
        assert ch._password == "env-pass"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback_token_auth(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://env.example.com")
        monkeypatch.setenv("ROCKETCHAT_AUTH_TOKEN", "env-tok")
        monkeypatch.setenv("ROCKETCHAT_USER_ID", "env-uid")
        ch = RocketChatChannel()
        assert ch._url == "https://env.example.com"
        assert ch._auth_token == "env-tok"
        assert ch._user_id == "env-uid"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_URL", "https://env.example.com")
        monkeypatch.setenv("ROCKETCHAT_USER", "env-bot")
        monkeypatch.setenv("ROCKETCHAT_PASSWORD", "env-pass")
        ch = RocketChatChannel(
            url="https://explicit.com",
            user="ex-bot",
            password="ex-pass",
        )
        assert ch._url == "https://explicit.com"
        assert ch._user == "ex-bot"
        assert ch._password == "ex-pass"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = RocketChatChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.rocketchat")
class TestConnect:
    def test_connect_no_url_sets_error(self):
        ch = RocketChatChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_credentials_sets_error(self):
        ch = RocketChatChannel(url="https://rc.example.com")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_url_only_no_auth_sets_error(self):
        ch = RocketChatChannel(url="https://rc.example.com", user="bot")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error_password_auth(self, monkeypatch):
        ch = RocketChatChannel(
            url="https://rc.example.com",
            user="bot",
            password="pass",
        )
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rocketchat_API" or name.startswith("rocketchat_API."):
                raise ImportError("No module named 'rocketchat_API'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="rocketchat_API"):
            ch.connect()

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error_token_auth(self, monkeypatch):
        ch = RocketChatChannel(
            url="https://rc.example.com",
            auth_token="tok",
            user_id="uid",
        )
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rocketchat_API" or name.startswith("rocketchat_API."):
                raise ImportError("No module named 'rocketchat_API'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="rocketchat_API"):
            ch.connect()


@pytest.mark.spec("REQ-channels.rocketchat")
class TestDisconnect:
    def test_disconnect(self):
        ch = RocketChatChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.rocketchat")
class TestStatus:
    def test_disconnected_initially(self):
        ch = RocketChatChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.rocketchat")
class TestListChannels:
    def test_list_channels(self):
        ch = RocketChatChannel()
        assert ch.list_channels() == ["rocketchat"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.rocketchat")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = RocketChatChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = RocketChatChannel()

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


@pytest.mark.spec("REQ-channels.rocketchat")
class TestSend:
    def test_send_no_url(self):
        ch = RocketChatChannel()
        result = ch.send("general", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_url_only_no_auth(self):
        ch = RocketChatChannel(url="https://rc.example.com")
        # send() only checks _url, not auth — but the library import will fail
        # In this case send returns False because rocketchat_API is not installed
        # or because the library call fails. The credential guard is only on _url.
        # Actually reading the source: `if not self._url:` is the only guard.
        # With url set, it tries to import rocketchat_API which will fail.
        result = ch.send("general", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.rocketchat")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = RocketChatChannel(bus=bus)
        ch._publish_sent("general", "hello rocket", "conv-rc")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "general"
        assert event.data["content"] == "hello rocket"
        assert event.data["conversation_id"] == "conv-rc"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = RocketChatChannel()
        ch._publish_sent("general", "hello", "")
