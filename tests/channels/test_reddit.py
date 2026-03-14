"""Tests for the RedditChannel adapter."""

from __future__ import annotations

import builtins

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.reddit_channel import RedditChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_reddit():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("reddit"):
        ChannelRegistry.register_value("reddit", RedditChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.reddit")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("reddit")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = RedditChannel(client_id="cid", client_secret="csec")
        assert ch.channel_id == "reddit"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.reddit")
class TestInit:
    def test_defaults(self):
        ch = RedditChannel()
        assert ch._client_id == ""
        assert ch._client_secret == ""
        assert ch._username == ""
        assert ch._password == ""
        assert ch._user_agent == "openjarvis:v1.0"
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None
        assert ch._reddit is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = RedditChannel(
            client_id="cid",
            client_secret="csec",
            username="user",
            password="pass",
            user_agent="mybot:v2.0",
        )
        assert ch._client_id == "cid"
        assert ch._client_secret == "csec"
        assert ch._username == "user"
        assert ch._password == "pass"
        assert ch._user_agent == "mybot:v2.0"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "env-cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env-csec")
        monkeypatch.setenv("REDDIT_USERNAME", "env-user")
        monkeypatch.setenv("REDDIT_PASSWORD", "env-pass")
        monkeypatch.setenv("REDDIT_USER_AGENT", "env-agent")
        ch = RedditChannel()
        assert ch._client_id == "env-cid"
        assert ch._client_secret == "env-csec"
        assert ch._username == "env-user"
        assert ch._password == "env-pass"
        assert ch._user_agent == "env-agent"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "env-cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env-csec")
        ch = RedditChannel(client_id="explicit-cid", client_secret="explicit-csec")
        assert ch._client_id == "explicit-cid"
        assert ch._client_secret == "explicit-csec"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = RedditChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.reddit")
class TestConnect:
    def test_connect_no_credentials_sets_error(self):
        ch = RedditChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_client_id_sets_error(self):
        ch = RedditChannel(client_secret="csec")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_no_client_secret_sets_error(self):
        ch = RedditChannel(client_id="cid")
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = RedditChannel(client_id="cid", client_secret="csec")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "praw" or name.startswith("praw."):
                raise ImportError("No module named 'praw'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="praw"):
            ch.connect()


@pytest.mark.spec("REQ-channels.reddit")
class TestDisconnect:
    def test_disconnect(self):
        ch = RedditChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_disconnect_clears_reddit_instance(self):
        ch = RedditChannel()
        ch._reddit = "fake-reddit-instance"
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch._reddit is None
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.reddit")
class TestStatus:
    def test_disconnected_initially(self):
        ch = RedditChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.reddit")
class TestListChannels:
    def test_list_channels(self):
        ch = RedditChannel()
        assert ch.list_channels() == ["reddit"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.reddit")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = RedditChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = RedditChannel()

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


@pytest.mark.spec("REQ-channels.reddit")
class TestSend:
    def test_send_no_credentials(self):
        ch = RedditChannel()
        result = ch.send("testsubreddit", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_no_client_id(self):
        ch = RedditChannel(client_secret="csec")
        result = ch.send("testsubreddit", "hello")
        assert result is False

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_no_client_secret(self):
        ch = RedditChannel(client_id="cid")
        result = ch.send("testsubreddit", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.reddit")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = RedditChannel(bus=bus)
        ch._publish_sent("testsubreddit", "hello reddit", "conv-789")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "testsubreddit"
        assert event.data["content"] == "hello reddit"
        assert event.data["conversation_id"] == "conv-789"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = RedditChannel()
        ch._publish_sent("testsubreddit", "hello", "")
