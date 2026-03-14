"""Tests for the TwitchChannel adapter."""

from __future__ import annotations

import builtins

import httpx
import pytest
import respx

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.twitch_channel import TwitchChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_twitch():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("twitch"):
        ChannelRegistry.register_value("twitch", TwitchChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("twitch")

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_channel_id(self):
        ch = TwitchChannel(access_token="tok")
        assert ch.channel_id == "twitch"


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestInit:
    def test_defaults(self):
        ch = TwitchChannel()
        assert ch._access_token == ""
        assert ch._client_id == ""
        assert ch._nick == ""
        assert ch._initial_channels == ""
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._handlers == []
        assert ch._bus is None

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_params(self):
        ch = TwitchChannel(
            access_token="my-tok",
            client_id="my-cid",
            nick="botname",
            initial_channels="chan1,chan2",
        )
        assert ch._access_token == "my-tok"
        assert ch._client_id == "my-cid"
        assert ch._nick == "botname"
        assert ch._initial_channels == "chan1,chan2"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("TWITCH_ACCESS_TOKEN", "env-tok")
        monkeypatch.setenv("TWITCH_CLIENT_ID", "env-cid")
        monkeypatch.setenv("TWITCH_NICK", "env-nick")
        monkeypatch.setenv("TWITCH_CHANNELS", "chan1,chan2")
        ch = TwitchChannel()
        assert ch._access_token == "env-tok"
        assert ch._client_id == "env-cid"
        assert ch._nick == "env-nick"
        assert ch._initial_channels == "chan1,chan2"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TWITCH_ACCESS_TOKEN", "env-tok")
        monkeypatch.setenv("TWITCH_CLIENT_ID", "env-cid")
        ch = TwitchChannel(access_token="explicit-tok", client_id="explicit-cid")
        assert ch._access_token == "explicit-tok"
        assert ch._client_id == "explicit-cid"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_bus_stored(self):
        bus = EventBus(record_history=True)
        ch = TwitchChannel(bus=bus)
        assert ch._bus is bus


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestConnect:
    def test_connect_no_token_sets_error(self):
        ch = TwitchChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_connect_import_error(self, monkeypatch):
        ch = TwitchChannel(access_token="tok")
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "twitchio" or name.startswith("twitchio."):
                raise ImportError("No module named 'twitchio'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="twitchio"):
            ch.connect()


@pytest.mark.spec("REQ-channels.twitch")
class TestDisconnect:
    def test_disconnect(self):
        ch = TwitchChannel()
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Status / list_channels
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestStatus:
    def test_disconnected_initially(self):
        ch = TwitchChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


@pytest.mark.spec("REQ-channels.twitch")
class TestListChannels:
    def test_list_channels(self):
        ch = TwitchChannel()
        assert ch.list_channels() == ["twitch"]


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestOnMessage:
    def test_on_message_registers_handler(self):
        ch = TwitchChannel()
        received = []

        def handler(msg):
            received.append(msg)

        ch.on_message(handler)
        assert handler in ch._handlers

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_multiple_handlers(self):
        ch = TwitchChannel()

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


@pytest.mark.spec("REQ-channels.twitch")
class TestSendNoCredentials:
    def test_send_no_token(self):
        ch = TwitchChannel()
        result = ch.send("broadcaster-id", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# Send (with credentials via respx)
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestSendWithCredentials:
    @respx.mock
    def test_send_success(self):
        ch = TwitchChannel(access_token="my-tok", client_id="my-cid", nick="botname")

        respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            return_value=httpx.Response(200, json={"data": [{"message_id": "123"}]})
        )

        result = ch.send("broadcaster-123", "Hello Twitch!")
        assert result is True

    @respx.mock
    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_includes_correct_headers(self):
        ch = TwitchChannel(access_token="my-tok", client_id="my-cid", nick="botname")

        route = respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            return_value=httpx.Response(200, json={})
        )

        ch.send("broadcaster-123", "Hello!")

        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer my-tok"
        assert request.headers["Client-Id"] == "my-cid"

    @respx.mock
    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_includes_correct_payload(self):
        ch = TwitchChannel(access_token="my-tok", client_id="my-cid", nick="botname")

        route = respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            return_value=httpx.Response(200, json={})
        )

        ch.send("broadcaster-123", "Test message")

        import json
        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["broadcaster_id"] == "broadcaster-123"
        assert body["sender_id"] == "botname"
        assert body["message"] == "Test message"

    @respx.mock
    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_failure_status(self):
        ch = TwitchChannel(access_token="my-tok", client_id="my-cid")

        respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        result = ch.send("broadcaster-123", "Hello!")
        assert result is False

    @respx.mock
    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_publishes_event(self):
        bus = EventBus(record_history=True)
        ch = TwitchChannel(
            access_token="my-tok",
            client_id="my-cid",
            nick="botname",
            bus=bus,
        )

        respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            return_value=httpx.Response(200, json={})
        )

        ch.send("broadcaster-123", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types

    @respx.mock
    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_no_event_on_failure(self):
        bus = EventBus(record_history=True)
        ch = TwitchChannel(access_token="my-tok", client_id="my-cid", bus=bus)

        respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            return_value=httpx.Response(500, json={"error": "Internal"})
        )

        ch.send("broadcaster-123", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT not in event_types

    @respx.mock
    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_send_network_error_returns_false(self):
        ch = TwitchChannel(access_token="my-tok", client_id="my-cid")

        respx.post("https://api.twitch.tv/helix/chat/messages").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = ch.send("broadcaster-123", "Hello!")
        assert result is False


# ---------------------------------------------------------------------------
# _publish_sent
# ---------------------------------------------------------------------------


@pytest.mark.spec("REQ-channels.twitch")
class TestPublishSent:
    def test_publish_sent_with_bus(self):
        bus = EventBus(record_history=True)
        ch = TwitchChannel(bus=bus)
        ch._publish_sent("broadcaster-123", "hello twitch", "conv-t1")
        assert len(bus.history) == 1
        event = bus.history[0]
        assert event.event_type == EventType.CHANNEL_MESSAGE_SENT
        assert event.data["channel"] == "broadcaster-123"
        assert event.data["content"] == "hello twitch"
        assert event.data["conversation_id"] == "conv-t1"

    @pytest.mark.spec("REQ-channels.protocol.lifecycle")
    def test_publish_sent_without_bus(self):
        ch = TwitchChannel()
        ch._publish_sent("broadcaster-123", "hello", "")
