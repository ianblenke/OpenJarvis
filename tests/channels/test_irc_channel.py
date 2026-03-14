"""Tests for the IRCChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.irc_channel import IRCChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

# ---------------------------------------------------------------------------
# Typed fake for socket.socket
# ---------------------------------------------------------------------------


class _FakeSocket:
    """Typed fake for socket.socket used by the IRC channel."""

    def __init__(
        self,
        connect_error: Exception | None = None,
    ) -> None:
        self._connect_error = connect_error
        self.connect_calls: list[tuple] = []
        self.sendall_calls: list[bytes] = []
        self.close_called = False

    def connect(self, address: tuple) -> None:
        self.connect_calls.append(address)
        if self._connect_error is not None:
            raise self._connect_error

    def sendall(self, data: bytes) -> None:
        self.sendall_calls.append(data)

    def close(self) -> None:
        self.close_called = True

    def settimeout(self, timeout: float | None) -> None:
        pass

    def recv(self, bufsize: int) -> bytes:
        return b""


@pytest.fixture(autouse=True)
def _register_irc():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("irc"):
        ChannelRegistry.register_value("irc", IRCChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.irc")
    def test_registry_key(self):
        assert ChannelRegistry.contains("irc")

    def test_channel_id(self):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")
        assert ch.channel_id == "irc"


class TestInit:
    @pytest.mark.spec("REQ-channels.irc")
    def test_defaults(self):
        ch = IRCChannel()
        assert ch._server == ""
        assert ch._nick == ""
        assert ch._password == ""
        assert ch._port == 6667
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.irc")
    def test_constructor_params(self):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")
        assert ch._server == "irc.example.com"
        assert ch._nick == "jarvis"
        assert ch._password == "pass123"

    @pytest.mark.spec("REQ-channels.irc")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.env.com")
        monkeypatch.setenv("IRC_NICK", "envbot")
        monkeypatch.setenv("IRC_PASSWORD", "envpass")
        monkeypatch.setenv("IRC_PORT", "6697")
        ch = IRCChannel()
        assert ch._server == "irc.env.com"
        assert ch._nick == "envbot"
        assert ch._password == "envpass"
        assert ch._port == 6697

    @pytest.mark.spec("REQ-channels.irc")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.env.com")
        monkeypatch.setenv("IRC_NICK", "envbot")
        monkeypatch.setenv("IRC_PASSWORD", "envpass")
        ch = IRCChannel(
            server="irc.explicit.com",
            nick="explicit",
            password="explicit-pass",
        )
        assert ch._server == "irc.explicit.com"
        assert ch._nick == "explicit"
        assert ch._password == "explicit-pass"


class TestSend:
    @pytest.mark.spec("REQ-channels.irc")
    def test_send_success(self, monkeypatch):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")

        fake_sock = _FakeSocket()
        monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)

        result = ch.send("#channel", "Hello!")
        assert result is True
        assert len(fake_sock.connect_calls) == 1
        assert len(fake_sock.sendall_calls) > 0

    @pytest.mark.spec("REQ-channels.irc")
    def test_send_failure_exception(self, monkeypatch):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")

        fake_sock = _FakeSocket(connect_error=ConnectionError("refused"))
        monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)

        result = ch.send("#channel", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.irc")
    def test_send_no_config(self):
        ch = IRCChannel()
        result = ch.send("#channel", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.irc")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = IRCChannel(
            server="irc.example.com",
            nick="jarvis",
            password="pass123",
            bus=bus,
        )

        fake_sock = _FakeSocket()
        monkeypatch.setattr("socket.socket", lambda *a, **kw: fake_sock)

        ch.send("#channel", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.irc")
    def test_list_channels(self):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")
        assert ch.list_channels() == ["irc"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.irc")
    def test_disconnected_initially(self):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.irc")
    def test_no_server_connect_error(self):
        ch = IRCChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.irc")
    def test_on_message(self):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.irc")
    def test_disconnect(self):
        ch = IRCChannel(server="irc.example.com", nick="jarvis", password="pass123")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
