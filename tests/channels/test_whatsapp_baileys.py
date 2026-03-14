"""Tests for the WhatsAppBaileysChannel adapter."""

from __future__ import annotations

import json
import threading
from io import StringIO
from typing import Any, List

import pytest

from openjarvis.channels._stubs import ChannelMessage, ChannelStatus
from openjarvis.channels.whatsapp_baileys import WhatsAppBaileysChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

# ---------------------------------------------------------------------------
# Typed fakes for subprocess boundary
# ---------------------------------------------------------------------------


class FakeStdin:
    """Typed fake for subprocess stdin pipe."""

    def __init__(self) -> None:
        self.written: List[str] = []

    def write(self, data: str) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeProcess:
    """Typed fake for subprocess.Popen result."""

    def __init__(
        self,
        *,
        pid: int = 12345,
        stdout_lines: List[str] | None = None,
    ) -> None:
        self.pid = pid
        self.stdin = FakeStdin()
        self.stdout = stdout_lines if stdout_lines is not None else []
        self.stderr = StringIO("")
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self.returncode or 0
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_whatsapp_baileys():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("whatsapp_baileys"):
        ChannelRegistry.register_value("whatsapp_baileys", WhatsAppBaileysChannel)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_registry_key(self):
        assert ChannelRegistry.contains("whatsapp_baileys")

    def test_channel_id(self):
        ch = WhatsAppBaileysChannel()
        assert ch.channel_id == "whatsapp_baileys"


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_defaults(self):
        ch = WhatsAppBaileysChannel()
        assert ch._auth_dir == ""
        assert ch._assistant_name == "Jarvis"
        assert ch._assistant_has_own_number is False
        assert ch._status == ChannelStatus.DISCONNECTED
        assert ch._process is None
        assert ch._handlers == []

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_custom_params(self):
        ch = WhatsAppBaileysChannel(
            auth_dir="/tmp/auth",
            assistant_name="Bot",
            assistant_has_own_number=True,
        )
        assert ch._auth_dir == "/tmp/auth"
        assert ch._assistant_name == "Bot"
        assert ch._assistant_has_own_number is True


# ---------------------------------------------------------------------------
# _ensure_bridge
# ---------------------------------------------------------------------------


class TestEnsureBridge:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_raises_when_node_not_found(self, monkeypatch):
        ch = WhatsAppBaileysChannel()
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        with pytest.raises(RuntimeError, match="Node.js is required"):
            ch._ensure_bridge()


# ---------------------------------------------------------------------------
# Connect / Disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_connect_spawns_subprocess(self, tmp_path, monkeypatch):
        ch = WhatsAppBaileysChannel()
        ch._runtime_dir = tmp_path

        fake_proc = FakeProcess()

        bridge_js = tmp_path / "dist" / "bridge.js"
        bridge_js.parent.mkdir(parents=True, exist_ok=True)
        bridge_js.write_text("// bridge")

        popen_calls: List[Any] = []

        def fake_popen(args, **kwargs):
            popen_calls.append((args, kwargs))
            return fake_proc

        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/node")
        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)

        # Pretend node_modules already exists to skip npm install.
        (tmp_path / "node_modules").mkdir()
        ch.connect()

        assert len(popen_calls) == 1
        call_args = popen_calls[0][0]
        assert "node" in call_args[0]
        assert str(bridge_js) in call_args[1]

        # Cleanup.
        ch._stop_event.set()
        ch._process = None

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_connect_sets_error_when_node_missing(self, monkeypatch):
        ch = WhatsAppBaileysChannel()
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_disconnect_terminates_process(self):
        ch = WhatsAppBaileysChannel()
        fake_proc = FakeProcess()
        ch._process = fake_proc
        ch._status = ChannelStatus.CONNECTED

        ch.disconnect()

        assert fake_proc.returncode == -15  # terminate was called
        assert ch.status() == ChannelStatus.DISCONNECTED
        assert ch._process is None

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_disconnect_when_not_connected(self):
        ch = WhatsAppBaileysChannel()
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_send_writes_json_to_stdin(self):
        ch = WhatsAppBaileysChannel()
        fake_proc = FakeProcess()
        ch._process = fake_proc
        ch._status = ChannelStatus.CONNECTED

        result = ch.send("123456@s.whatsapp.net", "Hello!")
        assert result is True

        written = fake_proc.stdin.written[0]
        payload = json.loads(written.strip())
        assert payload["type"] == "send"
        assert payload["jid"] == "123456@s.whatsapp.net"
        assert payload["text"] == "Hello!"

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_send_fails_when_not_connected(self):
        ch = WhatsAppBaileysChannel()
        result = ch.send("123456@s.whatsapp.net", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_send_publishes_event(self):
        bus = EventBus(record_history=True)
        ch = WhatsAppBaileysChannel(bus=bus)
        fake_proc = FakeProcess()
        ch._process = fake_proc
        ch._status = ChannelStatus.CONNECTED

        ch.send("123@s.whatsapp.net", "Hi!")
        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


# ---------------------------------------------------------------------------
# on_message
# ---------------------------------------------------------------------------


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_handler_registration(self):
        ch = WhatsAppBaileysChannel()
        received: List[ChannelMessage] = []
        ch.on_message(lambda msg: received.append(msg))
        assert len(ch._handlers) == 1


# ---------------------------------------------------------------------------
# list_channels / status
# ---------------------------------------------------------------------------


class TestListChannelsAndStatus:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_list_channels(self):
        ch = WhatsAppBaileysChannel()
        assert ch.list_channels() == ["whatsapp_baileys"]

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_initial_status(self):
        ch = WhatsAppBaileysChannel()
        assert ch.status() == ChannelStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# _reader_loop + _handle_bridge_event
# ---------------------------------------------------------------------------


class TestReaderLoop:
    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_parses_message_event(self):
        ch = WhatsAppBaileysChannel()
        received: List[ChannelMessage] = []
        ch.on_message(lambda msg: received.append(msg))

        event = {
            "type": "message",
            "jid": "123@s.whatsapp.net",
            "sender": "456@s.whatsapp.net",
            "text": "Hello from WhatsApp",
            "message_id": "msg-001",
        }
        ch._handle_bridge_event(event)

        assert len(received) == 1
        msg = received[0]
        assert msg.channel == "whatsapp_baileys"
        assert msg.sender == "456@s.whatsapp.net"
        assert msg.content == "Hello from WhatsApp"
        assert msg.message_id == "msg-001"
        assert msg.conversation_id == "123@s.whatsapp.net"

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_parses_status_connected(self):
        ch = WhatsAppBaileysChannel()
        ch._handle_bridge_event({"type": "status", "status": "connected"})
        assert ch.status() == ChannelStatus.CONNECTED

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_parses_status_disconnected(self):
        ch = WhatsAppBaileysChannel()
        ch._status = ChannelStatus.CONNECTED
        ch._handle_bridge_event({"type": "status", "status": "disconnected"})
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_parses_qr_event(self):
        ch = WhatsAppBaileysChannel()
        ch._handle_bridge_event({"type": "qr", "data": "qr-code-string"})
        assert ch._last_qr == "qr-code-string"

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_parses_error_event(self):
        ch = WhatsAppBaileysChannel()
        ch._handle_bridge_event({"type": "error", "message": "something broke"})
        assert ch.status() == ChannelStatus.ERROR

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_message_event_publishes_to_bus(self):
        bus = EventBus(record_history=True)
        ch = WhatsAppBaileysChannel(bus=bus)

        ch._handle_bridge_event({
            "type": "message",
            "jid": "123@s.whatsapp.net",
            "sender": "456@s.whatsapp.net",
            "text": "Bus test",
            "message_id": "msg-002",
        })

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_RECEIVED in event_types

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_reader_loop_processes_lines(self):
        ch = WhatsAppBaileysChannel()
        ch._stop_event = threading.Event()

        lines = [
            json.dumps({"type": "status", "status": "connected"}) + "\n",
            json.dumps({
                "type": "message",
                "jid": "j",
                "sender": "s",
                "text": "t",
                "message_id": "m",
            }) + "\n",
        ]

        fake_proc = FakeProcess(stdout_lines=lines)
        ch._process = fake_proc

        ch._reader_loop()

        assert ch.status() == ChannelStatus.CONNECTED

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_reader_loop_skips_non_json(self):
        ch = WhatsAppBaileysChannel()
        ch._stop_event = threading.Event()

        lines = [
            "not json at all\n",
            json.dumps({"type": "status", "status": "connected"}) + "\n",
        ]

        fake_proc = FakeProcess(stdout_lines=lines)
        ch._process = fake_proc

        ch._reader_loop()

        assert ch.status() == ChannelStatus.CONNECTED

    @pytest.mark.spec("REQ-channels.whatsapp-baileys")
    def test_handler_exception_does_not_crash(self):
        ch = WhatsAppBaileysChannel()

        def bad_handler(msg: ChannelMessage) -> None:
            raise ValueError("boom")

        ch.on_message(bad_handler)

        # Should not raise.
        ch._handle_bridge_event({
            "type": "message",
            "jid": "j",
            "sender": "s",
            "text": "t",
            "message_id": "m",
        })
        # Handler was registered and called (it raised, but didn't crash)
        assert len(ch._handlers) == 1
