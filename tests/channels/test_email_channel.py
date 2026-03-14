"""Tests for the EmailChannel adapter."""

from __future__ import annotations

from email.message import EmailMessage

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.email_channel import EmailChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

# ---------------------------------------------------------------------------
# Typed fake for smtplib.SMTP
# ---------------------------------------------------------------------------


class _FakeSMTP:
    """Typed fake for smtplib.SMTP used by the EmailChannel."""

    def __init__(self) -> None:
        self.starttls_calls = 0
        self.login_calls: list[tuple[str, str]] = []
        self.send_message_calls: list[EmailMessage] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self) -> None:
        self.starttls_calls += 1

    def login(self, user: str, password: str) -> None:
        self.login_calls.append((user, password))

    def send_message(self, msg: EmailMessage) -> None:
        self.send_message_calls.append(msg)


@pytest.fixture(autouse=True)
def _register_email():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("email"):
        ChannelRegistry.register_value("email", EmailChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.email")
    def test_registry_key(self):
        assert ChannelRegistry.contains("email")

    def test_channel_id(self):
        ch = EmailChannel(
            smtp_host="smtp.example.com", username="user@example.com",
        )
        assert ch.channel_id == "email"


class TestInit:
    @pytest.mark.spec("REQ-channels.email")
    def test_defaults(self):
        ch = EmailChannel()
        assert ch._smtp_host == ""
        assert ch._smtp_port == 587
        assert ch._imap_host == ""
        assert ch._imap_port == 993
        assert ch._username == ""
        assert ch._password == ""
        assert ch._use_tls is True
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.email")
    def test_constructor_params(self):
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=465,
            imap_host="imap.example.com",
            imap_port=143,
            username="user@example.com",
            password="pass123",
            use_tls=False,
        )
        assert ch._smtp_host == "smtp.example.com"
        assert ch._smtp_port == 465
        assert ch._imap_host == "imap.example.com"
        assert ch._imap_port == 143
        assert ch._username == "user@example.com"
        assert ch._password == "pass123"
        assert ch._use_tls is False

    @pytest.mark.spec("REQ-channels.email")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "env-pass")
        ch = EmailChannel()
        assert ch._username == "env@example.com"
        assert ch._password == "env-pass"

    @pytest.mark.spec("REQ-channels.email")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
        ch = EmailChannel(username="explicit@example.com")
        assert ch._username == "explicit@example.com"


class TestSend:
    @pytest.mark.spec("REQ-channels.email")
    def test_send_success_tls(self, monkeypatch):
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass123",
        )

        fake_smtp = _FakeSMTP()
        smtp_cls_calls: list[tuple] = []

        def _fake_smtp_cls(host, port):
            smtp_cls_calls.append((host, port))
            return fake_smtp

        monkeypatch.setattr("smtplib.SMTP", _fake_smtp_cls)

        result = ch.send("recipient@example.com", "Hello!")
        assert result is True
        assert smtp_cls_calls == [("smtp.example.com", 587)]
        assert fake_smtp.starttls_calls == 1
        assert fake_smtp.login_calls == [("user@example.com", "pass123")]
        assert len(fake_smtp.send_message_calls) == 1

    @pytest.mark.spec("REQ-channels.email")
    def test_send_success_no_tls(self, monkeypatch):
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass123",
            use_tls=False,
        )

        fake_smtp = _FakeSMTP()
        monkeypatch.setattr("smtplib.SMTP", lambda h, p: fake_smtp)

        result = ch.send("recipient@example.com", "Hello!")
        assert result is True
        assert fake_smtp.starttls_calls == 0

    @pytest.mark.spec("REQ-channels.email")
    def test_send_no_config(self):
        ch = EmailChannel()
        result = ch.send("recipient@example.com", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.email")
    def test_send_exception(self, monkeypatch):
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass123",
        )

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("smtplib.SMTP", _raise)

        result = ch.send("recipient@example.com", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.email")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass123",
            bus=bus,
        )

        fake_smtp = _FakeSMTP()
        monkeypatch.setattr("smtplib.SMTP", lambda h, p: fake_smtp)

        ch.send("recipient@example.com", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types

    @pytest.mark.spec("REQ-channels.email")
    def test_send_with_subject_metadata(self, monkeypatch):
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            username="user@example.com",
            password="pass123",
        )

        fake_smtp = _FakeSMTP()
        monkeypatch.setattr("smtplib.SMTP", lambda h, p: fake_smtp)

        result = ch.send(
            "recipient@example.com",
            "Hello!",
            metadata={"subject": "Custom Subject"},
        )
        assert result is True
        sent_msg = fake_smtp.send_message_calls[0]
        assert sent_msg["Subject"] == "Custom Subject"


class TestListChannels:
    @pytest.mark.spec("REQ-channels.email")
    def test_list_channels(self):
        ch = EmailChannel(smtp_host="smtp.example.com", username="user@example.com")
        assert ch.list_channels() == ["email"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.email")
    def test_disconnected_initially(self):
        ch = EmailChannel(smtp_host="smtp.example.com", username="user@example.com")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.email")
    def test_no_config_connect_error(self):
        ch = EmailChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestConnect:
    @pytest.mark.spec("REQ-channels.email")
    def test_connect_smtp_only(self):
        ch = EmailChannel(
            smtp_host="smtp.example.com",
            username="user@example.com",
        )
        ch.connect()
        assert ch.status() == ChannelStatus.CONNECTED
        # No IMAP, so no listener thread
        assert ch._listener_thread is None


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.email")
    def test_on_message(self):
        ch = EmailChannel(smtp_host="smtp.example.com", username="user@example.com")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.email")
    def test_disconnect(self):
        ch = EmailChannel(
            smtp_host="smtp.example.com", username="user@example.com",
        )
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
