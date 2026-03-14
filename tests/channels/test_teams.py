"""Tests for the TeamsChannel adapter."""

from __future__ import annotations

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.teams import TeamsChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

# ---------------------------------------------------------------------------
# Typed fake for httpx.Response
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    """Typed fake for httpx.Response replacing MagicMock."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.text = text


# ---------------------------------------------------------------------------
# Call-tracking helpers
# ---------------------------------------------------------------------------


class _HttpPostTracker:
    """Typed callable that records httpx.post calls for assertion."""

    def __init__(self, response: _FakeHttpResponse) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.call_count = 0

    def __call__(self, url, **kwargs):
        self.call_count += 1
        self.calls.append({"url": url, **kwargs})
        return self._response


@pytest.fixture(autouse=True)
def _register_teams():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("teams"):
        ChannelRegistry.register_value("teams", TeamsChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.teams")
    def test_registry_key(self):
        assert ChannelRegistry.contains("teams")

    def test_channel_id(self):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")
        assert ch.channel_id == "teams"


class TestInit:
    @pytest.mark.spec("REQ-channels.teams")
    def test_defaults(self):
        ch = TeamsChannel()
        assert ch._app_id == ""
        assert ch._app_password == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.teams")
    def test_constructor_param(self):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")
        assert ch._app_id == "test-id"
        assert ch._app_password == "test-pass"

    @pytest.mark.spec("REQ-channels.teams")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("TEAMS_APP_ID", "env-id")
        monkeypatch.setenv("TEAMS_APP_PASSWORD", "env-pass")
        ch = TeamsChannel()
        assert ch._app_id == "env-id"
        assert ch._app_password == "env-pass"

    @pytest.mark.spec("REQ-channels.teams")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TEAMS_APP_ID", "env-id")
        monkeypatch.setenv("TEAMS_APP_PASSWORD", "env-pass")
        ch = TeamsChannel(app_id="explicit-id", app_password="explicit-pass")
        assert ch._app_id == "explicit-id"
        assert ch._app_password == "explicit-pass"


class TestSend:
    @pytest.mark.spec("REQ-channels.teams")
    def test_send_success(self, monkeypatch):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")

        fake_response = _FakeHttpResponse(status_code=200)
        tracker = _HttpPostTracker(fake_response)
        monkeypatch.setattr("httpx.post", tracker)

        result = ch.send("general", "Hello!")
        assert result is True
        assert tracker.call_count == 1
        url = tracker.calls[0]["url"]
        assert "/v3/conversations/" in url
        assert "general" in url

    @pytest.mark.spec("REQ-channels.teams")
    def test_send_failure(self, monkeypatch):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")

        fake_response = _FakeHttpResponse(status_code=400, text="Bad Request")
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        result = ch.send("general", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.teams")
    def test_send_exception(self, monkeypatch):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("general", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.teams")
    def test_send_no_config(self):
        ch = TeamsChannel()
        result = ch.send("general", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.teams")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = TeamsChannel(app_id="test-id", app_password="test-pass", bus=bus)

        fake_response = _FakeHttpResponse(status_code=200)
        monkeypatch.setattr("httpx.post", lambda *a, **kw: fake_response)

        ch.send("general", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.teams")
    def test_list_channels(self):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")
        assert ch.list_channels() == ["teams"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.teams")
    def test_disconnected_initially(self):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.teams")
    def test_no_config_connect_error(self):
        ch = TeamsChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.teams")
    def test_on_message(self):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.teams")
    def test_disconnect(self):
        ch = TeamsChannel(app_id="test-id", app_password="test-pass")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
