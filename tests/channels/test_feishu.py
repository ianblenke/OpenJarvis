"""Tests for the FeishuChannel adapter."""

from __future__ import annotations

from typing import Any

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.feishu import FeishuChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

# ---------------------------------------------------------------------------
# Typed fake for httpx.Response
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    """Typed fake for httpx.Response with json() support."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        json_data: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def json(self) -> dict[str, Any]:
        return self._json_data


# ---------------------------------------------------------------------------
# Sequential response tracker
# ---------------------------------------------------------------------------


class _HttpPostSequence:
    """Typed callable that returns responses in sequence for httpx.post."""

    def __init__(self, responses: list[_FakeHttpResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.call_count += 1
        self.calls.append({"url": url, **kwargs})
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp


@pytest.fixture(autouse=True)
def _register_feishu():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("feishu"):
        ChannelRegistry.register_value("feishu", FeishuChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_registry_key(self):
        assert ChannelRegistry.contains("feishu")

    def test_channel_id(self):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")
        assert ch.channel_id == "feishu"


class TestInit:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_defaults(self):
        ch = FeishuChannel()
        assert ch._app_id == ""
        assert ch._app_secret == ""
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.feishu")
    def test_constructor_param(self):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")
        assert ch._app_id == "test-id"
        assert ch._app_secret == "test-secret"

    @pytest.mark.spec("REQ-channels.feishu")
    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "env-id")
        monkeypatch.setenv("FEISHU_APP_SECRET", "env-secret")
        ch = FeishuChannel()
        assert ch._app_id == "env-id"
        assert ch._app_secret == "env-secret"

    @pytest.mark.spec("REQ-channels.feishu")
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "env-id")
        monkeypatch.setenv("FEISHU_APP_SECRET", "env-secret")
        ch = FeishuChannel(app_id="explicit-id", app_secret="explicit-secret")
        assert ch._app_id == "explicit-id"
        assert ch._app_secret == "explicit-secret"


class TestSend:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_send_success(self, monkeypatch):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")

        token_resp = _FakeHttpResponse(
            status_code=200,
            json_data={"tenant_access_token": "fake-token"},
        )
        msg_resp = _FakeHttpResponse(status_code=200)

        seq = _HttpPostSequence([token_resp, msg_resp])
        monkeypatch.setattr("httpx.post", seq)

        result = ch.send("chat_id", "Hello!")
        assert result is True
        assert seq.call_count == 2

    @pytest.mark.spec("REQ-channels.feishu")
    def test_send_failure(self, monkeypatch):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")

        token_resp = _FakeHttpResponse(
            status_code=200,
            json_data={"tenant_access_token": "fake-token"},
        )
        msg_resp = _FakeHttpResponse(status_code=400, text="Bad Request")

        seq = _HttpPostSequence([token_resp, msg_resp])
        monkeypatch.setattr("httpx.post", seq)

        result = ch.send("chat_id", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.feishu")
    def test_send_exception(self, monkeypatch):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")

        def _raise(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.post", _raise)

        result = ch.send("chat_id", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.feishu")
    def test_send_no_token(self):
        ch = FeishuChannel()
        result = ch.send("chat_id", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.feishu")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret", bus=bus)

        token_resp = _FakeHttpResponse(
            status_code=200,
            json_data={"tenant_access_token": "fake-token"},
        )
        msg_resp = _FakeHttpResponse(status_code=200)

        seq = _HttpPostSequence([token_resp, msg_resp])
        monkeypatch.setattr("httpx.post", seq)

        ch.send("chat_id", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_list_channels(self):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")
        assert ch.list_channels() == ["feishu"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_disconnected_initially(self):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")
        assert ch.status() == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.feishu")
    def test_no_config_connect_error(self):
        ch = FeishuChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_on_message(self):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")
        def handler(msg):
            return None
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.feishu")
    def test_disconnect(self):
        ch = FeishuChannel(app_id="test-id", app_secret="test-secret")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
