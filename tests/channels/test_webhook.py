"""Tests for the WebhookChannel adapter."""

from __future__ import annotations

from typing import List

import httpx
import pytest

from openjarvis.channels._stubs import ChannelMessage, ChannelStatus
from openjarvis.channels.webhook import WebhookChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_webhook():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("webhook"):
        ChannelRegistry.register_value("webhook", WebhookChannel)


class TestRegistration:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_registry_key(self):
        assert ChannelRegistry.contains("webhook")

    def test_channel_id(self):
        ch = WebhookChannel(url="https://example.com/hook")
        assert ch.channel_id == "webhook"


class TestInit:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_defaults(self):
        ch = WebhookChannel()
        assert ch._url == ""
        assert ch._secret == ""
        assert ch._method == "POST"
        assert ch._status == ChannelStatus.DISCONNECTED

    @pytest.mark.spec("REQ-channels.webhook")
    def test_constructor_params(self):
        ch = WebhookChannel(
            url="https://example.com/hook",
            secret="s3cr3t",
            method="PUT",
        )
        assert ch._url == "https://example.com/hook"
        assert ch._secret == "s3cr3t"
        assert ch._method == "PUT"


class TestConnect:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_connect_with_url(self):
        ch = WebhookChannel(url="https://example.com/hook")
        ch.connect()
        assert ch.status() == ChannelStatus.CONNECTED

    @pytest.mark.spec("REQ-channels.webhook")
    def test_connect_no_url(self):
        ch = WebhookChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestSend:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_success(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook")

        request_log: List[dict] = []

        def fake_request(method, url, **kwargs):
            request_log.append({"method": method, "url": url, **kwargs})
            return httpx.Response(200)

        monkeypatch.setattr("httpx.request", fake_request)

        result = ch.send("target", "Hello!")
        assert result is True
        assert len(request_log) == 1
        assert request_log[0]["method"] == "POST"
        assert request_log[0]["url"] == "https://example.com/hook"
        payload = request_log[0]["json"]
        assert payload["channel"] == "target"
        assert payload["content"] == "Hello!"

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_put_method(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook", method="PUT")

        request_log: List[dict] = []

        def fake_request(method, url, **kwargs):
            request_log.append({"method": method, "url": url, **kwargs})
            return httpx.Response(200)

        monkeypatch.setattr("httpx.request", fake_request)

        ch.send("target", "Hello!")
        assert request_log[0]["method"] == "PUT"

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_with_secret(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook", secret="s3cr3t")

        request_log: List[dict] = []

        def fake_request(method, url, **kwargs):
            request_log.append({"method": method, "url": url, **kwargs})
            return httpx.Response(200)

        monkeypatch.setattr("httpx.request", fake_request)

        ch.send("target", "Hello!")
        headers = request_log[0]["headers"]
        assert headers["X-Webhook-Secret"] == "s3cr3t"

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_without_secret(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook")

        request_log: List[dict] = []

        def fake_request(method, url, **kwargs):
            request_log.append({"method": method, "url": url, **kwargs})
            return httpx.Response(200)

        monkeypatch.setattr("httpx.request", fake_request)

        ch.send("target", "Hello!")
        headers = request_log[0]["headers"]
        assert "X-Webhook-Secret" not in headers

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_with_metadata(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook")

        request_log: List[dict] = []

        def fake_request(method, url, **kwargs):
            request_log.append({"method": method, "url": url, **kwargs})
            return httpx.Response(200)

        monkeypatch.setattr("httpx.request", fake_request)

        ch.send("target", "Hello!", metadata={"key": "value"})
        payload = request_log[0]["json"]
        assert payload["metadata"] == {"key": "value"}

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_failure(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook")

        monkeypatch.setattr(
            "httpx.request",
            lambda *a, **kw: httpx.Response(500),
        )

        result = ch.send("target", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_exception(self, monkeypatch):
        ch = WebhookChannel(url="https://example.com/hook")

        def raise_conn_error(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr("httpx.request", raise_conn_error)

        result = ch.send("target", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_no_url(self):
        ch = WebhookChannel()
        result = ch.send("target", "Hello!")
        assert result is False

    @pytest.mark.spec("REQ-channels.webhook")
    def test_send_publishes_event(self, monkeypatch):
        bus = EventBus(record_history=True)
        ch = WebhookChannel(url="https://example.com/hook", bus=bus)

        monkeypatch.setattr(
            "httpx.request",
            lambda *a, **kw: httpx.Response(200),
        )

        ch.send("target", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_list_channels(self):
        ch = WebhookChannel(url="https://example.com/hook")
        assert ch.list_channels() == ["webhook"]


class TestStatus:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_disconnected_initially(self):
        ch = WebhookChannel(url="https://example.com/hook")
        assert ch.status() == ChannelStatus.DISCONNECTED


class TestOnMessage:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_on_message(self):
        ch = WebhookChannel(url="https://example.com/hook")
        received: List[ChannelMessage] = []
        ch.on_message(lambda msg: received.append(msg))
        assert len(ch._handlers) == 1


class TestDisconnect:
    @pytest.mark.spec("REQ-channels.webhook")
    def test_disconnect(self):
        ch = WebhookChannel(url="https://example.com/hook")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
