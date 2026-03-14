"""Tests for the ``jarvis channel`` CLI commands."""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

import pytest
from click.testing import CliRunner

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.cli import cli

_channel_cmd = importlib.import_module("openjarvis.cli.channel_cmd")


# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Typed fake for JarvisConfig used by channel commands."""

    class _ChannelSection:
        default_channel: str = ""

    channel = _ChannelSection()


class _FakeBridge:
    """Typed fake for channel bridge used by channel commands."""

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        send_ok: bool = True,
        status_val: ChannelStatus = ChannelStatus.DISCONNECTED,
        list_error: Optional[Exception] = None,
    ) -> None:
        self._channels = channels or []
        self._send_ok = send_ok
        self._status_val = status_val
        self._list_error = list_error

    def list_channels(self) -> List[str]:
        if self._list_error is not None:
            raise self._list_error
        return self._channels

    def send(self, channel: str, content: str, **kwargs: Any) -> bool:
        return self._send_ok

    def status(self, **kwargs: Any) -> ChannelStatus:
        return self._status_val


def _patch_channel(
    monkeypatch,
    list_channels=None,
    send_return=True,
    status_return=ChannelStatus.DISCONNECTED,
    list_error=None,
):
    """Patch channel_cmd with typed fakes via monkeypatch."""
    cfg = _FakeConfig()
    bridge = _FakeBridge(
        channels=list_channels,
        send_ok=send_return,
        status_val=status_return,
        list_error=list_error,
    )
    monkeypatch.setattr(
        "openjarvis.core.config.load_config", lambda: cfg,
    )
    monkeypatch.setattr(
        _channel_cmd, "_get_channel", lambda *a, **kw: bridge,
    )
    return bridge


class TestChannelHelp:
    @pytest.mark.spec("REQ-cli.channel")
    def test_subcommands_in_help(self) -> None:
        result = CliRunner().invoke(cli, ["channel", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "send" in result.output
        assert "status" in result.output


class TestChannelList:
    @pytest.mark.spec("REQ-cli.channel")
    def test_list_with_channels(self, monkeypatch) -> None:
        _patch_channel(
            monkeypatch, list_channels=["slack", "discord"],
        )
        result = CliRunner().invoke(cli, ["channel", "list"])
        assert result.exit_code == 0
        assert "slack" in result.output
        assert "discord" in result.output

    @pytest.mark.spec("REQ-cli.channel")
    def test_list_no_channels(self, monkeypatch) -> None:
        _patch_channel(monkeypatch, list_channels=[])
        result = CliRunner().invoke(cli, ["channel", "list"])
        assert result.exit_code == 0
        assert "No channels available" in result.output

    @pytest.mark.spec("REQ-cli.channel")
    def test_list_connection_error(self, monkeypatch) -> None:
        _patch_channel(
            monkeypatch,
            list_error=ConnectionError("refused"),
        )
        result = CliRunner().invoke(cli, ["channel", "list"])
        assert result.exit_code == 0
        assert "Failed" in result.output or "refused" in result.output


class TestChannelSend:
    @pytest.mark.spec("REQ-cli.channel")
    def test_send_success(self, monkeypatch) -> None:
        _patch_channel(monkeypatch, send_return=True)
        result = CliRunner().invoke(
            cli, ["channel", "send", "slack", "Hello!"],
        )
        assert result.exit_code == 0
        assert "Message sent" in result.output

    @pytest.mark.spec("REQ-cli.channel")
    def test_send_failure(self, monkeypatch) -> None:
        _patch_channel(monkeypatch, send_return=False)
        result = CliRunner().invoke(
            cli, ["channel", "send", "slack", "Hello!"],
        )
        assert result.exit_code == 0
        assert "Failed to send" in result.output


class TestChannelStatus:
    @pytest.mark.spec("REQ-cli.channel")
    def test_status_shows_info(self, monkeypatch) -> None:
        _patch_channel(
            monkeypatch,
            status_return=ChannelStatus.DISCONNECTED,
        )
        result = CliRunner().invoke(cli, ["channel", "status"])
        assert result.exit_code == 0
        assert "disconnected" in result.output
