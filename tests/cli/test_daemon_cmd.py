"""Tests for ``jarvis start|stop|restart|status`` daemon management commands."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.cli.daemon_cmd import _read_pid, _write_pid

_daemon_mod = importlib.import_module("openjarvis.cli.daemon_cmd")


# ---------------------------------------------------------------------------
# Typed fake (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Typed fake for JarvisConfig used by daemon status command."""

    class _ServerSection:
        host: str = "127.0.0.1"
        port: int = 8000

    server = _ServerSection()


class TestDaemonCommands:
    """Core daemon CLI tests."""

    @pytest.mark.spec("REQ-cli.daemon")
    def test_start_command_exists(self) -> None:
        """``jarvis start --help`` succeeds."""
        result = CliRunner().invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "daemon" in out or "start" in out or "background" in out

    @pytest.mark.spec("REQ-cli.daemon")
    def test_stop_no_server(self, monkeypatch) -> None:
        """``jarvis stop`` when no PID file shows 'not running'."""
        monkeypatch.setattr(_daemon_mod, "_read_pid", lambda: None)
        result = CliRunner().invoke(cli, ["stop"])
        assert result.exit_code != 0
        assert "No running server" in result.output

    @pytest.mark.spec("REQ-cli.daemon")
    def test_status_no_server(self, monkeypatch) -> None:
        """``jarvis status`` when no PID file shows 'not running'."""
        monkeypatch.setattr(_daemon_mod, "_read_pid", lambda: None)
        result = CliRunner().invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "not running" in result.output

    @pytest.mark.spec("REQ-cli.daemon")
    def test_read_pid_no_file(self, monkeypatch, tmp_path: Path) -> None:
        """``_read_pid()`` returns None when no PID file exists."""
        monkeypatch.setattr(
            _daemon_mod, "_PID_FILE", tmp_path / "nonexistent.pid",
        )
        assert _read_pid() is None

    @pytest.mark.spec("REQ-cli.daemon")
    def test_write_and_read_pid(self, monkeypatch, tmp_path: Path) -> None:
        """Write a PID, then read it back (monkeypatch os.kill to succeed)."""
        pid_file = tmp_path / "server.pid"
        monkeypatch.setattr(_daemon_mod, "_PID_FILE", pid_file)
        monkeypatch.setattr(_daemon_mod, "DEFAULT_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)
        _write_pid(12345)
        assert pid_file.exists()
        assert _read_pid() == 12345

    @pytest.mark.spec("REQ-cli.daemon")
    def test_status_shows_running(self, monkeypatch) -> None:
        """``jarvis status`` shows running info when PID exists."""
        monkeypatch.setattr(_daemon_mod, "_read_pid", lambda: 9999)
        monkeypatch.setattr(
            _daemon_mod, "load_config", lambda: _FakeConfig(),
        )
        result = CliRunner().invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "running" in result.output
        assert "9999" in result.output

    @pytest.mark.spec("REQ-cli.daemon")
    def test_start_already_running(self, monkeypatch) -> None:
        """``jarvis start`` exits with error when a server is already running."""
        monkeypatch.setattr(_daemon_mod, "_read_pid", lambda: 42)
        result = CliRunner().invoke(cli, ["start"])
        assert result.exit_code != 0
        assert "already running" in result.output
