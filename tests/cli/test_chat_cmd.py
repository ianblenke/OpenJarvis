"""Tests for ``jarvis chat`` interactive REPL command."""

from __future__ import annotations

import builtins

import pytest
from click.testing import CliRunner

from openjarvis.cli.chat_cmd import _read_input, chat


class TestChatCommand:
    """Test the Click command definition and help output."""

    @pytest.mark.spec("REQ-cli.chat")
    def test_command_exists(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "interactive" in result.output.lower() or "chat" in result.output.lower()

    @pytest.mark.spec("REQ-cli.chat")
    def test_options(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "--engine" in result.output
        assert "--model" in result.output
        assert "--agent" in result.output
        assert "--tools" in result.output
        assert "--system" in result.output

    @pytest.mark.spec("REQ-cli.chat")
    def test_slash_commands_listed(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "/quit" in result.output


class TestReadInput:
    """Test the _read_input helper function."""

    @pytest.mark.spec("REQ-cli.chat")
    def test_read_input_eof(self, monkeypatch) -> None:
        def _raise_eof(prompt=""):
            raise EOFError
        monkeypatch.setattr(builtins, "input", _raise_eof)
        assert _read_input() is None

    @pytest.mark.spec("REQ-cli.chat")
    def test_read_input_keyboard_interrupt(self, monkeypatch) -> None:
        def _raise_ki(prompt=""):
            raise KeyboardInterrupt
        monkeypatch.setattr(builtins, "input", _raise_ki)
        assert _read_input() is None

    @pytest.mark.spec("REQ-cli.chat")
    def test_read_input_normal(self, monkeypatch) -> None:
        monkeypatch.setattr(builtins, "input", lambda prompt="": "hello")
        assert _read_input() == "hello"
