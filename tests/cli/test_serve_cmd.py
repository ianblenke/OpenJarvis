"""Tests for ``jarvis serve`` CLI command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli


class TestServeCommand:
    """REQ-cli.serve: jarvis serve starts the FastAPI server."""

    @pytest.mark.spec("REQ-cli.serve")
    def test_serve_is_registered(self) -> None:
        """The serve command is registered in the CLI group."""
        result = CliRunner().invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the OpenAI-compatible API server" in result.output

    @pytest.mark.spec("REQ-cli.serve")
    def test_serve_accepts_host_option(self) -> None:
        result = CliRunner().invoke(cli, ["serve", "--help"])
        assert "--host" in result.output

    @pytest.mark.spec("REQ-cli.serve")
    def test_serve_accepts_port_option(self) -> None:
        result = CliRunner().invoke(cli, ["serve", "--help"])
        assert "--port" in result.output

    @pytest.mark.spec("REQ-cli.serve")
    def test_serve_accepts_engine_option(self) -> None:
        result = CliRunner().invoke(cli, ["serve", "--help"])
        assert "--engine" in result.output or "-e" in result.output

    @pytest.mark.spec("REQ-cli.serve")
    def test_serve_accepts_model_option(self) -> None:
        result = CliRunner().invoke(cli, ["serve", "--help"])
        assert "--model" in result.output or "-m" in result.output

    @pytest.mark.spec("REQ-cli.serve")
    def test_serve_exits_when_no_engine(self) -> None:
        """Serve exits with error when no inference engine is available."""
        result = CliRunner().invoke(cli, ["serve"])
        out = result.output.lower()
        # Exits non-zero or reports no engine / missing deps
        assert (
            result.exit_code != 0
            or "not installed" in out
            or "no inference" in out
        )
