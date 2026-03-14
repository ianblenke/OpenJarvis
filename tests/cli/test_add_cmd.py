"""Tests for the ``jarvis add`` CLI command."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from openjarvis.cli.add_cmd import _MCP_TEMPLATES, add

_add_mod = importlib.import_module("openjarvis.cli.add_cmd")


class TestAddCmd:
    @pytest.mark.spec("REQ-cli.add")
    def test_add_help(self) -> None:
        result = CliRunner().invoke(add, ["--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output

    @pytest.mark.spec("REQ-cli.add")
    def test_add_unknown_server(self) -> None:
        result = CliRunner().invoke(add, ["unknown_server"])
        assert result.exit_code != 0
        assert "Unknown MCP server" in result.output
        # Should list known servers
        assert "github" in result.output
        assert "filesystem" in result.output

    @pytest.mark.spec("REQ-cli.add")
    def test_add_known_server(self, monkeypatch, tmp_path: Path) -> None:
        mcp_dir = tmp_path / "mcp"
        monkeypatch.setattr(_add_mod, "_MCP_CONFIG_DIR", mcp_dir)
        result = CliRunner().invoke(add, ["filesystem"])
        assert result.exit_code == 0
        assert "Added MCP server: filesystem" in result.output

        config_file = mcp_dir / "filesystem.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["command"] == "npx"
        assert "@modelcontextprotocol/server-filesystem" in data["args"]

    @pytest.mark.spec("REQ-cli.add")
    def test_add_with_key(self, monkeypatch, tmp_path: Path) -> None:
        mcp_dir = tmp_path / "mcp"
        monkeypatch.setattr(_add_mod, "_MCP_CONFIG_DIR", mcp_dir)
        result = CliRunner().invoke(
            add, ["github", "--key", "test_token"],
        )
        assert result.exit_code == 0

        config_file = mcp_dir / "github.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert "env" in data
        assert data["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "test_token"

    @pytest.mark.spec("REQ-cli.add")
    def test_mcp_templates_complete(self) -> None:
        required_fields = {"command", "args", "env_key", "description"}
        for name, tmpl in _MCP_TEMPLATES.items():
            assert required_fields.issubset(
                tmpl.keys()
            ), f"Template '{name}' missing fields: {required_fields - tmpl.keys()}"
