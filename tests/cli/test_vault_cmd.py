"""Tests for the ``jarvis vault`` CLI commands."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from openjarvis.cli.vault_cmd import vault

_vault_mod = importlib.import_module("openjarvis.cli.vault_cmd")


class TestVaultCmd:
    @pytest.mark.spec("REQ-cli.vault")
    def test_vault_group_help(self) -> None:
        result = CliRunner().invoke(vault, ["--help"])
        assert result.exit_code == 0
        assert "set" in result.output
        assert "get" in result.output
        assert "list" in result.output
        assert "remove" in result.output

    @pytest.mark.spec("REQ-cli.vault")
    def test_vault_set_help(self) -> None:
        result = CliRunner().invoke(vault, ["set", "--help"])
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.vault")
    def test_vault_get_help(self) -> None:
        result = CliRunner().invoke(vault, ["get", "--help"])
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.vault")
    def test_vault_list_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _vault_mod, "_VAULT_FILE", Path("/nonexistent/vault.enc"),
        )
        result = CliRunner().invoke(vault, ["list"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    @pytest.mark.spec("REQ-cli.vault")
    def test_vault_roundtrip(self, monkeypatch, tmp_path: Path) -> None:
        pytest.importorskip("cryptography")

        vault_file = tmp_path / "vault.enc"
        key_file = tmp_path / ".vault_key"

        monkeypatch.setattr(_vault_mod, "_VAULT_FILE", vault_file)
        monkeypatch.setattr(_vault_mod, "_VAULT_KEY_FILE", key_file)
        monkeypatch.setattr(_vault_mod, "DEFAULT_CONFIG_DIR", tmp_path)

        runner = CliRunner()

        # Set a credential
        result = runner.invoke(vault, ["set", "MY_API_KEY", "secret123"])
        assert result.exit_code == 0

        # Get it back
        result = runner.invoke(vault, ["get", "MY_API_KEY"])
        assert result.exit_code == 0
        assert "secret123" in result.output

    @pytest.mark.spec("REQ-cli.vault")
    def test_vault_remove_not_found(self, monkeypatch) -> None:
        monkeypatch.setattr(
            _vault_mod, "_VAULT_FILE", Path("/nonexistent/vault.enc"),
        )
        result = CliRunner().invoke(vault, ["remove", "nonexistent"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()
