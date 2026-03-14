"""Tests for ``jarvis doctor`` optional dependency labels."""

from __future__ import annotations

import builtins
import json

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli

_real_import = builtins.__import__


def _selective_import_blocker(*blocked: str):
    """Return an __import__ replacement that blocks specific packages."""
    def _import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"mocked: {name} not installed")
        return _real_import(name, *args, **kwargs)
    return _import


class TestDoctorOptionalLabels:
    @pytest.mark.spec("REQ-cli.doctor")
    def test_labels_show_description(self) -> None:
        """Doctor output uses descriptive labels, not raw package names."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        data = json.loads(result.output)
        names = [c["name"] for c in data]
        # Should show descriptive labels
        assert "Optional: REST API server" in names
        assert "Optional: SFT/GRPO training" in names
        assert "Optional: NVIDIA energy monitoring" in names
        # Should NOT show old vague labels
        assert "Optional: torch (for learning)" not in names
        assert "Optional: pynvml (GPU monitoring)" not in names

    @pytest.mark.spec("REQ-cli.doctor")
    def test_labels_show_install_hint_on_missing(self, monkeypatch) -> None:
        """When a package is missing, show install hint in status."""
        blocker = _selective_import_blocker("zeus")
        monkeypatch.setattr(builtins, "__import__", blocker)
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        data = json.loads(result.output)
        apple_checks = [
            c for c in data
            if c["name"] == "Optional: Apple Silicon energy monitoring"
        ]
        assert len(apple_checks) == 1
        assert "Not installed (openjarvis[energy-apple])" == apple_checks[0]["message"]
