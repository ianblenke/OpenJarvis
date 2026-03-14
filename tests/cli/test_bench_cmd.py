"""Tests for the ``jarvis bench`` CLI commands."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli

_bench_mod = importlib.import_module("openjarvis.cli.bench_cmd")


# ---------------------------------------------------------------------------
# Typed fake (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Typed fake engine for bench CLI tests."""

    def __init__(self) -> None:
        self.engine_id = "mock"

    def health(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["test-model"]

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return {
            "content": "Hello",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


class TestBenchCLI:
    @pytest.mark.spec("REQ-cli.bench")
    def test_bench_group_in_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "bench" in result.output

    @pytest.mark.spec("REQ-cli.bench")
    def test_run_help(self):
        result = CliRunner().invoke(cli, ["bench", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--samples" in result.output

    @pytest.mark.spec("REQ-cli.bench")
    def test_run_no_engine_error(self, monkeypatch):
        monkeypatch.setattr(_bench_mod, "get_engine", lambda *a, **kw: None)
        result = CliRunner().invoke(cli, ["bench", "run"])
        assert result.exit_code != 0

    @pytest.mark.spec("REQ-cli.bench")
    def test_run_with_mock(self, monkeypatch):
        engine = _FakeEngine()
        monkeypatch.setattr(
            _bench_mod, "get_engine", lambda *a, **kw: ("mock", engine),
        )
        result = CliRunner().invoke(cli, ["bench", "run", "-n", "2"])
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.bench")
    def test_json_output(self, monkeypatch):
        engine = _FakeEngine()
        monkeypatch.setattr(
            _bench_mod, "get_engine", lambda *a, **kw: ("mock", engine),
        )
        result = CliRunner().invoke(
            cli, ["bench", "run", "-n", "2", "--json"],
        )
        assert result.exit_code == 0
        assert "benchmark_count" in result.output

    @pytest.mark.spec("REQ-cli.bench")
    def test_output_to_file(self, monkeypatch, tmp_path):
        engine = _FakeEngine()
        monkeypatch.setattr(
            _bench_mod, "get_engine", lambda *a, **kw: ("mock", engine),
        )
        out_file = tmp_path / "results.jsonl"
        result = CliRunner().invoke(
            cli, ["bench", "run", "-n", "2", "-o", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
