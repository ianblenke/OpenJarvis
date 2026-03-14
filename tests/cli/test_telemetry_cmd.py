"""Tests for the ``jarvis telemetry`` CLI commands."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.types import TelemetryRecord
from openjarvis.telemetry.store import TelemetryStore

_telem_mod = importlib.import_module("openjarvis.cli.telemetry_cmd")


# ---------------------------------------------------------------------------
# Typed fake (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Typed fake config with telemetry.db_path."""

    class _TelemetrySection:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    def __init__(self, db_path: str) -> None:
        self.telemetry = self._TelemetrySection(db_path)


def _populate_db(db_path: Path, n: int = 3) -> None:
    """Create a telemetry DB with *n* records."""
    store = TelemetryStore(db_path)
    for i in range(n):
        store.record(TelemetryRecord(
            timestamp=time.time() - (n - i),
            model_id=f"model-{i % 2}",
            engine="ollama",
            prompt_tokens=10 * (i + 1),
            completion_tokens=5 * (i + 1),
            total_tokens=15 * (i + 1),
            latency_seconds=0.5 * (i + 1),
            cost_usd=0.001 * (i + 1),
        ))
    store.close()


@pytest.fixture()
def patched_db(monkeypatch, tmp_path: Path):
    """Patch load_config to use a temp DB."""
    db_path = tmp_path / "telemetry.db"
    cfg = _FakeConfig(db_path=str(db_path))
    monkeypatch.setattr(_telem_mod, "load_config", lambda: cfg)
    return db_path


class TestTelemetrySubcommands:
    @pytest.mark.spec("REQ-cli.telemetry")
    def test_subcommands_exist_in_help(self) -> None:
        result = CliRunner().invoke(cli, ["telemetry", "--help"])
        assert result.exit_code == 0
        assert "stats" in result.output
        assert "export" in result.output
        assert "clear" in result.output


class TestTelemetryStats:
    @pytest.mark.spec("REQ-cli.telemetry")
    def test_stats_empty_db(self, patched_db) -> None:
        store = TelemetryStore(patched_db)
        store.close()
        result = CliRunner().invoke(cli, ["telemetry", "stats"])
        assert result.exit_code == 0
        assert "No telemetry data" in result.output

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_stats_with_data(self, patched_db) -> None:
        _populate_db(patched_db)
        result = CliRunner().invoke(cli, ["telemetry", "stats"])
        assert result.exit_code == 0
        assert "Total Calls" in result.output
        assert "3" in result.output

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_top_flag(self, patched_db) -> None:
        _populate_db(patched_db, n=5)
        result = CliRunner().invoke(cli, ["telemetry", "stats", "-n", "1"])
        assert result.exit_code == 0
        assert "Top 1 Models" in result.output


class TestTelemetryExport:
    @pytest.mark.spec("REQ-cli.telemetry")
    def test_export_json_empty(self, patched_db) -> None:
        store = TelemetryStore(patched_db)
        store.close()
        result = CliRunner().invoke(cli, ["telemetry", "export"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_export_json_with_data(self, patched_db) -> None:
        _populate_db(patched_db)
        result = CliRunner().invoke(cli, ["telemetry", "export"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_export_csv(self, patched_db) -> None:
        _populate_db(patched_db, n=2)
        result = CliRunner().invoke(cli, ["telemetry", "export", "-f", "csv"])
        assert result.exit_code == 0
        lines = result.output.strip().splitlines()
        assert len(lines) == 3  # header + 2 rows
        assert "model_id" in lines[0]

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_export_to_file(self, patched_db, tmp_path) -> None:
        _populate_db(patched_db)
        out_file = tmp_path / "export.json"
        result = CliRunner().invoke(
            cli, ["telemetry", "export", "-o", str(out_file)],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 3


class TestTelemetryClear:
    @pytest.mark.spec("REQ-cli.telemetry")
    def test_clear_empty(self, patched_db) -> None:
        store = TelemetryStore(patched_db)
        store.close()
        result = CliRunner().invoke(cli, ["telemetry", "clear", "--yes"])
        assert result.exit_code == 0
        assert "Deleted 0" in result.output

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_clear_with_yes(self, patched_db) -> None:
        _populate_db(patched_db)
        result = CliRunner().invoke(cli, ["telemetry", "clear", "--yes"])
        assert result.exit_code == 0
        assert "Deleted 3" in result.output

    @pytest.mark.spec("REQ-cli.telemetry")
    def test_clear_abort_without_yes(self, patched_db) -> None:
        _populate_db(patched_db)
        result = CliRunner().invoke(cli, ["telemetry", "clear"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
