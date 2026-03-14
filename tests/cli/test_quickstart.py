"""Tests for ``jarvis quickstart`` command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.config import GpuInfo, HardwareInfo


def _make_hw(*, gpu: bool = True) -> HardwareInfo:
    """Build a real HardwareInfo with or without GPU."""
    gpu_info = (
        GpuInfo(vendor="nvidia", name="Test GPU", vram_gb=24.0, count=1)
        if gpu
        else None
    )
    return HardwareInfo(
        platform="linux",
        cpu_brand="Test CPU",
        cpu_count=8,
        ram_gb=32.0,
        gpu=gpu_info,
    )


class TestQuickstartCommand:
    @pytest.mark.spec("REQ-cli.quickstart")
    def test_registered(self):
        """quickstart should be a registered CLI command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart", "--help"])
        assert result.exit_code == 0
        assert "quickstart" in result.output.lower() or "--help" in result.output

    @pytest.mark.spec("REQ-cli.quickstart")
    def test_happy_path(self, monkeypatch, tmp_path):
        """Full quickstart succeeds when hardware detected and engine healthy."""
        config_path = tmp_path / "config.toml"
        hw = _make_hw(gpu=True)

        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.detect_hardware", lambda: hw,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_PATH", config_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_DIR", tmp_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.generate_default_toml",
            lambda hw: "[engine]\n",
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.recommend_engine",
            lambda hw: "ollama",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: engine health requires live server
            "openjarvis.cli.quickstart_cmd._check_engine_health",
            lambda key: True,
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: model check requires live server
            "openjarvis.cli.quickstart_cmd._check_model_available",
            lambda key: True,
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: test query requires live server
            "openjarvis.cli.quickstart_cmd._test_query",
            lambda key: "Hello!",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart"])
        assert result.exit_code == 0
        assert "1/5" in result.output
        assert "5/5" in result.output

    @pytest.mark.spec("REQ-cli.quickstart")
    def test_skips_config_if_exists(self, monkeypatch, tmp_path):
        """Config step is skipped when config already exists."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[engine]\n")
        hw = _make_hw(gpu=False)

        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.detect_hardware", lambda: hw,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_PATH", config_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_DIR", tmp_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.generate_default_toml",
            lambda hw: "[engine]\n",
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.recommend_engine",
            lambda hw: "ollama",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: engine health requires live server
            "openjarvis.cli.quickstart_cmd._check_engine_health",
            lambda key: True,
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: model check requires live server
            "openjarvis.cli.quickstart_cmd._check_model_available",
            lambda key: True,
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: test query requires live server
            "openjarvis.cli.quickstart_cmd._test_query",
            lambda key: "Hello!",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart"])
        assert result.exit_code == 0
        assert (
            "already exists" in result.output.lower()
            or "skip" in result.output.lower()
        )

    @pytest.mark.spec("REQ-cli.quickstart")
    def test_force_regenerates_config(self, monkeypatch, tmp_path):
        """--force should regenerate config even if it exists."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[old]\n")
        hw = _make_hw(gpu=False)

        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.detect_hardware", lambda: hw,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_PATH", config_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_DIR", tmp_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.generate_default_toml",
            lambda hw: "[engine]\nnew = true\n",
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.recommend_engine",
            lambda hw: "ollama",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: engine health requires live server
            "openjarvis.cli.quickstart_cmd._check_engine_health",
            lambda key: True,
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: model check requires live server
            "openjarvis.cli.quickstart_cmd._check_model_available",
            lambda key: True,
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: test query requires live server
            "openjarvis.cli.quickstart_cmd._test_query",
            lambda key: "Hello!",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart", "--force"])
        assert result.exit_code == 0
        assert "new = true" in config_path.read_text()

    @pytest.mark.spec("REQ-cli.quickstart")
    def test_engine_not_found(self, monkeypatch, tmp_path):
        """Helpful message when engine is unreachable."""
        config_path = tmp_path / "config.toml"
        hw = _make_hw(gpu=False)

        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.detect_hardware", lambda: hw,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_PATH", config_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.DEFAULT_CONFIG_DIR", tmp_path,
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.generate_default_toml",
            lambda hw: "[engine]\n",
        )
        monkeypatch.setattr(
            "openjarvis.cli.quickstart_cmd.recommend_engine",
            lambda hw: "ollama",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: engine health requires live server
            "openjarvis.cli.quickstart_cmd._check_engine_health",
            lambda key: False,
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart"])
        assert result.exit_code == 1
        assert (
            "engine" in result.output.lower()
            or "not reachable" in result.output.lower()
        )
