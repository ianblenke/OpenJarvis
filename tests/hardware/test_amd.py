"""AMD-specific hardware tests."""

from __future__ import annotations

import pytest

import openjarvis.core.config as _config_mod
from openjarvis.core.config import (
    GpuInfo,
    HardwareInfo,
    _detect_amd_gpu,
    recommend_engine,
)

pytestmark = pytest.mark.amd


# ---------------------------------------------------------------------------
# Helpers: side-effect sequencer for _run_cmd
# ---------------------------------------------------------------------------


def _make_run_cmd_seq(outputs: list[str]):
    """Return a callable that yields successive *outputs* on each call."""
    it = iter(outputs)
    def _run_cmd(*args, **kwargs):
        return next(it)
    return _run_cmd


# ---------------------------------------------------------------------------
# Detection / rocm-smi parsing
# ---------------------------------------------------------------------------


class TestAMDDetection:
    """Tests for _detect_amd_gpu() against various rocm-smi outputs."""

    def test_rocm_smi_parsing(self, monkeypatch):
        monkeypatch.setattr("openjarvis.core.config.shutil.which",
                            lambda _n: "/usr/bin/rocm-smi")
        monkeypatch.setattr(_config_mod, "_run_cmd", _make_run_cmd_seq([
            "AMD Instinct MI300X",
            "GPU[0] : vram Total Memory (B): 206158430208",
            "GPU[0] : Some info",
        ]))
        gpu = _detect_amd_gpu()
        assert gpu is not None
        assert gpu.vendor == "amd"
        assert "MI300X" in gpu.name

    def test_rocm_smi_not_found(self, monkeypatch):
        monkeypatch.setattr("openjarvis.core.config.shutil.which", lambda _n: None)
        assert _detect_amd_gpu() is None

    def test_amd_gpu_model(self, monkeypatch):
        """First line of rocm-smi output is used as the GPU name."""
        monkeypatch.setattr("openjarvis.core.config.shutil.which",
                            lambda _n: "/usr/bin/rocm-smi")
        monkeypatch.setattr(_config_mod, "_run_cmd", _make_run_cmd_seq([
            "AMD Instinct MI250X\nAMD Instinct MI250X",
            "",
            "",
        ]))
        gpu = _detect_amd_gpu()
        assert gpu is not None
        assert "MI250X" in gpu.name

    def test_rocm_smi_empty_output(self, monkeypatch):
        """Empty output from rocm-smi --showproductname returns None."""
        monkeypatch.setattr("openjarvis.core.config.shutil.which",
                            lambda _n: "/usr/bin/rocm-smi")
        monkeypatch.setattr(_config_mod, "_run_cmd", _make_run_cmd_seq(["", "", ""]))
        assert _detect_amd_gpu() is None

    def test_amd_vram_parsing(self, monkeypatch):
        """VRAM is parsed from --showmeminfo vram output."""
        monkeypatch.setattr("openjarvis.core.config.shutil.which",
                            lambda _n: "/usr/bin/rocm-smi")
        monkeypatch.setattr(_config_mod, "_run_cmd", _make_run_cmd_seq([
            "AMD Instinct MI300X",
            "GPU[0] : vram Total Memory (B): 206158430208",
            "GPU[0] : Some info",
        ]))
        gpu = _detect_amd_gpu()
        assert gpu is not None
        assert gpu.vram_gb == 192.0

    def test_amd_multi_gpu_count(self, monkeypatch):
        """Multiple GPU entries in --showallinfo are counted."""
        monkeypatch.setattr("openjarvis.core.config.shutil.which",
                            lambda _n: "/usr/bin/rocm-smi")
        monkeypatch.setattr(_config_mod, "_run_cmd", _make_run_cmd_seq([
            "AMD Instinct MI300X",
            (
                "GPU[0] : vram Total Memory (B): 206158430208\n"
                "GPU[0] : vram Total Used Memory (B): 0\n"
                "GPU[1] : vram Total Memory (B): 206158430208\n"
                "GPU[1] : vram Total Used Memory (B): 0\n"
                "GPU[2] : vram Total Memory (B): 206158430208\n"
                "GPU[2] : vram Total Used Memory (B): 0\n"
                "GPU[3] : vram Total Memory (B): 206158430208\n"
                "GPU[3] : vram Total Used Memory (B): 0"
            ),
            (
                "GPU[0] : Info line\n"
                "GPU[1] : Info line\n"
                "GPU[2] : Info line\n"
                "GPU[3] : Info line"
            ),
        ]))
        gpu = _detect_amd_gpu()
        assert gpu is not None
        assert gpu.count == 4

    def test_amd_vram_parse_failure(self, monkeypatch):
        """Garbled VRAM output falls back to 0.0."""
        monkeypatch.setattr("openjarvis.core.config.shutil.which",
                            lambda _n: "/usr/bin/rocm-smi")
        monkeypatch.setattr(_config_mod, "_run_cmd", _make_run_cmd_seq([
            "AMD Instinct MI300X",
            "garbled output with no valid memory info",
            "GPU[0] : Some info",
        ]))
        gpu = _detect_amd_gpu()
        assert gpu is not None
        assert gpu.vram_gb == 0.0


# ---------------------------------------------------------------------------
# Engine recommendation
# ---------------------------------------------------------------------------


class TestAMDEngineRecommendation:
    """Tests that AMD cards map to vllm."""

    def test_mi300x_recommends_vllm(self):
        hw = HardwareInfo(
            platform="linux",
            cpu_brand="AMD EPYC 9654",
            cpu_count=96,
            ram_gb=768.0,
            gpu=GpuInfo(
                vendor="amd", name="AMD Instinct MI300X",
                vram_gb=192.0, count=1,
            ),
        )
        assert recommend_engine(hw) == "vllm"

    def test_amd_generic_recommends_vllm(self):
        hw = HardwareInfo(
            platform="linux",
            cpu_brand="AMD EPYC",
            cpu_count=64,
            ram_gb=256.0,
            gpu=GpuInfo(vendor="amd", name="AMD GPU", vram_gb=0.0, count=1),
        )
        assert recommend_engine(hw) == "vllm"

    def test_amd_multi_gpu_recommends_vllm(self):
        hw = HardwareInfo(
            platform="linux",
            cpu_brand="AMD EPYC 9654",
            cpu_count=128,
            ram_gb=1024.0,
            gpu=GpuInfo(
                vendor="amd", name="AMD Instinct MI300X",
                vram_gb=192.0, count=4,
            ),
        )
        assert recommend_engine(hw) == "vllm"
