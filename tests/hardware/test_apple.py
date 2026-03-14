"""Apple Silicon hardware tests."""

from __future__ import annotations

import pytest

import openjarvis.core.config as _config_mod
from openjarvis.core.config import (
    GpuInfo,
    HardwareInfo,
    _detect_apple_gpu,
    recommend_engine,
)

pytestmark = pytest.mark.apple


# ---------------------------------------------------------------------------
# Detection / system_profiler parsing
# ---------------------------------------------------------------------------


class TestAppleDetection:
    """Tests for _detect_apple_gpu() against system_profiler outputs."""

    def test_system_profiler_parsing(self, monkeypatch):
        monkeypatch.setattr("openjarvis.core.config.platform.system",
                            lambda: "Darwin")
        monkeypatch.setattr(_config_mod, "_run_cmd", lambda *a, **kw: (
            "Graphics/Displays:\n"
            "\n"
            "    Apple M4 Max:\n"
            "\n"
            "      Chipset Model: Apple M4 Max\n"
            "      Type: GPU\n"
            "      Bus: Built-In\n"
            "      Total Number of Cores: 40\n"
        ))
        gpu = _detect_apple_gpu()
        assert gpu is not None
        assert gpu.vendor == "apple"
        assert "M4 Max" in gpu.name

    def test_non_darwin_returns_none(self, monkeypatch):
        """On non-Darwin platforms, Apple GPU detection returns None."""
        monkeypatch.setattr("openjarvis.core.config.platform.system",
                            lambda: "Linux")
        assert _detect_apple_gpu() is None

    def test_no_apple_in_output(self, monkeypatch):
        """system_profiler output without 'Apple' returns None."""
        monkeypatch.setattr("openjarvis.core.config.platform.system",
                            lambda: "Darwin")
        monkeypatch.setattr(_config_mod, "_run_cmd", lambda *a, **kw: (
            "Graphics/Displays:\n"
            "\n"
            "    Intel UHD Graphics 630:\n"
            "\n"
            "      Chipset Model: Intel UHD Graphics 630\n"
            "      Type: GPU\n"
        ))
        assert _detect_apple_gpu() is None

    def test_apple_chip_model_extraction(self, monkeypatch):
        """'Chipset Model:' line is used to extract the chip name."""
        monkeypatch.setattr("openjarvis.core.config.platform.system",
                            lambda: "Darwin")
        monkeypatch.setattr(_config_mod, "_run_cmd", lambda *a, **kw: (
            "Graphics/Displays:\n"
            "\n"
            "    Apple M2 Ultra:\n"
            "\n"
            "      Chipset Model: Apple M2 Ultra\n"
            "      Type: GPU\n"
            "      Bus: Built-In\n"
        ))
        gpu = _detect_apple_gpu()
        assert gpu is not None
        assert gpu.name == "Apple M2 Ultra"

    def test_apple_no_chipset_line_falls_back(self, monkeypatch):
        """When no 'Chipset Model' line exists, falls back to 'Apple Silicon'."""
        monkeypatch.setattr("openjarvis.core.config.platform.system",
                            lambda: "Darwin")
        monkeypatch.setattr(_config_mod, "_run_cmd", lambda *a, **kw: (
            "Graphics/Displays:\n"
            "    Apple Silicon\n"
            "      Type: GPU\n"
        ))
        gpu = _detect_apple_gpu()
        assert gpu is not None
        assert gpu.vendor == "apple"
        assert gpu.name == "Apple Silicon"

    def test_empty_profiler_output(self, monkeypatch):
        """Empty system_profiler output returns None (no 'Apple' substring)."""
        monkeypatch.setattr("openjarvis.core.config.platform.system",
                            lambda: "Darwin")
        monkeypatch.setattr(_config_mod, "_run_cmd", lambda *a, **kw: "")
        assert _detect_apple_gpu() is None


# ---------------------------------------------------------------------------
# Engine recommendation
# ---------------------------------------------------------------------------


class TestAppleEngineRecommendation:
    """Tests that Apple Silicon hardware maps to mlx."""

    def test_m4_max_recommends_mlx(self):
        hw = HardwareInfo(
            platform="darwin",
            cpu_brand="Apple M4 Max",
            cpu_count=16,
            ram_gb=128.0,
            gpu=GpuInfo(vendor="apple", name="Apple M4 Max", vram_gb=128.0, count=1),
        )
        assert recommend_engine(hw) == "mlx"

    def test_unified_memory(self):
        """On Apple Silicon, GPU VRAM equals system RAM (unified memory)."""
        ram_gb = 192.0
        gpu = GpuInfo(vendor="apple", name="Apple M4 Ultra", vram_gb=ram_gb, count=1)
        hw = HardwareInfo(
            platform="darwin",
            cpu_brand="Apple M4 Ultra",
            cpu_count=24,
            ram_gb=ram_gb,
            gpu=gpu,
        )
        assert hw.gpu.vram_gb == hw.ram_gb
        assert recommend_engine(hw) == "mlx"
