"""Tests for AppleEnergyMonitor -- fake zeus (no real Apple Silicon required)."""

from __future__ import annotations

import platform
import time
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Tests: available()
# ---------------------------------------------------------------------------


class TestAvailable:
    @pytest.mark.spec("REQ-telemetry.energy.apple")
    def test_available_false_on_non_darwin(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        from openjarvis.telemetry.energy_apple import AppleEnergyMonitor

        assert AppleEnergyMonitor.available() is False

    @pytest.mark.spec("REQ-telemetry.energy.apple")
    def test_available_true_without_zeus(self, monkeypatch):
        """Monitor is available on Apple Silicon even without Zeus."""
        import openjarvis.telemetry.energy_apple as mod

        orig = mod._ZEUS_APPLE_AVAILABLE
        mod._ZEUS_APPLE_AVAILABLE = False
        try:
            monkeypatch.setattr(platform, "system", lambda: "Darwin")
            monkeypatch.setattr(platform, "machine", lambda: "arm64")
            assert mod.AppleEnergyMonitor.available() is True
            monitor = mod.AppleEnergyMonitor.__new__(mod.AppleEnergyMonitor)
            monitor._zeus_ok = False
            assert monitor.energy_method() == "cpu_time_estimate"
        finally:
            mod._ZEUS_APPLE_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Tests: energy_method()
# ---------------------------------------------------------------------------


class TestEnergyMethod:
    @pytest.mark.spec("REQ-telemetry.energy.apple")
    def test_returns_zeus(self):
        from openjarvis.telemetry.energy_apple import AppleEnergyMonitor

        monitor = AppleEnergyMonitor.__new__(AppleEnergyMonitor)
        monitor._zeus_ok = True
        assert monitor.energy_method() == "zeus"


# ---------------------------------------------------------------------------
# Tests: sample() component breakdown
# ---------------------------------------------------------------------------


class _FakeZeusMonitor:
    """Typed fake for AppleSiliconMonitor (zeus SDK)."""

    def __init__(
        self,
        cpu_energy: float = 1.5,
        gpu_energy: float = 3.0,
        dram_energy: float = 0.5,
        ane_energy: float = 2.0,
    ) -> None:
        self._measurement = SimpleNamespace(
            cpu_energy=cpu_energy,
            gpu_energy=gpu_energy,
            dram_energy=dram_energy,
            ane_energy=ane_energy,
        )
        self.begin_window_calls = 0
        self.end_window_calls = 0

    def begin_window(self, *args, **kwargs):
        self.begin_window_calls += 1

    def end_window(self, *args, **kwargs):
        self.end_window_calls += 1
        return self._measurement


class TestSampleComponentBreakdown:
    @pytest.mark.spec("REQ-telemetry.energy.apple")
    def test_component_energy_extraction(self):
        """Fake begin_window/end_window and verify cpu/gpu/dram/ane extraction."""
        fake_zeus_monitor = _FakeZeusMonitor(
            cpu_energy=1.5, gpu_energy=3.0, dram_energy=0.5, ane_energy=2.0,
        )

        from openjarvis.telemetry.energy_apple import AppleEnergyMonitor

        monitor = AppleEnergyMonitor.__new__(AppleEnergyMonitor)
        monitor._poll_interval_ms = 50
        monitor._monitor = fake_zeus_monitor
        monitor._zeus_ok = True
        monitor._chip_name = "M1"

        with monitor.sample() as result:
            time.sleep(0.01)

        assert fake_zeus_monitor.begin_window_calls == 1
        assert fake_zeus_monitor.end_window_calls == 1

        assert result.cpu_energy_joules == pytest.approx(1.5)
        assert result.gpu_energy_joules == pytest.approx(3.0)
        assert result.dram_energy_joules == pytest.approx(0.5)
        assert result.ane_energy_joules == pytest.approx(2.0)
        assert result.vendor == "apple"
        assert result.energy_method == "zeus"

    @pytest.mark.spec("REQ-telemetry.energy.apple")
    def test_total_energy_is_sum_of_components(self):
        """total = cpu + gpu + dram + ane."""
        fake_zeus_monitor = _FakeZeusMonitor(
            cpu_energy=1.0, gpu_energy=2.0, dram_energy=0.3, ane_energy=0.7,
        )

        from openjarvis.telemetry.energy_apple import AppleEnergyMonitor

        monitor = AppleEnergyMonitor.__new__(AppleEnergyMonitor)
        monitor._poll_interval_ms = 50
        monitor._monitor = fake_zeus_monitor
        monitor._zeus_ok = True
        monitor._chip_name = "M1"

        with monitor.sample() as result:
            pass

        expected_total = 1.0 + 2.0 + 0.3 + 0.7
        assert result.energy_joules == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# Tests: sample() with uninitialized monitor
# ---------------------------------------------------------------------------


class TestSampleUninitialized:
    @pytest.mark.spec("REQ-telemetry.energy.apple")
    def test_uninitialized_monitor_empty_result(self):
        """When monitor is not initialized, sample yields empty result."""
        from openjarvis.telemetry.energy_apple import AppleEnergyMonitor

        monitor = AppleEnergyMonitor.__new__(AppleEnergyMonitor)
        monitor._poll_interval_ms = 50
        monitor._monitor = None
        monitor._zeus_ok = False
        monitor._chip_name = "Apple Silicon"
        monitor._tdp_watts = 20.0

        with monitor.sample() as result:
            pass

        # CPU-time fallback produces small but non-zero estimates
        assert result.energy_joules >= 0.0
        assert result.cpu_energy_joules >= 0.0
        assert result.gpu_energy_joules >= 0.0
        assert result.dram_energy_joules >= 0.0
        assert result.ane_energy_joules >= 0.0
        assert result.duration_seconds >= 0
        assert result.vendor == "apple"
        assert result.energy_method == "cpu_time_estimate"
