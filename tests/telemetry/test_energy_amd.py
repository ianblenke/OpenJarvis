"""Tests for AmdEnergyMonitor -- fake amdsmi (no real GPU required)."""

from __future__ import annotations

import sys
import time
import types

import pytest

# ---------------------------------------------------------------------------
# Helpers: build a fake amdsmi module with typed callables
# ---------------------------------------------------------------------------


class _CallTracker:
    """Simple call tracker replacing MagicMock.assert_called()."""

    def __init__(self, return_value=None, side_effect=None):
        self._return_value = return_value
        self._side_effect = side_effect
        self._call_count = 0
        self._call_args_list: list = []

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        self._call_args_list.append((args, kwargs))
        if self._side_effect is not None:
            if callable(self._side_effect):
                return self._side_effect(*args, **kwargs)
            raise self._side_effect
        return self._return_value

    def assert_called(self):
        assert self._call_count > 0, "Expected to be called but was not"

    def assert_called_once(self):
        assert self._call_count == 1, (
            f"Expected exactly one call, got {self._call_count}"
        )

    def reset_mock(self):
        self._call_count = 0
        self._call_args_list = []


def _make_fake_amdsmi(device_count: int = 1):
    """Return a fake amdsmi module object with typed callables."""
    mod = types.ModuleType("amdsmi")
    handles = [f"amd-handle-{i}" for i in range(device_count)]
    mod.amdsmi_init = _CallTracker()
    mod.amdsmi_shut_down = _CallTracker()
    mod.amdsmi_get_processor_handles = _CallTracker(return_value=handles)
    mod.amdsmi_get_gpu_asic_info = _CallTracker(
        return_value={"market_name": "AMD Instinct MI300X"}
    )
    mod.amdsmi_get_energy_count = _CallTracker(
        return_value={"energy_accumulator": 1000.0, "counter_resolution": 15.3}
    )
    return mod


# ---------------------------------------------------------------------------
# Tests: available()
# ---------------------------------------------------------------------------


class TestAvailable:
    @pytest.mark.spec("REQ-telemetry.energy.amd")
    def test_available_true_when_amdsmi_works(self, monkeypatch):
        fake_amdsmi = _make_fake_amdsmi(device_count=1)

        monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)
        import openjarvis.telemetry.energy_amd as mod

        orig = mod._AMDSMI_AVAILABLE
        mod._AMDSMI_AVAILABLE = True
        mod.amdsmi = fake_amdsmi
        try:
            assert mod.AmdEnergyMonitor.available() is True
            fake_amdsmi.amdsmi_init.assert_called()
            fake_amdsmi.amdsmi_shut_down.assert_called()
        finally:
            mod._AMDSMI_AVAILABLE = orig

    @pytest.mark.spec("REQ-telemetry.energy.amd")
    def test_available_false_when_amdsmi_not_importable(self):
        import openjarvis.telemetry.energy_amd as mod

        orig = mod._AMDSMI_AVAILABLE
        mod._AMDSMI_AVAILABLE = False
        try:
            assert mod.AmdEnergyMonitor.available() is False
        finally:
            mod._AMDSMI_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Tests: energy_method()
# ---------------------------------------------------------------------------


class TestEnergyMethod:
    @pytest.mark.spec("REQ-telemetry.energy.amd")
    def test_returns_hw_counter(self, monkeypatch):
        fake_amdsmi = _make_fake_amdsmi(device_count=1)

        monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)
        import openjarvis.telemetry.energy_amd as mod

        orig = mod._AMDSMI_AVAILABLE
        mod._AMDSMI_AVAILABLE = True
        mod.amdsmi = fake_amdsmi
        try:
            monitor = mod.AmdEnergyMonitor(poll_interval_ms=50)
            assert monitor.energy_method() == "hw_counter"
        finally:
            mod._AMDSMI_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Tests: sample() counter delta math
# ---------------------------------------------------------------------------


class TestSampleCounterDelta:
    @pytest.mark.spec("REQ-telemetry.energy.amd")
    def test_counter_delta_microjoules_to_joules(self, monkeypatch):
        """acc_start=1000, acc_end=2000, resolution=15.3 =>
        delta=1000 * 15.3 = 15300 uJ => 0.0153 J."""
        fake_amdsmi = _make_fake_amdsmi(device_count=1)

        call_count = {"n": 0}
        readings = [
            {"energy_accumulator": 1000.0, "counter_resolution": 15.3},
            {"energy_accumulator": 2000.0, "counter_resolution": 15.3},
        ]

        def get_energy(handle):
            idx = min(call_count["n"], len(readings) - 1)
            val = readings[idx]
            call_count["n"] += 1
            return val

        fake_amdsmi.amdsmi_get_energy_count = _CallTracker(side_effect=get_energy)

        monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)
        import openjarvis.telemetry.energy_amd as mod

        orig = mod._AMDSMI_AVAILABLE
        mod._AMDSMI_AVAILABLE = True
        mod.amdsmi = fake_amdsmi
        try:
            monitor = mod.AmdEnergyMonitor(poll_interval_ms=50)
            # Reset for sample()
            call_count["n"] = 0

            with monitor.sample() as result:
                time.sleep(0.01)

            # delta = (2000 - 1000) * 15.3 = 15300 uJ = 0.0153 J
            expected_joules = (2000.0 - 1000.0) * 15.3 / 1e6
            assert result.energy_joules == pytest.approx(expected_joules)
            assert result.gpu_energy_joules == pytest.approx(expected_joules)
            assert result.vendor == "amd"
            assert result.energy_method == "hw_counter"
            assert result.duration_seconds > 0
        finally:
            mod._AMDSMI_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Tests: sample() with no devices
# ---------------------------------------------------------------------------


class TestSampleNoDevices:
    @pytest.mark.spec("REQ-telemetry.energy.amd")
    def test_no_devices_empty_result(self):
        """When no AMD GPUs present, sample yields empty result."""
        from openjarvis.telemetry.energy_amd import AmdEnergyMonitor

        monitor = AmdEnergyMonitor.__new__(AmdEnergyMonitor)
        monitor._poll_interval_ms = 50
        monitor._handles = []
        monitor._device_count = 0
        monitor._device_name = ""
        monitor._initialized = False

        with monitor.sample() as result:
            pass

        assert result.energy_joules == 0.0
        assert result.duration_seconds >= 0
        assert result.vendor == "amd"


# ---------------------------------------------------------------------------
# Tests: close()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.spec("REQ-telemetry.energy.amd")
    def test_close_calls_amdsmi_shut_down(self, monkeypatch):
        fake_amdsmi = _make_fake_amdsmi(device_count=1)

        monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)
        import openjarvis.telemetry.energy_amd as mod

        orig = mod._AMDSMI_AVAILABLE
        mod._AMDSMI_AVAILABLE = True
        mod.amdsmi = fake_amdsmi
        try:
            monitor = mod.AmdEnergyMonitor(poll_interval_ms=50)
            assert monitor._initialized is True

            fake_amdsmi.amdsmi_shut_down.reset_mock()
            monitor.close()

            fake_amdsmi.amdsmi_shut_down.assert_called_once()
            assert monitor._initialized is False
        finally:
            mod._AMDSMI_AVAILABLE = orig
