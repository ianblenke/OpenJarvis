"""Tests for EnergyMonitor ABC, EnergySample, EnergyVendor, and factory."""

from __future__ import annotations

import pytest

from openjarvis.telemetry.energy_monitor import (
    EnergyMonitor,
    EnergySample,
    EnergyVendor,
    create_energy_monitor,
)

# ---------------------------------------------------------------------------
# Tests: EnergySample defaults
# ---------------------------------------------------------------------------


class TestEnergySample:
    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_default_field_values(self):
        s = EnergySample()
        assert s.energy_joules == 0.0
        assert s.mean_power_watts == 0.0
        assert s.peak_power_watts == 0.0
        assert s.duration_seconds == 0.0
        assert s.num_snapshots == 0
        assert s.mean_utilization_pct == 0.0
        assert s.peak_utilization_pct == 0.0
        assert s.mean_memory_used_gb == 0.0
        assert s.peak_memory_used_gb == 0.0
        assert s.mean_temperature_c == 0.0
        assert s.peak_temperature_c == 0.0
        assert s.vendor == ""
        assert s.device_name == ""
        assert s.device_count == 0
        assert s.energy_method == ""
        assert s.cpu_energy_joules == 0.0
        assert s.gpu_energy_joules == 0.0
        assert s.dram_energy_joules == 0.0
        assert s.ane_energy_joules == 0.0


# ---------------------------------------------------------------------------
# Tests: EnergyVendor enum
# ---------------------------------------------------------------------------


class TestEnergyVendor:
    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_enum_values(self):
        assert EnergyVendor.NVIDIA.value == "nvidia"
        assert EnergyVendor.AMD.value == "amd"
        assert EnergyVendor.APPLE.value == "apple"
        assert EnergyVendor.CPU_RAPL.value == "cpu_rapl"

    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_enum_is_str(self):
        assert isinstance(EnergyVendor.NVIDIA, str)
        assert EnergyVendor.AMD == "amd"


# ---------------------------------------------------------------------------
# Tests: EnergyMonitor ABC
# ---------------------------------------------------------------------------


class TestEnergyMonitorABC:
    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            EnergyMonitor()


# ---------------------------------------------------------------------------
# Typed fakes for vendor monitors
# ---------------------------------------------------------------------------


class FakeVendorMonitor:
    """Typed fake for vendor-specific energy monitors.

    Replaces MagicMock for testing the create_energy_monitor factory.
    """

    def __init__(
        self,
        is_available: bool = False,
        poll_interval_ms: int = 50,
    ) -> None:
        self._is_available = is_available
        self.poll_interval_ms = poll_interval_ms
        self.init_called_with: int | None = None

    @classmethod
    def make_cls(
        cls, is_available: bool, call_tracker: list[str] | None = None, name: str = "",
    ) -> type:
        """Dynamically build a fake monitor class with controlled availability.

        Parameters
        ----------
        is_available:
            What ``available()`` should return.
        call_tracker:
            Optional list; appends *name* when ``available()`` is called.
        name:
            Label used in call_tracker.
        """
        tracker = call_tracker
        avail = is_available
        label = name

        class _Cls:
            def __init__(self, poll_interval_ms: int = 50) -> None:
                self.poll_interval_ms = poll_interval_ms

            @staticmethod
            def available() -> bool:
                if tracker is not None:
                    tracker.append(label)
                return avail

        return _Cls


# ---------------------------------------------------------------------------
# Tests: create_energy_monitor factory
# ---------------------------------------------------------------------------


class TestCreateEnergyMonitor:
    def _patch_vendors(
        self,
        monkeypatch,
        nvidia_avail: bool = False,
        amd_avail: bool = False,
        apple_avail: bool = False,
        rapl_avail: bool = False,
        call_tracker: list[str] | None = None,
    ) -> None:
        """Patch all vendor monitor classes with typed fakes via monkeypatch."""
        import openjarvis.telemetry.energy_amd as amd_mod
        import openjarvis.telemetry.energy_apple as apple_mod
        import openjarvis.telemetry.energy_nvidia as nvidia_mod
        import openjarvis.telemetry.energy_rapl as rapl_mod

        monkeypatch.setattr(
            nvidia_mod, "NvidiaEnergyMonitor",
            FakeVendorMonitor.make_cls(nvidia_avail, call_tracker, "nvidia"),
        )
        monkeypatch.setattr(
            amd_mod, "AmdEnergyMonitor",
            FakeVendorMonitor.make_cls(amd_avail, call_tracker, "amd"),
        )
        monkeypatch.setattr(
            apple_mod, "AppleEnergyMonitor",
            FakeVendorMonitor.make_cls(apple_avail, call_tracker, "apple"),
        )
        monkeypatch.setattr(
            rapl_mod, "RaplEnergyMonitor",
            FakeVendorMonitor.make_cls(rapl_avail, call_tracker, "rapl"),
        )

    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_returns_none_when_nothing_available(self, monkeypatch):
        self._patch_vendors(monkeypatch)
        result = create_energy_monitor()
        assert result is None

    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_prefer_vendor_parameter(self, monkeypatch):
        """When prefer_vendor is set, that vendor is tried first."""
        call_order: list[str] = []
        self._patch_vendors(
            monkeypatch, rapl_avail=True, call_tracker=call_order,
        )

        result = create_energy_monitor(prefer_vendor="cpu_rapl")
        # RaplEnergyMonitor was available and preferred
        assert result is not None
        assert result.poll_interval_ms == 50

    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_detection_order_nvidia_first(self, monkeypatch):
        """Default order: NVIDIA is tried before AMD."""
        call_order: list[str] = []
        self._patch_vendors(
            monkeypatch, nvidia_avail=True, amd_avail=True, call_tracker=call_order,
        )

        create_energy_monitor()
        # NVIDIA was tried first and returned True — stops there
        assert call_order == ["nvidia"]

    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_detection_order_falls_through(self, monkeypatch):
        """When NVIDIA unavailable, AMD is tried next."""
        call_order: list[str] = []
        self._patch_vendors(
            monkeypatch, nvidia_avail=False, amd_avail=True, call_tracker=call_order,
        )

        create_energy_monitor()
        assert call_order == ["nvidia", "amd"]

    @pytest.mark.spec("REQ-telemetry.energy.monitor")
    def test_prefer_vendor_tried_first_then_default_order(self, monkeypatch):
        """prefer_vendor=cpu_rapl puts RAPL first, then NVIDIA > AMD > Apple."""
        call_order: list[str] = []
        self._patch_vendors(monkeypatch, call_tracker=call_order)

        result = create_energy_monitor(prefer_vendor="cpu_rapl")
        assert result is None
        assert call_order == ["rapl", "nvidia", "amd", "apple"]
