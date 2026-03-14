"""Tests for Tier 2.1 — phase energy split: decode_latency, prefill/decode energy."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, Role, TelemetryRecord
from openjarvis.telemetry.aggregator import TelemetryAggregator
from openjarvis.telemetry.instrumented_engine import InstrumentedEngine
from openjarvis.telemetry.store import TelemetryStore
from tests.fixtures.engines import FakeEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TtftEngine(FakeEngine):
    """FakeEngine subclass that returns fixed usage + ttft in result."""

    def __init__(self, ttft: float = 0.1) -> None:
        super().__init__(engine_id="mock", responses=["hello world"])
        self._ttft = ttft

    def generate(self, messages, *, model, **kwargs):
        result = super().generate(messages, model=model, **kwargs)
        result["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "total_tokens": 60,
        }
        result["model"] = "test-model"
        result["ttft"] = self._ttft
        return result


@dataclass
class _FakeEnergySample:
    """Typed fake energy sample with all fields the InstrumentedEngine reads."""
    energy_joules: float = 10.0
    mean_power_watts: float = 200.0
    peak_power_watts: float = 200.0
    mean_utilization_pct: float = 80.0
    peak_utilization_pct: float = 95.0
    mean_memory_used_gb: float = 16.0
    peak_memory_used_gb: float = 20.0
    mean_temperature_c: float = 65.0
    peak_temperature_c: float = 72.0
    duration_seconds: float = 0.5
    num_snapshots: int = 10
    energy_method: str = "hw_counter"
    vendor: str = "nvidia"
    cpu_energy_joules: float = 0.0
    gpu_energy_joules: float = 10.0
    dram_energy_joules: float = 0.0


class _FakeEnergyMonitor:
    """Typed fake energy monitor implementing the sample() context manager."""

    def __init__(self, energy_joules: float = 10.0, power_watts: float = 200.0) -> None:
        self._energy_joules = energy_joules
        self._power_watts = power_watts

    @contextmanager
    def sample(self):
        yield _FakeEnergySample(
            energy_joules=self._energy_joules,
            mean_power_watts=self._power_watts,
            peak_power_watts=self._power_watts,
            gpu_energy_joules=self._energy_joules,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecodeLatency:
    """decode_latency = latency - ttft when ttft > 0."""

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_decode_latency_computed(self):
        bus = EventBus()
        engine = _TtftEngine(ttft=0.1)
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        # decode_latency = latency - ttft
        assert rec.decode_latency_seconds == pytest.approx(
            rec.latency_seconds - 0.1
        )

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_decode_latency_zero_when_no_ttft(self):
        bus = EventBus()
        engine = _TtftEngine(ttft=0.0)
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        assert rec.decode_latency_seconds == 0.0


class TestPhaseEnergySplit:
    """prefill_energy + decode_energy ≈ total energy."""

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_energy_split_proportional(self):
        bus = EventBus()
        engine = _TtftEngine(ttft=0.1)
        monitor = _FakeEnergyMonitor(energy_joules=10.0)
        ie = InstrumentedEngine(engine, bus, energy_monitor=monitor)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]

        # Sum should equal total energy
        assert rec.prefill_energy_joules + rec.decode_energy_joules == pytest.approx(
            rec.energy_joules
        )

        # Prefill fraction should be proportional to ttft/latency
        expected_prefill_frac = 0.1 / rec.latency_seconds
        actual_prefill_frac = rec.prefill_energy_joules / rec.energy_joules
        assert actual_prefill_frac == pytest.approx(expected_prefill_frac)

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_no_energy_no_split(self):
        bus = EventBus()
        engine = _TtftEngine(ttft=0.1)
        ie = InstrumentedEngine(engine, bus)  # no energy monitor

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        assert rec.prefill_energy_joules == 0.0
        assert rec.decode_energy_joules == 0.0

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_no_ttft_no_split(self):
        bus = EventBus()
        engine = _TtftEngine(ttft=0.0)
        monitor = _FakeEnergyMonitor(energy_joules=10.0)
        ie = InstrumentedEngine(engine, bus, energy_monitor=monitor)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        # No ttft → no prefill_latency → no split
        assert rec.prefill_energy_joules == 0.0
        assert rec.decode_energy_joules == 0.0

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_latency_equals_ttft_decode_energy_zero(self):
        """When latency == ttft, all energy is prefill."""
        bus = EventBus()
        # We'll use a ttft that's close to the measured latency
        engine = _TtftEngine(ttft=0.001)
        monitor = _FakeEnergyMonitor(energy_joules=5.0)
        ie = InstrumentedEngine(engine, bus, energy_monitor=monitor)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        # prefill + decode should still sum to total
        assert rec.prefill_energy_joules + rec.decode_energy_joules == pytest.approx(
            rec.energy_joules
        )


class TestPhaseEnergyInTelemetryDict:
    """Phase energy fields appear in result['_telemetry']."""

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_telemetry_dict_contains_phase_energy(self):
        bus = EventBus()
        engine = _TtftEngine(ttft=0.1)
        monitor = _FakeEnergyMonitor(energy_joules=10.0)
        ie = InstrumentedEngine(engine, bus, energy_monitor=monitor)

        result = ie.generate([Message(role=Role.USER, content="hi")], model="m")
        t = result["_telemetry"]
        assert "prefill_energy_joules" in t
        assert "decode_energy_joules" in t
        assert "decode_latency_seconds" in t
        assert t["prefill_energy_joules"] + t["decode_energy_joules"] == pytest.approx(
            t["energy_joules"] if t["energy_joules"] > 0 else 0.0
        )


class TestPhaseEnergyStorage:
    """Phase energy fields are stored and queryable."""

    @pytest.mark.spec("REQ-telemetry.store.record")
    def test_store_and_aggregate(self, tmp_path):
        store = TelemetryStore(tmp_path / "test.db")
        store.record(TelemetryRecord(
            timestamp=time.time(),
            model_id="m1",
            engine="mock",
            energy_joules=10.0,
            prefill_energy_joules=3.0,
            decode_energy_joules=7.0,
        ))

        agg = TelemetryAggregator(tmp_path / "test.db")
        stats = agg.per_model_stats()
        assert len(stats) == 1
        assert stats[0].total_prefill_energy_joules == pytest.approx(3.0)
        assert stats[0].total_decode_energy_joules == pytest.approx(7.0)
        agg.close()
        store.close()
