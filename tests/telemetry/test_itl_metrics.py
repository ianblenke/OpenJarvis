"""Tests for Tier 3 — per-token timestamps, ITL percentiles, streaming telemetry."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, Role, TelemetryRecord
from openjarvis.telemetry.aggregator import TelemetryAggregator
from openjarvis.telemetry.instrumented_engine import (
    InstrumentedEngine,
    _compute_itl_stats,
    _percentile,
)
from openjarvis.telemetry.store import TelemetryStore

# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeStreamEngine:
    """Typed fake engine that supports both generate() and stream()."""

    def __init__(self, tokens: Optional[List[str]] = None) -> None:
        self.engine_id = "mock"
        self._tokens = tokens if tokens is not None else ["Hello", " ", "world", "!"]

    async def stream(self, *args, **kwargs):
        for tok in self._tokens:
            yield tok

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return {
            "content": "".join(self._tokens),
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": len(self._tokens),
                "total_tokens": 5 + len(self._tokens),
            },
            "model": "m",
            "ttft": 0.01,
        }


class _FakeGenerateEngine:
    """Typed fake engine with configurable generate() behavior."""

    def __init__(
        self,
        generate_fn=None,
        generate_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine_id = "mock"
        self._generate_fn = generate_fn
        self._generate_result = generate_result or {
            "content": "hi",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "m",
            "ttft": 0.0,
        }

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        if self._generate_fn is not None:
            return self._generate_fn(messages, **kwargs)
        return self._generate_result


@dataclass
class _FakeEnergySample:
    """Typed fake for energy monitor sample result."""

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
    """Typed fake energy monitor with configurable sample values."""

    def __init__(
        self,
        energy_joules: float = 10.0,
        power_watts: float = 200.0,
    ) -> None:
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


def _mock_engine_with_stream(tokens=None):
    """Return a typed fake engine whose stream() yields the given tokens."""
    return _FakeStreamEngine(tokens=tokens)


def _mock_energy_monitor(energy_joules=10.0, power_watts=200.0):
    return _FakeEnergyMonitor(
        energy_joules=energy_joules, power_watts=power_watts,
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestPercentile:
    """_percentile() computes interpolated percentiles."""

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_simple_median(self):
        assert _percentile([1, 2, 3, 4, 5], 0.50) == pytest.approx(3.0)

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_p90(self):
        data = list(range(1, 101))  # 1..100
        assert _percentile(data, 0.90) == pytest.approx(90.1)

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_single_value(self):
        assert _percentile([42.0], 0.99) == pytest.approx(42.0)

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_two_values(self):
        assert _percentile([10, 20], 0.50) == pytest.approx(15.0)

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_unsorted_input(self):
        """Input doesn't need to be sorted."""
        assert _percentile([5, 3, 1, 4, 2], 0.50) == pytest.approx(3.0)


class TestComputeItlStats:
    """_compute_itl_stats() computes ITL summary statistics."""

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_empty_list(self):
        stats = _compute_itl_stats([])
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["p90"] == 0.0
        assert stats["p95"] == 0.0
        assert stats["p99"] == 0.0
        assert stats["std"] == 0.0

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_single_value(self):
        stats = _compute_itl_stats([10.0])
        assert stats["mean"] == pytest.approx(10.0)
        assert stats["median"] == pytest.approx(10.0)
        assert stats["std"] == 0.0  # single value

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_known_sequence(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = _compute_itl_stats(values)
        assert stats["mean"] == pytest.approx(30.0)
        assert stats["median"] == pytest.approx(30.0)
        assert stats["p90"] == pytest.approx(46.0)
        assert stats["p95"] == pytest.approx(48.0)
        assert stats["p99"] == pytest.approx(49.6)
        assert stats["std"] > 0


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestStreamTelemetry:
    """InstrumentedEngine.stream() records telemetry with ITL."""

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_stream_creates_telemetry_record(self):
        bus = EventBus()
        engine = _mock_engine_with_stream(["a", "b", "c"])
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        async def run():
            tokens = []
            async for tok in ie.stream(
                [Message(role=Role.USER, content="hi")], model="m"
            ):
                tokens.append(tok)
            return tokens

        tokens = asyncio.run(run())
        assert tokens == ["a", "b", "c"]
        assert len(records) == 1
        rec = records[0]
        assert rec.is_streaming is True
        assert rec.completion_tokens == 3

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_stream_computes_itl(self):
        bus = EventBus()
        engine = _mock_engine_with_stream(["a", "b", "c", "d", "e"])
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        async def run():
            async for _ in ie.stream(
                [Message(role=Role.USER, content="hi")], model="m"
            ):
                pass

        asyncio.run(run())
        rec = records[0]
        # 5 tokens → 4 ITL deltas
        assert rec.mean_itl_ms >= 0
        assert rec.median_itl_ms >= 0
        assert rec.p90_itl_ms >= 0
        assert rec.p95_itl_ms >= 0
        assert rec.p99_itl_ms >= 0

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_stream_with_energy_monitor(self):
        bus = EventBus()
        engine = _mock_engine_with_stream(["x", "y"])
        monitor = _mock_energy_monitor(energy_joules=5.0, power_watts=100.0)
        ie = InstrumentedEngine(engine, bus, energy_monitor=monitor)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        async def run():
            async for _ in ie.stream(
                [Message(role=Role.USER, content="hi")], model="m"
            ):
                pass

        asyncio.run(run())
        rec = records[0]
        assert rec.energy_joules == 5.0
        assert rec.energy_per_output_token_joules == pytest.approx(5.0 / 2)

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_stream_empty_tokens(self):
        bus = EventBus()
        engine = _mock_engine_with_stream([])
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        async def run():
            async for _ in ie.stream(
                [Message(role=Role.USER, content="hi")], model="m"
            ):
                pass

        asyncio.run(run())
        rec = records[0]
        assert rec.completion_tokens == 0
        assert rec.mean_itl_ms == 0.0
        assert rec.ttft == 0.0

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_stream_single_token(self):
        bus = EventBus()
        engine = _mock_engine_with_stream(["only"])
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        async def run():
            async for _ in ie.stream(
                [Message(role=Role.USER, content="hi")], model="m"
            ):
                pass

        asyncio.run(run())
        rec = records[0]
        assert rec.completion_tokens == 1
        # No ITL deltas with single token
        assert rec.mean_itl_ms == 0.0
        assert rec.std_itl_ms == 0.0


class TestGenerateMeanItlApproximation:
    """generate() computes mean_itl_ms from decode_latency/completion_tokens."""

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_mean_itl_computed(self):
        bus = EventBus()

        def _slow_generate(messages, **kwargs):
            time.sleep(0.05)  # ensure latency > ttft
            return {
                "content": "hi",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "model": "m",
                "ttft": 0.01,
            }

        engine = _FakeGenerateEngine(generate_fn=_slow_generate)
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        # decode_latency > 0 because latency > ttft
        assert rec.decode_latency_seconds > 0
        # mean_itl_ms = (decode_latency / completion_tokens) * 1000
        expected = (rec.decode_latency_seconds / 20) * 1000
        assert rec.mean_itl_ms == pytest.approx(expected)
        assert rec.is_streaming is False

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_no_ttft_no_itl(self):
        bus = EventBus()
        engine = _FakeGenerateEngine(generate_result={
            "content": "hi",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "m",
            "ttft": 0.0,
        })
        ie = InstrumentedEngine(engine, bus)

        records = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: records.append(e.data["record"]),
        )

        ie.generate([Message(role=Role.USER, content="hi")], model="m")
        rec = records[0]
        assert rec.mean_itl_ms == 0.0  # no decode_latency -> no ITL


class TestItlStorage:
    """ITL fields are stored and queryable."""

    @pytest.mark.spec("REQ-telemetry.itl")
    def test_store_and_query_itl(self, tmp_path):
        store = TelemetryStore(tmp_path / "test.db")
        store.record(TelemetryRecord(
            timestamp=time.time(),
            model_id="m1",
            engine="mock",
            mean_itl_ms=15.0,
            median_itl_ms=14.0,
            p90_itl_ms=20.0,
            p95_itl_ms=25.0,
            p99_itl_ms=30.0,
            std_itl_ms=5.0,
            is_streaming=True,
        ))

        agg = TelemetryAggregator(tmp_path / "test.db")
        stats = agg.per_model_stats()
        assert len(stats) == 1
        assert stats[0].avg_mean_itl_ms == pytest.approx(15.0)
        assert stats[0].avg_median_itl_ms == pytest.approx(14.0)
        assert stats[0].avg_p95_itl_ms == pytest.approx(25.0)
        agg.close()
        store.close()
