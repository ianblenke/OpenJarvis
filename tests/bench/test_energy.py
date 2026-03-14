"""Tests for the energy benchmark."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from openjarvis.bench.energy import EnergyBenchmark
from openjarvis.core.registry import BenchmarkRegistry
from openjarvis.telemetry.energy_monitor import EnergySample
from tests.fixtures.engines import FakeEngine


@pytest.fixture(autouse=True)
def _register_energy():
    """Re-register energy benchmark after registry clear."""
    from openjarvis.bench.energy import ensure_registered

    ensure_registered()


def _make_engine(completion_tokens=10):
    return FakeEngine(
        engine_id="mock",
        responses=["Hello world"],
        # FakeEngine computes usage from content length; override via wrapper
    )


class _FixedUsageEngine:
    """Thin wrapper around FakeEngine that returns fixed usage tokens."""

    def __init__(self, completion_tokens: int = 10) -> None:
        self._inner = FakeEngine(engine_id="mock", responses=["Hello world"])
        self.engine_id = self._inner.engine_id
        self._completion_tokens = completion_tokens
        self._call_count = 0
        self._side_effect = None

    def generate(self, messages, *, model, **kwargs):
        self._call_count += 1
        if self._side_effect is not None:
            raise self._side_effect
        result = self._inner.generate(messages, model=model, **kwargs)
        result["usage"] = {
            "prompt_tokens": 5,
            "completion_tokens": self._completion_tokens,
            "total_tokens": 5 + self._completion_tokens,
        }
        return result

    @property
    def call_count(self):
        return self._call_count


class _FakeEnergyMonitor:
    """Typed fake energy monitor implementing the sample() context manager."""

    def __init__(self, energy_joules: float = 5.0, power_watts: float = 100.0):
        self._energy_joules = energy_joules
        self._power_watts = power_watts

    def energy_method(self):
        return "polling"

    @contextmanager
    def sample(self):
        yield EnergySample(
            energy_joules=self._energy_joules,
            mean_power_watts=self._power_watts,
        )


class TestEnergyBenchmark:
    def test_registration(self):
        assert BenchmarkRegistry.contains("energy")
        assert BenchmarkRegistry.get("energy") is EnergyBenchmark

    def test_name_and_description(self):
        b = EnergyBenchmark()
        assert b.name == "energy"
        assert "energy" in b.description.lower()

    def test_run_without_energy_monitor(self):
        """Running without an energy monitor should still return metrics."""
        engine = _FixedUsageEngine()
        b = EnergyBenchmark()
        result = b.run(engine, "test-model", num_samples=3, warmup_samples=0)

        assert result.benchmark_name == "energy"
        assert result.model == "test-model"
        assert result.engine == "mock"
        assert result.samples == 3
        assert result.errors == 0
        assert "mean_tokens_per_second" in result.metrics
        assert "total_energy_joules" in result.metrics
        assert result.metrics["total_energy_joules"] == 0.0
        assert result.energy_method == ""

    def test_run_with_fake_energy_monitor(self):
        """Running with a fake energy monitor should populate energy fields."""
        engine = _FixedUsageEngine(completion_tokens=10)
        monitor = _FakeEnergyMonitor(energy_joules=5.0, power_watts=100.0)

        b = EnergyBenchmark()
        result = b.run(
            engine, "test-model", num_samples=3, warmup_samples=0,
            energy_monitor=monitor,
        )

        assert result.benchmark_name == "energy"
        assert result.total_energy_joules == 15.0
        assert result.energy_method == "polling"
        assert result.energy_per_token_joules > 0.0

    def test_warmup_samples_excluded(self):
        """Warmup samples should not be included in measurement metrics."""
        engine = _FixedUsageEngine()
        b = EnergyBenchmark()

        result = b.run(engine, "test-model", num_samples=3, warmup_samples=2)

        assert result.warmup_samples == 2
        assert result.samples == 3
        # warmup (2) + measurement (3) = 5 total calls
        assert engine.call_count == 5

    def test_run_with_errors(self):
        """All errors should result in zero metrics."""
        engine = _FixedUsageEngine()
        engine._side_effect = RuntimeError("fail")
        b = EnergyBenchmark()
        result = b.run(engine, "test-model", num_samples=3, warmup_samples=0)

        assert result.errors == 3
        assert result.metrics.get("mean_tokens_per_second", 0.0) == 0.0
        assert result.metrics.get("total_energy_joules", 0.0) == 0.0

    def test_ensure_registered(self):
        from openjarvis.bench.energy import ensure_registered

        ensure_registered()  # should not raise
        assert BenchmarkRegistry.contains("energy")
