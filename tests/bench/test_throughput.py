"""Tests for the throughput benchmark."""

from __future__ import annotations

import pytest

from openjarvis.bench.throughput import ThroughputBenchmark
from openjarvis.core.registry import BenchmarkRegistry
from tests.fixtures.engines import FakeEngine


@pytest.fixture(autouse=True)
def _register_throughput():
    """Re-register throughput benchmark after registry clear."""
    from openjarvis.bench.throughput import ensure_registered

    ensure_registered()


class _FixedUsageEngine:
    """Thin wrapper around FakeEngine returning fixed completion_tokens."""

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


class TestThroughputBenchmark:
    def test_registration(self):
        assert BenchmarkRegistry.contains("throughput")
        assert BenchmarkRegistry.get("throughput") is ThroughputBenchmark

    def test_run_with_fake(self):
        engine = _FixedUsageEngine()
        b = ThroughputBenchmark()
        result = b.run(engine, "test-model", num_samples=3)
        assert result.benchmark_name == "throughput"
        assert result.model == "test-model"
        assert result.engine == "mock"
        assert result.samples == 3

    def test_metrics_keys(self):
        engine = _FixedUsageEngine()
        b = ThroughputBenchmark()
        result = b.run(engine, "test-model", num_samples=3)
        assert "mean_tokens_per_second" in result.metrics
        assert "total_tokens" in result.metrics
        assert "total_time_seconds" in result.metrics

    def test_tokens_per_second_calc(self):
        engine = _FixedUsageEngine(completion_tokens=10)
        b = ThroughputBenchmark()
        result = b.run(engine, "test-model", num_samples=5)
        # 5 samples * 10 tokens each = 50 total tokens
        assert result.metrics["total_tokens"] == 50.0
        assert result.metrics["mean_tokens_per_second"] > 0

    def test_sample_count(self):
        engine = _FixedUsageEngine()
        b = ThroughputBenchmark()
        b.run(engine, "test-model", num_samples=7)
        assert engine.call_count == 7

    def test_zero_latency_handling(self):
        """All errors should result in 0 tokens_per_second."""
        engine = _FixedUsageEngine()
        engine._side_effect = RuntimeError("fail")
        b = ThroughputBenchmark()
        result = b.run(engine, "test-model", num_samples=3)
        assert result.errors == 3
        assert result.metrics.get("mean_tokens_per_second", 0.0) == 0.0
