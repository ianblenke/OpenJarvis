"""Tests for the DSPy agent optimizer (no dspy dependency required)."""

from __future__ import annotations

import time
from typing import Any, List

import pytest


class _FakeTraceStore:
    """Typed fake for trace store used by optimizers."""

    def __init__(self, traces: List[Any] = None) -> None:
        self._traces = traces or []

    def list_traces(self, **kwargs) -> List[Any]:
        return self._traces


class TestDSPyOptimizerConfig:
    @pytest.mark.spec("REQ-learning.dspy-optimizer")
    def test_default_config(self) -> None:
        from openjarvis.core.config import DSPyOptimizerConfig

        cfg = DSPyOptimizerConfig()
        assert cfg.optimizer == "BootstrapFewShotWithRandomSearch"
        assert cfg.max_bootstrapped_demos == 4
        assert cfg.min_traces == 20

    @pytest.mark.spec("REQ-learning.dspy-optimizer")
    def test_optimizer_init(self) -> None:
        from openjarvis.core.config import DSPyOptimizerConfig
        from openjarvis.learning.agents.dspy_optimizer import DSPyAgentOptimizer

        cfg = DSPyOptimizerConfig()
        optimizer = DSPyAgentOptimizer(cfg)
        assert optimizer.config is cfg


class TestDSPyOptimizerTraceConversion:
    @pytest.mark.spec("REQ-learning.dspy-optimizer")
    def test_too_few_traces_skipped(self) -> None:
        from openjarvis.core.config import DSPyOptimizerConfig
        from openjarvis.learning.agents.dspy_optimizer import DSPyAgentOptimizer

        optimizer = DSPyAgentOptimizer(DSPyOptimizerConfig(min_traces=10))
        store = _FakeTraceStore(traces=[])

        result = optimizer.optimize(store)
        assert result["status"] == "skipped"

    @pytest.mark.spec("REQ-learning.dspy-optimizer")
    def test_optimize_returns_toml_updates(self, monkeypatch) -> None:
        import openjarvis.learning.agents.dspy_optimizer as mod
        from openjarvis.core.config import DSPyOptimizerConfig
        from openjarvis.core.types import StepType, Trace, TraceStep
        from openjarvis.learning.agents.dspy_optimizer import DSPyAgentOptimizer

        cfg = DSPyOptimizerConfig(min_traces=1)
        optimizer = DSPyAgentOptimizer(cfg)

        now = time.time()
        traces = []
        for i in range(5):
            traces.append(Trace(
                query=f"test query {i}",
                agent="native_react",
                model="qwen3:8b",
                result=f"result {i}",
                outcome="success",
                feedback=0.9,
                started_at=now,
                ended_at=now + 1,
                total_tokens=100,
                total_latency_seconds=1.0,
                steps=[TraceStep(
                    step_type=StepType.GENERATE,
                    timestamp=now,
                    duration_seconds=0.5,
                )],
            ))

        store = _FakeTraceStore(traces=traces)

        monkeypatch.setattr(mod, "HAS_DSPY", True)
        monkeypatch.setattr(optimizer, "_run_dspy_optimization", lambda *a, **kw: {
            "system_prompt": "You are a helpful assistant.",
            "few_shot_examples": [{"input": "hi", "output": "hello"}],
        })

        result = optimizer.optimize(store)
        assert result["status"] == "completed"
        assert "config_updates" in result
