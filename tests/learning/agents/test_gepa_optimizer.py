"""Tests for the GEPA agent optimizer (no gepa dependency required)."""

from __future__ import annotations

import sys
import time
from typing import Any, List

import pytest


class _FakeTraceStore:
    """Typed fake for trace store used by optimizers."""

    def __init__(self, traces: List[Any] = None) -> None:
        self._traces = traces or []

    def list_traces(self, **kwargs) -> List[Any]:
        return self._traces


class TestGEPAOptimizerConfig:
    @pytest.mark.spec("REQ-learning.gepa-optimizer")
    def test_default_config(self) -> None:
        from openjarvis.core.config import GEPAOptimizerConfig

        cfg = GEPAOptimizerConfig()
        assert cfg.max_metric_calls == 150
        assert cfg.population_size == 10
        assert cfg.min_traces == 20

    @pytest.mark.spec("REQ-learning.gepa-optimizer")
    def test_optimizer_init(self) -> None:
        from openjarvis.core.config import GEPAOptimizerConfig
        from openjarvis.learning.agents.gepa_optimizer import GEPAAgentOptimizer

        cfg = GEPAOptimizerConfig()
        optimizer = GEPAAgentOptimizer(cfg)
        assert optimizer.config is cfg


class TestGEPAOptimizerOptimize:
    @pytest.mark.spec("REQ-learning.gepa-optimizer")
    def test_too_few_traces_skipped(self) -> None:
        from openjarvis.core.config import GEPAOptimizerConfig
        from openjarvis.learning.agents.gepa_optimizer import GEPAAgentOptimizer

        optimizer = GEPAAgentOptimizer(GEPAOptimizerConfig(min_traces=10))
        store = _FakeTraceStore(traces=[])

        result = optimizer.optimize(store)
        assert result["status"] == "skipped"

    @pytest.mark.spec("REQ-learning.gepa-optimizer")
    def test_no_gepa_reports_error(self, monkeypatch) -> None:
        import openjarvis.learning.agents.gepa_optimizer as gepa_mod
        from openjarvis.core.config import GEPAOptimizerConfig
        from openjarvis.core.types import StepType, Trace, TraceStep
        from openjarvis.learning.agents.gepa_optimizer import GEPAAgentOptimizer

        cfg = GEPAOptimizerConfig(min_traces=1)
        optimizer = GEPAAgentOptimizer(cfg)

        now = time.time()
        traces = [Trace(
            query="test", agent="native_react", model="qwen3:8b",
            result="result", outcome="success", feedback=0.9,
            started_at=now, ended_at=now + 1,
            total_tokens=100, total_latency_seconds=1.0,
            steps=[TraceStep(
                step_type=StepType.GENERATE,
                timestamp=now,
                duration_seconds=0.5,
            )],
        )]

        store = _FakeTraceStore(traces=traces)

        # Ensure gepa is not available
        monkeypatch.setitem(sys.modules, "gepa", None)
        monkeypatch.setattr(gepa_mod, "HAS_GEPA", False)

        result = optimizer.optimize(store)
        assert result["status"] == "error"
        assert "gepa" in result["reason"].lower()


class TestOpenJarvisGEPAAdapter:
    @pytest.mark.spec("REQ-learning.gepa-optimizer")
    def test_adapter_init(self) -> None:
        from openjarvis.core.config import GEPAOptimizerConfig
        from openjarvis.learning.agents.gepa_optimizer import (
            OpenJarvisGEPAAdapter,
        )

        store = _FakeTraceStore()
        adapter = OpenJarvisGEPAAdapter(
            store, "native_react", GEPAOptimizerConfig(),
        )
        assert adapter.agent_name == "native_react"
