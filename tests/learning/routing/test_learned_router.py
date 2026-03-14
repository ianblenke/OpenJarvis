"""Tests for LearnedRouterPolicy (merged trace-driven + SFT routing)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from openjarvis.core.types import StepType, Trace, TraceStep
from openjarvis.learning._stubs import RoutingContext
from openjarvis.learning.routing.learned_router import LearnedRouterPolicy
from openjarvis.traces.analyzer import TraceAnalyzer
from openjarvis.traces.store import TraceStore


def _make_trace(
    query: str = "test",
    model: str = "qwen3:8b",
    outcome: str | None = "success",
    feedback: float | None = 0.8,
) -> Trace:
    now = time.time()
    return Trace(
        query=query,
        agent="orchestrator",
        model=model,
        engine="ollama",
        result="result",
        outcome=outcome,
        feedback=feedback,
        started_at=now,
        ended_at=now + 0.5,
        total_tokens=100,
        total_latency_seconds=0.5,
        steps=[
            TraceStep(
                step_type=StepType.GENERATE,
                timestamp=now,
                duration_seconds=0.5,
                output={"tokens": 100},
            ),
        ],
    )


class TestLearnedRouterPolicy:
    @pytest.mark.spec("REQ-learning.learned-router")
    def test_registered_as_learned(self) -> None:
        from openjarvis.core.registry import RouterPolicyRegistry
        from openjarvis.learning.routing.learned_router import ensure_registered
        ensure_registered()
        assert RouterPolicyRegistry.contains("learned")

    @pytest.mark.spec("REQ-learning.learned-router")
    def test_fallback_no_traces(self) -> None:
        policy = LearnedRouterPolicy(default_model="qwen3:8b")
        ctx = RoutingContext(query="hello")
        assert policy.select_model(ctx) == "qwen3:8b"

    @pytest.mark.spec("REQ-learning.learned-router")
    def test_fallback_chain(self) -> None:
        policy = LearnedRouterPolicy(
            default_model="missing",
            fallback_model="llama3:8b",
            available_models=["llama3:8b"],
        )
        ctx = RoutingContext(query="hello")
        assert policy.select_model(ctx) == "llama3:8b"

    @pytest.mark.spec("REQ-learning.learned-router")
    def test_update_from_traces(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        for _ in range(6):
            store.save(_make_trace(
                query="def foo(): pass",
                model="codestral",
                outcome="success",
                feedback=0.9,
            ))
        for _ in range(6):
            store.save(_make_trace(
                query="def bar(): return 1",
                model="qwen3:8b",
                outcome="failure",
                feedback=0.3,
            ))

        analyzer = TraceAnalyzer(store)
        policy = LearnedRouterPolicy(
            analyzer=analyzer,
            default_model="qwen3:8b",
        )
        policy.min_samples = 3
        result = policy.update_from_traces()
        assert result["updated"] is True

        ctx = RoutingContext(query="import os; def main(): pass")
        assert policy.select_model(ctx) == "codestral"
        store.close()

    @pytest.mark.spec("REQ-learning.learned-router")
    def test_policy_map_readable(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        for _ in range(5):
            store.save(_make_trace(
                query="hello", model="small-model",
                outcome="success",
            ))

        analyzer = TraceAnalyzer(store)
        policy = LearnedRouterPolicy(analyzer=analyzer, default_model="default")
        policy.min_samples = 3
        policy.update_from_traces()

        pmap = policy.policy_map
        assert isinstance(pmap, dict)
        assert "short" in pmap
        assert pmap["short"] == "small-model"
        store.close()

    @pytest.mark.spec("REQ-learning.learned-router")
    def test_observe_online(self) -> None:
        policy = LearnedRouterPolicy(default_model="default")
        policy.min_samples = 3
        policy.observe("hello", "fast-model", "success", 0.9)
        assert policy.policy_map.get("short") == "fast-model"

    @pytest.mark.spec("REQ-learning.learned-router")
    def test_batch_update(self, tmp_path: Path) -> None:
        """Test the batch update() method inherited from SFT routing logic."""
        store = TraceStore(tmp_path / "router.db")
        for query in [
            "def foo(): pass",
            "def bar(): pass",
            "def baz(): pass",
            "def qux(): pass",
            "def quux(): pass",
        ]:
            store.save(_make_trace(
                query=query, model="code-model",
                outcome="success", feedback=0.9,
            ))

        policy = LearnedRouterPolicy(default_model="default")
        result = policy.update(store)
        assert isinstance(result, dict)
        assert "policy_map" in result
        store.close()
