"""Tests for MonitorOperativeAgent."""

import pytest

from openjarvis.agents.monitor_operative import MonitorOperativeAgent
from openjarvis.core.registry import AgentRegistry
from tests.fixtures.engines import FakeEngine


def _make_engine(content: str = "Hello") -> FakeEngine:
    return FakeEngine(engine_id="mock", responses=[content])


class TestMonitorOperativeAgent:
    @pytest.mark.spec("REQ-agents.base.registration")
    def test_registration(self) -> None:
        # Import triggers registration; re-register after autouse fixture
        # clears the registry (same pattern as test_monitor.py)
        import openjarvis.agents.monitor_operative  # noqa: F401
        if not AgentRegistry.contains("monitor_operative"):
            AgentRegistry.register_value("monitor_operative", MonitorOperativeAgent)
        assert AgentRegistry.contains("monitor_operative")
        cls = AgentRegistry.get("monitor_operative")
        assert cls is MonitorOperativeAgent

    @pytest.mark.spec("REQ-agents.monitor")
    def test_instantiation(self) -> None:
        engine = _make_engine()
        agent = MonitorOperativeAgent(engine, "test-model")
        assert agent.agent_id == "monitor_operative"
        assert agent.accepts_tools is True

    @pytest.mark.spec("REQ-agents.monitor")
    def test_default_strategies(self) -> None:
        engine = _make_engine()
        agent = MonitorOperativeAgent(engine, "test-model")
        assert agent._memory_extraction == "causality_graph"
        assert agent._observation_compression == "summarize"
        assert agent._retrieval_strategy == "hybrid_with_self_eval"
        assert agent._task_decomposition == "phased"

    @pytest.mark.spec("REQ-agents.monitor")
    def test_custom_strategies(self) -> None:
        engine = _make_engine()
        agent = MonitorOperativeAgent(
            engine, "test-model",
            memory_extraction="scratchpad",
            observation_compression="none",
            retrieval_strategy="keyword",
            task_decomposition="monolithic",
        )
        assert agent._memory_extraction == "scratchpad"
        assert agent._observation_compression == "none"

    @pytest.mark.spec("REQ-agents.monitor")
    def test_simple_run(self) -> None:
        engine = _make_engine("The answer is 42.")
        agent = MonitorOperativeAgent(engine, "test-model")
        result = agent.run("What is the answer?")
        assert result.content == "The answer is 42."
        assert result.turns >= 1
