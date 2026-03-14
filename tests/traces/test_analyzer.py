"""Tests for the TraceAnalyzer."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from openjarvis.core.types import StepType, Trace, TraceStep
from openjarvis.traces.analyzer import TraceAnalyzer
from openjarvis.traces.store import TraceStore


def _make_trace(
    query: str = "test",
    agent: str = "orchestrator",
    model: str = "qwen3:8b",
    outcome: str | None = None,
    feedback: float | None = None,
    latency: float = 1.0,
    tokens: int = 100,
    tool_name: str | None = None,
    energy: float = 0.0,
) -> Trace:
    now = time.time()
    steps = [
        TraceStep(
            step_type=StepType.GENERATE,
            timestamp=now,
            duration_seconds=latency * 0.8,
            input={"model": model},
            output={
                "tokens": tokens,
                "prompt_tokens": tokens // 2,
                "completion_tokens": tokens // 2,
            },
            metadata={"energy_joules": energy},
        ),
    ]
    if tool_name:
        steps.append(TraceStep(
            step_type=StepType.TOOL_CALL,
            timestamp=now + 0.1,
            duration_seconds=latency * 0.2,
            input={"tool": tool_name},
            output={"success": True},
        ))
    steps.append(TraceStep(
        step_type=StepType.RESPOND,
        timestamp=now + latency,
        duration_seconds=0.0,
        output={"content": "result"},
    ))
    return Trace(
        query=query,
        agent=agent,
        model=model,
        engine="ollama",
        result="result",
        outcome=outcome,
        feedback=feedback,
        started_at=now,
        ended_at=now + latency,
        total_tokens=tokens,
        total_latency_seconds=latency,
        steps=steps,
    )


class TestTraceAnalyzer:
    @pytest.mark.spec("REQ-traces.analyzer.summary-empty")
    def test_empty_summary(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        analyzer = TraceAnalyzer(store)
        summary = analyzer.summary()
        assert summary.total_traces == 0
        assert summary.total_steps == 0
        assert summary.avg_latency == 0.0
        assert summary.avg_tokens == 0.0
        assert summary.success_rate == 0.0
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.summary")
    def test_summary(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(outcome="success", latency=1.0, tokens=100))
        store.save(_make_trace(outcome="success", latency=2.0, tokens=200))
        store.save(_make_trace(outcome="failure", latency=0.5, tokens=50))

        analyzer = TraceAnalyzer(store)
        summary = analyzer.summary()
        assert summary.total_traces == 3
        assert summary.avg_latency > 0
        assert summary.avg_tokens > 0
        assert summary.success_rate == pytest.approx(2 / 3)
        assert "generate" in summary.step_type_distribution
        assert "respond" in summary.step_type_distribution
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.summary-steps-per-trace")
    def test_summary_steps_per_trace(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        # Each trace has 2 steps (generate + respond)
        store.save(_make_trace())
        store.save(_make_trace())

        analyzer = TraceAnalyzer(store)
        summary = analyzer.summary()
        assert summary.total_traces == 2
        assert summary.total_steps == 4
        assert summary.avg_steps_per_trace == pytest.approx(2.0)
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.summary-energy")
    def test_summary_energy_tracking(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(energy=1.5))
        store.save(_make_trace(energy=2.5))

        analyzer = TraceAnalyzer(store)
        summary = analyzer.summary()
        assert summary.total_energy_joules == pytest.approx(4.0)
        assert summary.total_generate_energy_joules == pytest.approx(4.0)
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.summary-step-type-stats")
    def test_summary_step_type_stats(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(latency=1.0))
        store.save(_make_trace(latency=2.0))

        analyzer = TraceAnalyzer(store)
        summary = analyzer.summary()
        assert "generate" in summary.step_type_stats
        gen = summary.step_type_stats["generate"]
        assert gen.count == 2
        assert gen.avg_duration > 0
        assert gen.min_duration <= gen.max_duration
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.per-route-stats")
    def test_per_route_stats(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(
            model="qwen3:8b", agent="simple",
            outcome="success", feedback=0.9,
        ))
        store.save(_make_trace(
            model="qwen3:8b", agent="simple",
            outcome="success", feedback=0.8,
        ))
        store.save(_make_trace(
            model="llama3:70b", agent="orchestrator",
            outcome="failure",
        ))

        analyzer = TraceAnalyzer(store)
        stats = analyzer.per_route_stats()
        assert len(stats) == 2

        qwen_stats = [s for s in stats if s.model == "qwen3:8b"][0]
        assert qwen_stats.count == 2
        assert qwen_stats.agent == "simple"
        assert qwen_stats.success_rate == 1.0
        assert abs(qwen_stats.avg_feedback - 0.85) < 1e-9

        llama_stats = [s for s in stats if s.model == "llama3:70b"][0]
        assert llama_stats.count == 1
        assert llama_stats.success_rate == 0.0
        assert llama_stats.avg_feedback is None
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.per-route-stats-no-evaluated")
    def test_per_route_stats_no_evaluated(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(outcome=None))  # unknown outcome

        analyzer = TraceAnalyzer(store)
        stats = analyzer.per_route_stats()
        assert len(stats) == 1
        assert stats[0].success_rate == 0.0  # no evaluated traces
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.per-tool-stats")
    def test_per_tool_stats(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(tool_name="calculator"))
        store.save(_make_trace(tool_name="calculator"))
        store.save(_make_trace(tool_name="web_search"))
        store.save(_make_trace())  # no tool

        analyzer = TraceAnalyzer(store)
        stats = analyzer.per_tool_stats()
        assert len(stats) == 2

        calc = [s for s in stats if s.tool_name == "calculator"][0]
        assert calc.call_count == 2
        assert calc.success_rate == 1.0
        assert calc.avg_latency > 0

        web = [s for s in stats if s.tool_name == "web_search"][0]
        assert web.call_count == 1
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.per-tool-stats-empty")
    def test_per_tool_stats_empty(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace())  # no tool

        analyzer = TraceAnalyzer(store)
        stats = analyzer.per_tool_stats()
        assert len(stats) == 0
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.export-traces")
    def test_export_traces(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(query="q1"))
        store.save(_make_trace(query="q2"))

        analyzer = TraceAnalyzer(store)
        exported = analyzer.export_traces()
        assert len(exported) == 2
        assert all(isinstance(e, dict) for e in exported)
        assert exported[0]["query"] in ("q1", "q2")
        assert "steps" in exported[0]
        assert len(exported[0]["steps"]) > 0
        # Verify exported dict structure
        for e in exported:
            assert "trace_id" in e
            assert "agent" in e
            assert "model" in e
            assert "engine" in e
            assert "result" in e
            assert "started_at" in e
            assert "ended_at" in e
            assert "total_tokens" in e
            assert "total_latency_seconds" in e
            assert "metadata" in e
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.export-traces-limit")
    def test_export_traces_limit(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        for i in range(10):
            store.save(_make_trace(query=f"q{i}"))

        analyzer = TraceAnalyzer(store)
        exported = analyzer.export_traces(limit=3)
        assert len(exported) == 3
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.export-traces-step-structure")
    def test_export_traces_step_structure(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(tool_name="calculator"))

        analyzer = TraceAnalyzer(store)
        exported = analyzer.export_traces()
        steps = exported[0]["steps"]
        for s in steps:
            assert "step_type" in s
            assert "timestamp" in s
            assert "duration_seconds" in s
            assert "input" in s
            assert "output" in s
            assert "metadata" in s
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.traces-for-query-type-code")
    def test_traces_for_query_type_code(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(query="def foo(): pass"))
        store.save(_make_trace(query="what is the weather"))

        analyzer = TraceAnalyzer(store)
        code_traces = analyzer.traces_for_query_type(has_code=True)
        assert len(code_traces) == 1
        assert "def foo" in code_traces[0].query
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.traces-for-query-type-length")
    def test_traces_for_query_type_length(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        store.save(_make_trace(query="hi"))
        store.save(_make_trace(query="a" * 200))

        analyzer = TraceAnalyzer(store)
        long_traces = analyzer.traces_for_query_type(min_length=100)
        assert len(long_traces) == 1

        short_traces = analyzer.traces_for_query_type(max_length=10)
        assert len(short_traces) == 1
        assert short_traces[0].query == "hi"
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.per-tool-stats-failed")
    def test_per_tool_stats_with_failed_tool(self, tmp_path: Path) -> None:
        """Exercise branch 221->212: tool step with success=False."""
        store = TraceStore(tmp_path / "test.db")
        now = time.time()
        # Create a trace with a tool step that has success=False
        trace = Trace(
            query="test",
            agent="orchestrator",
            model="qwen3:8b",
            engine="ollama",
            result="error",
            outcome="failure",
            started_at=now,
            ended_at=now + 1.0,
            total_tokens=50,
            total_latency_seconds=1.0,
            steps=[
                TraceStep(
                    step_type=StepType.GENERATE,
                    timestamp=now,
                    duration_seconds=0.5,
                    input={"model": "qwen3:8b"},
                    output={"tokens": 50, "prompt_tokens": 25, "completion_tokens": 25},
                ),
                TraceStep(
                    step_type=StepType.TOOL_CALL,
                    timestamp=now + 0.5,
                    duration_seconds=0.3,
                    input={"tool": "failing_tool"},
                    output={"success": False},
                ),
                TraceStep(
                    step_type=StepType.RESPOND,
                    timestamp=now + 0.8,
                    duration_seconds=0.0,
                    output={"content": "error"},
                ),
            ],
        )
        store.save(trace)

        analyzer = TraceAnalyzer(store)
        stats = analyzer.per_tool_stats()
        assert len(stats) == 1
        failing = stats[0]
        assert failing.tool_name == "failing_tool"
        assert failing.call_count == 1
        assert failing.success_rate == 0.0  # success was False
        store.close()

    @pytest.mark.spec("REQ-traces.analyzer.summary-time-range")
    def test_summary_time_range(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "test.db")
        now = time.time()
        t1 = _make_trace(query="old", outcome="success")
        t1.started_at = now - 7200
        t2 = _make_trace(query="recent", outcome="failure")
        t2.started_at = now
        store.save(t1)
        store.save(t2)

        analyzer = TraceAnalyzer(store)
        summary = analyzer.summary(since=now - 60)
        assert summary.total_traces == 1
        store.close()
