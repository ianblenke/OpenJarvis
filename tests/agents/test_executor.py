"""Tests for AgentExecutor single-tick execution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from openjarvis.agents._stubs import AgentResult
from openjarvis.agents.errors import FatalError, RetryableError
from openjarvis.core.events import EventBus, EventType

# ---------------------------------------------------------------------------
# Typed fake for the "system" object passed to AgentExecutor
# ---------------------------------------------------------------------------


class _FakeSystem:
    """Minimal typed stub replacing MagicMock for the system dependency."""
    pass


@pytest.fixture
def manager():
    from openjarvis.agents.manager import AgentManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AgentManager(db_path=str(Path(tmpdir) / "agents.db"))
        yield mgr
        mgr.close()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def executor(manager, event_bus):
    from openjarvis.agents.executor import AgentExecutor

    mock_system = _FakeSystem()
    ex = AgentExecutor(manager=manager, event_bus=event_bus)
    ex.set_system(mock_system)
    return ex


class TestExecutorBasic:
    @pytest.mark.spec("REQ-agents.executor.events")
    def test_execute_tick_publishes_start_end_events(
        self, executor, manager, event_bus, monkeypatch
    ):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        events = []
        event_bus.subscribe(EventType.AGENT_TICK_START, lambda e: events.append(e))
        event_bus.subscribe(EventType.AGENT_TICK_END, lambda e: events.append(e))

        rv = AgentResult(content="result text")
        monkeypatch.setattr(executor, "_invoke_agent", lambda *a, **kw: rv)
        executor.execute_tick(agent["id"])

        assert len(events) == 2
        assert events[0].event_type == EventType.AGENT_TICK_START
        assert events[1].event_type == EventType.AGENT_TICK_END

    @pytest.mark.spec("REQ-agents.executor.tick")
    def test_execute_tick_updates_run_stats(self, executor, manager, monkeypatch):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")

        rv = AgentResult(content="result text")
        monkeypatch.setattr(executor, "_invoke_agent", lambda *a, **kw: rv)
        executor.execute_tick(agent["id"])

        updated = manager.get_agent(agent["id"])
        assert updated["total_runs"] == 1
        assert updated["status"] == "idle"

    @pytest.mark.spec("REQ-agents.executor.tick")
    def test_execute_tick_sets_running_then_idle(self, executor, manager, monkeypatch):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        statuses = []

        original_start = manager.start_tick

        def track_start(aid):
            original_start(aid)
            statuses.append(manager.get_agent(aid)["status"])

        manager.start_tick = track_start

        rv = AgentResult(content="result")
        monkeypatch.setattr(executor, "_invoke_agent", lambda *a, **kw: rv)
        executor.execute_tick(agent["id"])

        assert statuses == ["running"]
        assert manager.get_agent(agent["id"])["status"] == "idle"

    @pytest.mark.spec("REQ-agents.executor.retry")
    def test_execute_tick_handles_fatal_error(
        self, executor, manager, event_bus, monkeypatch
    ):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        errors = []
        event_bus.subscribe(EventType.AGENT_TICK_ERROR, lambda e: errors.append(e))

        def _raise_fatal(*a, **kw):
            raise FatalError("bad config")

        monkeypatch.setattr(executor, "_invoke_agent", _raise_fatal)
        executor.execute_tick(agent["id"])

        assert manager.get_agent(agent["id"])["status"] == "error"
        assert len(errors) == 1

    @pytest.mark.spec("REQ-agents.executor.retry")
    def test_execute_tick_retries_retryable_error(self, executor, manager, monkeypatch):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        call_count = 0

        def flaky_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("rate limit")
            return AgentResult(content="success")

        monkeypatch.setattr(executor, "_invoke_agent", flaky_invoke)
        monkeypatch.setattr(
            "openjarvis.agents.executor.retry_delay", lambda *a, **kw: 0,
        )
        executor.execute_tick(agent["id"])

        assert call_count == 3
        assert manager.get_agent(agent["id"])["status"] == "idle"

    @pytest.mark.spec("REQ-agents.executor.retry")
    def test_execute_tick_gives_up_after_max_retries(
        self, executor, manager, monkeypatch
    ):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")

        def _always_fail(*a, **kw):
            raise RetryableError("always fails")

        monkeypatch.setattr(executor, "_invoke_agent", _always_fail)
        monkeypatch.setattr(
            "openjarvis.agents.executor.retry_delay", lambda *a, **kw: 0,
        )
        executor.execute_tick(agent["id"])

        assert manager.get_agent(agent["id"])["status"] == "error"

    @pytest.mark.spec("REQ-agents.manager.concurrency")
    def test_execute_tick_concurrency_guard(self, executor, manager, monkeypatch):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        manager.start_tick(agent["id"])  # Simulate already running

        # Second tick should handle the ValueError from start_tick
        rv = AgentResult(content="result")
        monkeypatch.setattr(executor, "_invoke_agent", lambda *a, **kw: rv)
        executor.execute_tick(agent["id"])

        # Agent should still be running (first tick owns it)
        assert manager.get_agent(agent["id"])["status"] == "running"


@pytest.mark.spec("REQ-agents.executor.tick")
def test_finalize_tick_reads_agent_result_metadata(tmp_path):
    """_finalize_tick() accumulates cost/tokens from AgentResult.metadata."""
    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.agents.manager import AgentManager

    mgr = AgentManager(str(tmp_path / "test.db"))
    bus = EventBus()
    executor = AgentExecutor(mgr, bus)

    agent = mgr.create_agent("budget-agent")
    mgr.start_tick(agent["id"])

    result = AgentResult(
        content="done",
        metadata={"tokens_used": 500, "cost": 0.05},
    )
    executor._finalize_tick(agent["id"], result, error=None, duration=1.0)

    updated = mgr.get_agent(agent["id"])
    assert updated["total_tokens"] == 500
    assert updated["total_cost"] == 0.05
    assert updated["stall_retries"] == 0
    mgr.close()
