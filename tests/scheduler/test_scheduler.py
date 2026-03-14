"""Tests for TaskScheduler -- scheduling logic and lifecycle management.

Tests focus on synchronous methods (create, list, pause, resume, cancel)
and the ScheduledTask dataclass. The async polling loop is not tested here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from openjarvis.scheduler.scheduler import ScheduledTask, TaskScheduler
from openjarvis.scheduler.store import SchedulerStore


@pytest.fixture()
def store(tmp_path):
    s = SchedulerStore(tmp_path / "scheduler_test.db")
    yield s
    s.close()


@pytest.fixture()
def scheduler(store):
    sched = TaskScheduler(store, poll_interval=1)
    yield sched
    sched.stop()


# ---------------------------------------------------------------------------
# ScheduledTask dataclass
# ---------------------------------------------------------------------------


class TestScheduledTask:
    @pytest.mark.spec("REQ-scheduler.task")
    @pytest.mark.spec("REQ-scheduler.task.round-trip")
    def test_round_trip_to_dict_from_dict(self) -> None:
        task = ScheduledTask(
            id="abc123",
            prompt="hello",
            schedule_type="interval",
            schedule_value="60",
            agent="orchestrator",
            tools="calculator,think",
            metadata={"key": "value"},
        )
        d = task.to_dict()
        restored = ScheduledTask.from_dict(d)
        assert restored.id == "abc123"
        assert restored.prompt == "hello"
        assert restored.schedule_type == "interval"
        assert restored.schedule_value == "60"
        assert restored.agent == "orchestrator"
        assert restored.tools == "calculator,think"
        assert restored.metadata == {"key": "value"}

    @pytest.mark.spec("REQ-scheduler.task.defaults")
    def test_defaults(self) -> None:
        task = ScheduledTask(
            id="x",
            prompt="p",
            schedule_type="once",
            schedule_value="2026-01-01T00:00:00",
        )
        assert task.context_mode == "isolated"
        assert task.status == "active"
        assert task.agent == "simple"
        assert task.tools == ""
        assert task.metadata == {}
        assert task.next_run is None
        assert task.last_run is None

    @pytest.mark.spec("REQ-scheduler.task.round-trip")
    def test_to_dict_contains_all_keys(self) -> None:
        task = ScheduledTask(
            id="t1", prompt="test", schedule_type="interval", schedule_value="300"
        )
        d = task.to_dict()
        expected_keys = {
            "id", "prompt", "schedule_type", "schedule_value",
            "context_mode", "status", "next_run", "last_run",
            "agent", "tools", "metadata",
        }
        assert set(d.keys()) == expected_keys

    @pytest.mark.spec("REQ-scheduler.task.round-trip")
    def test_from_dict_with_minimal_data(self) -> None:
        d = {
            "id": "t1",
            "prompt": "hello",
            "schedule_type": "once",
            "schedule_value": "2026-01-01T00:00:00",
        }
        task = ScheduledTask.from_dict(d)
        assert task.id == "t1"
        assert task.context_mode == "isolated"
        assert task.status == "active"
        assert task.agent == "simple"
        assert task.tools == ""
        assert task.metadata == {}


# ---------------------------------------------------------------------------
# TaskScheduler create and list
# ---------------------------------------------------------------------------


class TestCreateAndList:
    @pytest.mark.spec("REQ-scheduler.create")
    @pytest.mark.spec("REQ-scheduler.scheduler.create")
    def test_create_task_returns_task_with_id(self, scheduler) -> None:
        task = scheduler.create_task(
            prompt="hello world",
            schedule_type="interval",
            schedule_value="3600",
        )
        assert task.id
        assert len(task.id) == 16
        assert task.prompt == "hello world"
        assert task.schedule_type == "interval"
        assert task.status == "active"

    @pytest.mark.spec("REQ-scheduler.scheduler.create")
    def test_create_task_computes_next_run(self, scheduler) -> None:
        task = scheduler.create_task(
            prompt="test",
            schedule_type="interval",
            schedule_value="3600",
        )
        assert task.next_run is not None

    @pytest.mark.spec("REQ-scheduler.scheduler.create")
    def test_create_task_with_agent_and_tools(self, scheduler) -> None:
        task = scheduler.create_task(
            prompt="hello",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
            agent="orchestrator",
            tools="calculator,think",
        )
        assert task.agent == "orchestrator"
        assert task.tools == "calculator,think"

    @pytest.mark.spec("REQ-scheduler.scheduler.create")
    def test_create_task_persisted_in_store(self, scheduler, store) -> None:
        task = scheduler.create_task("persist test", "interval", "60")
        got = store.get_task(task.id)
        assert got is not None
        assert got["prompt"] == "persist test"

    @pytest.mark.spec("REQ-scheduler.scheduler.list")
    def test_list_tasks_empty(self, scheduler) -> None:
        assert scheduler.list_tasks() == []

    @pytest.mark.spec("REQ-scheduler.scheduler.list")
    def test_list_tasks_returns_all(self, scheduler) -> None:
        scheduler.create_task("a", "interval", "60")
        scheduler.create_task("b", "interval", "120")
        tasks = scheduler.list_tasks()
        assert len(tasks) == 2
        prompts = {t.prompt for t in tasks}
        assert prompts == {"a", "b"}

    @pytest.mark.spec("REQ-scheduler.scheduler.list")
    def test_list_tasks_filter_status(self, scheduler) -> None:
        t1 = scheduler.create_task("a", "interval", "60")
        scheduler.create_task("b", "interval", "120")
        scheduler.pause_task(t1.id)
        active = scheduler.list_tasks(status="active")
        paused = scheduler.list_tasks(status="paused")
        assert len(active) == 1
        assert len(paused) == 1

    @pytest.mark.spec("REQ-scheduler.scheduler.list")
    def test_list_tasks_returns_scheduled_task_objects(self, scheduler) -> None:
        scheduler.create_task("test", "interval", "60")
        tasks = scheduler.list_tasks()
        assert all(isinstance(t, ScheduledTask) for t in tasks)


# ---------------------------------------------------------------------------
# Pause / resume / cancel
# ---------------------------------------------------------------------------


class TestPauseResumeCancel:
    @pytest.mark.spec("REQ-scheduler.management")
    @pytest.mark.spec("REQ-scheduler.lifecycle")
    @pytest.mark.spec("REQ-scheduler.scheduler.pause")
    def test_pause_task(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "60")
        scheduler.pause_task(task.id)
        tasks = scheduler.list_tasks(status="paused")
        assert len(tasks) == 1
        assert tasks[0].status == "paused"

    @pytest.mark.spec("REQ-scheduler.scheduler.resume")
    def test_resume_task(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "60")
        scheduler.pause_task(task.id)
        scheduler.resume_task(task.id)
        tasks = scheduler.list_tasks(status="active")
        assert len(tasks) == 1
        assert tasks[0].status == "active"

    @pytest.mark.spec("REQ-scheduler.scheduler.resume")
    def test_resume_task_recomputes_next_run(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "300")
        scheduler.pause_task(task.id)
        scheduler.resume_task(task.id)
        resumed = scheduler.list_tasks(status="active")
        # next_run should be recomputed (likely different timestamp)
        assert resumed[0].next_run is not None

    @pytest.mark.spec("REQ-scheduler.scheduler.cancel")
    def test_cancel_task(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "60")
        scheduler.cancel_task(task.id)
        tasks = scheduler.list_tasks(status="cancelled")
        assert len(tasks) == 1
        assert tasks[0].next_run is None

    @pytest.mark.spec("REQ-scheduler.scheduler.pause")
    def test_pause_nonexistent_raises(self, scheduler) -> None:
        with pytest.raises(KeyError):
            scheduler.pause_task("nonexistent")

    @pytest.mark.spec("REQ-scheduler.scheduler.resume")
    def test_resume_nonexistent_raises(self, scheduler) -> None:
        with pytest.raises(KeyError):
            scheduler.resume_task("nonexistent")

    @pytest.mark.spec("REQ-scheduler.scheduler.cancel")
    def test_cancel_nonexistent_raises(self, scheduler) -> None:
        with pytest.raises(KeyError):
            scheduler.cancel_task("nonexistent")


# ---------------------------------------------------------------------------
# _compute_next_run
# ---------------------------------------------------------------------------


class TestComputeNextRun:
    @pytest.mark.spec("REQ-scheduler.schedule-types")
    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_interval_computes_future_time(self, scheduler) -> None:
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="interval", schedule_value="300"
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is not None
        parsed = datetime.fromisoformat(next_run)
        diff = (parsed - datetime.now(timezone.utc)).total_seconds()
        assert 295 <= diff <= 310

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_once_not_yet_run(self, scheduler) -> None:
        target = "2099-06-15T12:00:00+00:00"
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="once",
            schedule_value=target, last_run=None,
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run == target

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_once_already_run_returns_none(self, scheduler) -> None:
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="once",
            schedule_value="2099-06-15T12:00:00+00:00",
            last_run="2099-06-15T12:01:00+00:00",
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is None

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_cron_fallback_returns_value(self, scheduler) -> None:
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="cron",
            schedule_value="30 2 * * *",
        )
        next_run = scheduler._compute_next_run(task)
        assert next_run is not None

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_unknown_type_returns_none(self, scheduler) -> None:
        task = ScheduledTask(
            id="t", prompt="p", schedule_type="unknown", schedule_value="x"
        )
        assert scheduler._compute_next_run(task) is None


# ---------------------------------------------------------------------------
# _execute_task (dry-run, no system)
# ---------------------------------------------------------------------------


class TestExecuteTask:
    @pytest.mark.spec("REQ-scheduler.events")
    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_without_system_dry_run(self, store) -> None:
        sched = TaskScheduler(store, poll_interval=1)
        task = sched.create_task("dry run", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert "dry-run" in logs[0]["result"]

    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_once_task_completed_after_run(self, store) -> None:
        sched = TaskScheduler(store, poll_interval=1)
        task = sched.create_task("one-shot", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        updated = store.get_task(task.id)
        assert updated["status"] == "completed"
        assert updated["next_run"] is None
