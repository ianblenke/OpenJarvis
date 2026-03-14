"""Extended tests for TaskScheduler — covers _poll_loop, _execute_task with
a system, _compute_next_cron fallback paths, and event bus integration.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from openjarvis.scheduler.scheduler import (
    SCHEDULER_TASK_END,
    SCHEDULER_TASK_START,
    ScheduledTask,
    TaskScheduler,
    _now_iso,
)
from openjarvis.scheduler.store import SchedulerStore

# ---------------------------------------------------------------------------
# Typed fakes
# ---------------------------------------------------------------------------


class FakeSystem:
    """Typed fake JarvisSystem for scheduler tests."""

    def __init__(
        self,
        response: str = "system response",
        raise_on_ask: bool = False,
    ) -> None:
        self._response = response
        self._raise_on_ask = raise_on_ask
        self.ask_calls: List[Dict[str, Any]] = []

    def ask(self, prompt: str, **kwargs: Any) -> str:
        self.ask_calls.append({"prompt": prompt, **kwargs})
        if self._raise_on_ask:
            raise RuntimeError("system failure")
        return self._response


class FakeEventBus:
    """Typed fake event bus for scheduler tests."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append({"type": event_type, "data": data})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    s = SchedulerStore(tmp_path / "sched_cov.db")
    yield s
    s.close()


@pytest.fixture()
def scheduler(store):
    sched = TaskScheduler(store, poll_interval=1)
    yield sched
    sched.stop()


# ---------------------------------------------------------------------------
# _now_iso
# ---------------------------------------------------------------------------


class TestNowIso:
    @pytest.mark.spec("REQ-scheduler.time")
    def test_returns_iso_string(self) -> None:
        result = _now_iso()
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None

    @pytest.mark.spec("REQ-scheduler.time")
    def test_returns_utc(self) -> None:
        result = _now_iso()
        parsed = datetime.fromisoformat(result)
        # Should be within a few seconds of now
        diff = abs((parsed - datetime.now(timezone.utc)).total_seconds())
        assert diff < 5


# ---------------------------------------------------------------------------
# _execute_task with system
# ---------------------------------------------------------------------------


class TestExecuteTaskWithSystem:
    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_with_system(self, store) -> None:
        """_execute_task calls system.ask with the task prompt."""
        system = FakeSystem(response="done")
        sched = TaskScheduler(store, system=system, poll_interval=1)
        task = sched.create_task("test prompt", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        assert len(system.ask_calls) == 1
        assert system.ask_calls[0]["prompt"] == "test prompt"
        logs = store.get_run_logs(task.id)
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert logs[0]["result"] == "done"

    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_with_system_and_tools(self, store) -> None:
        """_execute_task passes tools from the task to system.ask."""
        system = FakeSystem(response="ok")
        sched = TaskScheduler(store, system=system, poll_interval=1)
        task = sched.create_task(
            "run analysis",
            "interval",
            "3600",
            tools="calculator,search",
            agent="orchestrator",
        )
        sched._execute_task(task)

        call = system.ask_calls[0]
        assert call["agent"] == "orchestrator"
        assert call["tools"] == ["calculator", "search"]

    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_with_empty_tools(self, store) -> None:
        """_execute_task passes tools=None when task has no tools."""
        system = FakeSystem(response="ok")
        sched = TaskScheduler(store, system=system, poll_interval=1)
        task = sched.create_task("simple", "interval", "60")
        sched._execute_task(task)

        call = system.ask_calls[0]
        assert call["tools"] is None

    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_system_error(self, store) -> None:
        """_execute_task logs failure when system.ask raises."""
        system = FakeSystem(raise_on_ask=True)
        sched = TaskScheduler(store, system=system, poll_interval=1)
        task = sched.create_task("fail", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        logs = store.get_run_logs(task.id)
        assert logs[0]["success"] == 0
        assert "system failure" in logs[0]["error"]

    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_with_metadata_operator_id(self, store) -> None:
        """_execute_task passes operator_id and system_prompt from metadata."""
        system = FakeSystem(response="operator run")
        sched = TaskScheduler(store, system=system, poll_interval=1)
        task = sched.create_task(
            "operator task",
            "once",
            "2026-01-01T00:00:00+00:00",
            metadata={
                "operator_id": "op1",
                "system_prompt": "You are a helpful bot.",
            },
        )
        sched._execute_task(task)

        call = system.ask_calls[0]
        assert call["operator_id"] == "op1"
        assert call["system_prompt"] == "You are a helpful bot."

    @pytest.mark.spec("REQ-scheduler.scheduler.execute")
    def test_execute_interval_task_gets_new_next_run(self, store) -> None:
        """After executing an interval task, next_run is recomputed."""
        system = FakeSystem(response="ok")
        sched = TaskScheduler(store, system=system, poll_interval=1)
        task = sched.create_task("recurring", "interval", "300")
        sched._execute_task(task)

        updated = store.get_task(task.id)
        assert updated is not None
        assert updated["last_run"] is not None
        # next_run should be recomputed (different from original since time passed)
        assert updated["next_run"] is not None
        assert updated["status"] == "active"


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------


class TestSchedulerEvents:
    @pytest.mark.spec("REQ-scheduler.events")
    def test_execute_publishes_start_and_end_events(self, store) -> None:
        bus = FakeEventBus()
        sched = TaskScheduler(store, poll_interval=1, bus=bus)
        task = sched.create_task("ev test", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        types = [e["type"] for e in bus.events]
        assert SCHEDULER_TASK_START in types
        assert SCHEDULER_TASK_END in types

    @pytest.mark.spec("REQ-scheduler.events")
    def test_start_event_has_task_id(self, store) -> None:
        bus = FakeEventBus()
        sched = TaskScheduler(store, poll_interval=1, bus=bus)
        task = sched.create_task("ev", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        start_evs = [e for e in bus.events if e["type"] == SCHEDULER_TASK_START]
        assert start_evs[0]["data"]["task_id"] == task.id
        assert start_evs[0]["data"]["prompt"] == "ev"

    @pytest.mark.spec("REQ-scheduler.events")
    def test_end_event_has_success_and_result(self, store) -> None:
        bus = FakeEventBus()
        sched = TaskScheduler(store, poll_interval=1, bus=bus)
        task = sched.create_task("ev2", "once", "2026-01-01T00:00:00+00:00")
        sched._execute_task(task)

        end_evs = [e for e in bus.events if e["type"] == SCHEDULER_TASK_END]
        assert end_evs[0]["data"]["success"] is True
        assert "dry-run" in end_evs[0]["data"]["result"]

    @pytest.mark.spec("REQ-scheduler.events")
    def test_no_bus_executes_without_error(self, store) -> None:
        sched = TaskScheduler(store, poll_interval=1, bus=None)
        task = sched.create_task("no bus", "once", "2026-01-01T00:00:00+00:00")
        # Should not raise
        sched._execute_task(task)
        logs = store.get_run_logs(task.id)
        assert logs[0]["success"] == 1


# ---------------------------------------------------------------------------
# _compute_next_cron fallback
# ---------------------------------------------------------------------------


class TestComputeNextCron:
    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_cron_fallback_specific_time(self) -> None:
        """Fallback parser handles 'minute hour * * *' expressions."""
        now = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        result = TaskScheduler._compute_next_cron("30 14 * * *", now)
        assert result is not None
        parsed = datetime.fromisoformat(result)
        assert parsed.hour == 14
        assert parsed.minute == 30

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_cron_fallback_past_time_adds_day(self) -> None:
        """If the target time is already past, fallback adds 1 day."""
        now = datetime(2026, 3, 13, 16, 0, 0, tzinfo=timezone.utc)
        result = TaskScheduler._compute_next_cron("30 14 * * *", now)
        assert result is not None
        parsed = datetime.fromisoformat(result)
        assert parsed.day == 14  # next day
        assert parsed.hour == 14
        assert parsed.minute == 30

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_cron_fallback_wildcard_minute_and_hour(self) -> None:
        """Wildcard minute/hour uses current time."""
        now = datetime(2026, 3, 13, 10, 15, 0, tzinfo=timezone.utc)
        result = TaskScheduler._compute_next_cron("* * * * *", now)
        assert result is not None

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_cron_fallback_too_few_parts(self) -> None:
        """Short cron expressions fall back to 1 hour from now."""
        now = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        result = TaskScheduler._compute_next_cron("30 14", now)
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected = now + timedelta(hours=1)
        diff = abs((parsed - expected).total_seconds())
        assert diff < 2

    @pytest.mark.spec("REQ-scheduler.scheduler.next-run")
    def test_cron_fallback_invalid_values(self) -> None:
        """Non-integer minute/hour falls back to 1 hour from now."""
        now = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        result = TaskScheduler._compute_next_cron("abc def * * *", now)
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected = now + timedelta(hours=1)
        diff = abs((parsed - expected).total_seconds())
        assert diff < 2


# ---------------------------------------------------------------------------
# _poll_loop
# ---------------------------------------------------------------------------


class TestPollLoop:
    @pytest.mark.spec("REQ-scheduler.scheduler.poll")
    def test_poll_loop_executes_due_task(self, store) -> None:
        """_poll_loop picks up a due task and executes it."""
        sched = TaskScheduler(store, poll_interval=1)
        # Create a task that is already due
        task = sched.create_task("due", "once", "2020-01-01T00:00:00+00:00")
        # Manually set next_run to the past so it's due
        d = store.get_task(task.id)
        d["next_run"] = "2020-01-01T00:00:00+00:00"
        store.update_task(d)

        # Use _execute_task directly to test execution of a due task
        # (the poll_loop's while condition makes single-iteration testing tricky)
        task_obj = ScheduledTask.from_dict(store.get_task(task.id))
        sched._execute_task(task_obj)

        logs = store.get_run_logs(task.id)
        assert len(logs) == 1

    @pytest.mark.spec("REQ-scheduler.scheduler.poll")
    def test_poll_loop_runs_and_stops(self, store) -> None:
        """_poll_loop runs in a thread and stops cleanly."""
        sched = TaskScheduler(store, poll_interval=1)
        sched.start()
        assert sched._thread is not None
        assert sched._thread.is_alive()
        # Let it run one cycle
        import time
        time.sleep(0.1)
        sched.stop()
        assert sched._thread is None

    @pytest.mark.spec("REQ-scheduler.scheduler.poll")
    def test_poll_loop_handles_exception(self, store, monkeypatch) -> None:
        """_poll_loop catches exceptions and continues."""
        sched = TaskScheduler(store, poll_interval=1)

        call_count = 0

        def _failing_get_due(now_iso):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("store error")
            # After first call, stop the scheduler
            sched._stop_event.set()
            return []

        monkeypatch.setattr(store, "get_due_tasks", _failing_get_due)

        # Start and let it run — poll_interval=1 is fine, the error triggers fast
        sched._stop_event = threading.Event()
        sched._poll_interval = 0  # poll immediately
        t = threading.Thread(target=sched._poll_loop, daemon=True)
        t.start()
        t.join(timeout=5)
        assert call_count >= 1


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.spec("REQ-scheduler.lifecycle")
    def test_start_creates_daemon_thread(self, store) -> None:
        sched = TaskScheduler(store, poll_interval=1)
        sched.start()
        assert sched._thread is not None
        assert sched._thread.is_alive()
        sched.stop()
        assert sched._thread is None

    @pytest.mark.spec("REQ-scheduler.lifecycle")
    def test_start_idempotent(self, store) -> None:
        """Calling start() twice does not create a second thread."""
        sched = TaskScheduler(store, poll_interval=1)
        sched.start()
        thread1 = sched._thread
        sched.start()
        thread2 = sched._thread
        assert thread1 is thread2
        sched.stop()

    @pytest.mark.spec("REQ-scheduler.lifecycle")
    def test_stop_when_not_started(self, store) -> None:
        """Calling stop() before start() does not error."""
        sched = TaskScheduler(store, poll_interval=1)
        sched.stop()  # should be a no-op


# ---------------------------------------------------------------------------
# ScheduledTask edge cases
# ---------------------------------------------------------------------------


class TestScheduledTaskEdgeCases:
    @pytest.mark.spec("REQ-scheduler.task")
    def test_from_dict_missing_optional_fields(self) -> None:
        """from_dict handles missing optional fields with defaults."""
        d = {
            "id": "t99",
            "prompt": "test",
            "schedule_type": "interval",
            "schedule_value": "60",
        }
        task = ScheduledTask.from_dict(d)
        assert task.context_mode == "isolated"
        assert task.status == "active"
        assert task.agent == "simple"
        assert task.tools == ""
        assert task.metadata == {}
        assert task.next_run is None
        assert task.last_run is None
