"""Tests for scheduler MCP tools using real TaskScheduler (no mocks)."""

from __future__ import annotations

import json

import pytest

from openjarvis.scheduler.scheduler import TaskScheduler
from openjarvis.scheduler.store import SchedulerStore
from openjarvis.scheduler.tools import (
    CancelScheduledTaskTool,
    ListScheduledTasksTool,
    PauseScheduledTaskTool,
    ResumeScheduledTaskTool,
    ScheduleTaskTool,
)


@pytest.fixture()
def store(tmp_path):
    s = SchedulerStore(tmp_path / "scheduler_tools_test.db")
    yield s
    s.close()


@pytest.fixture()
def scheduler(store):
    sched = TaskScheduler(store, poll_interval=1)
    yield sched
    sched.stop()


# ---------------------------------------------------------------------------
# Spec correctness
# ---------------------------------------------------------------------------


class TestToolSpecs:
    @pytest.mark.spec("REQ-scheduler.tools.spec")
    def test_schedule_task_spec(self) -> None:
        tool = ScheduleTaskTool()
        assert tool.spec.name == "schedule_task"
        assert tool.tool_id == "schedule_task"
        assert "prompt" in tool.spec.parameters["properties"]
        assert "schedule_type" in tool.spec.parameters["properties"]
        assert "schedule_value" in tool.spec.parameters["properties"]
        assert tool.spec.parameters["required"] == [
            "prompt", "schedule_type", "schedule_value"
        ]

    @pytest.mark.spec("REQ-scheduler.tools.spec")
    def test_list_spec(self) -> None:
        tool = ListScheduledTasksTool()
        assert tool.spec.name == "list_scheduled_tasks"
        assert tool.tool_id == "list_scheduled_tasks"

    @pytest.mark.spec("REQ-scheduler.tools.spec")
    def test_pause_spec(self) -> None:
        tool = PauseScheduledTaskTool()
        assert tool.spec.name == "pause_scheduled_task"
        assert tool.spec.parameters["required"] == ["task_id"]

    @pytest.mark.spec("REQ-scheduler.tools.spec")
    def test_resume_spec(self) -> None:
        tool = ResumeScheduledTaskTool()
        assert tool.spec.name == "resume_scheduled_task"
        assert tool.spec.parameters["required"] == ["task_id"]

    @pytest.mark.spec("REQ-scheduler.tools.spec")
    def test_cancel_spec(self) -> None:
        tool = CancelScheduledTaskTool()
        assert tool.spec.name == "cancel_scheduled_task"
        assert tool.spec.parameters["required"] == ["task_id"]

    @pytest.mark.spec("REQ-scheduler.tools.spec")
    def test_all_tools_have_scheduler_category(self) -> None:
        for cls in [
            ScheduleTaskTool,
            ListScheduledTasksTool,
            PauseScheduledTaskTool,
            ResumeScheduledTaskTool,
            CancelScheduledTaskTool,
        ]:
            assert cls().spec.category == "scheduler"


# ---------------------------------------------------------------------------
# Scheduler not available
# ---------------------------------------------------------------------------


class TestNoScheduler:
    @pytest.mark.spec("REQ-scheduler.tools.no-scheduler")
    def test_schedule_task_no_scheduler(self) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = None
        result = tool.execute(
            prompt="hello", schedule_type="once", schedule_value="2026-01-01"
        )
        assert not result.success
        assert "not available" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.no-scheduler")
    def test_list_no_scheduler(self) -> None:
        tool = ListScheduledTasksTool()
        tool._scheduler = None
        result = tool.execute()
        assert not result.success
        assert "not available" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.no-scheduler")
    def test_pause_no_scheduler(self) -> None:
        tool = PauseScheduledTaskTool()
        tool._scheduler = None
        result = tool.execute(task_id="abc")
        assert not result.success

    @pytest.mark.spec("REQ-scheduler.tools.no-scheduler")
    def test_resume_no_scheduler(self) -> None:
        tool = ResumeScheduledTaskTool()
        tool._scheduler = None
        result = tool.execute(task_id="abc")
        assert not result.success

    @pytest.mark.spec("REQ-scheduler.tools.no-scheduler")
    def test_cancel_no_scheduler(self) -> None:
        tool = CancelScheduledTaskTool()
        tool._scheduler = None
        result = tool.execute(task_id="abc")
        assert not result.success


# ---------------------------------------------------------------------------
# Missing required parameters
# ---------------------------------------------------------------------------


class TestMissingParams:
    @pytest.mark.spec("REQ-scheduler.tools.validation")
    def test_schedule_task_missing_prompt(self, scheduler) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(schedule_type="once", schedule_value="2026-01-01")
        assert not result.success
        assert "Missing" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.validation")
    def test_schedule_task_missing_schedule_type(self, scheduler) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(prompt="hello", schedule_value="2026-01-01")
        assert not result.success
        assert "Missing" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.validation")
    def test_schedule_task_missing_schedule_value(self, scheduler) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(prompt="hello", schedule_type="once")
        assert not result.success
        assert "Missing" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.validation")
    def test_pause_missing_task_id(self, scheduler) -> None:
        tool = PauseScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute()
        assert not result.success
        assert "Missing" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.validation")
    def test_resume_missing_task_id(self, scheduler) -> None:
        tool = ResumeScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute()
        assert not result.success
        assert "Missing" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.validation")
    def test_cancel_missing_task_id(self, scheduler) -> None:
        tool = CancelScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute()
        assert not result.success
        assert "Missing" in result.content


# ---------------------------------------------------------------------------
# With real scheduler (no mocks)
# ---------------------------------------------------------------------------


class TestWithRealScheduler:
    @pytest.mark.spec("REQ-scheduler.tools.schedule")
    def test_schedule_task_success(self, scheduler) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(
            prompt="hello",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
        )
        assert result.success
        data = json.loads(result.content)
        assert "task_id" in data
        assert data["status"] == "active"
        assert data["next_run"] == "2099-01-01T00:00:00+00:00"

    @pytest.mark.spec("REQ-scheduler.tools.schedule")
    def test_schedule_task_with_agent_and_tools(self, scheduler) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(
            prompt="review code",
            schedule_type="interval",
            schedule_value="3600",
            agent="orchestrator",
            tools="calculator,think",
        )
        assert result.success
        data = json.loads(result.content)
        # Verify the task was actually persisted
        tasks = scheduler.list_tasks()
        matched = [t for t in tasks if t.id == data["task_id"]]
        assert len(matched) == 1
        assert matched[0].agent == "orchestrator"
        assert matched[0].tools == "calculator,think"

    @pytest.mark.spec("REQ-scheduler.tools.list")
    def test_list_scheduled_tasks_empty(self, scheduler) -> None:
        tool = ListScheduledTasksTool()
        tool._scheduler = scheduler
        result = tool.execute()
        assert result.success
        items = json.loads(result.content)
        assert items == []

    @pytest.mark.spec("REQ-scheduler.tools.list")
    def test_list_scheduled_tasks(self, scheduler) -> None:
        scheduler.create_task("task a", "interval", "60")
        scheduler.create_task("task b", "interval", "120")
        tool = ListScheduledTasksTool()
        tool._scheduler = scheduler
        result = tool.execute()
        assert result.success
        items = json.loads(result.content)
        assert len(items) == 2

    @pytest.mark.spec("REQ-scheduler.tools.list")
    def test_list_scheduled_tasks_filtered(self, scheduler) -> None:
        t1 = scheduler.create_task("task a", "interval", "60")
        scheduler.create_task("task b", "interval", "120")
        scheduler.pause_task(t1.id)
        tool = ListScheduledTasksTool()
        tool._scheduler = scheduler
        result = tool.execute(status="paused")
        assert result.success
        items = json.loads(result.content)
        assert len(items) == 1

    @pytest.mark.spec("REQ-scheduler.tools.pause")
    def test_pause_task_success(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "60")
        tool = PauseScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(task_id=task.id)
        assert result.success
        assert task.id in result.content
        # Verify actually paused
        tasks = scheduler.list_tasks(status="paused")
        assert len(tasks) == 1

    @pytest.mark.spec("REQ-scheduler.tools.pause")
    def test_pause_nonexistent_task(self, scheduler) -> None:
        tool = PauseScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(task_id="nonexistent")
        assert not result.success
        assert "not found" in result.content.lower()

    @pytest.mark.spec("REQ-scheduler.tools.resume")
    def test_resume_task_success(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "60")
        scheduler.pause_task(task.id)
        tool = ResumeScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(task_id=task.id)
        assert result.success
        assert task.id in result.content
        # Verify actually resumed
        tasks = scheduler.list_tasks(status="active")
        assert len(tasks) == 1

    @pytest.mark.spec("REQ-scheduler.tools.resume")
    def test_resume_nonexistent_task(self, scheduler) -> None:
        tool = ResumeScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(task_id="nonexistent")
        assert not result.success
        assert "not found" in result.content.lower()

    @pytest.mark.spec("REQ-scheduler.tools.cancel")
    def test_cancel_task_success(self, scheduler) -> None:
        task = scheduler.create_task("test", "interval", "60")
        tool = CancelScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(task_id=task.id)
        assert result.success
        assert task.id in result.content
        # Verify actually cancelled
        tasks = scheduler.list_tasks(status="cancelled")
        assert len(tasks) == 1

    @pytest.mark.spec("REQ-scheduler.tools.cancel")
    def test_cancel_nonexistent_task(self, scheduler) -> None:
        tool = CancelScheduledTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(task_id="nonexistent")
        assert not result.success
        assert "not found" in result.content.lower()

    @pytest.mark.spec("REQ-scheduler.tools.schedule-exception")
    def test_schedule_task_generic_exception(self, scheduler) -> None:
        """Exercise lines 103-104: create_task raises generic Exception."""
        tool = ScheduleTaskTool()

        class _FailingScheduler:
            def create_task(self, **kwargs):
                raise RuntimeError("DB connection lost")

        tool._scheduler = _FailingScheduler()
        result = tool.execute(
            prompt="test",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
        )
        assert not result.success
        assert "Failed to schedule task" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.list-exception")
    def test_list_tasks_generic_exception(self, scheduler) -> None:
        """Exercise lines 154-155: list_tasks raises generic Exception."""
        tool = ListScheduledTasksTool()

        class _FailingScheduler:
            def list_tasks(self, **kwargs):
                raise RuntimeError("DB error")

        tool._scheduler = _FailingScheduler()
        result = tool.execute()
        assert not result.success
        assert "Failed to list tasks" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.pause-exception")
    def test_pause_task_generic_exception(self, scheduler) -> None:
        """Exercise generic Exception in pause_task."""
        tool = PauseScheduledTaskTool()

        class _FailingScheduler:
            def pause_task(self, task_id):
                raise RuntimeError("Cannot pause")

        tool._scheduler = _FailingScheduler()
        result = tool.execute(task_id="some-id")
        assert not result.success
        assert "Failed to pause task" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.resume-exception")
    def test_resume_task_generic_exception(self, scheduler) -> None:
        """Exercise generic Exception in resume_task."""
        tool = ResumeScheduledTaskTool()

        class _FailingScheduler:
            def resume_task(self, task_id):
                raise RuntimeError("Cannot resume")

        tool._scheduler = _FailingScheduler()
        result = tool.execute(task_id="some-id")
        assert not result.success
        assert "Failed to resume task" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.cancel-exception")
    def test_cancel_task_generic_exception(self, scheduler) -> None:
        """Exercise generic Exception in cancel_task."""
        tool = CancelScheduledTaskTool()

        class _FailingScheduler:
            def cancel_task(self, task_id):
                raise RuntimeError("Cannot cancel")

        tool._scheduler = _FailingScheduler()
        result = tool.execute(task_id="some-id")
        assert not result.success
        assert "Failed to cancel task" in result.content

    @pytest.mark.spec("REQ-scheduler.tools.schedule")
    def test_tool_result_has_correct_tool_name(self, scheduler) -> None:
        tool = ScheduleTaskTool()
        tool._scheduler = scheduler
        result = tool.execute(
            prompt="test",
            schedule_type="once",
            schedule_value="2099-01-01T00:00:00+00:00",
        )
        assert result.tool_name == "schedule_task"
