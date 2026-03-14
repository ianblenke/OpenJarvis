"""Tests for SchedulerStore -- SQLite CRUD for scheduled tasks and run logs."""

from __future__ import annotations

import pytest

from openjarvis.scheduler.store import SchedulerStore


@pytest.fixture()
def store(tmp_path):
    """Create a SchedulerStore backed by a temporary SQLite database."""
    s = SchedulerStore(tmp_path / "scheduler_test.db")
    yield s
    s.close()


def _make_task(task_id: str = "t1", **overrides) -> dict:
    base = {
        "id": task_id,
        "prompt": "summarize the news",
        "schedule_type": "interval",
        "schedule_value": "3600",
        "context_mode": "isolated",
        "status": "active",
        "next_run": "2026-01-01T00:00:00+00:00",
        "last_run": None,
        "agent": "simple",
        "tools": "",
        "metadata": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------


class TestSaveAndGet:
    @pytest.mark.spec("REQ-scheduler.store.save")
    def test_save_and_get(self, store) -> None:
        task = _make_task()
        store.save_task(task)
        got = store.get_task("t1")
        assert got is not None
        assert got["id"] == "t1"
        assert got["prompt"] == "summarize the news"
        assert got["schedule_type"] == "interval"
        assert got["schedule_value"] == "3600"

    @pytest.mark.spec("REQ-scheduler.store.save")
    def test_save_task_preserves_all_fields(self, store) -> None:
        task = _make_task(
            context_mode="shared",
            agent="orchestrator",
            tools="calculator,think",
        )
        store.save_task(task)
        got = store.get_task("t1")
        assert got["context_mode"] == "shared"
        assert got["agent"] == "orchestrator"
        assert got["tools"] == "calculator,think"

    @pytest.mark.spec("REQ-scheduler.store.get")
    def test_get_missing_returns_none(self, store) -> None:
        assert store.get_task("nonexistent") is None

    @pytest.mark.spec("REQ-scheduler.store.save")
    def test_save_task_upserts(self, store) -> None:
        task = _make_task()
        store.save_task(task)
        task["prompt"] = "updated prompt"
        store.save_task(task)
        got = store.get_task("t1")
        assert got["prompt"] == "updated prompt"


class TestListTasks:
    @pytest.mark.spec("REQ-scheduler.store.list")
    def test_list_tasks_all(self, store) -> None:
        store.save_task(_make_task("t1"))
        store.save_task(_make_task("t2", status="paused"))
        store.save_task(_make_task("t3", status="completed"))
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 3

    @pytest.mark.spec("REQ-scheduler.store.list")
    def test_list_tasks_filtered_by_status(self, store) -> None:
        store.save_task(_make_task("t1", status="active"))
        store.save_task(_make_task("t2", status="paused"))
        store.save_task(_make_task("t3", status="active"))
        active = store.list_tasks(status="active")
        assert len(active) == 2
        paused = store.list_tasks(status="paused")
        assert len(paused) == 1

    @pytest.mark.spec("REQ-scheduler.store.list")
    def test_list_tasks_empty(self, store) -> None:
        assert store.list_tasks() == []


class TestUpdateTask:
    @pytest.mark.spec("REQ-scheduler.store.update")
    def test_update_task(self, store) -> None:
        task = _make_task()
        store.save_task(task)
        task["status"] = "paused"
        store.update_task(task)
        got = store.get_task("t1")
        assert got["status"] == "paused"

    @pytest.mark.spec("REQ-scheduler.store.update")
    def test_update_task_next_run(self, store) -> None:
        task = _make_task()
        store.save_task(task)
        task["next_run"] = "2026-06-01T00:00:00+00:00"
        task["last_run"] = "2026-01-01T01:00:00+00:00"
        store.update_task(task)
        got = store.get_task("t1")
        assert got["next_run"] == "2026-06-01T00:00:00+00:00"
        assert got["last_run"] == "2026-01-01T01:00:00+00:00"


class TestDeleteTask:
    @pytest.mark.spec("REQ-scheduler.store.delete")
    def test_delete_task(self, store) -> None:
        store.save_task(_make_task())
        store.delete_task("t1")
        assert store.get_task("t1") is None

    @pytest.mark.spec("REQ-scheduler.store.delete")
    def test_delete_nonexistent_is_noop(self, store) -> None:
        # Should not raise
        store.delete_task("nonexistent")


class TestMetadata:
    @pytest.mark.spec("REQ-scheduler.store.metadata")
    def test_metadata_serialized_as_json(self, store) -> None:
        task = _make_task(metadata={"key": "value", "count": 42})
        store.save_task(task)
        got = store.get_task("t1")
        assert got["metadata"] == {"key": "value", "count": 42}

    @pytest.mark.spec("REQ-scheduler.store.metadata")
    def test_empty_metadata(self, store) -> None:
        task = _make_task(metadata={})
        store.save_task(task)
        got = store.get_task("t1")
        assert got["metadata"] == {}

    @pytest.mark.spec("REQ-scheduler.store.metadata")
    def test_nested_metadata(self, store) -> None:
        meta = {"config": {"retry": 3, "timeout": 30}, "tags": ["daily", "report"]}
        task = _make_task(metadata=meta)
        store.save_task(task)
        got = store.get_task("t1")
        assert got["metadata"]["config"]["retry"] == 3
        assert got["metadata"]["tags"] == ["daily", "report"]


# ---------------------------------------------------------------------------
# Due tasks
# ---------------------------------------------------------------------------


class TestDueTasks:
    @pytest.mark.spec("REQ-scheduler.store.due-tasks")
    def test_get_due_tasks(self, store) -> None:
        store.save_task(_make_task("t1", next_run="2026-01-01T00:00:00+00:00"))
        store.save_task(_make_task("t2", next_run="2026-06-01T00:00:00+00:00"))
        store.save_task(_make_task("t3", next_run="2026-03-01T00:00:00+00:00"))
        due = store.get_due_tasks("2026-03-15T00:00:00+00:00")
        ids = {d["id"] for d in due}
        assert "t1" in ids
        assert "t3" in ids
        assert "t2" not in ids

    @pytest.mark.spec("REQ-scheduler.store.due-tasks")
    def test_due_tasks_excludes_paused(self, store) -> None:
        store.save_task(
            _make_task("t1", next_run="2026-01-01T00:00:00+00:00", status="paused")
        )
        due = store.get_due_tasks("2026-06-01T00:00:00+00:00")
        assert len(due) == 0

    @pytest.mark.spec("REQ-scheduler.store.due-tasks")
    def test_due_tasks_excludes_null_next_run(self, store) -> None:
        store.save_task(_make_task("t1", next_run=None))
        due = store.get_due_tasks("2026-06-01T00:00:00+00:00")
        assert len(due) == 0

    @pytest.mark.spec("REQ-scheduler.store.due-tasks")
    def test_due_tasks_boundary_exact_match(self, store) -> None:
        store.save_task(_make_task("t1", next_run="2026-03-15T00:00:00+00:00"))
        due = store.get_due_tasks("2026-03-15T00:00:00+00:00")
        assert len(due) == 1
        assert due[0]["id"] == "t1"


# ---------------------------------------------------------------------------
# Run logs
# ---------------------------------------------------------------------------


class TestRunLogs:
    @pytest.mark.spec("REQ-scheduler.store.log-run")
    def test_log_run_and_retrieve(self, store) -> None:
        store.save_task(_make_task())
        store.log_run(
            task_id="t1",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:01:00+00:00",
            success=True,
            result="Done",
            error="",
        )
        logs = store.get_run_logs("t1")
        assert len(logs) == 1
        assert logs[0]["success"] == 1
        assert logs[0]["result"] == "Done"

    @pytest.mark.spec("REQ-scheduler.store.log-run")
    def test_log_run_failure(self, store) -> None:
        store.save_task(_make_task())
        store.log_run(
            task_id="t1",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:01:00+00:00",
            success=False,
            result="",
            error="Connection timeout",
        )
        logs = store.get_run_logs("t1")
        assert len(logs) == 1
        assert logs[0]["success"] == 0
        assert logs[0]["error"] == "Connection timeout"

    @pytest.mark.spec("REQ-scheduler.store.log-run")
    def test_log_run_limit(self, store) -> None:
        store.save_task(_make_task())
        for i in range(20):
            store.log_run(
                task_id="t1",
                started_at=f"2026-01-{i + 1:02d}T00:00:00+00:00",
                finished_at=f"2026-01-{i + 1:02d}T00:01:00+00:00",
                success=True,
            )
        logs = store.get_run_logs("t1", limit=5)
        assert len(logs) == 5

    @pytest.mark.spec("REQ-scheduler.store.log-run")
    def test_get_run_logs_empty(self, store) -> None:
        logs = store.get_run_logs("nonexistent")
        assert logs == []

    @pytest.mark.spec("REQ-scheduler.store.metadata-corrupt")
    def test_corrupt_metadata_falls_back_to_empty_dict(self, store) -> None:
        """Exercise lines 167-168: JSONDecodeError in _row_to_dict."""
        task = _make_task()
        store.save_task(task)
        # Corrupt the metadata field directly in the database
        store._conn.execute(
            "UPDATE scheduled_tasks SET metadata = 'not-valid-json' WHERE id = ?",
            ("t1",),
        )
        store._conn.commit()
        got = store.get_task("t1")
        assert got is not None
        assert got["metadata"] == {}

    @pytest.mark.spec("REQ-scheduler.store.log-run")
    def test_multiple_logs_for_same_task(self, store) -> None:
        store.save_task(_make_task())
        store.log_run(
            task_id="t1",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:01:00+00:00",
            success=True,
            result="Run 1",
        )
        store.log_run(
            task_id="t1",
            started_at="2026-01-02T00:00:00+00:00",
            finished_at="2026-01-02T00:01:00+00:00",
            success=False,
            error="Oops",
        )
        logs = store.get_run_logs("t1")
        assert len(logs) == 2
