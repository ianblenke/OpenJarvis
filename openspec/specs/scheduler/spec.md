# Scheduler Module Spec

Task scheduling and automation with cron, interval, and one-shot execution.

## TaskScheduler

### REQ-scheduler.lifecycle: Scheduler lifecycle
`TaskScheduler.start()` begins background polling. `stop()` halts execution.

### REQ-scheduler.create: Task creation
`create_task(prompt, schedule_type, schedule_value, **kwargs) -> ScheduledTask`.

### REQ-scheduler.schedule-types: Schedule types
Three types: `"once"` (ISO datetime), `"interval"` (seconds), `"cron"` (cron expression).

### REQ-scheduler.management: Task management
`list_tasks()`, `pause_task()`, `resume_task()`, `cancel_task()`.

### REQ-scheduler.events: Scheduler events
Publishes `SCHEDULER_TASK_START` and `SCHEDULER_TASK_END` events.

## ScheduledTask

### REQ-scheduler.task: Task data
`ScheduledTask` with `id`, `prompt`, `schedule_type`, `schedule_value`, `context_mode`, `status`, `next_run`, `last_run`, `agent`, `tools`, `metadata`.

## Scheduler Operations (detailed)

### REQ-scheduler.scheduler.create: Task creation via scheduler
`TaskScheduler.create_task()` creates a scheduled task, computes next_run, persists to store, and returns the task.

### REQ-scheduler.scheduler.list: Task listing via scheduler
`TaskScheduler.list_tasks()` returns all tasks with optional status filtering.

### REQ-scheduler.scheduler.pause: Task pausing
`TaskScheduler.pause_task()` sets task status to paused; raises for nonexistent tasks.

### REQ-scheduler.scheduler.resume: Task resuming
`TaskScheduler.resume_task()` reactivates a paused task and recomputes next_run; raises for nonexistent tasks.

### REQ-scheduler.scheduler.cancel: Task cancellation
`TaskScheduler.cancel_task()` sets task status to cancelled; raises for nonexistent tasks.

### REQ-scheduler.scheduler.execute: Task execution
`TaskScheduler` executes due tasks by invoking the configured agent with the task prompt and tools.

### REQ-scheduler.scheduler.next-run: Next run computation
Scheduler computes next_run based on schedule type: interval adds seconds, cron uses next cron match, once uses the specified datetime.

### REQ-scheduler.scheduler.poll: Background poll loop
`TaskScheduler._poll_loop()` runs in a background thread, queries for due tasks, executes them, and handles exceptions without crashing.

## Time Utilities

### REQ-scheduler.time: Timestamp generation
`_now_iso()` returns the current time as an ISO 8601 string with UTC timezone info.

## ScheduledTask (detailed)

### REQ-scheduler.task.defaults: Task default values
ScheduledTask provides defaults for optional fields (status="active", empty tools list, no metadata).

### REQ-scheduler.task.round-trip: Task serialization roundtrip
ScheduledTask supports `to_dict()` / `from_dict()` roundtrip with all fields preserved.

## Scheduler Store

### REQ-scheduler.store.save: Task persistence
Scheduler store saves tasks with all fields to the backing storage.

### REQ-scheduler.store.get: Task retrieval
Scheduler store retrieves a task by ID.

### REQ-scheduler.store.list: Task listing from store
Scheduler store lists all tasks with optional filtering.

### REQ-scheduler.store.update: Task update
Scheduler store updates task fields (status, next_run, etc.).

### REQ-scheduler.store.delete: Task deletion
Scheduler store deletes a task by ID.

### REQ-scheduler.store.due-tasks: Due task query
Scheduler store queries for tasks whose next_run is in the past and status is active.

### REQ-scheduler.store.log-run: Run logging
Scheduler store logs task execution runs with timestamp, outcome, and duration.

### REQ-scheduler.store.metadata: Store metadata
Scheduler store tracks metadata about the store itself (schema version, creation time).

### REQ-scheduler.store.metadata-corrupt: Corrupt metadata fallback
Scheduler store falls back to an empty dict when task metadata contains invalid JSON.

## Scheduler Tools

### REQ-scheduler.tools.schedule: Schedule tool
Tool interface for creating scheduled tasks via agent tool calls.

### REQ-scheduler.tools.list: List tasks tool
Tool interface for listing scheduled tasks via agent tool calls.

### REQ-scheduler.tools.pause: Pause task tool
Tool interface for pausing scheduled tasks via agent tool calls.

### REQ-scheduler.tools.resume: Resume task tool
Tool interface for resuming scheduled tasks via agent tool calls.

### REQ-scheduler.tools.cancel: Cancel task tool
Tool interface for cancelling scheduled tasks via agent tool calls.

### REQ-scheduler.tools.spec: Scheduler tool specifications
Scheduler tools provide OpenAI function-calling compatible tool specs.

### REQ-scheduler.tools.validation: Scheduler tool input validation
Scheduler tools validate input parameters and return clear error messages for invalid inputs.

### REQ-scheduler.tools.no-scheduler: Tools without scheduler
Scheduler tools return appropriate errors when no scheduler instance is configured.

### REQ-scheduler.tools.schedule-exception: Schedule tool exception handling
Schedule tool returns a failure result when `create_task` raises a generic exception.

### REQ-scheduler.tools.list-exception: List tool exception handling
List tool returns a failure result when `list_tasks` raises a generic exception.

### REQ-scheduler.tools.pause-exception: Pause tool exception handling
Pause tool returns a failure result when `pause_task` raises a generic exception.

### REQ-scheduler.tools.resume-exception: Resume tool exception handling
Resume tool returns a failure result when `resume_task` raises a generic exception.

### REQ-scheduler.tools.cancel-exception: Cancel tool exception handling
Cancel tool returns a failure result when `cancel_task` raises a generic exception.

## Tests

- `tests/scheduler/test_scheduler.py` - Scheduler tests
- `tests/scheduler/test_tools.py` - Scheduler tool integration
