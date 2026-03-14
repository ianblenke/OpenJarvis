"""Tests for EvalRunner — suite loading, execution flow, result aggregation."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from openjarvis.evals.core.backend import InferenceBackend
from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.runner import (
    EvalRunner,
    _metric_stats,
    _strip_think_tags,
)
from openjarvis.evals.core.scorer import Scorer
from openjarvis.evals.core.tracker import ResultTracker
from openjarvis.evals.core.types import (
    EvalRecord,
    EvalResult,
    RunConfig,
    RunSummary,
)

# ---------------------------------------------------------------------------
# Typed fakes (no MagicMock)
# ---------------------------------------------------------------------------


class FakeBackend(InferenceBackend):
    """Deterministic inference backend for testing."""

    backend_id = "fake"

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "42",
    ) -> None:
        self._responses = responses or {}
        self._default = default_response
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        model: str = "",
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        return self._responses.get(prompt, self._default)

    def generate_full(
        self,
        prompt: str,
        *,
        model: str = "",
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        self.calls.append({
            "prompt": prompt,
            "model": model,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        content = self._responses.get(prompt, self._default)
        return {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "latency_seconds": 0.1,
            "cost_usd": 0.001,
        }


class FakeErrorBackend(InferenceBackend):
    """Backend that raises on every call."""

    backend_id = "fake-error"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("backend failure")

    def generate_full(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("backend failure")


class FakeDataset(DatasetProvider):
    """In-memory dataset for testing."""

    dataset_id = "fake"
    dataset_name = "Fake"

    def __init__(self, records: Optional[List[EvalRecord]] = None) -> None:
        self._records = records or []
        self.load_called = False
        self.load_kwargs: Dict[str, Any] = {}

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.load_called = True
        self.load_kwargs = {
            "max_samples": max_samples,
            "split": split,
            "seed": seed,
        }

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)


class FakeEpisodeDataset(FakeDataset):
    """Dataset that groups records into episodes."""

    def __init__(
        self,
        episodes: List[List[EvalRecord]],
    ) -> None:
        flat = [r for ep in episodes for r in ep]
        super().__init__(flat)
        self._episodes = episodes

    def iter_episodes(self) -> Iterable[List[EvalRecord]]:
        return iter(self._episodes)


class FakeScorer(Scorer):
    """Scorer that compares model answer to reference for exact match."""

    scorer_id = "fake"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        is_correct = model_answer.strip() == record.reference.strip()
        return is_correct, {"match_type": "exact"}


class AlwaysTrueScorer(Scorer):
    """Scorer that always returns True."""

    scorer_id = "always-true"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        return True, {}


class RecordingTracker(ResultTracker):
    """Records all lifecycle calls."""

    def __init__(self) -> None:
        self.events: List[str] = []
        self.results: List[EvalResult] = []
        self.summary: Optional[RunSummary] = None
        self.config: Optional[RunConfig] = None

    def on_run_start(self, config: RunConfig) -> None:
        self.events.append("start")
        self.config = config

    def on_result(self, result: EvalResult, config: RunConfig) -> None:
        self.events.append("result")
        self.results.append(result)

    def on_summary(self, summary: RunSummary) -> None:
        self.events.append("summary")
        self.summary = summary

    def on_run_end(self) -> None:
        self.events.append("end")


class FailingTracker(ResultTracker):
    """Tracker that raises on every call, to verify runner resilience."""

    def on_run_start(self, config: RunConfig) -> None:
        raise RuntimeError("tracker start boom")

    def on_result(self, result: EvalResult, config: RunConfig) -> None:
        raise RuntimeError("tracker result boom")

    def on_summary(self, summary: RunSummary) -> None:
        raise RuntimeError("tracker summary boom")

    def on_run_end(self) -> None:
        raise RuntimeError("tracker end boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    record_id: str = "r1",
    problem: str = "What is 6 * 7?",
    reference: str = "42",
    category: str = "chat",
    subject: str = "",
) -> EvalRecord:
    return EvalRecord(
        record_id=record_id,
        problem=problem,
        reference=reference,
        category=category,
        subject=subject,
    )


def _make_config(tmp_path, **overrides: Any) -> RunConfig:
    defaults: Dict[str, Any] = {
        "benchmark": "test-bench",
        "backend": "jarvis-direct",
        "model": "test-model",
        "output_path": str(tmp_path / "out.jsonl"),
        "max_workers": 1,
    }
    defaults.update(overrides)
    return RunConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStripThinkTags:
    """Unit tests for _strip_think_tags helper."""

    @pytest.mark.spec("REQ-evals.runner.think-tag-stripping")
    def test_removes_think_block(self):
        text = "Hello <think>some reasoning</think> World"
        assert _strip_think_tags(text) == "Hello  World"

    @pytest.mark.spec("REQ-evals.runner.think-tag-stripping")
    def test_no_think_tags_unchanged(self):
        text = "No thinking here"
        assert _strip_think_tags(text) == "No thinking here"

    @pytest.mark.spec("REQ-evals.runner.think-tag-stripping")
    def test_multiline_think_block(self):
        text = "Start <think>\nline1\nline2\n</think> End"
        assert _strip_think_tags(text) == "Start  End"


class TestMetricStats:
    """Tests for _metric_stats helper."""

    @pytest.mark.spec("REQ-evals.runner.metric-stats")
    def test_empty_list_returns_none(self):
        assert _metric_stats([]) is None

    @pytest.mark.spec("REQ-evals.runner.metric-stats")
    def test_single_value(self):
        stats = _metric_stats([5.0])
        assert stats is not None
        assert stats.mean == 5.0
        assert stats.median == 5.0
        assert stats.min == 5.0
        assert stats.max == 5.0
        assert stats.std == 0.0

    @pytest.mark.spec("REQ-evals.runner.metric-stats")
    def test_multiple_values(self):
        stats = _metric_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats is not None
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p90 >= 4.0
        assert stats.p99 >= 4.0


class TestEvalRunnerDatasetLoading:
    """Tests that EvalRunner calls dataset.load with correct parameters."""

    @pytest.mark.spec("REQ-evals.runner.dataset-loading")
    def test_load_called_with_config_params(self, tmp_path):
        dataset = FakeDataset([_make_record()])
        config = _make_config(
            tmp_path, max_samples=5, dataset_split="test", seed=99,
        )
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run()

        assert dataset.load_called
        assert dataset.load_kwargs["max_samples"] == 5
        assert dataset.load_kwargs["split"] == "test"
        assert dataset.load_kwargs["seed"] == 99

    @pytest.mark.spec("REQ-evals.runner.dataset-loading")
    def test_empty_dataset_produces_empty_summary(self, tmp_path):
        dataset = FakeDataset([])
        config = _make_config(tmp_path)
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 0
        assert summary.accuracy == 0.0


class TestEvalRunnerExecution:
    """Tests for EvalRunner execution flow."""

    @pytest.mark.spec("REQ-evals.runner.execution")
    def test_single_sample_correct(self, tmp_path):
        record = _make_record(reference="42")
        dataset = FakeDataset([record])
        backend = FakeBackend(default_response="42")
        scorer = FakeScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 1
        assert summary.scored_samples == 1
        assert summary.correct == 1
        assert summary.accuracy == 1.0

    @pytest.mark.spec("REQ-evals.runner.execution")
    def test_single_sample_incorrect(self, tmp_path):
        record = _make_record(reference="99")
        dataset = FakeDataset([record])
        backend = FakeBackend(default_response="42")
        scorer = FakeScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 1
        assert summary.correct == 0
        assert summary.accuracy == 0.0

    @pytest.mark.spec("REQ-evals.runner.execution")
    def test_multiple_samples(self, tmp_path):
        records = [
            _make_record("r1", reference="42"),
            _make_record("r2", reference="wrong"),
            _make_record("r3", reference="42"),
        ]
        dataset = FakeDataset(records)
        backend = FakeBackend(default_response="42")
        scorer = FakeScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 3
        assert summary.correct == 2
        # Two correct out of three scored
        assert abs(summary.accuracy - 2 / 3) < 0.01

    @pytest.mark.spec("REQ-evals.runner.execution")
    def test_backend_error_produces_error_result(self, tmp_path):
        record = _make_record()
        dataset = FakeDataset([record])
        backend = FakeErrorBackend()
        scorer = FakeScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 1
        assert summary.errors == 1
        results = runner.results
        assert len(results) == 1
        assert results[0].error is not None
        assert "backend failure" in results[0].error


class TestEvalRunnerResultAggregation:
    """Tests for result aggregation and summary computation."""

    @pytest.mark.spec("REQ-evals.runner.result-aggregation")
    def test_summary_has_benchmark_and_model(self, tmp_path):
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path, benchmark="my-bench", model="my-model")

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.benchmark == "my-bench"
        assert summary.model == "my-model"

    @pytest.mark.spec("REQ-evals.runner.result-aggregation")
    def test_summary_timing(self, tmp_path):
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.started_at > 0
        assert summary.ended_at >= summary.started_at

    @pytest.mark.spec("REQ-evals.runner.result-aggregation")
    def test_per_subject_breakdown(self, tmp_path):
        records = [
            _make_record("r1", subject="math", reference="42"),
            _make_record("r2", subject="math", reference="42"),
            _make_record("r3", subject="science", reference="42"),
        ]
        dataset = FakeDataset(records)
        backend = FakeBackend(default_response="42")
        scorer = FakeScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert "math" in summary.per_subject
        assert "science" in summary.per_subject
        assert summary.per_subject["math"]["accuracy"] == 1.0
        assert summary.per_subject["math"]["correct"] == 2.0
        assert summary.per_subject["science"]["correct"] == 1.0

    @pytest.mark.spec("REQ-evals.runner.result-aggregation")
    def test_total_cost_aggregated(self, tmp_path):
        records = [_make_record("r1"), _make_record("r2")]
        dataset = FakeDataset(records)
        backend = FakeBackend()  # Each call returns cost_usd=0.001
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        # Two samples at $0.001 each
        assert summary.total_cost_usd == pytest.approx(0.002, abs=1e-6)

    @pytest.mark.spec("REQ-evals.runner.result-aggregation")
    def test_results_property_returns_copy(self, tmp_path):
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run()

        results1 = runner.results
        results2 = runner.results
        assert results1 is not results2
        assert len(results1) == len(results2) == 1


class TestEvalRunnerOutput:
    """Tests for JSONL and summary file output."""

    @pytest.mark.spec("REQ-evals.runner.output")
    def test_jsonl_output_written(self, tmp_path):
        output_path = tmp_path / "results.jsonl"
        record = _make_record()
        dataset = FakeDataset([record])
        backend = FakeBackend(default_response="42")
        scorer = FakeScorer()
        config = _make_config(tmp_path, output_path=str(output_path))

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run()

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["record_id"] == "r1"
        assert data["is_correct"] is True

    @pytest.mark.spec("REQ-evals.runner.output")
    def test_summary_json_written(self, tmp_path):
        output_path = tmp_path / "results.jsonl"
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path, output_path=str(output_path))

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run()

        summary_path = output_path.with_suffix(".summary.json")
        assert summary_path.exists()
        summary_data = json.loads(summary_path.read_text())
        assert "accuracy" in summary_data
        assert "benchmark" in summary_data

    @pytest.mark.spec("REQ-evals.runner.output")
    def test_traces_directory_created(self, tmp_path):
        output_path = tmp_path / "results.jsonl"
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path, output_path=str(output_path))

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run()

        traces_dir = tmp_path / "traces"
        assert traces_dir.exists()


class TestEvalRunnerTrackers:
    """Tests for tracker lifecycle integration."""

    @pytest.mark.spec("REQ-evals.runner.tracker-lifecycle")
    def test_tracker_receives_all_lifecycle_events(self, tmp_path):
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        tracker = RecordingTracker()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer, trackers=[tracker])
        runner.run()

        assert tracker.events == ["start", "result", "summary", "end"]

    @pytest.mark.spec("REQ-evals.runner.tracker-lifecycle")
    def test_tracker_receives_correct_result_count(self, tmp_path):
        records = [_make_record("r1"), _make_record("r2"), _make_record("r3")]
        dataset = FakeDataset(records)
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        tracker = RecordingTracker()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer, trackers=[tracker])
        runner.run()

        assert len(tracker.results) == 3
        assert tracker.summary is not None
        assert tracker.summary.total_samples == 3

    @pytest.mark.spec("REQ-evals.runner.tracker-lifecycle")
    def test_failing_tracker_does_not_abort_run(self, tmp_path):
        output_path = tmp_path / "out.jsonl"
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path, output_path=str(output_path))

        runner = EvalRunner(
            config, dataset, backend, scorer, trackers=[FailingTracker()],
        )
        summary = runner.run()

        assert summary.total_samples == 1
        assert output_path.exists()

    @pytest.mark.spec("REQ-evals.runner.tracker-lifecycle")
    def test_multiple_trackers(self, tmp_path):
        dataset = FakeDataset([_make_record()])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        t1 = RecordingTracker()
        t2 = RecordingTracker()
        config = _make_config(tmp_path)

        runner = EvalRunner(
            config, dataset, backend, scorer, trackers=[t1, t2],
        )
        runner.run()

        for tracker in (t1, t2):
            assert "start" in tracker.events
            assert "end" in tracker.events
            assert len(tracker.results) == 1


class TestEvalRunnerProgressCallback:
    """Tests for progress_callback invocation."""

    @pytest.mark.spec("REQ-evals.runner.progress-callback")
    def test_progress_callback_called(self, tmp_path):
        records = [_make_record("r1"), _make_record("r2")]
        dataset = FakeDataset(records)
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)

        progress_updates: List[Tuple[int, int]] = []

        def on_progress(completed: int, total: int) -> None:
            progress_updates.append((completed, total))

        runner = EvalRunner(config, dataset, backend, scorer)
        runner.run(progress_callback=on_progress)

        assert len(progress_updates) == 2
        # Total should always be 2
        assert all(total == 2 for _, total in progress_updates)
        # Final call should show completed == 2
        completed_vals = sorted(c for c, _ in progress_updates)
        assert completed_vals[-1] == 2


class TestEvalRunnerWarmup:
    """Tests for warmup sample exclusion."""

    @pytest.mark.spec("REQ-evals.runner.warmup")
    def test_warmup_samples_processed_but_excluded(self, tmp_path):
        records = [
            _make_record("r1"),
            _make_record("r2"),
            _make_record("r3"),
        ]
        dataset = FakeDataset(records)
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path, warmup_samples=1)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        # Warmup runs the first sample but discards it from results.
        # The runner still processes all records from iter_records.
        assert summary.warmup_samples_excluded == 1
        # Backend should be called: 1 warmup + 3 actual = 4
        assert len(backend.calls) == 4


class TestEvalRunnerEpisodeMode:
    """Tests for episode mode (sequential processing)."""

    @pytest.mark.spec("REQ-evals.runner.episode-mode")
    def test_episode_mode_auto_enabled(self, tmp_path):
        episodes = [
            [_make_record("r1"), _make_record("r2")],
            [_make_record("r3")],
        ]
        dataset = FakeEpisodeDataset(episodes)
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)

        runner = EvalRunner(config, dataset, backend, scorer)
        summary = runner.run()

        assert summary.total_samples == 3

    @pytest.mark.spec("REQ-evals.runner.episode-mode")
    def test_inject_examples_prepends_prior_examples(self, tmp_path):
        dataset = FakeDataset([])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)
        runner = EvalRunner(config, dataset, backend, scorer)

        record = _make_record("r1", problem="New task")
        examples = [
            {"problem": "Prior task 1", "answer": "Answer 1"},
        ]
        augmented = runner._inject_examples(record, examples)

        assert "Previously Completed Tasks" in augmented.problem
        assert "Prior task 1" in augmented.problem
        assert "New task" in augmented.problem
        # record_id should be preserved
        assert augmented.record_id == "r1"

    @pytest.mark.spec("REQ-evals.runner.episode-mode")
    def test_inject_examples_empty_examples_returns_same(self, tmp_path):
        dataset = FakeDataset([])
        backend = FakeBackend()
        scorer = AlwaysTrueScorer()
        config = _make_config(tmp_path)
        runner = EvalRunner(config, dataset, backend, scorer)

        record = _make_record("r1", problem="My task")
        result = runner._inject_examples(record, [])

        assert result is record


class TestFormatMessagesAsPrompt:
    """Tests for _format_messages_as_prompt static method."""

    @pytest.mark.spec("REQ-evals.runner.format-messages")
    def test_user_and_assistant_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        prompt = EvalRunner._format_messages_as_prompt(messages)
        assert "[User]" in prompt
        assert "Hello" in prompt
        assert "[Assistant]" in prompt
        assert "Hi there" in prompt

    @pytest.mark.spec("REQ-evals.runner.format-messages")
    def test_system_messages_excluded(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = EvalRunner._format_messages_as_prompt(messages)
        assert "You are helpful" not in prompt
        assert "[User]" in prompt

    @pytest.mark.spec("REQ-evals.runner.format-messages")
    def test_ends_with_assistant_prompt(self):
        messages = [{"role": "user", "content": "Q"}]
        prompt = EvalRunner._format_messages_as_prompt(messages)
        assert prompt.strip().endswith("[Assistant]")
