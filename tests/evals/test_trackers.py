"""Tests for eval result trackers (W&B + Google Sheets).

Refactored to use typed fakes instead of MagicMock / unittest.mock.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from openjarvis.evals.core.backend import InferenceBackend
from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.scorer import Scorer
from openjarvis.evals.core.tracker import ResultTracker
from openjarvis.evals.core.types import (
    EvalRecord,
    EvalResult,
    RunConfig,
    RunSummary,
)

# ---------------------------------------------------------------------------
# Typed fakes
# ---------------------------------------------------------------------------


class FakeBackend(InferenceBackend):
    """Deterministic backend for integration tests."""

    backend_id = "fake"

    def __init__(self, content: str = "42") -> None:
        self._content = content

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._content

    def generate_full(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "content": self._content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "latency_seconds": 0.5,
        }


class FakeDataset(DatasetProvider):
    """In-memory dataset for testing."""

    dataset_id = "fake"
    dataset_name = "Fake"

    def __init__(self, records: List[EvalRecord]) -> None:
        self._records = records

    def load(self, **kwargs: Any) -> None:
        pass

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)


class FakeScorer(Scorer):
    """Scorer that always returns True."""

    scorer_id = "fake"

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        return True, {}


class RecordingTracker(ResultTracker):
    """Records all lifecycle calls for testing."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.results: List[EvalResult] = []
        self.summary: Optional[RunSummary] = None

    def on_run_start(self, config: RunConfig) -> None:
        self.calls.append("on_run_start")

    def on_result(self, result: EvalResult, config: RunConfig) -> None:
        self.calls.append("on_result")
        self.results.append(result)

    def on_summary(self, summary: RunSummary) -> None:
        self.calls.append("on_summary")
        self.summary = summary

    def on_run_end(self) -> None:
        self.calls.append("on_run_end")


class CrashingTracker(ResultTracker):
    """Raises on every lifecycle call."""

    def on_run_start(self, config: RunConfig) -> None:
        raise RuntimeError("boom start")

    def on_result(self, result: EvalResult, config: RunConfig) -> None:
        raise RuntimeError("boom result")

    def on_summary(self, summary: RunSummary) -> None:
        raise RuntimeError("boom summary")

    def on_run_end(self) -> None:
        raise RuntimeError("boom end")


class FakeWandbRun:
    """Typed fake for a wandb run object."""

    def __init__(self) -> None:
        self.summary = FakeWandbSummary()
        self.finished = False

    def finish(self) -> None:
        self.finished = True


class FakeWandbSummary:
    """Typed fake for wandb.run.summary."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

    def update(self, d: Dict[str, Any]) -> None:
        self.data.update(d)


class FakeWandbModule:
    """Typed fake for the wandb module.

    Replaces the real wandb module to avoid external dependencies.
    """

    def __init__(self) -> None:
        self.init_calls: List[Dict[str, Any]] = []
        self.log_calls: List[tuple] = []  # (data, step)
        self.run: Optional[FakeWandbRun] = None

    def init(self, **kwargs: Any) -> FakeWandbRun:
        self.init_calls.append(kwargs)
        self.run = FakeWandbRun()
        return self.run

    def log(self, data: Dict[str, Any], step: int = 0) -> None:
        self.log_calls.append((data, step))


class FakeGspreadModule:
    """Typed fake for the gspread module."""

    authorize_called = False

    def authorize(self, creds: Any) -> None:
        self.authorize_called = True


class FakeCredentials:
    """Typed fake for google.oauth2.service_account.Credentials."""

    @classmethod
    def from_service_account_file(
        cls, path: str, scopes: Any = None,
    ) -> "FakeCredentials":
        return cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> RunConfig:
    defaults: Dict[str, Any] = {
        "benchmark": "test",
        "backend": "jarvis-direct",
        "model": "test-model",
    }
    defaults.update(overrides)
    return RunConfig(**defaults)


def _make_summary(**overrides: Any) -> RunSummary:
    defaults: Dict[str, Any] = {
        "benchmark": "test",
        "category": "chat",
        "backend": "jarvis-direct",
        "model": "test-model",
        "total_samples": 10,
        "scored_samples": 10,
        "correct": 8,
        "accuracy": 0.8,
        "errors": 0,
        "mean_latency_seconds": 1.0,
        "total_cost_usd": 0.01,
    }
    defaults.update(overrides)
    return RunSummary(**defaults)


def _make_result(**overrides: Any) -> EvalResult:
    defaults: Dict[str, Any] = {
        "record_id": "r1",
        "model_answer": "answer",
        "is_correct": True,
    }
    defaults.update(overrides)
    return EvalResult(**defaults)


def _make_record(
    record_id: str = "r1",
    problem: str = "What is 1+1?",
    reference: str = "2",
) -> EvalRecord:
    return EvalRecord(
        record_id=record_id,
        problem=problem,
        reference=reference,
        category="chat",
    )


# ---------------------------------------------------------------------------
# Tests: ResultTracker ABC
# ---------------------------------------------------------------------------


class TestResultTrackerABC:
    """Tests for the ResultTracker abstract base class."""

    @pytest.mark.spec("REQ-evals.tracker.base-protocol")
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ResultTracker()  # type: ignore[abstract]

    @pytest.mark.spec("REQ-evals.tracker.base-protocol")
    def test_recording_tracker_is_valid(self):
        tracker = RecordingTracker()
        assert isinstance(tracker, ResultTracker)


# ---------------------------------------------------------------------------
# RecordingTracker through EvalRunner lifecycle
# ---------------------------------------------------------------------------


class TestRecordingTrackerIntegration:
    """Test that trackers receive all lifecycle calls through EvalRunner."""

    @pytest.mark.spec("REQ-evals.tracker.lifecycle")
    def test_tracker_lifecycle(self, tmp_path):
        """RecordingTracker receives start, result, summary, end calls."""
        from openjarvis.evals.core.runner import EvalRunner

        record = _make_record()
        dataset = FakeDataset([record])
        backend = FakeBackend(content="2")
        scorer = FakeScorer()

        tracker = RecordingTracker()
        config = _make_config(output_path=str(tmp_path / "out.jsonl"))

        runner = EvalRunner(config, dataset, backend, scorer, trackers=[tracker])
        runner.run()

        assert "on_run_start" in tracker.calls
        assert "on_result" in tracker.calls
        assert "on_summary" in tracker.calls
        assert "on_run_end" in tracker.calls
        # Order matters
        assert tracker.calls.index("on_run_start") < tracker.calls.index("on_result")
        assert tracker.calls.index("on_result") < tracker.calls.index("on_summary")
        assert tracker.calls.index("on_summary") < tracker.calls.index("on_run_end")
        assert len(tracker.results) == 1
        assert tracker.summary is not None

    @pytest.mark.spec("REQ-evals.tracker.lifecycle")
    def test_crashing_tracker_does_not_abort(self, tmp_path):
        """A tracker that raises exceptions must not prevent JSONL output."""
        from openjarvis.evals.core.runner import EvalRunner

        record = _make_record()
        dataset = FakeDataset([record])
        backend = FakeBackend(content="yes")
        scorer = FakeScorer()

        output = tmp_path / "out.jsonl"
        config = _make_config(output_path=str(output))

        crasher = CrashingTracker()
        runner = EvalRunner(config, dataset, backend, scorer, trackers=[crasher])
        summary = runner.run()

        # Run completed, JSONL written despite crashing tracker
        assert summary.total_samples == 1
        assert output.exists()
        assert output.read_text().strip() != ""

    @pytest.mark.spec("REQ-evals.tracker.lifecycle")
    def test_multiple_trackers_all_notified(self, tmp_path):
        """All trackers receive events even when one crashes."""
        from openjarvis.evals.core.runner import EvalRunner

        record = _make_record()
        dataset = FakeDataset([record])
        backend = FakeBackend()
        scorer = FakeScorer()

        healthy = RecordingTracker()
        crasher = CrashingTracker()
        config = _make_config(output_path=str(tmp_path / "out.jsonl"))

        runner = EvalRunner(
            config, dataset, backend, scorer,
            trackers=[crasher, healthy],
        )
        runner.run()

        # Healthy tracker should still receive all events
        assert "on_run_start" in healthy.calls
        assert "on_run_end" in healthy.calls


# ---------------------------------------------------------------------------
# WandbTracker unit tests
# ---------------------------------------------------------------------------


class TestWandbTracker:
    """Unit tests for WandbTracker using typed fakes."""

    @pytest.mark.spec("REQ-evals.trackers.wandb")
    @pytest.mark.spec("REQ-evals.tracker.wandb-import")
    def test_import_error_when_wandb_missing(self):
        """WandbTracker raises ImportError when wandb is not installed."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        original = wt_mod.wandb
        wt_mod.wandb = None
        try:
            with pytest.raises(ImportError, match="wandb is not installed"):
                wt_mod.WandbTracker(project="test")
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-log")
    def test_on_result_logs_metrics(self):
        """on_result logs sample/ prefixed keys via wandb.log."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(project="test-proj")
            config = _make_config()
            tracker.on_run_start(config)

            result = _make_result(latency_seconds=0.5, energy_joules=1.0)
            tracker.on_result(result, config)

            assert len(fake_wandb.log_calls) == 1
            log_data, step = fake_wandb.log_calls[0]
            assert "sample/is_correct" in log_data
            assert "sample/latency_seconds" in log_data
            assert log_data["sample/is_correct"] == 1.0
            assert step == 1

            tracker.on_run_end()
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-summary")
    def test_on_summary_updates_run_summary(self):
        """on_summary updates wandb.run.summary with flat dict."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(project="test-proj")
            config = _make_config()
            tracker.on_run_start(config)

            summary = _make_summary()
            tracker.on_summary(summary)

            assert fake_wandb.run is not None
            assert fake_wandb.run.summary.data["accuracy"] == 0.8
            assert fake_wandb.run.summary.data["total_samples"] == 10

            tracker.on_run_end()
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-init")
    def test_reinit_true_for_suite_mode(self):
        """wandb.init is called with reinit=True."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(project="test-proj", entity="team")
            config = _make_config()
            tracker.on_run_start(config)

            assert len(fake_wandb.init_calls) == 1
            call_kwargs = fake_wandb.init_calls[0]
            assert call_kwargs["reinit"] is True
            assert call_kwargs["project"] == "test-proj"
            assert call_kwargs["entity"] == "team"

            tracker.on_run_end()
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-init")
    def test_tags_parsed_from_comma_string(self):
        """Tags are parsed from comma-separated string."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(
                project="p", tags="tag1, tag2, tag3",
            )
            config = _make_config()
            tracker.on_run_start(config)

            call_kwargs = fake_wandb.init_calls[0]
            assert call_kwargs["tags"] == ["tag1", "tag2", "tag3"]

            tracker.on_run_end()
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-log")
    def test_on_result_increments_step(self):
        """Each on_result call increments the step counter."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(project="p")
            config = _make_config()
            tracker.on_run_start(config)

            tracker.on_result(_make_result(record_id="r1"), config)
            tracker.on_result(_make_result(record_id="r2"), config)
            tracker.on_result(_make_result(record_id="r3"), config)

            steps = [step for _, step in fake_wandb.log_calls]
            assert steps == [1, 2, 3]

            tracker.on_run_end()
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-log")
    def test_on_result_before_run_start_is_noop(self):
        """on_result does nothing if on_run_start was not called."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(project="p")
            config = _make_config()
            # Don't call on_run_start
            tracker.on_result(_make_result(), config)

            assert len(fake_wandb.log_calls) == 0
        finally:
            wt_mod.wandb = original

    @pytest.mark.spec("REQ-evals.tracker.wandb-finish")
    def test_on_run_end_finishes_run(self):
        """on_run_end calls run.finish()."""
        import openjarvis.evals.trackers.wandb_tracker as wt_mod

        fake_wandb = FakeWandbModule()
        original = wt_mod.wandb
        wt_mod.wandb = fake_wandb
        try:
            tracker = wt_mod.WandbTracker(project="p")
            tracker.on_run_start(_make_config())

            run = fake_wandb.run
            tracker.on_run_end()

            assert run is not None
            assert run.finished is True
        finally:
            wt_mod.wandb = original


# ---------------------------------------------------------------------------
# SheetsTracker unit tests
# ---------------------------------------------------------------------------


class TestSheetsTracker:
    """Unit tests for SheetsTracker using typed fakes."""

    @pytest.mark.spec("REQ-evals.trackers.sheets")
    @pytest.mark.spec("REQ-evals.tracker.sheets-import")
    def test_import_error_when_gspread_missing(self):
        """SheetsTracker raises ImportError when gspread not installed."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        original = st_mod.gspread
        st_mod.gspread = None
        try:
            with pytest.raises(ImportError, match="gspread is not installed"):
                st_mod.SheetsTracker(spreadsheet_id="abc123")
        finally:
            st_mod.gspread = original

    @pytest.mark.spec("REQ-evals.tracker.sheets-noop")
    def test_on_result_is_noop(self):
        """on_result does nothing (no API calls for individual samples)."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        fake_gspread = FakeGspreadModule()
        original = st_mod.gspread
        st_mod.gspread = fake_gspread
        original_creds = st_mod.Credentials
        st_mod.Credentials = FakeCredentials
        try:
            tracker = st_mod.SheetsTracker(spreadsheet_id="abc123")
            result = _make_result()
            config = _make_config()

            tracker.on_result(result, config)

            # on_result should not trigger authorize
            assert not fake_gspread.authorize_called
        finally:
            st_mod.gspread = original
            st_mod.Credentials = original_creds

    @pytest.mark.spec("REQ-evals.tracker.sheets-row")
    def test_build_row_matches_columns(self):
        """_build_row returns a list matching SHEET_COLUMNS length."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        fake_gspread = FakeGspreadModule()
        original = st_mod.gspread
        st_mod.gspread = fake_gspread
        original_creds = st_mod.Credentials
        st_mod.Credentials = FakeCredentials
        try:
            tracker = st_mod.SheetsTracker(spreadsheet_id="abc123")
            summary = _make_summary()
            row = tracker._build_row(summary)
            assert len(row) == len(st_mod.SHEET_COLUMNS), (
                f"Row length {len(row)} != columns length {len(st_mod.SHEET_COLUMNS)}"
            )
        finally:
            st_mod.gspread = original
            st_mod.Credentials = original_creds

    @pytest.mark.spec("REQ-evals.tracker.sheets-row")
    def test_build_row_contains_benchmark_and_model(self):
        """_build_row includes the benchmark name and model."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        fake_gspread = FakeGspreadModule()
        original = st_mod.gspread
        st_mod.gspread = fake_gspread
        original_creds = st_mod.Credentials
        st_mod.Credentials = FakeCredentials
        try:
            tracker = st_mod.SheetsTracker(spreadsheet_id="abc123")
            summary = _make_summary(benchmark="gaia", model="qwen3:8b")
            row = tracker._build_row(summary)

            assert "gaia" in row
            assert "qwen3:8b" in row
        finally:
            st_mod.gspread = original
            st_mod.Credentials = original_creds

    @pytest.mark.spec("REQ-evals.tracker.sheets-row")
    def test_build_row_accuracy_value(self):
        """_build_row includes the accuracy value."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        fake_gspread = FakeGspreadModule()
        original = st_mod.gspread
        st_mod.gspread = fake_gspread
        original_creds = st_mod.Credentials
        st_mod.Credentials = FakeCredentials
        try:
            tracker = st_mod.SheetsTracker(spreadsheet_id="abc123")
            summary = _make_summary(accuracy=0.95)
            row = tracker._build_row(summary)

            assert 0.95 in row
        finally:
            st_mod.gspread = original
            st_mod.Credentials = original_creds

    @pytest.mark.spec("REQ-evals.tracker.sheets-lifecycle")
    def test_on_run_start_is_noop(self):
        """on_run_start does nothing."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        fake_gspread = FakeGspreadModule()
        original = st_mod.gspread
        st_mod.gspread = fake_gspread
        original_creds = st_mod.Credentials
        st_mod.Credentials = FakeCredentials
        try:
            tracker = st_mod.SheetsTracker(spreadsheet_id="abc123")
            # Should not raise
            tracker.on_run_start(_make_config())
        finally:
            st_mod.gspread = original
            st_mod.Credentials = original_creds

    @pytest.mark.spec("REQ-evals.tracker.sheets-lifecycle")
    def test_on_run_end_is_noop(self):
        """on_run_end does nothing."""
        import openjarvis.evals.trackers.sheets_tracker as st_mod

        fake_gspread = FakeGspreadModule()
        original = st_mod.gspread
        st_mod.gspread = fake_gspread
        original_creds = st_mod.Credentials
        st_mod.Credentials = FakeCredentials
        try:
            tracker = st_mod.SheetsTracker(spreadsheet_id="abc123")
            # Should not raise
            tracker.on_run_end()
        finally:
            st_mod.gspread = original
            st_mod.Credentials = original_creds


# ---------------------------------------------------------------------------
# Tests: _flatten_metric_stats helper
# ---------------------------------------------------------------------------


class TestFlattenMetricStats:
    """Tests for the _flatten_metric_stats helper in wandb_tracker."""

    @pytest.mark.spec("REQ-evals.tracker.flatten-metric-stats")
    def test_none_returns_empty(self):
        from openjarvis.evals.trackers.wandb_tracker import _flatten_metric_stats

        result = _flatten_metric_stats("test", None)
        assert result == {}

    @pytest.mark.spec("REQ-evals.tracker.flatten-metric-stats")
    def test_prefixed_keys(self):
        from openjarvis.evals.core.types import MetricStats
        from openjarvis.evals.trackers.wandb_tracker import _flatten_metric_stats

        ms = MetricStats(
            mean=1.0, median=2.0, min=0.0, max=3.0,
            std=0.5, p90=2.5, p95=2.8, p99=2.9,
        )
        result = _flatten_metric_stats("latency", ms)

        assert result["latency_mean"] == 1.0
        assert result["latency_median"] == 2.0
        assert result["latency_min"] == 0.0
        assert result["latency_max"] == 3.0
        assert result["latency_std"] == 0.5
        assert result["latency_p90"] == 2.5
        assert result["latency_p95"] == 2.8
        assert result["latency_p99"] == 2.9
        assert len(result) == 8


# ---------------------------------------------------------------------------
# Tests: _stat_val helper in sheets_tracker
# ---------------------------------------------------------------------------


class TestStatVal:
    """Tests for the _stat_val helper in sheets_tracker."""

    @pytest.mark.spec("REQ-evals.tracker.stat-val")
    def test_none_returns_empty_string(self):
        from openjarvis.evals.trackers.sheets_tracker import _stat_val

        assert _stat_val(None, "mean") == ""

    @pytest.mark.spec("REQ-evals.tracker.stat-val")
    def test_extracts_attribute(self):
        from openjarvis.evals.core.types import MetricStats
        from openjarvis.evals.trackers.sheets_tracker import _stat_val

        ms = MetricStats(mean=3.14, p90=5.0)
        assert _stat_val(ms, "mean") == 3.14
        assert _stat_val(ms, "p90") == 5.0

    @pytest.mark.spec("REQ-evals.tracker.stat-val")
    def test_missing_attribute_returns_empty_string(self):
        from openjarvis.evals.core.types import MetricStats
        from openjarvis.evals.trackers.sheets_tracker import _stat_val

        ms = MetricStats()
        assert _stat_val(ms, "nonexistent") == ""
