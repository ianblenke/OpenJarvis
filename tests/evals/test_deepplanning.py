"""Tests for DeepPlanning benchmark."""

import pytest

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.datasets.deepplanning import DeepPlanningDataset
from openjarvis.evals.scorers.deepplanning_scorer import DeepPlanningScorer
from tests.fixtures.engines import FakeInferenceBackend


def _mock_backend():
    return FakeInferenceBackend(responses=["CORRECT"])


class TestDeepPlanningDataset:
    @pytest.mark.spec("REQ-evals.datasets")
    def test_instantiation(self) -> None:
        ds = DeepPlanningDataset()
        assert ds.dataset_id == "deepplanning"
        assert ds.dataset_name == "DeepPlanning"

    @pytest.mark.spec("REQ-evals.datasets")
    def test_has_required_methods(self) -> None:
        ds = DeepPlanningDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")


class TestDeepPlanningScorer:
    @pytest.mark.spec("REQ-evals.scorers")
    def test_instantiation(self) -> None:
        s = DeepPlanningScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "deepplanning"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_correct_plan(self) -> None:
        s = DeepPlanningScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="dp-1",
            problem="## Task (travel)\nPlan a trip to Paris",
            reference="Day 1: Flight to Paris...",
            category="agentic",
            subject="travel",
            metadata={"task_type": "travel"},
        )
        is_correct, meta = s.score(record, "Day 1: Fly to Paris, visit Eiffel Tower")
        assert is_correct is True
        assert meta["match_type"] == "llm_judge"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_incorrect_plan(self) -> None:
        backend = FakeInferenceBackend(
            responses=["INCORRECT - Missing budget constraint"],
        )
        s = DeepPlanningScorer(backend, "test-model")
        record = EvalRecord(
            record_id="dp-2",
            problem="## Task (shopping)\nBuild a cart",
            reference="Cart: item A, item B, total $50",
            category="agentic",
            subject="shopping",
            metadata={"task_type": "shopping"},
        )
        is_correct, meta = s.score(record, "Cart: item C, total $100")
        assert is_correct is False

    @pytest.mark.spec("REQ-evals.scorers")
    def test_empty_response(self) -> None:
        s = DeepPlanningScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="dp-3", problem="task",
            reference="answer", category="agentic",
        )
        is_correct, meta = s.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"


class TestDeepPlanningCLI:
    @pytest.mark.spec("REQ-evals.datasets")
    def test_in_benchmarks(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "deepplanning" in BENCHMARKS

    @pytest.mark.spec("REQ-evals.datasets")
    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("deepplanning")
        assert ds.dataset_id == "deepplanning"

    @pytest.mark.spec("REQ-evals.datasets")
    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("deepplanning", _mock_backend(), "test-model")
        assert s.scorer_id == "deepplanning"
