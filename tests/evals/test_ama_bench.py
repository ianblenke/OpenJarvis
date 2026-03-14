"""Tests for AMA-Bench dataset provider and scorer."""

import pytest

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.datasets.ama_bench import AMABenchDataset
from openjarvis.evals.scorers.ama_bench_judge import (
    AMABenchScorer,
    _compute_token_f1,
    _parse_judge_label,
)
from tests.fixtures.engines import FakeInferenceBackend


class TestAMABenchDataset:
    @pytest.mark.spec("REQ-evals.datasets")
    def test_instantiation(self) -> None:
        ds = AMABenchDataset()
        assert ds.dataset_id == "ama-bench"
        assert ds.dataset_name == "AMA-Bench"

    @pytest.mark.spec("REQ-evals.datasets")
    def test_has_required_methods(self) -> None:
        ds = AMABenchDataset()
        assert hasattr(ds, "load")
        assert hasattr(ds, "iter_records")
        assert hasattr(ds, "size")
        assert hasattr(ds, "iter_episodes")

    @pytest.mark.spec("REQ-evals.datasets")
    def test_row_to_episode(self) -> None:
        ds = AMABenchDataset()
        row = {
            "episode_id": "test-ep-1",
            "task": "Navigate to the store",
            "domain": "web",
            "task_type": "navigation",
            "source": "webarena",
            "success": True,
            "num_turns": 3,
            "total_tokens": 500,
            "trajectory": [
                {"turn_idx": 0, "action": "click button", "observation": "Page loaded"},
                {"turn_idx": 1, "action": "type query", "observation": "Results shown"},
                {"turn_idx": 2, "action": "click result", "observation": "Item found"},
            ],
            "qa_pairs": [
                {
                    "question": "What action was taken at turn 1?",
                    "answer": "type query",
                    "question_uuid": "q-uuid-1",
                    "type": "A",
                },
                {
                    "question": "Why did the agent click the result?",
                    "answer": "To find the item",
                    "question_uuid": "q-uuid-2",
                    "type": "B",
                },
            ],
        }
        records = ds._row_to_episode(row)
        assert len(records) == 2
        assert records[0].record_id == "ama-test-ep-1-q-uuid-1"
        assert records[0].subject == "recall"
        assert records[1].subject == "causal_inference"
        assert "## Trajectory" in records[0].problem
        assert "## Question" in records[0].problem
        assert records[0].reference == "type query"

    @pytest.mark.spec("REQ-evals.datasets")
    def test_format_trajectory(self) -> None:
        trajectory = [
            {"turn_idx": 0, "action": "look around", "observation": "You see a room"},
            {"turn_idx": 1, "action": "go north", "observation": "You enter a hallway"},
        ]
        text = AMABenchDataset._format_trajectory(trajectory)
        assert "Turn 0" in text
        assert "Action: look around" in text
        assert "Observation:" in text
        assert "You see a room" in text
        assert "Turn 1" in text

    @pytest.mark.spec("REQ-evals.datasets")
    def test_truncate_trajectory_text(self) -> None:
        long_text = "A" * 1000
        truncated = AMABenchDataset._truncate_trajectory_text(long_text, 200)
        assert len(truncated) <= 200
        assert "truncated" in truncated
        assert truncated.startswith("A")
        assert truncated.endswith("A")

    @pytest.mark.spec("REQ-evals.datasets")
    def test_truncate_preserves_short_text(self) -> None:
        short_text = "Hello world"
        result = AMABenchDataset._truncate_trajectory_text(short_text, 1000)
        assert len(result) <= 1000

    @pytest.mark.spec("REQ-evals.datasets")
    def test_question_types_mapped(self) -> None:
        from openjarvis.evals.datasets.ama_bench import _QUESTION_TYPE_TO_SUBJECT
        assert _QUESTION_TYPE_TO_SUBJECT["A"] == "recall"
        assert _QUESTION_TYPE_TO_SUBJECT["B"] == "causal_inference"
        assert _QUESTION_TYPE_TO_SUBJECT["C"] == "state_updating"
        assert _QUESTION_TYPE_TO_SUBJECT["D"] == "state_abstraction"


def _mock_backend():
    return FakeInferenceBackend(responses=["yes"])


class TestAMABenchScorer:
    @pytest.mark.spec("REQ-evals.scorers")
    def test_instantiation(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        assert s.scorer_id == "ama-bench"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_empty_response(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-1", problem="question",
            reference="answer", category="agentic",
        )
        is_correct, meta = s.score(record, "")
        assert is_correct is False
        assert meta["reason"] == "empty_response"
        assert meta["f1"] == 0.0

    @pytest.mark.spec("REQ-evals.scorers")
    def test_judge_yes(self) -> None:
        s = AMABenchScorer(_mock_backend(), "test-model")
        record = EvalRecord(
            record_id="test-2",
            problem="## Question\nWhat color is the sky?",
            reference="blue",
            category="agentic",
            metadata={"capability": "recall"},
        )
        is_correct, meta = s.score(record, "The sky is blue")
        assert is_correct is True
        assert meta["match_type"] == "llm_judge"
        assert meta["f1"] > 0

    @pytest.mark.spec("REQ-evals.scorers")
    def test_judge_no(self) -> None:
        backend = FakeInferenceBackend(responses=["no"])
        s = AMABenchScorer(backend, "test-model")
        record = EvalRecord(
            record_id="test-3",
            problem="## Question\nWhat color is the sky?",
            reference="blue",
            category="agentic",
            metadata={"capability": "recall"},
        )
        is_correct, meta = s.score(record, "The sky is green")
        assert is_correct is False

    @pytest.mark.spec("REQ-evals.scorers")
    def test_judge_invalid_falls_back_to_f1(self) -> None:
        backend = FakeInferenceBackend(responses=["I cannot determine this"])
        s = AMABenchScorer(backend, "test-model")
        record = EvalRecord(
            record_id="test-4",
            problem="## Question\nWhat is X?",
            reference="blue sky",
            category="agentic",
            metadata={"capability": "recall"},
        )
        is_correct, meta = s.score(record, "blue sky is the answer")
        assert meta["match_type"] == "f1_fallback"
        assert meta["judge_parse_failed"] is True
        assert isinstance(is_correct, bool)


class TestJudgeParsing:
    @pytest.mark.spec("REQ-evals.scorers")
    def test_direct_yes(self) -> None:
        assert _parse_judge_label("yes") == "yes"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_direct_no(self) -> None:
        assert _parse_judge_label("no") == "no"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_with_period(self) -> None:
        assert _parse_judge_label("yes.") == "yes"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_with_thinking_tags(self) -> None:
        assert _parse_judge_label(
            "<think>Let me check...</think>yes"
        ) == "yes"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_with_thinking_tags_no(self) -> None:
        assert _parse_judge_label(
            "<think>The answer doesn't match</think>\nno"
        ) == "no"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_multiline(self) -> None:
        assert _parse_judge_label("  \n  yes\n") == "yes"

    @pytest.mark.spec("REQ-evals.scorers")
    def test_garbage_returns_none(self) -> None:
        assert _parse_judge_label("maybe") is None

    @pytest.mark.spec("REQ-evals.scorers")
    def test_embedded_yes(self) -> None:
        assert _parse_judge_label("The answer is yes based on the reference") == "yes"


class TestTokenF1:
    @pytest.mark.spec("REQ-evals.scorers")
    def test_exact_match(self) -> None:
        assert _compute_token_f1("hello world", "hello world") == 1.0

    @pytest.mark.spec("REQ-evals.scorers")
    def test_no_overlap(self) -> None:
        assert _compute_token_f1("hello", "goodbye") == 0.0

    @pytest.mark.spec("REQ-evals.scorers")
    def test_partial_overlap(self) -> None:
        f1 = _compute_token_f1("the blue sky", "blue sky today")
        assert 0.0 < f1 < 1.0

    @pytest.mark.spec("REQ-evals.scorers")
    def test_empty_reference(self) -> None:
        assert _compute_token_f1("", "") == 1.0
        assert _compute_token_f1("hello", "") == 0.0

    @pytest.mark.spec("REQ-evals.scorers")
    def test_empty_prediction(self) -> None:
        assert _compute_token_f1("", "hello") == 0.0


class TestAMABenchCLI:
    @pytest.mark.spec("REQ-evals.datasets")
    def test_in_benchmarks_dict(self) -> None:
        from openjarvis.evals.cli import BENCHMARKS
        assert "ama-bench" in BENCHMARKS

    @pytest.mark.spec("REQ-evals.datasets")
    def test_build_dataset(self) -> None:
        from openjarvis.evals.cli import _build_dataset
        ds = _build_dataset("ama-bench")
        assert ds.dataset_id == "ama-bench"

    @pytest.mark.spec("REQ-evals.datasets")
    def test_build_scorer(self) -> None:
        from openjarvis.evals.cli import _build_scorer
        s = _build_scorer("ama-bench", _mock_backend(), "test-model")
        assert s.scorer_id == "ama-bench"
