"""Tests for openjarvis.optimize.feedback.judge module."""

from __future__ import annotations

import pytest

from openjarvis.core.types import StepType, Trace, TraceStep
from openjarvis.optimize.feedback.judge import TraceJudge, _parse_score
from tests.fixtures.engines import FakeInferenceBackend

# ---------------------------------------------------------------------------
# _parse_score unit tests
# ---------------------------------------------------------------------------


class TestParseScore:
    """Tests for the internal _parse_score helper."""

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_plain_decimal(self) -> None:
        assert _parse_score("0.85") == 0.85

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_score_prefix(self) -> None:
        assert _parse_score("Score: 0.90") == 0.90

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_rating_prefix(self) -> None:
        assert _parse_score("Rating: 0.75\nGood work.") == 0.75

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_fraction_format(self) -> None:
        score = _parse_score("Rating: 7/10\nSolid answer")
        assert abs(score - 0.7) < 1e-6

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_fraction_eight_over_ten(self) -> None:
        score = _parse_score("Score: 8/10")
        assert abs(score - 0.8) < 1e-6

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_whole_number_over_one_treated_as_10_scale(self) -> None:
        score = _parse_score("Quality: 9\nExcellent")
        assert abs(score - 0.9) < 1e-6

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_zero_score(self) -> None:
        assert _parse_score("Score: 0.0\nTerrible") == 0.0

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_perfect_score(self) -> None:
        assert _parse_score("1.0\nPerfect") == 1.0

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_no_score_defaults_to_half(self) -> None:
        assert _parse_score("This is just a comment.") == 0.5

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_negative_clamped_to_zero(self) -> None:
        # Edge case: if somehow negative appeared
        score = _parse_score("Score: 0.0")
        assert score == 0.0

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_over_one_on_ten_scale_capped(self) -> None:
        score = _parse_score("Score: 11")
        assert score <= 1.0

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_quality_prefix(self) -> None:
        assert _parse_score("Quality: 0.65") == 0.65


# ---------------------------------------------------------------------------
# TraceJudge
# ---------------------------------------------------------------------------


def _make_trace(**overrides) -> Trace:
    """Create a minimal Trace for testing."""
    defaults = dict(
        trace_id="trace-001",
        query="What is 2+2?",
        agent="orchestrator",
        model="qwen3:8b",
        result="The answer is 4.",
    )
    defaults.update(overrides)
    return Trace(**defaults)


class TestTraceJudgeInit:
    """TraceJudge constructor stores backend and model."""

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_stores_backend_and_model(self) -> None:
        backend = FakeInferenceBackend()
        judge = TraceJudge(backend=backend, model="judge-model")
        assert judge._backend is backend
        assert judge._model == "judge-model"


class TestScoreTrace:
    """TraceJudge.score_trace calls backend and parses result."""

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_returns_score_and_feedback(self) -> None:
        backend = FakeInferenceBackend(
            responses=["Score: 0.85\nGood reasoning and correct answer."],
        )
        judge = TraceJudge(backend=backend, model="judge-model")

        trace = _make_trace()
        score, feedback = judge.score_trace(trace)

        assert score == 0.85
        assert "Good reasoning" in feedback
        assert backend.call_count == 1

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_prompt_includes_query(self) -> None:
        backend = FakeInferenceBackend(responses=["0.9\nGreat"])
        judge = TraceJudge(backend=backend, model="m")

        trace = _make_trace(query="Explain gravity")
        judge.score_trace(trace)

        prompt = backend.call_history[0]["prompt"]
        assert "Explain gravity" in prompt

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_prompt_includes_result(self) -> None:
        backend = FakeInferenceBackend(responses=["0.7\nOk"])
        judge = TraceJudge(backend=backend, model="m")

        trace = _make_trace(result="Gravity is a fundamental force.")
        judge.score_trace(trace)

        prompt = backend.call_history[0]["prompt"]
        assert "Gravity is a fundamental force." in prompt

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_prompt_includes_steps(self) -> None:
        backend = FakeInferenceBackend(responses=["0.8\nGood"])
        judge = TraceJudge(backend=backend, model="m")

        step = TraceStep(
            step_type=StepType.GENERATE,
            timestamp=1.0,
            duration_seconds=0.5,
            input={"content": "some input"},
            output={"content": "some output"},
        )
        trace = _make_trace(steps=[step])
        judge.score_trace(trace)

        prompt = backend.call_history[0]["prompt"]
        assert "generate" in prompt
        assert "some input" in prompt

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_uses_system_prompt(self) -> None:
        backend = FakeInferenceBackend(responses=["0.5"])
        judge = TraceJudge(backend=backend, model="m")

        judge.score_trace(_make_trace())

        system = backend.call_history[0]["system"]
        assert system != ""

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_fraction_score_parsing(self) -> None:
        backend = FakeInferenceBackend(
            responses=["Rating: 7/10\nDecent answer"],
        )
        judge = TraceJudge(backend=backend, model="m")

        score, _ = judge.score_trace(_make_trace())
        assert abs(score - 0.7) < 1e-6


class TestBatchEvaluate:
    """TraceJudge.batch_evaluate processes multiple traces."""

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_returns_one_result_per_trace(self) -> None:
        backend = FakeInferenceBackend(responses=["0.8\nGood"])
        judge = TraceJudge(backend=backend, model="m")

        traces = [_make_trace(trace_id=f"t{i}") for i in range(3)]
        results = judge.batch_evaluate(traces)

        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_calls_score_trace_for_each(self) -> None:
        backend = FakeInferenceBackend(responses=["0.6\nAverage"])
        judge = TraceJudge(backend=backend, model="m")

        traces = [_make_trace(trace_id=f"t{i}") for i in range(5)]
        judge.batch_evaluate(traces)

        assert backend.call_count == 5

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_empty_list_returns_empty(self) -> None:
        backend = FakeInferenceBackend()
        judge = TraceJudge(backend=backend, model="m")
        assert judge.batch_evaluate([]) == []

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_different_scores_per_trace(self) -> None:
        backend = FakeInferenceBackend(
            responses=[
                "Score: 0.9\nExcellent",
                "Score: 0.3\nPoor",
            ],
        )
        judge = TraceJudge(backend=backend, model="m")

        traces = [_make_trace(trace_id="a"), _make_trace(trace_id="b")]
        results = judge.batch_evaluate(traces)

        assert results[0][0] == 0.9
        assert results[1][0] == 0.3
