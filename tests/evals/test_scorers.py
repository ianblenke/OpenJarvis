"""Tests for scorer base classes and concrete scorers.

Covers the Scorer / LLMJudgeScorer ABCs, GAIA exact match, MCQ scorers
(GPQA, MMLU-Pro, SuperGPQA), and the checklist text utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from openjarvis.evals.core.backend import InferenceBackend
from openjarvis.evals.core.scorer import LLMJudgeScorer, Scorer
from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.scorers._checklist import (
    contains_key_phrases,
    normalize_number_str,
    normalize_str,
)
from openjarvis.evals.scorers.gaia_exact import GAIAScorer, exact_match

# ---------------------------------------------------------------------------
# Typed fakes
# ---------------------------------------------------------------------------


class FakeJudgeBackend(InferenceBackend):
    """Deterministic backend that returns canned responses for judge calls."""

    backend_id = "fake-judge"

    def __init__(self, response: str = "") -> None:
        self._response = response
        self.prompts_received: List[str] = []

    def generate(
        self,
        prompt: str,
        *,
        model: str = "",
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        self.prompts_received.append(prompt)
        return self._response

    def generate_full(
        self,
        prompt: str,
        *,
        model: str = "",
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        return {
            "content": self.generate(
                prompt, model=model, system=system,
                temperature=temperature, max_tokens=max_tokens,
            ),
            "usage": {},
        }


class ErrorJudgeBackend(InferenceBackend):
    """Backend that always raises, to test scorer error handling."""

    backend_id = "error-judge"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("judge unavailable")

    def generate_full(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("judge unavailable")


class ConstantScorer(Scorer):
    """Scorer that always returns a fixed result."""

    scorer_id = "constant"

    def __init__(self, is_correct: Optional[bool] = True) -> None:
        self._is_correct = is_correct

    def score(
        self, record: EvalRecord, model_answer: str,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        return self._is_correct, {"constant": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    problem: str = "Question?",
    reference: str = "answer",
    category: str = "chat",
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalRecord:
    return EvalRecord(
        record_id="test-1",
        problem=problem,
        reference=reference,
        category=category,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Tests: Scorer ABC
# ---------------------------------------------------------------------------


class TestScorerABC:
    """Tests for the Scorer abstract base class."""

    @pytest.mark.spec("REQ-evals.scorer.base-protocol")
    def test_cannot_instantiate_scorer_directly(self):
        with pytest.raises(TypeError):
            Scorer()  # type: ignore[abstract]

    @pytest.mark.spec("REQ-evals.scorer.base-protocol")
    def test_concrete_scorer_works(self):
        scorer = ConstantScorer(is_correct=True)
        record = _make_record()
        is_correct, meta = scorer.score(record, "any answer")
        assert is_correct is True
        assert meta == {"constant": True}

    @pytest.mark.spec("REQ-evals.scorer.base-protocol")
    def test_scorer_can_return_none(self):
        scorer = ConstantScorer(is_correct=None)
        record = _make_record()
        is_correct, _ = scorer.score(record, "any answer")
        assert is_correct is None


class TestLLMJudgeScorerABC:
    """Tests for the LLMJudgeScorer abstract base class."""

    @pytest.mark.spec("REQ-evals.scorer.llm-judge-base")
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMJudgeScorer(  # type: ignore[abstract]
                judge_backend=FakeJudgeBackend(),
                judge_model="test-model",
            )

    @pytest.mark.spec("REQ-evals.scorer.llm-judge-base")
    def test_ask_judge_delegates_to_backend(self):
        backend = FakeJudgeBackend(response="judge says yes")

        class ConcreteJudge(LLMJudgeScorer):
            scorer_id = "test-judge"

            def score(
                self, record: EvalRecord, model_answer: str,
            ) -> Tuple[Optional[bool], Dict[str, Any]]:
                result = self._ask_judge("test prompt")
                return result == "judge says yes", {"raw": result}

        scorer = ConcreteJudge(backend, "test-model")
        is_correct, meta = scorer.score(_make_record(), "anything")

        assert is_correct is True
        assert meta["raw"] == "judge says yes"
        assert len(backend.prompts_received) == 1


# ---------------------------------------------------------------------------
# Tests: normalize_str
# ---------------------------------------------------------------------------


class TestNormalizeStr:
    """Tests for normalize_str utility."""

    @pytest.mark.spec("REQ-evals.scorer.normalize-str")
    def test_lowercase(self):
        assert normalize_str("HELLO") == "hello"

    @pytest.mark.spec("REQ-evals.scorer.normalize-str")
    def test_remove_punctuation(self):
        assert normalize_str("hello, world!") == "hello world"

    @pytest.mark.spec("REQ-evals.scorer.normalize-str")
    def test_collapse_whitespace(self):
        assert normalize_str("  lots   of   spaces  ") == "lots of spaces"

    @pytest.mark.spec("REQ-evals.scorer.normalize-str")
    def test_empty_string(self):
        assert normalize_str("") == ""


class TestNormalizeNumberStr:
    """Tests for normalize_number_str utility."""

    @pytest.mark.spec("REQ-evals.scorer.normalize-number")
    def test_plain_number(self):
        assert normalize_number_str("42") == 42.0

    @pytest.mark.spec("REQ-evals.scorer.normalize-number")
    def test_dollar_sign(self):
        assert normalize_number_str("$1,000") == 1000.0

    @pytest.mark.spec("REQ-evals.scorer.normalize-number")
    def test_percent_sign(self):
        assert normalize_number_str("95%") == 95.0

    @pytest.mark.spec("REQ-evals.scorer.normalize-number")
    def test_invalid_returns_inf(self):
        result = normalize_number_str("not a number")
        assert result == float("inf")


class TestContainsKeyPhrases:
    """Tests for contains_key_phrases utility."""

    @pytest.mark.spec("REQ-evals.scorer.contains-phrases")
    def test_above_threshold(self):
        answer = "The system uses SQLite and FAISS for memory storage."
        reference = "SQLite; FAISS; BM25; hybrid"
        # 2 of 4 = 50%, meets default threshold
        assert contains_key_phrases(answer, reference) is True

    @pytest.mark.spec("REQ-evals.scorer.contains-phrases")
    def test_below_threshold(self):
        answer = "The system uses PostgreSQL."
        reference = "SQLite; FAISS; BM25; hybrid"
        assert contains_key_phrases(answer, reference) is False

    @pytest.mark.spec("REQ-evals.scorer.contains-phrases")
    def test_empty_reference(self):
        assert contains_key_phrases("anything", "") is False

    @pytest.mark.spec("REQ-evals.scorer.contains-phrases")
    def test_custom_threshold(self):
        answer = "Uses SQLite"
        reference = "SQLite; FAISS; BM25"
        # 1 of 3 = 33%, below 0.5 but above 0.3
        assert contains_key_phrases(answer, reference, threshold=0.3) is True
        assert contains_key_phrases(answer, reference, threshold=0.5) is False

    @pytest.mark.spec("REQ-evals.scorer.contains-phrases")
    def test_case_insensitive(self):
        answer = "Uses SQLITE and faiss"
        reference = "SQLite; FAISS"
        assert contains_key_phrases(answer, reference) is True


# ---------------------------------------------------------------------------
# Tests: GAIA exact_match
# ---------------------------------------------------------------------------


class TestGAIAExactMatch:
    """Tests for the GAIA exact_match function."""

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_exact_string_match(self):
        assert exact_match("Paris", "Paris") is True

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_case_insensitive_match(self):
        assert exact_match("PARIS", "paris") is True

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_punctuation_ignored(self):
        assert exact_match("Hello, World!", "hello world") is True

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_numeric_match(self):
        assert exact_match("42", "42") is True

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_numeric_with_formatting(self):
        assert exact_match("$1,000", "1000") is True

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_numeric_mismatch(self):
        assert exact_match("42", "43") is False

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_list_match(self):
        assert exact_match("a, b, c", "a, b, c") is True

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_list_mismatch_length(self):
        assert exact_match("a, b", "a, b, c") is False

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_list_element_mismatch(self):
        assert exact_match("a, b, d", "a, b, c") is False

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_none_model_answer(self):
        # Should not crash; the function converts None to "None"
        assert exact_match(None, "None") is True  # type: ignore[arg-type]

    @pytest.mark.spec("REQ-evals.scorer.gaia-exact")
    def test_string_mismatch(self):
        assert exact_match("London", "Paris") is False


class TestGAIAScorer:
    """Tests for the GAIAScorer (exact match + LLM fallback)."""

    @pytest.mark.spec("REQ-evals.scorer.gaia-scorer")
    def test_exact_match_returns_true(self):
        backend = FakeJudgeBackend()
        scorer = GAIAScorer(backend, "test-model")
        record = _make_record(reference="Paris")

        is_correct, meta = scorer.score(record, "Paris")

        assert is_correct is True
        assert meta["match_type"] == "exact"
        # No LLM call needed for exact match
        assert len(backend.prompts_received) == 0

    @pytest.mark.spec("REQ-evals.scorer.gaia-scorer")
    def test_empty_response_returns_false(self):
        backend = FakeJudgeBackend()
        scorer = GAIAScorer(backend, "test-model")
        record = _make_record(reference="Paris")

        is_correct, meta = scorer.score(record, "")

        assert is_correct is False
        assert meta["reason"] == "empty_response"

    @pytest.mark.spec("REQ-evals.scorer.gaia-scorer")
    def test_no_reference_returns_none(self):
        backend = FakeJudgeBackend()
        scorer = GAIAScorer(backend, "test-model")
        record = _make_record(reference="")

        is_correct, meta = scorer.score(record, "some answer")

        assert is_correct is None
        assert meta["reason"] == "no_ground_truth"

    @pytest.mark.spec("REQ-evals.scorer.gaia-scorer")
    def test_llm_fallback_correct(self):
        response = (
            "extracted_final_answer: Paris\n"
            "reasoning: The answer matches.\n"
            "correct: yes"
        )
        backend = FakeJudgeBackend(response=response)
        scorer = GAIAScorer(backend, "test-model")
        record = _make_record(
            problem="What is the capital of France?",
            reference="Paris",
        )

        # Model gave a different form that fails exact match
        is_correct, meta = scorer.score(record, "The capital is Paris, France.")

        assert is_correct is True
        assert meta["match_type"] == "llm_fallback"
        assert len(backend.prompts_received) == 1

    @pytest.mark.spec("REQ-evals.scorer.gaia-scorer")
    def test_llm_fallback_incorrect(self):
        response = (
            "extracted_final_answer: London\n"
            "reasoning: The answer does not match.\n"
            "correct: no"
        )
        backend = FakeJudgeBackend(response=response)
        scorer = GAIAScorer(backend, "test-model")
        record = _make_record(reference="Paris")

        is_correct, meta = scorer.score(record, "London is the answer")

        assert is_correct is False
        assert meta["match_type"] == "llm_fallback"

    @pytest.mark.spec("REQ-evals.scorer.gaia-scorer")
    def test_llm_fallback_error_returns_false(self):
        backend = ErrorJudgeBackend()
        scorer = GAIAScorer(backend, "test-model")
        record = _make_record(reference="Paris")

        is_correct, meta = scorer.score(record, "some non-matching answer")

        assert is_correct is False
        assert meta["match_type"] == "llm_fallback_error"


# ---------------------------------------------------------------------------
# Tests: MCQ Scorers (GPQA, MMLU-Pro, SuperGPQA)
# ---------------------------------------------------------------------------


class TestGPQAScorer:
    """Tests for GPQA MCQ scorer."""

    @pytest.mark.spec("REQ-evals.scorer.gpqa-mcq")
    def test_correct_answer(self):
        from openjarvis.evals.scorers.gpqa_mcq import GPQAScorer

        backend = FakeJudgeBackend(response="B")
        scorer = GPQAScorer(backend, "test-model")
        record = _make_record(
            reference="B",
            metadata={"options": ["opt1", "opt2", "opt3", "opt4"]},
        )

        is_correct, meta = scorer.score(record, "I think the answer is B")

        assert is_correct is True
        assert meta["reference_letter"] == "B"
        assert meta["candidate_letter"] == "B"

    @pytest.mark.spec("REQ-evals.scorer.gpqa-mcq")
    def test_incorrect_answer(self):
        from openjarvis.evals.scorers.gpqa_mcq import GPQAScorer

        backend = FakeJudgeBackend(response="A")
        scorer = GPQAScorer(backend, "test-model")
        record = _make_record(
            reference="C",
            metadata={"options": ["opt1", "opt2", "opt3", "opt4"]},
        )

        is_correct, meta = scorer.score(record, "The answer is A")

        assert is_correct is False
        assert meta["candidate_letter"] == "A"
        assert meta["reference_letter"] == "C"

    @pytest.mark.spec("REQ-evals.scorer.gpqa-mcq")
    def test_missing_reference(self):
        from openjarvis.evals.scorers.gpqa_mcq import GPQAScorer

        backend = FakeJudgeBackend(response="A")
        scorer = GPQAScorer(backend, "test-model")
        record = _make_record(reference="")

        is_correct, meta = scorer.score(record, "answer")

        assert is_correct is None
        assert meta["reason"] == "missing_reference_letter"

    @pytest.mark.spec("REQ-evals.scorer.gpqa-mcq")
    def test_extraction_failure_returns_none(self):
        from openjarvis.evals.scorers.gpqa_mcq import GPQAScorer

        # Backend returns gibberish that cannot be parsed as a letter
        backend = FakeJudgeBackend(response="NONE")
        scorer = GPQAScorer(backend, "test-model")
        record = _make_record(
            reference="A",
            metadata={"options": ["opt1", "opt2", "opt3", "opt4"]},
        )

        is_correct, meta = scorer.score(record, "no clear answer")

        # "NONE" contains "N" which is outside ABCD, so extraction returns None
        # But actually re.search will match N from NONE, and N not in "ABCD"
        # so it returns None -> no_choice_letter_extracted
        assert is_correct is None
        assert meta["reason"] == "no_choice_letter_extracted"

    @pytest.mark.spec("REQ-evals.scorer.gpqa-mcq")
    def test_valid_letters_from_options(self):
        from openjarvis.evals.scorers.gpqa_mcq import GPQAScorer

        backend = FakeJudgeBackend()
        scorer = GPQAScorer(backend, "test-model")

        # 3 options -> ABC
        assert scorer._valid_letters_from_options(
            {"options": ["a", "b", "c"]}
        ) == "ABC"

        # No options -> default ABCD
        assert scorer._valid_letters_from_options({}) == "ABCD"


class TestMMLUProScorer:
    """Tests for MMLU-Pro MCQ scorer."""

    @pytest.mark.spec("REQ-evals.scorer.mmlu-pro-mcq")
    def test_correct_answer(self):
        from openjarvis.evals.scorers.mmlu_pro_mcq import MMLUProScorer

        backend = FakeJudgeBackend(response="D")
        scorer = MMLUProScorer(backend, "test-model")
        record = _make_record(
            reference="D",
            metadata={"options": [f"opt{i}" for i in range(10)]},
        )

        is_correct, meta = scorer.score(record, "The answer is D")

        assert is_correct is True
        assert meta["candidate_letter"] == "D"

    @pytest.mark.spec("REQ-evals.scorer.mmlu-pro-mcq")
    def test_default_valid_letters(self):
        from openjarvis.evals.scorers.mmlu_pro_mcq import MMLUProScorer

        backend = FakeJudgeBackend()
        scorer = MMLUProScorer(backend, "test-model")

        # No options -> default ABCDEFGHIJ (10 options)
        assert scorer._valid_letters_from_options({}) == "ABCDEFGHIJ"


class TestSuperGPQAScorer:
    """Tests for SuperGPQA MCQ scorer."""

    @pytest.mark.spec("REQ-evals.scorer.supergpqa-mcq")
    def test_correct_answer(self):
        from openjarvis.evals.scorers.supergpqa_mcq import SuperGPQAScorer

        backend = FakeJudgeBackend(response="C")
        scorer = SuperGPQAScorer(backend, "test-model")
        record = _make_record(
            reference="C",
            metadata={"options": ["opt1", "opt2", "opt3", "opt4"]},
        )

        is_correct, meta = scorer.score(record, "C is correct")

        assert is_correct is True

    @pytest.mark.spec("REQ-evals.scorer.supergpqa-mcq")
    def test_default_valid_letters(self):
        from openjarvis.evals.scorers.supergpqa_mcq import SuperGPQAScorer

        backend = FakeJudgeBackend()
        scorer = SuperGPQAScorer(backend, "test-model")
        assert scorer._valid_letters_from_options({}) == "ABCD"


# ---------------------------------------------------------------------------
# Tests: Email triage scorer
# ---------------------------------------------------------------------------


class TestEmailTriageScorer:
    """Tests for EmailTriageScorer exact-match path."""

    @pytest.mark.spec("REQ-evals.scorer.email-triage")
    def test_exact_match_both_fields(self):
        from openjarvis.evals.scorers.email_triage import EmailTriageScorer

        backend = FakeJudgeBackend()
        scorer = EmailTriageScorer(backend, "test-model")
        record = _make_record(
            metadata={"urgency": "high", "category": "action"},
        )
        model_answer = "urgency: high\ncategory: action\nDraft: Will do!"

        is_correct, meta = scorer.score(record, model_answer)

        assert is_correct is True
        assert meta["match_type"] == "exact"
        assert meta["urgency_correct"] is True
        assert meta["category_correct"] is True
        # No LLM call needed
        assert len(backend.prompts_received) == 0

    @pytest.mark.spec("REQ-evals.scorer.email-triage")
    def test_empty_response(self):
        from openjarvis.evals.scorers.email_triage import EmailTriageScorer

        backend = FakeJudgeBackend()
        scorer = EmailTriageScorer(backend, "test-model")
        record = _make_record(
            metadata={"urgency": "low", "category": "info"},
        )

        is_correct, meta = scorer.score(record, "")

        assert is_correct is False
        assert meta["reason"] == "empty_response"

    @pytest.mark.spec("REQ-evals.scorer.email-triage")
    def test_partial_match_triggers_llm_fallback(self):
        from openjarvis.evals.scorers.email_triage import EmailTriageScorer

        response = (
            "urgency_correct: yes\n"
            "category_correct: no\n"
            "draft_quality: good\n"
            "reasoning: Category was wrong\n"
            "overall_correct: no"
        )
        backend = FakeJudgeBackend(response=response)
        scorer = EmailTriageScorer(backend, "test-model")
        record = _make_record(
            metadata={"urgency": "high", "category": "action"},
        )
        # Urgency matches but category does not
        model_answer = "urgency: high\ncategory: info"

        is_correct, meta = scorer.score(record, model_answer)

        assert is_correct is False
        assert meta["match_type"] == "llm_judge"
        assert len(backend.prompts_received) == 1
