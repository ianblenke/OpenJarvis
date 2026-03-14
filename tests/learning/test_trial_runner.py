"""Tests for openjarvis.optimize.trial_runner module."""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from openjarvis.evals.core.types import EvalResult, RunConfig, RunSummary
from openjarvis.optimize.trial_runner import TrialRunner
from openjarvis.optimize.types import TrialConfig, TrialResult

# ---------------------------------------------------------------------------
# Typed fakes replacing MagicMock
# ---------------------------------------------------------------------------


class FakeDataset:
    """Typed fake dataset for eval runner construction."""

    def load(self, **kwargs: Any) -> None:
        pass

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0


class FakeBackend:
    """Typed fake inference backend with close tracking."""

    def __init__(self) -> None:
        self.closed = False
        self.close_count = 0

    def generate(self, **kwargs: Any) -> dict:
        return {"content": "fake", "usage": {}}

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class FakeScorer:
    """Typed fake scorer for eval runner construction."""

    def score(self, **kwargs: Any) -> float:
        return 1.0


class FakeEvalRunner:
    """Typed fake EvalRunner that returns a preconfigured summary.

    Replaces the MagicMock pattern of ``mock_runner_cls.return_value.run.return_value``.
    """

    def __init__(
        self,
        config: Any = None,
        dataset: Any = None,
        backend: Any = None,
        scorer: Any = None,
        trackers: Any = None,
    ) -> None:
        self._config = config
        self._dataset = dataset
        self._backend = backend
        self._scorer = scorer
        self._trackers = trackers
        self._results: List[EvalResult] = []
        self._summary: Optional[RunSummary] = None
        # Track that __init__ was called
        FakeEvalRunner.last_instance = self

    @property
    def results(self) -> List[EvalResult]:
        return list(self._results)

    def run(self, **kwargs: Any) -> RunSummary:
        assert self._summary is not None, "FakeEvalRunner.summary not set"
        FakeEvalRunner.run_called = True
        return self._summary

    # Class-level state for assertions
    last_instance: Optional["FakeEvalRunner"] = None
    run_called: bool = False


class TestTrialRunnerInit:
    """TrialRunner.__init__ stores parameters correctly."""

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_default_params(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        assert runner.benchmark == "supergpqa"
        assert runner.max_samples == 50
        assert runner.judge_model == "gpt-5-mini-2025-08-07"
        assert runner.output_dir == "results/optimize/"

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_custom_params(self) -> None:
        runner = TrialRunner(
            benchmark="gaia",
            max_samples=100,
            judge_model="custom-judge",
            output_dir="/tmp/results/",
        )
        assert runner.benchmark == "gaia"
        assert runner.max_samples == 100
        assert runner.judge_model == "custom-judge"
        assert runner.output_dir == "/tmp/results/"


class TestBuildRunConfig:
    """TrialRunner._build_run_config maps recipe fields correctly."""

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_model_mapping(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(
            trial_id="t1",
            params={"intelligence.model": "qwen3:8b"},
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert isinstance(cfg, RunConfig)
        assert cfg.model == "qwen3:8b"
        assert cfg.benchmark == "supergpqa"

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_agent_maps_to_agent_backend(self) -> None:
        runner = TrialRunner(benchmark="gaia")
        trial = TrialConfig(
            trial_id="t2",
            params={
                "intelligence.model": "llama3.1:8b",
                "agent.type": "native_react",
            },
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.backend == "jarvis-agent"
        assert cfg.agent_name == "native_react"

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_no_agent_maps_to_direct_backend(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(
            trial_id="t3",
            params={"intelligence.model": "qwen3:8b"},
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.backend == "jarvis-direct"
        assert cfg.agent_name is None

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_tools_mapping(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(
            trial_id="t4",
            params={
                "agent.type": "orchestrator",
                "tools.tool_set": ["calculator", "think"],
            },
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.tools == ["calculator", "think"]

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_temperature_mapping(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(
            trial_id="t5",
            params={"intelligence.temperature": 0.7},
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.temperature == 0.7

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_engine_key_mapping(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(
            trial_id="t6",
            params={"engine.backend": "vllm"},
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.engine_key == "vllm"

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_max_samples_from_runner(self) -> None:
        runner = TrialRunner(benchmark="supergpqa", max_samples=25)
        trial = TrialConfig(trial_id="t7")
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.max_samples == 25

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_judge_model_from_runner(self) -> None:
        runner = TrialRunner(benchmark="supergpqa", judge_model="my-judge")
        trial = TrialConfig(trial_id="t8")
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.judge_model == "my-judge"

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_output_path_contains_trial_id(self) -> None:
        runner = TrialRunner(benchmark="supergpqa", output_dir="out/")
        trial = TrialConfig(
            trial_id="trial-abc",
            params={"intelligence.model": "qwen3:8b"},
        )
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert "trial-abc" in cfg.output_path
        assert cfg.output_path.startswith("out/")

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_default_model_fallback(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t9")
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.model == "default"

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_default_temperature_fallback(self) -> None:
        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t10")
        recipe = trial.to_recipe()
        cfg = runner._build_run_config(trial, recipe)

        assert cfg.temperature == 0.0


class TestRunTrial:
    """TrialRunner.run_trial integration (all eval deps replaced with fakes)."""

    def _make_summary(self, **overrides) -> RunSummary:
        defaults = dict(
            benchmark="supergpqa",
            category="reasoning",
            backend="jarvis-direct",
            model="qwen3:8b",
            total_samples=50,
            scored_samples=48,
            correct=40,
            accuracy=0.8333,
            errors=2,
            mean_latency_seconds=1.5,
            total_cost_usd=0.10,
            total_energy_joules=500.0,
            total_input_tokens=10000,
            total_output_tokens=5000,
        )
        defaults.update(overrides)
        return RunSummary(**defaults)

    def _patch_trial_deps(
        self,
        monkeypatch,
        summary: RunSummary,
        eval_results: Optional[List[EvalResult]] = None,
    ) -> tuple[FakeBackend, FakeBackend]:
        """Patch all eval dependencies with typed fakes via monkeypatch.

        Returns (fake_backend, fake_judge) so callers can assert close() calls.
        """
        fake_backend = FakeBackend()
        fake_judge = FakeBackend()

        # Reset class-level state
        FakeEvalRunner.last_instance = None
        FakeEvalRunner.run_called = False

        def _fake_eval_runner_factory(config, dataset, backend, scorer, **kw):
            runner = FakeEvalRunner(config, dataset, backend, scorer)
            runner._summary = summary
            if eval_results is not None:
                runner._results = eval_results
            return runner

        import openjarvis.evals.cli as cli_mod
        import openjarvis.evals.core.runner as runner_mod

        monkeypatch.setattr(
            runner_mod, "EvalRunner", _fake_eval_runner_factory,
        )
        monkeypatch.setattr(
            cli_mod, "_build_backend", lambda *a, **kw: fake_backend,
        )
        monkeypatch.setattr(
            cli_mod, "_build_dataset", lambda *a, **kw: FakeDataset(),
        )
        monkeypatch.setattr(
            cli_mod, "_build_judge_backend", lambda *a, **kw: fake_judge,
        )
        monkeypatch.setattr(
            cli_mod, "_build_scorer", lambda *a, **kw: FakeScorer(),
        )

        return fake_backend, fake_judge

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_returns_trial_result(self, monkeypatch) -> None:
        summary = self._make_summary()
        self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa", max_samples=50)
        trial = TrialConfig(
            trial_id="t-run",
            params={"intelligence.model": "qwen3:8b"},
        )

        result = runner.run_trial(trial)

        assert isinstance(result, TrialResult)
        assert result.trial_id == "t-run"
        assert result.config is trial
        assert FakeEvalRunner.run_called is True

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_accuracy_from_summary(self, monkeypatch) -> None:
        summary = self._make_summary(accuracy=0.92)
        self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t-acc", params={})
        result = runner.run_trial(trial)

        assert result.accuracy == 0.92

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_tokens_summed(self, monkeypatch) -> None:
        summary = self._make_summary(
            total_input_tokens=3000,
            total_output_tokens=2000,
        )
        self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t-tok", params={})
        result = runner.run_trial(trial)

        assert result.total_tokens == 5000

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_summary_attached(self, monkeypatch) -> None:
        summary = self._make_summary()
        self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t-sum", params={})
        result = runner.run_trial(trial)

        assert result.summary is summary

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_failure_modes_on_errors(self, monkeypatch) -> None:
        summary = self._make_summary(errors=5)
        self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t-err", params={})
        result = runner.run_trial(trial)

        assert len(result.failure_modes) == 1
        assert "5" in result.failure_modes[0]

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_no_failure_modes_when_clean(self, monkeypatch) -> None:
        summary = self._make_summary(errors=0)
        self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t-ok", params={})
        result = runner.run_trial(trial)

        assert result.failure_modes == []

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_closes_backends(self, monkeypatch) -> None:
        summary = self._make_summary()
        fake_backend, fake_judge = self._patch_trial_deps(monkeypatch, summary)

        runner = TrialRunner(benchmark="supergpqa")
        trial = TrialConfig(trial_id="t-close", params={})
        runner.run_trial(trial)

        assert fake_backend.closed is True
        assert fake_backend.close_count == 1
        assert fake_judge.closed is True
        assert fake_judge.close_count == 1

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_run_trial_populates_sample_scores(self, monkeypatch) -> None:
        summary = self._make_summary()
        eval_results = [
            EvalResult(
                record_id="r1",
                model_answer="42",
                is_correct=True,
                score=1.0,
                latency_seconds=0.5,
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd=0.001,
            ),
            EvalResult(
                record_id="r2",
                model_answer="wrong",
                is_correct=False,
                score=0.0,
                latency_seconds=1.2,
                prompt_tokens=120,
                completion_tokens=60,
                cost_usd=0.002,
                error="parse error",
            ),
        ]
        self._patch_trial_deps(monkeypatch, summary, eval_results=eval_results)

        runner = TrialRunner(benchmark="supergpqa", max_samples=50)
        trial = TrialConfig(
            trial_id="t-scores",
            params={"intelligence.model": "qwen3:8b"},
        )
        result = runner.run_trial(trial)

        assert len(result.sample_scores) == 2
        assert result.sample_scores[0].record_id == "r1"
        assert result.sample_scores[0].is_correct is True
        assert result.sample_scores[0].latency_seconds == 0.5
        assert result.sample_scores[1].record_id == "r2"
        assert result.sample_scores[1].is_correct is False
        assert result.sample_scores[1].error == "parse error"
