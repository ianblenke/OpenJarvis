"""Tests for optimize/ backward-compatibility shim layer.

The optimize/ module re-exports everything from learning.optimize/.
These tests verify the shims work and point to the correct implementations.
"""

from __future__ import annotations

import pytest


class TestOptimizeShimImports:
    """Verify all re-exports from openjarvis.optimize work."""

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_import_optimization_engine(self) -> None:
        from openjarvis.optimize import OptimizationEngine

        assert OptimizationEngine is not None

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_import_llm_optimizer(self) -> None:
        from openjarvis.optimize import LLMOptimizer

        assert LLMOptimizer is not None

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_import_optimization_store(self) -> None:
        from openjarvis.optimize import OptimizationStore

        assert OptimizationStore is not None

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_import_trial_runner(self) -> None:
        from openjarvis.optimize import TrialRunner

        assert TrialRunner is not None

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_import_multi_bench_trial_runner(self) -> None:
        from openjarvis.optimize import MultiBenchTrialRunner

        assert MultiBenchTrialRunner is not None


class TestOptimizeShimTypes:
    """Verify type re-exports and basic instantiation."""

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_trial_config(self) -> None:
        from openjarvis.optimize import TrialConfig

        config = TrialConfig(
            trial_id="test-1",
            params={"model": "test-model"},
        )
        assert config.trial_id == "test-1"
        assert config.params["model"] == "test-model"

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_trial_result(self) -> None:
        from openjarvis.optimize import TrialConfig, TrialResult

        config = TrialConfig(trial_id="test-1", params={"model": "m"})
        result = TrialResult(
            trial_id="test-1",
            config=config,
            accuracy=0.95,
        )
        assert result.trial_id == "test-1"
        assert result.accuracy == 0.95

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_search_dimension(self) -> None:
        from openjarvis.optimize import SearchDimension

        dim = SearchDimension(
            name="temperature",
            dim_type="continuous",
            low=0.0,
            high=2.0,
        )
        assert dim.name == "temperature"
        assert dim.dim_type == "continuous"

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_search_space(self) -> None:
        from openjarvis.optimize import SearchDimension, SearchSpace

        dims = [
            SearchDimension(name="temp", dim_type="continuous", low=0.0, high=2.0),
        ]
        space = SearchSpace(dimensions=dims)
        assert len(space.dimensions) == 1

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_benchmark_score(self) -> None:
        from openjarvis.optimize import BenchmarkScore

        score = BenchmarkScore(benchmark="mmlu", accuracy=0.85)
        assert score.benchmark == "mmlu"
        assert score.accuracy == 0.85

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_objective_spec(self) -> None:
        from openjarvis.optimize import ObjectiveSpec

        obj = ObjectiveSpec(metric="accuracy", direction="maximize")
        assert obj.metric == "accuracy"
        assert obj.direction == "maximize"


class TestOptimizeShimFunctions:
    """Verify function re-exports."""

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_build_search_space(self) -> None:
        from openjarvis.optimize import build_search_space

        assert callable(build_search_space)

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_compute_pareto_frontier(self) -> None:
        from openjarvis.optimize import compute_pareto_frontier

        assert callable(compute_pareto_frontier)


class TestOptimizeShimConfig:
    """Verify config re-exports."""

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_import_config_loaders(self) -> None:
        from openjarvis.optimize.config import load_optimize_config

        assert callable(load_optimize_config)


class TestOptimizeShimStore:
    """Verify OptimizationStore basic operations."""

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_store_init(self, tmp_path) -> None:
        from openjarvis.optimize import OptimizationStore

        db_path = tmp_path / "test_opt.db"
        store = OptimizationStore(str(db_path))
        try:
            assert db_path.exists()
        finally:
            store.close()


class TestOptimizeShimIdentity:
    """Verify shims point to the exact same objects as learning.optimize."""

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_optimization_engine_identity(self) -> None:
        from openjarvis.learning.optimize import OptimizationEngine as Real
        from openjarvis.optimize import OptimizationEngine as Shim

        assert Shim is Real

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_trial_config_identity(self) -> None:
        from openjarvis.learning.optimize import TrialConfig as Real
        from openjarvis.optimize import TrialConfig as Shim

        assert Shim is Real

    @pytest.mark.spec("REQ-learning.trial-runner")
    def test_trial_runner_identity(self) -> None:
        from openjarvis.learning.optimize import TrialRunner as Real
        from openjarvis.optimize import TrialRunner as Shim

        assert Shim is Real


class TestOptimizeFeedbackShims:
    """Verify feedback submodule re-exports."""

    @pytest.mark.spec("REQ-learning.feedback-collector")
    def test_import_feedback_collector(self) -> None:
        from openjarvis.optimize.feedback.collector import FeedbackCollector

        assert FeedbackCollector is not None

    @pytest.mark.spec("REQ-learning.trace-judge")
    def test_import_trace_judge(self) -> None:
        from openjarvis.optimize.feedback.judge import TraceJudge

        assert TraceJudge is not None


class TestOptimizePersonalShims:
    """Verify personal submodule re-exports."""

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_import_personal_synthesizer(self) -> None:
        from openjarvis.optimize.personal.synthesizer import (
            PersonalBenchmarkSynthesizer,
        )

        assert PersonalBenchmarkSynthesizer is not None

    @pytest.mark.spec("REQ-learning.optimizer-engine")
    def test_import_personal_dataset(self) -> None:
        from openjarvis.optimize.personal.dataset import PersonalBenchmarkDataset

        assert PersonalBenchmarkDataset is not None
