"""Typed fakes for the optimization subsystem.

Replaces MagicMock-based patterns for LLMOptimizer and TrialRunner with
real classes that implement the same interface. Catches interface drift
and exercises real code paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openjarvis.evals.core.types import RunSummary
from openjarvis.learning.optimize.types import (
    TrialConfig,
    TrialFeedback,
    TrialResult,
)


class FakeOptimizer:
    """Typed fake implementing the LLMOptimizer interface.

    Provides controllable responses for propose_initial, propose_next,
    analyze_trial, propose_targeted, and propose_merge without using
    MagicMock.
    """

    def __init__(
        self,
        *,
        optimizer_model: str = "fake-model",
        initial_config: Optional[TrialConfig] = None,
        next_configs: Optional[List[TrialConfig]] = None,
        feedback: Optional[TrialFeedback] = None,
        feedbacks: Optional[List[TrialFeedback]] = None,
        targeted_config: Optional[TrialConfig] = None,
        merge_config: Optional[TrialConfig] = None,
    ) -> None:
        self.optimizer_model = optimizer_model
        self._initial_config = initial_config or TrialConfig(
            trial_id="init", params={},
        )
        self._next_configs = list(next_configs or [])
        self._next_idx = 0
        self._feedback = feedback or TrialFeedback(summary_text="ok")
        self._feedbacks = list(feedbacks or [])
        self._feedback_idx = 0
        self._targeted_config = targeted_config
        self._merge_config = merge_config

        # Call tracking
        self.propose_initial_count = 0
        self.propose_next_count = 0
        self.analyze_trial_count = 0
        self.propose_targeted_count = 0
        self.propose_merge_count = 0
        self.analyze_trial_calls: List[Dict[str, Any]] = []

    def propose_initial(self) -> TrialConfig:
        self.propose_initial_count += 1
        return self._initial_config

    def propose_next(
        self,
        history: List[TrialResult],
        traces: Optional[list] = None,
        frontier_ids: Optional[set] = None,
    ) -> TrialConfig:
        self.propose_next_count += 1
        if self._next_configs:
            idx = min(self._next_idx, len(self._next_configs) - 1)
            self._next_idx += 1
            return self._next_configs[idx]
        return TrialConfig(
            trial_id=f"next-{self.propose_next_count}", params={},
        )

    def analyze_trial(
        self,
        trial: TrialConfig,
        summary: RunSummary,
        traces: Optional[list] = None,
        sample_scores: Optional[list] = None,
        per_benchmark: Optional[list] = None,
    ) -> TrialFeedback:
        self.analyze_trial_count += 1
        self.analyze_trial_calls.append({
            "trial": trial,
            "summary": summary,
        })
        if self._feedbacks:
            idx = min(self._feedback_idx, len(self._feedbacks) - 1)
            self._feedback_idx += 1
            return self._feedbacks[idx]
        return self._feedback

    def propose_targeted(
        self,
        history: List[TrialResult],
        base_config: TrialConfig,
        target_primitive: str,
        frontier_ids: Optional[set] = None,
    ) -> TrialConfig:
        self.propose_targeted_count += 1
        if self._targeted_config:
            return self._targeted_config
        return TrialConfig(
            trial_id=f"targeted-{self.propose_targeted_count}", params={},
        )

    def propose_merge(
        self,
        candidates: List[TrialResult],
        history: List[TrialResult],
        frontier_ids: Optional[set] = None,
    ) -> TrialConfig:
        self.propose_merge_count += 1
        if self._merge_config:
            return self._merge_config
        return TrialConfig(
            trial_id=f"merged-{self.propose_merge_count}", params={},
        )


class FakeTrialRunner:
    """Typed fake implementing the TrialRunner interface.

    Returns pre-configured TrialResult instances from run_trial().
    """

    def __init__(
        self,
        *,
        benchmark: str = "test",
        results: Optional[List[TrialResult]] = None,
        default_result: Optional[TrialResult] = None,
    ) -> None:
        self.benchmark = benchmark
        self._results = list(results or [])
        self._default_result = default_result
        self._call_count = 0
        self.run_trial_calls: List[TrialConfig] = []

    def run_trial(self, config: TrialConfig) -> TrialResult:
        self.run_trial_calls.append(config)
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._results):
            return self._results[idx]
        if self._default_result is not None:
            return self._default_result
        # Fallback: return a simple result matching the config
        return TrialResult(
            trial_id=config.trial_id,
            config=config,
            accuracy=0.5,
            mean_latency_seconds=1.0,
            total_cost_usd=0.01,
            samples_evaluated=50,
        )

    @property
    def call_count(self) -> int:
        return self._call_count


class FakeExecutor:
    """Typed fake implementing the AgentExecutor.execute_tick interface.

    Tracks which agent IDs were ticked and how many times.
    """

    def __init__(self) -> None:
        self.tick_calls: List[str] = []

    def execute_tick(self, agent_id: str) -> None:
        self.tick_calls.append(agent_id)

    @property
    def call_count(self) -> int:
        return len(self.tick_calls)

    def assert_called_with(self, agent_id: str) -> None:
        assert self.tick_calls, "execute_tick was never called"
        assert self.tick_calls[-1] == agent_id, (
            f"Expected last call with {agent_id!r}, got {self.tick_calls[-1]!r}"
        )

    def assert_not_called(self) -> None:
        assert not self.tick_calls, (
            f"execute_tick was called {len(self.tick_calls)} time(s), expected 0"
        )
