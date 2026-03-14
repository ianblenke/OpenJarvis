"""Tests for the general-purpose GRPO trainer."""

from __future__ import annotations

from typing import Any, List

import pytest


class _FakeTraceStore:
    """Typed fake for trace store used by trainers."""

    def __init__(self, traces: List[Any] = None) -> None:
        self._traces = traces or []

    def list_traces(self, **kwargs) -> List[Any]:
        return self._traces


class TestGRPOConfig:
    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_default_config(self) -> None:
        from openjarvis.core.config import GRPOConfig

        cfg = GRPOConfig()
        assert cfg.model_name == "Qwen/Qwen3-1.7B"
        assert cfg.num_samples_per_prompt == 8
        assert cfg.kl_coef == 0.0001
        assert cfg.clip_ratio == 0.2
        assert cfg.min_prompts == 10


class TestDefaultRewardFn:
    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_score_returns_float(self) -> None:
        from openjarvis.learning.intelligence.grpo_trainer import DefaultRewardFn

        reward = DefaultRewardFn()
        score = reward.score("prompt", "response", None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_score_with_ground_truth(self) -> None:
        from openjarvis.learning.intelligence.grpo_trainer import DefaultRewardFn

        reward = DefaultRewardFn()
        # When response matches ground truth, score should be higher
        score_match = reward.score("what is 2+2?", "4", "4")
        score_no_match = reward.score("what is 2+2?", "5", "4")
        assert score_match > score_no_match


class TestGRPOTrainer:
    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_init(self) -> None:
        from openjarvis.core.config import GRPOConfig
        from openjarvis.learning.intelligence.grpo_trainer import GRPOTrainer

        cfg = GRPOConfig()
        trainer = GRPOTrainer(cfg)
        assert trainer.config is cfg

    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_train_on_prompts_empty(self) -> None:
        from openjarvis.core.config import GRPOConfig
        from openjarvis.learning.intelligence.grpo_trainer import GRPOTrainer

        trainer = GRPOTrainer(GRPOConfig())
        result = trainer.train_on_prompts([])
        assert result["status"] == "skipped"

    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_train_on_prompts_too_few(self) -> None:
        from openjarvis.core.config import GRPOConfig
        from openjarvis.learning.intelligence.grpo_trainer import GRPOTrainer

        trainer = GRPOTrainer(GRPOConfig(min_prompts=5))
        result = trainer.train_on_prompts(["hello"])
        assert result["status"] == "skipped"
        assert "min_prompts" in result.get("reason", "")

    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_custom_reward_fn(self) -> None:
        from openjarvis.core.config import GRPOConfig
        from openjarvis.learning.intelligence.grpo_trainer import GRPOTrainer

        class MyReward:
            def score(
                self, prompt: str, response: str, ground_truth: str | None
            ) -> float:
                return 0.42

        trainer = GRPOTrainer(GRPOConfig(), reward_fn=MyReward())
        assert trainer.reward_fn.score("a", "b", None) == 0.42

    @pytest.mark.spec("REQ-learning.grpo-trainer")
    def test_train_delegates_to_miner(self, monkeypatch) -> None:
        from openjarvis.core.config import GRPOConfig
        from openjarvis.learning.intelligence.grpo_trainer import GRPOTrainer

        trainer = GRPOTrainer(GRPOConfig(min_prompts=1))
        store = _FakeTraceStore()

        mine_called = False

        def _fake_mine(s):
            nonlocal mine_called
            mine_called = True
            return []

        monkeypatch.setattr(trainer, "_mine_prompts", _fake_mine)
        result = trainer.train(store)
        assert mine_called
        assert result["status"] == "skipped"
