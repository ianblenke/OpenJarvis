"""Tests for the general-purpose SFT trainer."""

from __future__ import annotations

from typing import Any, List

import pytest


class _FakeTraceStore:
    """Typed fake for trace store used by trainers."""

    def __init__(self, traces: List[Any] = None) -> None:
        self._traces = traces or []

    def list_traces(self, **kwargs) -> List[Any]:
        return self._traces


class TestSFTTrainerConfig:
    @pytest.mark.spec("REQ-learning.sft-trainer")
    def test_default_config(self) -> None:
        from openjarvis.core.config import SFTConfig

        cfg = SFTConfig()
        assert cfg.model_name == "Qwen/Qwen3-1.7B"
        assert cfg.use_lora is True
        assert cfg.lora_rank == 16
        assert cfg.min_pairs == 10

    @pytest.mark.spec("REQ-learning.sft-trainer")
    def test_trainer_init(self) -> None:
        from openjarvis.core.config import SFTConfig
        from openjarvis.learning.intelligence.sft_trainer import SFTTrainer

        cfg = SFTConfig()
        trainer = SFTTrainer(cfg)
        assert trainer.config is cfg

    @pytest.mark.spec("REQ-learning.sft-trainer")
    def test_target_modules_parsing(self) -> None:
        from openjarvis.core.config import SFTConfig
        from openjarvis.learning.intelligence.sft_trainer import SFTTrainer

        cfg = SFTConfig(target_modules="q_proj,v_proj,k_proj")
        trainer = SFTTrainer(cfg)
        assert trainer.target_module_list == ["q_proj", "v_proj", "k_proj"]


class TestSFTTrainerTrainOnPairs:
    @pytest.mark.spec("REQ-learning.sft-trainer")
    def test_empty_pairs_skipped(self) -> None:
        from openjarvis.core.config import SFTConfig
        from openjarvis.learning.intelligence.sft_trainer import SFTTrainer

        trainer = SFTTrainer(SFTConfig())
        result = trainer.train_on_pairs([])
        assert result["status"] == "skipped"

    @pytest.mark.spec("REQ-learning.sft-trainer")
    def test_too_few_pairs_skipped(self) -> None:
        from openjarvis.core.config import SFTConfig
        from openjarvis.learning.intelligence.sft_trainer import SFTTrainer

        trainer = SFTTrainer(SFTConfig(min_pairs=5))
        pairs = [{"input": "hi", "output": "hello"}]
        result = trainer.train_on_pairs(pairs)
        assert result["status"] == "skipped"
        assert "min_pairs" in result.get("reason", "")


class TestSFTTrainerTraceMining:
    @pytest.mark.spec("REQ-learning.sft-trainer")
    def test_train_delegates_to_miner(self, monkeypatch) -> None:
        from openjarvis.core.config import SFTConfig
        from openjarvis.learning.intelligence.sft_trainer import SFTTrainer

        trainer = SFTTrainer(SFTConfig(min_pairs=1))
        store = _FakeTraceStore()

        mine_called = False

        def _fake_mine(s):
            nonlocal mine_called
            mine_called = True
            return []

        monkeypatch.setattr(trainer, "_mine_pairs", _fake_mine)
        result = trainer.train(store)
        assert mine_called
        assert result["status"] == "skipped"
