"""Tests for orchestrator SFT trainer."""

from __future__ import annotations

from typing import Any, Dict

from openjarvis.learning.intelligence.orchestrator.sft_trainer import (
    OrchestratorSFTConfig,
    OrchestratorSFTDataset,
)

# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal fake tensor that supports squeeze(0) and clone()."""

    def __init__(self, data: list) -> None:
        self._data = data

    def squeeze(self, dim: int = 0):
        return _FakeTensor(self._data)

    def clone(self):
        return _FakeTensor(list(self._data))

    def __len__(self):
        return len(self._data)


class _FakeTokenizer:
    """Typed fake tokenizer for SFT dataset tests.

    Supports eos_token and __call__ for tokenization.
    Does NOT have apply_chat_template by default (to test fallback).
    """

    def __init__(self, eos_token: str = "</s>") -> None:
        self.eos_token = eos_token

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        # Return fake tensors that support .squeeze(0)
        tokens = text.split()
        return {
            "input_ids": _FakeTensor(list(range(len(tokens)))),
            "attention_mask": _FakeTensor([1] * len(tokens)),
        }


class _FakeTokenizerWithTemplate(_FakeTokenizer):
    """Fake tokenizer that has apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=False, **kwargs) -> str:
        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        return "\n".join(parts)


class TestOrchestratorSFTConfig:
    def test_defaults(self):
        cfg = OrchestratorSFTConfig()
        assert cfg.model_name == "Qwen/Qwen3-1.7B"
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 2e-5
        assert cfg.max_seq_length == 4096
        assert cfg.gradient_checkpointing is True

    def test_custom_values(self):
        cfg = OrchestratorSFTConfig(
            model_name="test-model",
            num_epochs=5,
            batch_size=16,
        )
        assert cfg.model_name == "test-model"
        assert cfg.num_epochs == 5
        assert cfg.batch_size == 16

    def test_default_tools(self):
        cfg = OrchestratorSFTConfig()
        assert "calculator" in cfg.available_tools
        assert "think" in cfg.available_tools


class TestOrchestratorSFTDataset:
    def test_empty_on_missing_file(self):
        tok = _FakeTokenizer()
        ds = OrchestratorSFTDataset(
            trace_path="/nonexistent/path.jsonl",
            tokenizer=tok,
        )
        assert len(ds) == 0

    def test_format_conversation_fallback(self, tmp_path):
        """Test manual formatting when tokenizer has no chat template."""
        import json

        trace_file = tmp_path / "traces.jsonl"
        trace = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        }
        trace_file.write_text(json.dumps(trace) + "\n")

        tok = _FakeTokenizer(eos_token="</s>")

        ds = OrchestratorSFTDataset(
            trace_path=str(trace_file),
            tokenizer=tok,
        )
        assert len(ds) == 1

        text = ds._format_conversation(trace["conversations"])
        assert "<|user|>" in text
        assert "Hello" in text
        assert "<|assistant|>" in text
        assert "Hi there" in text
        assert text.endswith("</s>")

    def test_format_tool_message(self):
        tok = _FakeTokenizer(eos_token="")

        ds = OrchestratorSFTDataset(
            trace_path="/nonexistent",
            tokenizer=tok,
        )
        convs = [
            {"role": "tool", "name": "calculator", "content": "42"},
        ]
        text = ds._format_conversation(convs)
        assert "calculator" in text
        assert "42" in text

    def test_iter_batches(self, tmp_path):
        import json

        trace_file = tmp_path / "traces.jsonl"
        traces = []
        for i in range(5):
            traces.append({
                "conversations": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            })
        trace_file.write_text(
            "\n".join(json.dumps(t) for t in traces) + "\n"
        )

        tok = _FakeTokenizer(eos_token="")

        ds = OrchestratorSFTDataset(
            trace_path=str(trace_file),
            tokenizer=tok,
        )
        batches = list(ds.iter_batches(batch_size=2))
        assert len(batches) == 3  # 2+2+1


class TestSFTRegistration:
    def test_registered_in_learning_registry(self):
        # Import to trigger registration
        import openjarvis.learning.intelligence.orchestrator.sft_trainer  # noqa: F401
        from openjarvis.core.registry import LearningRegistry
        assert LearningRegistry.contains("orchestrator_sft")
