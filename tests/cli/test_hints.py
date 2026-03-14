"""Tests for CLI error hint functions."""

from __future__ import annotations

import pytest

from openjarvis.cli.hints import (
    hint_no_config,
    hint_no_engine,
    hint_no_model,
)


class TestHintFunctions:
    @pytest.mark.spec("REQ-cli.ask")
    def test_hint_no_config_returns_string(self):
        msg = hint_no_config()
        assert isinstance(msg, str)
        assert len(msg) > 0
        assert "init" in msg.lower() or "config" in msg.lower()

    @pytest.mark.spec("REQ-cli.ask")
    def test_hint_no_engine_returns_string(self):
        msg = hint_no_engine()
        assert isinstance(msg, str)
        assert len(msg) > 0
        assert "engine" in msg.lower() or "ollama" in msg.lower()

    @pytest.mark.spec("REQ-cli.ask")
    def test_hint_no_engine_with_name(self):
        msg = hint_no_engine("vllm")
        assert "vllm" in msg.lower()

    @pytest.mark.spec("REQ-cli.ask")
    def test_hint_no_model_returns_string(self):
        msg = hint_no_model()
        assert isinstance(msg, str)
        assert len(msg) > 0
        assert "model" in msg.lower() or "pull" in msg.lower()

    @pytest.mark.spec("REQ-cli.ask")
    def test_hint_no_model_with_name(self):
        msg = hint_no_model("qwen3:8b")
        assert "qwen3:8b" in msg
