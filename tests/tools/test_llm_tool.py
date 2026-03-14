"""Tests for the LLM tool."""

from __future__ import annotations

import pytest

from openjarvis.tools.llm_tool import LLMTool
from tests.fixtures.engines import FakeEngine


def _make_engine(content: str = "response") -> FakeEngine:
    return FakeEngine(responses=[content])


class _ErrorEngine(FakeEngine):
    """Engine that raises on generate()."""

    def generate(self, messages, **kwargs):
        raise RuntimeError("connection failed")


class TestLLMTool:
    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_spec(self):
        tool = LLMTool()
        assert tool.spec.name == "llm"
        assert tool.spec.category == "inference"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_engine(self):
        tool = LLMTool()
        result = tool.execute(prompt="hello")
        assert result.success is False
        assert "No inference engine" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_model(self):
        tool = LLMTool(engine=_make_engine(), model="")
        result = tool.execute(prompt="hello")
        assert result.success is False
        assert "No model" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_prompt(self):
        tool = LLMTool(engine=_make_engine(), model="test-model")
        result = tool.execute(prompt="")
        assert result.success is False

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_successful_generation(self):
        engine = _make_engine("The answer is 42.")
        tool = LLMTool(engine=engine, model="test-model")
        result = tool.execute(prompt="What is the answer?")
        assert result.success is True
        assert result.content == "The answer is 42."
        assert engine.call_count == 1

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_with_system_message(self):
        engine = _make_engine()
        tool = LLMTool(engine=engine, model="test-model")
        tool.execute(prompt="hello", system="You are helpful.")
        messages = engine.call_history[-1]["messages"]
        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert messages[1].role.value == "user"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_engine_error(self):
        engine = _ErrorEngine()
        tool = LLMTool(engine=engine, model="test-model")
        result = tool.execute(prompt="hello")
        assert result.success is False
        assert "LLM error" in result.content
