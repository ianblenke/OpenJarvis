"""Tests for the think tool.

Covers ThinkTool spec properties and execute behaviour.
Uses the real Rust backend.
"""

from __future__ import annotations

import pytest

from openjarvis.tools.think import ThinkTool


class TestThinkTool:
    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_name(self):
        tool = ThinkTool()
        assert tool.spec.name == "think"

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_category(self):
        tool = ThinkTool()
        assert tool.spec.category == "reasoning"

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_cost_estimate_zero(self):
        tool = ThinkTool()
        assert tool.spec.cost_estimate == 0.0

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_latency_estimate_zero(self):
        tool = ThinkTool()
        assert tool.spec.latency_estimate == 0.0

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_has_thought_param(self):
        tool = ThinkTool()
        props = tool.spec.parameters["properties"]
        assert "thought" in props

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_thought_required(self):
        tool = ThinkTool()
        assert "thought" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_tool_id(self):
        tool = ThinkTool()
        assert tool.tool_id == "think"

    @pytest.mark.spec("REQ-tools.think.spec")
    def test_spec_description_non_empty(self):
        tool = ThinkTool()
        assert len(tool.spec.description) > 0

    @pytest.mark.spec("REQ-tools.think.execute")
    def test_echoes_thought(self):
        tool = ThinkTool()
        result = tool.execute(thought="Let me think step by step...")
        assert result.success is True
        assert result.content == "Let me think step by step..."

    @pytest.mark.spec("REQ-tools.think.execute")
    def test_empty_thought(self):
        tool = ThinkTool()
        result = tool.execute(thought="")
        assert result.success is True
        assert result.content == ""

    @pytest.mark.spec("REQ-tools.think.execute")
    def test_no_thought_param(self):
        tool = ThinkTool()
        result = tool.execute()
        assert result.success is True
        assert result.content == ""

    @pytest.mark.spec("REQ-tools.think.execute")
    def test_multiline_thought(self):
        tool = ThinkTool()
        thought = "Step 1: Analyze.\nStep 2: Plan.\nStep 3: Execute."
        result = tool.execute(thought=thought)
        assert result.success is True
        assert result.content == thought

    @pytest.mark.spec("REQ-tools.think.execute")
    def test_tool_name_in_result(self):
        tool = ThinkTool()
        result = tool.execute(thought="test")
        assert result.tool_name == "think"

    @pytest.mark.spec("REQ-tools.think.execute")
    def test_long_thought(self):
        tool = ThinkTool()
        long_text = "reasoning " * 1000
        result = tool.execute(thought=long_text)
        assert result.success is True
        assert result.content == long_text

    @pytest.mark.spec("REQ-tools.think.openai")
    def test_openai_function(self):
        tool = ThinkTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "think"
        assert "thought" in fn["function"]["parameters"]["properties"]
