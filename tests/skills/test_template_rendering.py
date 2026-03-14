"""Tests for skill executor template rendering."""

from __future__ import annotations

import pytest

from openjarvis.core.events import EventBus
from openjarvis.skills.executor import SkillExecutor
from openjarvis.skills.types import SkillManifest, SkillStep
from openjarvis.tools._stubs import ToolExecutor
from tests.fixtures.tools import CalculatorStub, EchoTool


class TestTemplateRendering:
    """Test SkillExecutor._render_template()."""

    @pytest.mark.spec("REQ-skills.executor.context")
    def test_render_simple_placeholder(self) -> None:
        result = SkillExecutor._render_template(
            '{"text": "{input}"}',
            {"input": "hello"},
        )
        assert result == '{"text": "hello"}'

    @pytest.mark.spec("REQ-skills.executor.context")
    def test_render_multiple_placeholders(self) -> None:
        result = SkillExecutor._render_template(
            '{"a": "{x}", "b": "{y}"}',
            {"x": "foo", "y": "bar"},
        )
        assert '"a": "foo"' in result
        assert '"b": "bar"' in result

    @pytest.mark.spec("REQ-skills.executor.context")
    def test_render_missing_key_preserves_placeholder(self) -> None:
        result = SkillExecutor._render_template(
            '{"text": "{missing}"}',
            {},
        )
        assert "{missing}" in result


class TestSkillExecutorRun:
    """Test full skill execution with real tools."""

    @pytest.mark.spec("REQ-skills.executor.run")
    def test_single_step_execution(self) -> None:
        tools = [EchoTool()]
        executor = ToolExecutor(tools)
        skill_executor = SkillExecutor(executor)
        manifest = SkillManifest(
            name="echo-skill",
            version="1.0",
            description="echo test",
            author="test",
            steps=[
                SkillStep(
                    tool_name="echo",
                    arguments_template='{"text": "{input}"}',
                    output_key="result",
                ),
            ],
        )
        result = skill_executor.run(manifest, initial_context={"input": "hello"})
        assert result.success

    @pytest.mark.spec("REQ-skills.executor.run")
    def test_multi_step_context_chaining(self) -> None:
        tools = [EchoTool(), CalculatorStub()]
        executor = ToolExecutor(tools)
        skill_executor = SkillExecutor(executor)
        manifest = SkillManifest(
            name="chain-skill",
            version="1.0",
            description="chain test",
            author="test",
            steps=[
                SkillStep(
                    tool_name="echo",
                    arguments_template='{"text": "2+2"}',
                    output_key="expression",
                ),
                SkillStep(
                    tool_name="calculator",
                    arguments_template='{"expression": "{expression}"}',
                    output_key="answer",
                ),
            ],
        )
        result = skill_executor.run(manifest)
        assert result.success
        assert "answer" in result.context

    @pytest.mark.spec("REQ-skills.executor.run")
    def test_skill_events_emitted(self) -> None:
        from openjarvis.core.events import EventType

        bus = EventBus(record_history=True)
        tools = [EchoTool()]
        executor = ToolExecutor(tools, bus=bus)
        skill_executor = SkillExecutor(executor, bus=bus)
        manifest = SkillManifest(
            name="event-skill",
            version="1.0",
            description="event test",
            author="test",
            steps=[
                SkillStep(
                    tool_name="echo",
                    arguments_template='{"text": "test"}',
                    output_key="result",
                ),
            ],
        )
        skill_executor.run(manifest)
        event_types = [e.event_type for e in bus.history]
        assert EventType.SKILL_EXECUTE_START in event_types
        assert EventType.SKILL_EXECUTE_END in event_types
