"""Tests for the RLM agent."""

from __future__ import annotations

import pytest

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.rlm import RLMAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from tests.fixtures.engines import FakeEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CalcStub(BaseTool):
    tool_id = "calculator"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Math calculator.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

    def execute(self, **params) -> ToolResult:
        expr = params.get("expression", "0")
        try:
            val = eval(expr)  # noqa: S307
        except Exception as e:
            return ToolResult(tool_name="calculator", content=str(e), success=False)
        return ToolResult(tool_name="calculator", content=str(val), success=True)


def _make_engine(content: str = "Final answer.") -> FakeEngine:
    """Engine that returns plain content (no code block)."""
    return FakeEngine(engine_id="fake", responses=[content])


def _make_engine_with_responses(*responses: str) -> FakeEngine:
    """Engine that returns a sequence of responses."""
    return FakeEngine(engine_id="fake", responses=list(responses))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRLMAgentRegistration:
    @pytest.mark.spec("REQ-agents.base.registration")
    def test_registered(self):
        # Re-register after conftest clears all registries
        AgentRegistry.register_value("rlm", RLMAgent)
        assert AgentRegistry.contains("rlm")

    @pytest.mark.spec("REQ-agents.base.registration")
    def test_agent_id(self):
        engine = _make_engine()
        agent = RLMAgent(engine, "test-model")
        assert agent.agent_id == "rlm"


class TestRLMCodeExtraction:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_extract_python_block(self):
        text = "Here is code:\n```python\nx = 1\n```\nDone."
        code = RLMAgent._extract_code(text)
        assert code == "x = 1"

    @pytest.mark.spec("REQ-agents.base.run")
    def test_extract_bare_block(self):
        text = "Here is code:\n```\nx = 1\n```\nDone."
        code = RLMAgent._extract_code(text)
        assert code == "x = 1"

    @pytest.mark.spec("REQ-agents.base.run")
    def test_no_block(self):
        text = "No code here, just text."
        code = RLMAgent._extract_code(text)
        assert code is None

    @pytest.mark.spec("REQ-agents.base.run")
    def test_python_preferred_over_bare(self):
        text = "```python\nx = 1\n```\n\n```\ny = 2\n```"
        code = RLMAgent._extract_code(text)
        assert code == "x = 1"


class TestRLMStripThink:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_strip_think(self):
        text = "<think>thinking...</think>Answer here."
        result = RLMAgent._strip_think_tags(text)
        assert result == "Answer here."

    @pytest.mark.spec("REQ-agents.base.run")
    def test_no_think_tags(self):
        text = "Just text."
        result = RLMAgent._strip_think_tags(text)
        assert result == "Just text."


class TestRLMDirectAnswer:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_no_code_block_returns_content(self):
        """When model returns no code block, treat content as final answer."""
        engine = _make_engine("The answer is 42.")
        agent = RLMAgent(engine, "test-model")
        result = agent.run("What is the answer?")
        assert result.content == "The answer is 42."
        assert result.turns == 1
        assert result.tool_results == []


class TestRLMFinalTermination:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_final_terminates(self):
        """FINAL() in code should terminate the agent."""
        engine = _make_engine("```python\nFINAL('hello world')\n```")
        agent = RLMAgent(engine, "test-model")
        result = agent.run("Test")
        assert result.content == "hello world"
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "rlm_repl"

    @pytest.mark.spec("REQ-agents.base.run")
    def test_final_var_terminates(self):
        engine = _make_engine("```python\nresult = 42\nFINAL_VAR('result')\n```")
        agent = RLMAgent(engine, "test-model")
        result = agent.run("Test")
        assert result.content == "42"


class TestRLMContextInjection:
    @pytest.mark.spec("REQ-agents.context")
    def test_context_from_metadata(self):
        engine = _make_engine("The answer is 42.")
        agent = RLMAgent(engine, "test-model")
        ctx = AgentContext(metadata={"context": "Some long document text."})
        result = agent.run("Summarize", context=ctx)
        assert result.content == "The answer is 42."

    @pytest.mark.spec("REQ-agents.context")
    def test_context_from_memory_results(self):
        engine = _make_engine("Summary.")
        agent = RLMAgent(engine, "test-model")
        ctx = AgentContext(memory_results=["chunk1", "chunk2"])
        result = agent.run("Summarize", context=ctx)
        assert result.content == "Summary."


class TestRLMSubLMCalls:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_sub_lm_called_from_repl(self):
        """Verify that llm_query() inside REPL code calls engine.generate."""
        engine = _make_engine_with_responses(
            # First call: root LM generates code that calls llm_query
            "```python\n"
            "result = llm_query('What is 2+2?')\n"
            "FINAL(result)\n```",
            # Second call: sub-LM responds to llm_query
            "4",
        )
        agent = RLMAgent(engine, "test-model")
        result = agent.run("Calculate")
        assert result.content == "4"
        # engine.generate should be called at least twice (root + sub)
        assert engine.call_count >= 2


class TestRLMMultiTurn:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_multi_turn_loop(self):
        """Agent should loop: generate -> execute -> feed -> repeat."""
        engine = _make_engine_with_responses(
            # Turn 1: code that sets a variable
            "```python\n"
            "x = 10\nprint(f'x = {x}')\n```",
            # Turn 2: code that uses the variable and terminates
            "```python\ny = x * 2\nFINAL(y)\n```",
        )
        agent = RLMAgent(engine, "test-model")
        result = agent.run("Calculate")
        assert result.content == "20"
        assert result.turns == 2
        assert len(result.tool_results) == 2

    @pytest.mark.spec("REQ-agents.tool-using.loop-guard")
    def test_max_turns_exceeded(self):
        """Agent should stop after max_turns."""
        engine = _make_engine("```python\nprint('looping')\n```")
        agent = RLMAgent(engine, "test-model", max_turns=3)
        result = agent.run("Loop")
        assert result.turns == 3
        assert result.metadata.get("max_turns_exceeded") is True

    @pytest.mark.spec("REQ-agents.tool-using.loop-guard")
    def test_max_turns_with_partial_answer(self):
        """When max turns exceeded but answer dict has value, use it."""
        code = (
            "```python\n"
            "answer['value'] = 'partial'\n"
            "print('working')\n```"
        )
        engine = _make_engine(code)
        agent = RLMAgent(engine, "test-model", max_turns=2)
        result = agent.run("Work")
        assert result.content == "partial"
        assert result.metadata.get("max_turns_exceeded") is True


class TestRLMEventBus:
    @pytest.mark.spec("REQ-agents.base.events")
    def test_agent_events(self):
        bus = EventBus(record_history=True)
        engine = _make_engine("Direct answer.")
        agent = RLMAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    @pytest.mark.spec("REQ-agents.base.events")
    def test_agent_events_with_code(self):
        bus = EventBus(record_history=True)
        engine = _make_engine("```python\nFINAL('done')\n```")
        agent = RLMAgent(engine, "test-model", bus=bus)
        agent.run("Test")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types


class TestRLMSubLMWithTools:
    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_sub_lm_tool_resolution(self):
        """When sub-LM returns tool_calls, agent resolves them."""
        engine = FakeEngine(
            engine_id="fake",
            responses=[
                # Root LM: code that calls llm_query
                "```python\n"
                "result = llm_query('Calculate 2+2')\n"
                "FINAL(result)\n```",
                # Sub-LM: returns empty content (tool call in tool_calls)
                "",
                # Sub-LM follow-up after tool result
                "The answer is 4.",
            ],
            tool_calls=[
                [],  # No tool calls for root response
                [    # Sub-LM returns tool call
                    {
                        "id": "sub_0",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                ],
                [],  # No tool calls for follow-up
            ],
        )
        agent = RLMAgent(engine, "test-model", tools=[_CalcStub()])
        result = agent.run("Calculate")
        assert result.content == "The answer is 4."


class TestRLMBlockedCode:
    @pytest.mark.spec("REQ-agents.base.run")
    def test_blocked_code_returns_error(self):
        engine = _make_engine_with_responses(
            # Code with blocked pattern
            "```python\nos.system('ls')\n```",
            # After error feedback, model gives direct answer
            "I apologize, let me answer directly.",
        )
        agent = RLMAgent(engine, "test-model")
        result = agent.run("Test")
        assert result.content == "I apologize, let me answer directly."
        # The blocked code should produce a failed tool result
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "Blocked" in result.tool_results[0].content


class TestRLMToolSectionInjection:
    """Verify that tool descriptions are injected into the RLM system prompt."""

    @pytest.mark.spec("REQ-agents.tool-using.protocol")
    def test_system_prompt_includes_tool_section(self):
        """Tools provided -> system prompt includes descriptions."""
        engine = _make_engine("Direct answer.")
        agent = RLMAgent(engine, "test-model", tools=[_CalcStub()])
        agent.run("Hello")
        call = engine.call_history[0]
        messages = call["messages"]
        system_msg = messages[0].content
        assert "## Available Tools" in system_msg
        assert "### calculator" in system_msg
        assert "expression" in system_msg

    @pytest.mark.spec("REQ-agents.tool-using.protocol")
    def test_system_prompt_no_tool_section_without_tools(self):
        """No tools -> system prompt has no tool section."""
        engine = _make_engine("Direct answer.")
        agent = RLMAgent(engine, "test-model")
        agent.run("Hello")
        call = engine.call_history[0]
        messages = call["messages"]
        system_msg = messages[0].content
        assert "## Available Tools" not in system_msg


class TestRLMReplResults:
    @pytest.mark.spec("REQ-agents.result")
    def test_repl_results_in_tool_results(self):
        engine = _make_engine("```python\nprint('hello')\nFINAL('done')\n```")
        agent = RLMAgent(engine, "test-model")
        result = agent.run("Test")
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "rlm_repl"
        assert "hello" in result.tool_results[0].content
