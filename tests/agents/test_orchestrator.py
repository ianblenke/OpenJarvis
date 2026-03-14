"""Tests for the OrchestratorAgent."""

from __future__ import annotations

import pytest

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Conversation, Message, Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from tests.fixtures.engines import FakeEngine
from tests.fixtures.tools import CalculatorStub, ThinkStub

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SlowTool(BaseTool):
    """Tool with simulated delay for parallel execution tests."""

    tool_id = "slow"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow",
            description="Slow tool.",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        )

    def execute(self, **params) -> ToolResult:
        import time

        time.sleep(0.1)  # Simulate slow operation
        return ToolResult(
            tool_name="slow",
            content=f"result_{params.get('id', '')}",
            success=True,
        )


def _engine_no_tools(content: str = "Final answer.") -> FakeEngine:
    """FakeEngine that never returns tool calls."""
    return FakeEngine(
        engine_id="mock",
        responses=[content],
        models=["test-model"],
    )


def _engine_with_tool_call(
    tool_name: str = "calculator",
    arguments: str = '{"expression":"2+2"}',
    tool_call_id: str = "call_1",
    final_content: str = "The answer is 4.",
) -> FakeEngine:
    """FakeEngine that returns one tool call then a final answer.

    Uses the FakeEngine's tool_calls support to emit a tool call on the
    first generate() then a plain text response on the second.
    """
    return FakeEngine(
        engine_id="mock",
        responses=["", final_content],
        tool_calls=[
            [{"id": tool_call_id, "name": tool_name, "arguments": arguments}],
            [],  # second call: no tool calls
        ],
        models=["test-model"],
    )


def _engine_multi_tool() -> FakeEngine:
    """FakeEngine that calls multiple tools in one turn then answers."""
    return FakeEngine(
        engine_id="mock",
        responses=["", "Done."],
        tool_calls=[
            [
                {
                    "id": "call_1",
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
                {
                    "id": "call_2",
                    "name": "think",
                    "arguments": '{"thought":"thinking..."}',
                },
            ],
            [],  # second call: no tool calls
        ],
        models=["test-model"],
    )


def _engine_always_tool_call() -> FakeEngine:
    """FakeEngine that always returns tool calls (for max_turns tests)."""
    # FakeEngine cycles responses; with one response + one tool_calls entry,
    # every call returns a tool call.
    return FakeEngine(
        engine_id="mock",
        responses=[""],
        tool_calls=[
            [{"id": "c1", "name": "calculator", "arguments": '{"expression":"1+1"}'}],
        ],
        models=["test-model"],
    )


def _engine_always_tool_call_with_partial() -> FakeEngine:
    """FakeEngine that always returns tool calls with partial content."""
    return FakeEngine(
        engine_id="mock",
        responses=["partial"],
        tool_calls=[
            [{"id": "c1", "name": "calculator", "arguments": '{"expression":"1"}'}],
        ],
        models=["test-model"],
    )


def _engine_three_turn() -> FakeEngine:
    """FakeEngine that calls a tool twice before answering."""
    return FakeEngine(
        engine_id="mock",
        responses=["", "", "2+2=4, 4*3=12"],
        tool_calls=[
            [
                {
                    "id": "c1",
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
            ],
            [
                {
                    "id": "c2",
                    "name": "calculator",
                    "arguments": '{"expression":"4*3"}',
                },
            ],
            [],  # third call: no tool calls
        ],
        models=["test-model"],
    )


def _engine_structured_final(content: str = "FINAL_ANSWER: ok") -> FakeEngine:
    """FakeEngine for structured mode — single turn with final answer."""
    return FakeEngine(
        engine_id="mock",
        responses=[content],
        models=["test-model"],
    )


def _engine_structured_tool_call() -> FakeEngine:
    """FakeEngine for structured mode — TOOL then FINAL_ANSWER."""
    return FakeEngine(
        engine_id="mock",
        responses=[
            (
                "THOUGHT: Need to calculate.\n"
                'TOOL: calculator\n'
                'INPUT: {"expression":"2+2"}'
            ),
            (
                "THOUGHT: Got 4.\n"
                "FINAL_ANSWER: The answer is 4."
            ),
        ],
        models=["test-model"],
    )


def _engine_parallel_tools() -> FakeEngine:
    """FakeEngine for parallel tool execution tests."""
    return FakeEngine(
        engine_id="mock",
        responses=["", "All done."],
        tool_calls=[
            [
                {"id": "c1", "name": "slow", "arguments": '{"id":"1"}'},
                {"id": "c2", "name": "slow", "arguments": '{"id":"2"}'},
                {"id": "c3", "name": "slow", "arguments": '{"id":"3"}'},
            ],
            [],
        ],
        models=["test-model"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestratorAgent:
    @pytest.mark.spec("REQ-agents.base.registration")
    def test_agent_id(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model")
        assert agent.agent_id == "orchestrator"

    @pytest.mark.spec("REQ-agents.base.run")
    def test_no_tools_single_turn(self):
        engine = _engine_no_tools("Hello!")
        agent = OrchestratorAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "Hello!"
        assert result.turns == 1
        assert result.tool_results == []

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_single_tool_call(self):
        engine = _engine_with_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calculator"
        assert result.tool_results[0].content == "4"

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_multiple_tool_calls_same_turn(self):
        engine = _engine_multi_tool()
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub(), ThinkStub()],
        )
        result = agent.run("Think and calculate.")
        assert result.content == "Done."
        assert result.turns == 2
        assert len(result.tool_results) == 2

    @pytest.mark.spec("REQ-agents.context")
    def test_with_context_conversation(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model")
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="Be helpful."))
        ctx = AgentContext(conversation=conv)
        agent.run("Hi", context=ctx)
        # FakeEngine records call history; verify messages include system
        assert engine.call_count == 1
        messages = engine.call_history[0]["messages"]
        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM

    @pytest.mark.spec("REQ-agents.tool-using.protocol")
    def test_tools_passed_to_engine(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        agent.run("Hello")
        call_kwargs = engine.call_history[0]["kwargs"]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    @pytest.mark.spec("REQ-agents.tool-using.protocol")
    def test_no_tools_no_tools_kwarg(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model")
        agent.run("Hello")
        call_kwargs = engine.call_history[0]["kwargs"]
        assert "tools" not in call_kwargs

    @pytest.mark.spec("REQ-agents.tool-using.loop-guard")
    def test_max_turns_exceeded(self):
        """When the engine keeps returning tool calls, stop after max_turns."""
        engine = _engine_always_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
            max_turns=3,
        )
        result = agent.run("Loop forever")
        assert result.turns == 3
        assert result.metadata.get("max_turns_exceeded") is True

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_unknown_tool_in_response(self):
        engine = _engine_with_tool_call(
            tool_name="unknown_tool",
            arguments="{}",
            final_content="Handled.",
        )
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        result = agent.run("Use unknown tool")
        assert result.content == "Handled."
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False

    @pytest.mark.spec("REQ-agents.base.protocol")
    def test_temperature_passthrough(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", temperature=0.1)
        agent.run("Hello")
        assert engine.call_history[0]["temperature"] == 0.1

    @pytest.mark.spec("REQ-agents.base.protocol")
    def test_max_tokens_passthrough(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", max_tokens=256)
        agent.run("Hello")
        assert engine.call_history[0]["max_tokens"] == 256

    @pytest.mark.spec("REQ-agents.base.events")
    def test_event_bus_agent_events(self):
        bus = EventBus(record_history=True)
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    @pytest.mark.spec("REQ-agents.base.events")
    def test_event_bus_inference_events(self):
        """INFERENCE_START/END are now published by InstrumentedEngine,
        not by agents directly.  Agent tests verify agent-level events."""
        bus = EventBus(record_history=True)
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    @pytest.mark.spec("REQ-agents.base.events")
    def test_event_bus_tool_events(self):
        bus = EventBus(record_history=True)
        engine = _engine_with_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()], bus=bus,
        )
        agent.run("Calc 2+2")
        event_types = [e.event_type for e in bus.history]
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_messages_accumulate(self):
        """After tool call, messages include assistant + tool messages."""
        engine = _engine_with_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        agent.run("What is 2+2?")
        # Second call should include accumulated messages
        assert engine.call_count == 2
        messages = engine.call_history[1]["messages"]
        roles = [m.role for m in messages]
        assert Role.ASSISTANT in roles
        assert Role.TOOL in roles

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_tool_message_has_tool_call_id(self):
        engine = _engine_with_tool_call(tool_call_id="abc123")
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        agent.run("What is 2+2?")
        messages = engine.call_history[1]["messages"]
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "abc123"

    @pytest.mark.spec("REQ-agents.base.run")
    def test_no_bus_works(self):
        engine = _engine_with_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."

    @pytest.mark.spec("REQ-agents.base.run")
    def test_empty_tools_list(self):
        engine = _engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", tools=[])
        result = agent.run("Hello")
        assert result.content == "Final answer."

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_three_turn_conversation(self):
        """Engine calls a tool twice before answering."""
        engine = _engine_three_turn()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        result = agent.run("Calculate")
        assert result.turns == 3
        assert len(result.tool_results) == 2
        assert result.tool_results[0].content == "4"
        assert result.tool_results[1].content == "12"

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_tool_result_latency_tracked(self):
        engine = _engine_with_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.tool_results[0].latency_seconds >= 0

    @pytest.mark.spec("REQ-agents.tool-using.loop-guard")
    def test_max_turns_1(self):
        """With max_turns=1 and a tool call, should stop after 1 turn."""
        engine = _engine_always_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
            max_turns=1,
        )
        result = agent.run("Calc")
        assert result.turns == 1
        assert result.metadata.get("max_turns_exceeded") is True

    @pytest.mark.spec("REQ-agents.base.events")
    def test_agent_turn_end_data_no_tools(self):
        bus = EventBus(record_history=True)
        engine = _engine_no_tools("reply")
        agent = OrchestratorAgent(engine, "test-model", bus=bus)
        agent.run("Hi")
        end = [e for e in bus.history if e.event_type == EventType.AGENT_TURN_END][0]
        assert end.data["turns"] == 1
        assert end.data["content_length"] == 5

    @pytest.mark.spec("REQ-agents.tool-using.loop-guard")
    def test_result_content_on_max_turns(self):
        engine = _engine_always_tool_call_with_partial()
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
            max_turns=2,
        )
        result = agent.run("Calc")
        # Should use the partial content if available
        assert result.content == "partial"


class TestOrchestratorStructuredMode:
    """Tests for the structured (THOUGHT/TOOL/INPUT/FINAL_ANSWER) mode."""

    @pytest.mark.spec("REQ-agents.base.run")
    def test_structured_mode_final_answer(self):
        """Structured mode should parse FINAL_ANSWER: correctly."""
        engine = _engine_structured_final(
            "THOUGHT: Easy question.\nFINAL_ANSWER: Paris",
        )
        agent = OrchestratorAgent(
            engine, "test-model", mode="structured",
        )
        result = agent.run("What is the capital of France?")
        assert result.content == "Paris"
        assert result.turns == 1
        assert result.tool_results == []

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_structured_mode_tool_call(self):
        """Parse TOOL:/INPUT:, execute tool, return final answer."""
        engine = _engine_structured_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
            mode="structured",
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calculator"
        assert result.tool_results[0].content == "4"

    @pytest.mark.spec("REQ-agents.tool-using.protocol")
    def test_structured_mode_enriched_descriptions(self):
        """Structured mode system prompt should contain enriched tool descriptions."""
        engine = _engine_structured_final("FINAL_ANSWER: ok")
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
            mode="structured",
        )
        agent.run("Hello")
        messages = engine.call_history[0]["messages"]
        system_msg = messages[0].content
        assert "### calculator" in system_msg
        assert "expression" in system_msg


class TestOrchestratorParallelTools:
    """Tests for parallel tool execution."""

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_parallel_tool_execution(self):
        """Multiple tool calls execute in parallel and return in correct order."""
        import time

        engine = _engine_parallel_tools()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[_SlowTool()], parallel_tools=True,
        )
        t0 = time.time()
        result = agent.run("Do things")
        elapsed = time.time() - t0

        assert result.content == "All done."
        assert len(result.tool_results) == 3
        # Results should be in original order
        assert result.tool_results[0].content == "result_1"
        assert result.tool_results[1].content == "result_2"
        assert result.tool_results[2].content == "result_3"
        # Should be parallel — 3 tools at 0.1s each should take < 0.25s, not 0.3s+
        assert elapsed < 0.25

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_sequential_tool_execution(self):
        """parallel_tools=False runs tools sequentially."""
        engine = _engine_multi_tool()
        agent = OrchestratorAgent(
            engine, "test-model",
            tools=[CalculatorStub(), ThinkStub()],
            parallel_tools=False,
        )
        result = agent.run("Do things")
        assert result.content == "Done."
        assert len(result.tool_results) == 2

    @pytest.mark.spec("REQ-agents.tool-using.executor")
    def test_single_tool_call_no_parallel(self):
        """Single tool call should not use parallel path even if parallel_tools=True."""
        engine = _engine_with_tool_call()
        agent = OrchestratorAgent(
            engine, "test-model", tools=[CalculatorStub()],
            parallel_tools=True,
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
