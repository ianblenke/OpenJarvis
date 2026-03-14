"""Tests for NativeOpenHandsAgent (formerly OpenHandsAgent)."""

from __future__ import annotations

import pytest

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.native_openhands import NativeOpenHandsAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Conversation, Message, Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from tests.fixtures.engines import FakeEngine
from tests.fixtures.tools import CalculatorStub

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CodeInterpreterStub(BaseTool):
    """Stub code_interpreter tool for testing."""

    tool_id = "code_interpreter"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="code_interpreter",
            description="Execute Python code.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        )

    def execute(self, **params) -> ToolResult:
        code = params.get("code", "")
        # Simple simulation: if it contains print(), capture the content
        if "print(" in code:
            import re
            match = re.search(r"print\((.+?)\)", code)
            if match:
                try:
                    val = eval(match.group(1))  # noqa: S307
                    return ToolResult(
                        tool_name="code_interpreter",
                        content=str(val),
                        success=True,
                    )
                except Exception:
                    pass
        return ToolResult(
            tool_name="code_interpreter",
            content=f"Executed: {code}",
            success=True,
        )


# ---------------------------------------------------------------------------
# Typed fake for httpx.Response (URL expansion tests)
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    """Typed fake for httpx response, replacing MagicMock in URL expansion tests."""

    def __init__(
        self,
        text: str = "",
        headers: dict | None = None,
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.headers = headers or {"content-type": "text/html"}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestNativeOpenHandsRegistration:
    @pytest.mark.spec("REQ-agents.registration")
    def test_registration(self):
        AgentRegistry.register_value("native_openhands", NativeOpenHandsAgent)
        assert AgentRegistry.contains("native_openhands")

    @pytest.mark.spec("REQ-agents.registration")
    def test_agent_id(self):
        engine = FakeEngine(engine_id="fake")
        agent = NativeOpenHandsAgent(engine, "test-model")
        assert agent.agent_id == "native_openhands"

    @pytest.mark.spec("REQ-agents.registration")
    def test_accepts_tools(self):
        assert NativeOpenHandsAgent.accepts_tools is True


# ---------------------------------------------------------------------------
# Agent execution tests
# ---------------------------------------------------------------------------


class TestNativeOpenHandsAgent:
    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_simple_response(self):
        """No code -> direct answer."""
        engine = FakeEngine(responses=["The answer is 42."])
        bus = EventBus(record_history=True)
        agent = NativeOpenHandsAgent(engine, "test-model", bus=bus)
        result = agent.run("What is the meaning of life?")
        assert result.content == "The answer is 42."
        assert result.turns == 1
        assert result.tool_results == []

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_code_generation_execution(self):
        """Turn 1: returns code block -> code_interpreter executed. Turn 2: final."""
        engine = FakeEngine(
            responses=[
                "Let me calculate:\n```python\nprint(2+2)\n```",
                "The result is 4.",
            ],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The result is 4."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "code_interpreter"

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_multi_step_code(self):
        """Multiple code blocks across turns."""
        engine = FakeEngine(
            responses=[
                "Step 1:\n```python\nprint(1+1)\n```",
                "Step 2:\n```python\nprint(3*3)\n```",
                "First was 2, second was 9.",
            ],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
        )
        result = agent.run("Two calculations")
        assert result.turns == 3
        assert len(result.tool_results) == 2
        assert result.content == "First was 2, second was 9."

    @pytest.mark.spec("REQ-agents.max-turns")
    def test_max_turns(self):
        """Engine keeps generating code -> hits max_turns."""
        engine = FakeEngine(
            responses=["More code:\n```python\nprint('hello')\n```"],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
            max_turns=3,
        )
        result = agent.run("Keep coding")
        assert result.turns == 3
        assert result.metadata.get("max_turns_exceeded") is True

    @pytest.mark.spec("REQ-agents.events")
    def test_event_bus_emissions(self):
        """Verify AGENT_TURN_START and AGENT_TURN_END events."""
        bus = EventBus(record_history=True)
        engine = FakeEngine(responses=["Direct answer."])
        agent = NativeOpenHandsAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    @pytest.mark.spec("REQ-agents.events")
    def test_event_bus_tool_events(self):
        """Tool execution should trigger TOOL_CALL_START/END events."""
        bus = EventBus(record_history=True)
        engine = FakeEngine(
            responses=[
                "Code:\n```python\nprint(1)\n```",
                "Done.",
            ],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
            bus=bus,
        )
        agent.run("Run code")
        event_types = [e.event_type for e in bus.history]
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types

    @pytest.mark.spec("REQ-agents.context")
    def test_context_passing(self):
        """Pass AgentContext with conversation history."""
        engine = FakeEngine(responses=["Hello!"])
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="Previous"))
        conv.add(Message(role=Role.ASSISTANT, content="Previous response"))
        ctx = AgentContext(conversation=conv)
        agent = NativeOpenHandsAgent(engine, "test-model")
        agent.run("Hello", context=ctx)
        call = engine.call_history[0]
        messages = call["messages"]
        # System prompt + 2 context + user input
        assert len(messages) == 4
        assert messages[0].role == Role.SYSTEM
        assert messages[1].content == "Previous"
        assert messages[3].content == "Hello"

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_tool_fallback(self):
        """Non-code tool use via Action: syntax."""
        engine = FakeEngine(
            responses=[
                'Action: calculator\nAction Input: {"expression": "7*6"}',
                "The answer is 42.",
            ],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
        )
        result = agent.run("What is 7 times 6?")
        assert result.content == "The answer is 42."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calculator"
        assert result.tool_results[0].content == "42"

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_no_code_interpreter_tool(self):
        """Agent has code but no code_interpreter -> tool not found."""
        engine = FakeEngine(
            responses=[
                "```python\nprint(1)\n```",
                "Could not run code.",
            ],
        )
        # No tools at all
        agent = NativeOpenHandsAgent(engine, "test-model")
        result = agent.run("Run code")
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "Unknown tool" in result.tool_results[0].content

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_no_bus_works(self):
        """Agent runs correctly without an event bus."""
        engine = FakeEngine(responses=["Works!"])
        agent = NativeOpenHandsAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "Works!"

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_system_prompt_includes_tool_names(self):
        """System prompt should list available tool names."""
        engine = FakeEngine(responses=["Ok"])
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub(), CalculatorStub()],
        )
        agent.run("Hello")
        call = engine.call_history[0]
        messages = call["messages"]
        system_msg = messages[0]
        assert "code_interpreter" in system_msg.content
        assert "calculator" in system_msg.content

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_system_prompt_enriched_descriptions(self):
        """System prompt has enriched descriptions with param schemas."""
        engine = FakeEngine(responses=["Ok"])
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub(), CalculatorStub()],
        )
        agent.run("Hello")
        call = engine.call_history[0]
        messages = call["messages"]
        system_msg = messages[0].content
        assert "### calculator" in system_msg
        assert "### code_interpreter" in system_msg
        assert "expression" in system_msg
        assert "string" in system_msg

    @pytest.mark.spec("REQ-agents.max-turns")
    def test_max_turns_content_preserved(self):
        """When max turns exceeded, last content should be preserved."""
        engine = FakeEngine(
            responses=["Still working:\n```python\nx = 1\n```"],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
            max_turns=2,
        )
        result = agent.run("Loop")
        assert result.metadata.get("max_turns_exceeded") is True
        # Should have the last content, not the fallback message
        assert "Still working" in result.content

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_observation_appended_to_messages(self):
        """Code output is sent back to the engine as observation."""
        engine = FakeEngine(
            responses=[
                "```python\nprint(42)\n```",
                "Got 42.",
            ],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[_CodeInterpreterStub()],
        )
        agent.run("Print 42")
        second_call = engine.call_history[1]
        messages = second_call["messages"]
        last_msg = messages[-1]
        assert last_msg.role == Role.USER
        assert "Output:" in last_msg.content

    @pytest.mark.spec("REQ-agents.events")
    def test_event_data_agent_turn_start(self):
        """AGENT_TURN_START event data should include agent id and input."""
        bus = EventBus(record_history=True)
        engine = FakeEngine(responses=["Hi"])
        agent = NativeOpenHandsAgent(engine, "test-model", bus=bus)
        agent.run("test input")
        start_events = [
            e for e in bus.history
            if e.event_type == EventType.AGENT_TURN_START
        ]
        assert len(start_events) == 1
        assert start_events[0].data["agent"] == "native_openhands"
        assert start_events[0].data["input"] == "test input"

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_empty_input(self):
        """Agent handles empty input."""
        engine = FakeEngine(responses=["Empty input received."])
        agent = NativeOpenHandsAgent(engine, "test-model")
        result = agent.run("")
        assert result.content == "Empty input received."

    @pytest.mark.spec("REQ-agents.error-handling")
    def test_error_400_handling(self):
        """Agent catches 400 errors and returns friendly message."""
        engine = FakeEngine()
        # Override generate to raise

        def _raising_generate(*args, **kwargs):
            raise RuntimeError("HTTP 400 Bad Request")

        engine.generate = _raising_generate
        agent = NativeOpenHandsAgent(engine, "test-model")
        result = agent.run("Hello")
        assert "too long" in result.content
        assert result.metadata.get("error") is True

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_xml_tool_call_extraction(self):
        """Agent parses XML-style tool calls."""
        engine = FakeEngine(
            responses=[
                '<tool_call>calculator\n$expression=7*6</calculator>',
                "The answer is 42.",
            ],
        )
        agent = NativeOpenHandsAgent(
            engine, "test-model",
            tools=[CalculatorStub()],
        )
        result = agent.run("What is 7 times 6?")
        assert result.content == "The answer is 42."
        assert len(result.tool_results) == 1
        assert result.tool_results[0].content == "42"

    @pytest.mark.spec("REQ-agents.openhands-run")
    def test_observation_truncation(self):
        """Long tool results are truncated in observations."""
        engine = FakeEngine(
            responses=[
                'Action: calculator\nAction Input: {"expression": "1+1"}',
                "Done.",
            ],
        )
        # Make calculator return very long output
        long_calc = CalculatorStub()
        orig_execute = long_calc.execute

        def _long_execute(**params):
            r = orig_execute(**params)
            return ToolResult(
                tool_name=r.tool_name,
                content="x" * 10000,
                success=True,
            )

        long_calc.execute = _long_execute
        agent = NativeOpenHandsAgent(engine, "test-model", tools=[long_calc])
        agent.run("Compute")
        # Check the observation message sent to the engine
        second_call = engine.call_history[1]
        messages = second_call["messages"]
        last_msg = messages[-1]
        assert len(last_msg.content) < 5000
        assert "[Output truncated]" in last_msg.content


# ---------------------------------------------------------------------------
# Truncation tests
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.spec("REQ-agents.truncation")
    def test_short_messages_unchanged(self):
        """Messages under limit are not modified."""
        engine = FakeEngine()
        agent = NativeOpenHandsAgent(engine, "test-model")
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="Short query"),
        ]
        result = agent._truncate_if_needed(messages, max_prompt_tokens=1000)
        assert result[1].content == "Short query"

    @pytest.mark.spec("REQ-agents.truncation")
    def test_long_messages_truncated(self):
        """Messages over limit get the last user message truncated."""
        engine = FakeEngine()
        agent = NativeOpenHandsAgent(engine, "test-model")
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="x" * 20000),
        ]
        result = agent._truncate_if_needed(messages, max_prompt_tokens=1000)
        assert len(result[1].content) < 20000
        assert "[Input truncated to fit context window]" in result[1].content

    @pytest.mark.spec("REQ-agents.truncation")
    def test_truncation_preserves_system_prompt(self):
        """Truncation only modifies user message, not system prompt."""
        engine = FakeEngine()
        agent = NativeOpenHandsAgent(engine, "test-model")
        system = "Important system prompt"
        messages = [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content="y" * 20000),
        ]
        result = agent._truncate_if_needed(messages, max_prompt_tokens=1000)
        assert result[0].content == system


# ---------------------------------------------------------------------------
# URL expansion tests
# ---------------------------------------------------------------------------


class TestUrlExpansion:
    @pytest.mark.spec("REQ-agents.url-expansion")
    def test_no_url_returns_false(self):
        text, expanded = NativeOpenHandsAgent._expand_urls("What is 2+2?")
        assert text == "What is 2+2?"
        assert expanded is False

    @pytest.mark.spec("REQ-agents.url-expansion")
    def test_url_detected_returns_true(self, monkeypatch):
        import httpx

        fake_resp = _FakeHttpResponse(
            text="<html><body>Page content</body></html>",
            headers={"content-type": "text/html"},
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        text, expanded = NativeOpenHandsAgent._expand_urls(
            "Summarize: https://example.com/article"
        )
        assert expanded is True
        assert "Page content" in text
        assert "Content from" in text

    @pytest.mark.spec("REQ-agents.url-expansion")
    def test_url_expansion_failure_returns_false(self, monkeypatch):
        import httpx

        def _raise(*a, **kw):
            raise Exception("Connection error")

        monkeypatch.setattr(httpx, "get", _raise)
        text, expanded = NativeOpenHandsAgent._expand_urls(
            "Read https://example.com/broken"
        )
        assert expanded is False

    @pytest.mark.spec("REQ-agents.url-expansion")
    def test_url_expanded_uses_direct_path(self, monkeypatch):
        """When URL is expanded, agent bypasses tool loop."""
        import httpx

        fake_resp = _FakeHttpResponse(
            text="<html><body>Article text here</body></html>",
            headers={"content-type": "text/html"},
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        engine = FakeEngine(responses=["Summary of article."])
        agent = NativeOpenHandsAgent(engine, "test-model")
        result = agent.run("Summarize: https://example.com/article")
        assert result.content == "Summary of article."
        assert result.turns == 1
        # Only one generate call (direct, no tool loop)
        assert engine.call_count == 1
        # The message should contain the fetched content, not tool descriptions
        call_messages = engine.call_history[0]["messages"]
        system_msg = call_messages[0].content
        assert "tool" not in system_msg.lower()
