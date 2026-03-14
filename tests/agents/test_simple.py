"""Tests for the SimpleAgent."""

from __future__ import annotations

import pytest

from openjarvis.agents._stubs import AgentContext, AgentResult
from openjarvis.agents.simple import SimpleAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Conversation, Message, Role
from tests.fixtures.engines import FakeEngine


def _make_engine(content: str = "Hello there!") -> FakeEngine:
    return FakeEngine(engine_id="mock", responses=[content])


class TestSimpleAgent:
    @pytest.mark.spec("REQ-agents.simple")
    def test_basic_run(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        result = agent.run("Hello")
        assert isinstance(result, AgentResult)
        assert result.content == "Hello there!"
        assert result.turns == 1
        assert engine.call_count == 1

    @pytest.mark.spec("REQ-agents.simple")
    def test_agent_id(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        assert agent.agent_id == "simple"

    @pytest.mark.spec("REQ-agents.context")
    def test_with_context_conversation(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="You are helpful."))
        ctx = AgentContext(conversation=conv)
        agent.run("Hello", context=ctx)
        last_call = engine.call_history[-1]
        messages = last_call["messages"]
        # Should have system message + user message
        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER

    @pytest.mark.spec("REQ-agents.simple")
    def test_without_context(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        agent.run("Hello")
        last_call = engine.call_history[-1]
        messages = last_call["messages"]
        assert len(messages) == 1
        assert messages[0].role == Role.USER

    @pytest.mark.spec("REQ-agents.base.protocol")
    def test_custom_temperature(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model", temperature=0.1)
        agent.run("Hello")
        last_call = engine.call_history[-1]
        assert last_call["temperature"] == 0.1

    @pytest.mark.spec("REQ-agents.base.protocol")
    def test_custom_max_tokens(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model", max_tokens=256)
        agent.run("Hello")
        last_call = engine.call_history[-1]
        assert last_call["max_tokens"] == 256

    @pytest.mark.spec("REQ-agents.base.events")
    def test_event_bus_integration(self):
        bus = EventBus(record_history=True)
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types
        # INFERENCE_START/END are now published by InstrumentedEngine,
        # not by agents directly

    @pytest.mark.spec("REQ-agents.base.events")
    def test_turn_start_event_data(self):
        bus = EventBus(record_history=True)
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model", bus=bus)
        agent.run("test input")
        evts = bus.history
        start = [
            e for e in evts
            if e.event_type == EventType.AGENT_TURN_START
        ][0]
        assert start.data["agent"] == "simple"
        assert start.data["input"] == "test input"

    @pytest.mark.spec("REQ-agents.base.events")
    def test_turn_end_event_data(self):
        bus = EventBus(record_history=True)
        engine = _make_engine("response text")
        agent = SimpleAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        end = [e for e in bus.history if e.event_type == EventType.AGENT_TURN_END][0]
        assert end.data["agent"] == "simple"
        assert end.data["content_length"] == len("response text")

    @pytest.mark.spec("REQ-agents.simple")
    def test_no_bus_works(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "Hello there!"

    @pytest.mark.spec("REQ-agents.simple")
    def test_empty_content_response(self):
        engine = _make_engine("")
        agent = SimpleAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == ""
        assert result.turns == 1

    @pytest.mark.spec("REQ-agents.simple")
    def test_empty_context(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        ctx = AgentContext()
        result = agent.run("Hello", context=ctx)
        assert result.content == "Hello there!"

    @pytest.mark.spec("REQ-agents.result")
    def test_result_has_no_tool_results(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.tool_results == []

    @pytest.mark.spec("REQ-agents.simple")
    def test_model_passthrough(self):
        engine = _make_engine()
        agent = SimpleAgent(engine, "qwen3:8b")
        agent.run("Hello")
        last_call = engine.call_history[-1]
        assert last_call["model"] == "qwen3:8b"
