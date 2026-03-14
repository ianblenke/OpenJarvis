"""Tests for core data types."""

from __future__ import annotations

import time

import pytest

from openjarvis.core.types import (
    Conversation,
    Message,
    ModelSpec,
    Quantization,
    Role,
    RoutingContext,
    StepType,
    TelemetryRecord,
    ToolCall,
    ToolResult,
    Trace,
    TraceStep,
)


class TestRole:
    @pytest.mark.spec("REQ-core.types.message")
    def test_values(self) -> None:
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"


class TestQuantization:
    @pytest.mark.spec("REQ-core.types.model-spec")
    def test_none_and_variants(self) -> None:
        assert Quantization.NONE == "none"
        assert Quantization.GGUF_Q4 == "gguf_q4"


class TestMessage:
    @pytest.mark.spec("REQ-core.types.message")
    def test_basic_message(self) -> None:
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.metadata == {}

    @pytest.mark.spec("REQ-core.types.message")
    def test_tool_calls(self) -> None:
        tc = ToolCall(id="1", name="calc", arguments='{"x": 1}')
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "calc"


class TestConversation:
    @pytest.mark.spec("REQ-core.types.conversation")
    def test_add_and_window(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="a"))
        conv.add(Message(role=Role.ASSISTANT, content="b"))
        conv.add(Message(role=Role.USER, content="c"))
        assert len(conv.messages) == 3
        assert [m.content for m in conv.window(2)] == ["b", "c"]

    @pytest.mark.spec("REQ-core.types.conversation")
    def test_window_zero_returns_empty(self) -> None:
        conv = Conversation()
        conv.add(Message(role=Role.USER, content="a"))
        conv.add(Message(role=Role.ASSISTANT, content="b"))
        assert conv.window(0) == []

    @pytest.mark.spec("REQ-core.types.conversation")
    def test_max_messages(self) -> None:
        conv = Conversation(max_messages=2)
        for i in range(5):
            conv.add(Message(role=Role.USER, content=str(i)))
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "3"
        assert conv.messages[1].content == "4"


class TestModelSpec:
    @pytest.mark.spec("REQ-core.types.model-spec")
    def test_defaults(self) -> None:
        spec = ModelSpec(
            model_id="m1",
            name="Model One",
            parameter_count_b=7.0,
            context_length=8192,
        )
        assert spec.quantization == Quantization.NONE
        assert spec.min_vram_gb == 0.0
        assert spec.active_parameter_count_b is None
        assert spec.metadata == {}


class TestToolResult:
    @pytest.mark.spec("REQ-core.types.tool-result")
    def test_success_defaults(self) -> None:
        tr = ToolResult(tool_name="calc", content="42")
        assert tr.success is True
        assert tr.cost_usd == 0.0


class TestTelemetryRecord:
    @pytest.mark.spec("REQ-core.types.telemetry-record")
    def test_fields(self) -> None:
        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="m1",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert rec.total_tokens == 30
        assert rec.energy_joules == 0.0


class TestTrace:
    @pytest.mark.spec("REQ-core.types.trace")
    def test_trace_defaults(self) -> None:
        t = Trace()
        assert t.trace_id  # auto-generated, non-empty
        assert t.query == ""
        assert t.agent == ""
        assert t.model == ""
        assert t.engine == ""
        assert t.steps == []
        assert t.result == ""
        assert t.outcome is None
        assert t.feedback is None
        assert t.total_tokens == 0
        assert t.total_latency_seconds == 0.0

    @pytest.mark.spec("REQ-core.types.trace")
    def test_trace_add_step_updates_totals(self) -> None:
        t = Trace(query="hello", agent="simple", model="llama3:8b")
        step = TraceStep(
            step_type=StepType.GENERATE,
            timestamp=time.time(),
            duration_seconds=1.5,
            output={"tokens": 42},
        )
        t.add_step(step)
        assert len(t.steps) == 1
        assert t.total_latency_seconds == 1.5
        assert t.total_tokens == 42

    @pytest.mark.spec("REQ-core.types.trace")
    def test_trace_add_multiple_steps(self) -> None:
        t = Trace()
        s1 = TraceStep(
            step_type=StepType.ROUTE,
            timestamp=time.time(),
            duration_seconds=0.1,
            output={"tokens": 0},
        )
        s2 = TraceStep(
            step_type=StepType.GENERATE,
            timestamp=time.time(),
            duration_seconds=2.0,
            output={"tokens": 100},
        )
        t.add_step(s1)
        t.add_step(s2)
        assert len(t.steps) == 2
        assert t.total_latency_seconds == pytest.approx(2.1)
        assert t.total_tokens == 100

    @pytest.mark.spec("REQ-core.types.trace")
    def test_trace_unique_ids(self) -> None:
        t1 = Trace()
        t2 = Trace()
        assert t1.trace_id != t2.trace_id

    @pytest.mark.spec("REQ-core.types.trace")
    def test_trace_step_types(self) -> None:
        assert StepType.ROUTE == "route"
        assert StepType.RETRIEVE == "retrieve"
        assert StepType.GENERATE == "generate"
        assert StepType.TOOL_CALL == "tool_call"
        assert StepType.RESPOND == "respond"


class TestRoutingContext:
    @pytest.mark.spec("REQ-core.types.routing-context")
    def test_routing_context_defaults(self) -> None:
        ctx = RoutingContext()
        assert ctx.query == ""
        assert ctx.query_length == 0
        assert ctx.has_code is False
        assert ctx.has_math is False
        assert ctx.language == "en"
        assert ctx.urgency == 0.5
        assert ctx.metadata == {}

    @pytest.mark.spec("REQ-core.types.routing-context")
    def test_routing_context_custom_fields(self) -> None:
        ctx = RoutingContext(
            query="Write a function",
            query_length=17,
            has_code=True,
            has_math=False,
            language="en",
            urgency=0.8,
            metadata={"source": "cli"},
        )
        assert ctx.query == "Write a function"
        assert ctx.has_code is True
        assert ctx.urgency == 0.8
        assert ctx.metadata["source"] == "cli"

    @pytest.mark.spec("REQ-core.types.routing-context")
    def test_routing_context_metadata_isolation(self) -> None:
        ctx1 = RoutingContext()
        ctx2 = RoutingContext()
        ctx1.metadata["key"] = "val"
        assert "key" not in ctx2.metadata
