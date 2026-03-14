"""Tests for the typed test fakes themselves.

Ensures the fakes correctly implement the real protocols.
"""

from __future__ import annotations

import pytest

from openjarvis.core.types import Message, Role, ToolCall
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool, ToolExecutor
from tests.fixtures.engines import FakeEngine
from tests.fixtures.stores import InMemoryStore
from tests.fixtures.tools import (
    CalculatorStub,
    ConfirmationTool,
    EchoTool,
    FailingTool,
    SlowTool,
    ThinkStub,
    make_tool_set,
)


class TestFakeEngine:
    """Verify FakeEngine implements InferenceEngine protocol."""

    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_is_inference_engine(self) -> None:
        engine = FakeEngine()
        assert isinstance(engine, InferenceEngine)

    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_generate_returns_expected_format(self) -> None:
        engine = FakeEngine(responses=["Hello!"])
        messages = [Message(role=Role.USER, content="Hi")]
        result = engine.generate(messages, model="test")
        assert "content" in result
        assert "usage" in result
        assert "model" in result
        assert "finish_reason" in result
        assert result["content"] == "Hello!"

    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_generate_cycles_responses(self) -> None:
        engine = FakeEngine(responses=["first", "second"])
        msg = [Message(role=Role.USER, content="Hi")]
        assert engine.generate(msg, model="test")["content"] == "first"
        assert engine.generate(msg, model="test")["content"] == "second"
        assert engine.generate(msg, model="test")["content"] == "first"

    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_generate_with_tool_calls(self) -> None:
        tool_calls = [[{"id": "tc1", "name": "calc", "arguments": '{"x": 1}'}]]
        engine = FakeEngine(responses=[""], tool_calls=tool_calls)
        msg = [Message(role=Role.USER, content="calc")]
        result = engine.generate(msg, model="test")
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1

    @pytest.mark.spec("REQ-engine.protocol.list-models")
    def test_list_models(self) -> None:
        engine = FakeEngine(models=["model-a", "model-b"])
        assert engine.list_models() == ["model-a", "model-b"]

    @pytest.mark.spec("REQ-engine.protocol.health")
    def test_health_true(self) -> None:
        engine = FakeEngine(healthy=True)
        assert engine.health() is True

    @pytest.mark.spec("REQ-engine.protocol.health")
    def test_health_false(self) -> None:
        engine = FakeEngine(healthy=False)
        assert engine.health() is False

    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_call_history(self) -> None:
        engine = FakeEngine()
        msg = [Message(role=Role.USER, content="Hi")]
        engine.generate(msg, model="test-model")
        assert len(engine.call_history) == 1
        assert engine.call_history[0]["model"] == "test-model"
        assert engine.call_count == 1

    @pytest.mark.spec("REQ-engine.protocol.lifecycle")
    def test_reset(self) -> None:
        engine = FakeEngine()
        msg = [Message(role=Role.USER, content="Hi")]
        engine.generate(msg, model="test")
        engine.reset()
        assert engine.call_count == 0
        assert len(engine.call_history) == 0

    @pytest.mark.spec("REQ-engine.protocol.stream")
    @pytest.mark.asyncio
    async def test_stream(self) -> None:
        engine = FakeEngine(responses=["hello world"])
        msg = [Message(role=Role.USER, content="Hi")]
        tokens = []
        async for token in engine.stream(msg, model="test"):
            tokens.append(token)
        assert len(tokens) == 2
        assert "hello" in tokens[0]


class TestToolStubs:
    """Verify tool stubs implement BaseTool protocol."""

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_calculator_is_base_tool(self) -> None:
        tool = CalculatorStub()
        assert isinstance(tool, BaseTool)

    @pytest.mark.spec("REQ-tools.calculator")
    def test_calculator_evaluates(self) -> None:
        tool = CalculatorStub()
        result = tool.execute(expression="2 + 3")
        assert result.success
        assert result.content == "5"

    @pytest.mark.spec("REQ-tools.calculator")
    def test_calculator_error(self) -> None:
        tool = CalculatorStub()
        result = tool.execute(expression="invalid")
        assert not result.success

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_think_stub(self) -> None:
        tool = ThinkStub()
        result = tool.execute(thought="consider options")
        assert result.success
        assert "consider options" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_echo_tool(self) -> None:
        tool = EchoTool()
        result = tool.execute(text="hello world")
        assert result.success
        assert result.content == "hello world"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_failing_tool(self) -> None:
        tool = FailingTool(error_message="custom error")
        result = tool.execute()
        assert not result.success
        assert "custom error" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_confirmation_tool_spec(self) -> None:
        tool = ConfirmationTool()
        assert tool.spec.requires_confirmation is True

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_make_tool_set(self) -> None:
        tools = make_tool_set()
        assert len(tools) == 3
        names = {t.spec.name for t in tools}
        assert "calculator" in names
        assert "think" in names
        assert "echo" in names


class TestToolExecutorWithFakes:
    """Test ToolExecutor with typed fakes instead of mocks."""

    @pytest.mark.spec("REQ-tools.executor.dispatch")
    def test_execute_known_tool(self) -> None:
        tools = [CalculatorStub()]
        executor = ToolExecutor(tools)
        call = ToolCall(id="tc1", name="calculator", arguments='{"expression": "3*4"}')
        result = executor.execute(call)
        assert result.success
        assert result.content == "12"

    @pytest.mark.spec("REQ-tools.executor.dispatch")
    def test_execute_unknown_tool(self) -> None:
        executor = ToolExecutor([])
        call = ToolCall(id="tc1", name="nonexistent", arguments="{}")
        result = executor.execute(call)
        assert not result.success
        assert "Unknown tool" in result.content

    @pytest.mark.spec("REQ-tools.executor.timeout")
    def test_execute_timeout(self) -> None:
        tools = [SlowTool(delay_seconds=10.0)]
        executor = ToolExecutor(tools, default_timeout=0.5)
        call = ToolCall(id="tc1", name="slow_tool", arguments="{}")
        result = executor.execute(call)
        assert not result.success
        assert "timed out" in result.content

    @pytest.mark.spec("REQ-tools.executor.listing")
    def test_available_tools(self) -> None:
        tools = make_tool_set()
        executor = ToolExecutor(tools)
        specs = executor.available_tools()
        assert len(specs) == 3

    @pytest.mark.spec("REQ-tools.executor.listing")
    def test_get_openai_tools(self) -> None:
        tools = [CalculatorStub()]
        executor = ToolExecutor(tools)
        openai_tools = executor.get_openai_tools()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "calculator"


class TestInMemoryStore:
    """Test the in-memory store fake."""

    @pytest.mark.spec("REQ-storage.protocol.store")
    def test_put_and_get(self) -> None:
        store = InMemoryStore()
        store.put("key1", "value1")
        assert store.get("key1") == "value1"

    @pytest.mark.spec("REQ-storage.protocol.retrieve")
    def test_get_missing_key(self) -> None:
        store = InMemoryStore()
        assert store.get("missing") is None

    @pytest.mark.spec("REQ-storage.protocol.delete")
    def test_delete(self) -> None:
        store = InMemoryStore()
        store.put("key1", "value1")
        store.delete("key1")
        assert store.get("key1") is None

    @pytest.mark.spec("REQ-storage.protocol.store")
    def test_count(self) -> None:
        store = InMemoryStore()
        store.put("a", "1")
        store.put("b", "2")
        assert store.count() == 2

    @pytest.mark.spec("REQ-storage.protocol.store")
    def test_keys(self) -> None:
        store = InMemoryStore()
        store.put("b", "2")
        store.put("a", "1")
        assert store.keys() == ["a", "b"]

    @pytest.mark.spec("REQ-storage.protocol.clear")
    def test_close(self) -> None:
        store = InMemoryStore()
        store.put("x", "y")
        store.close()
