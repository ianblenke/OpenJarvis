"""Tests for the LiteLLM engine backend."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine.litellm import LiteLLMEngine

# ---------------------------------------------------------------------------
# Typed fake for the litellm module
# ---------------------------------------------------------------------------


class _CallTracker:
    """Tracks calls and returns preconfigured values."""

    def __init__(self, return_value=None, side_effect=None):
        self._return_value = return_value
        self._side_effect = side_effect
        self._call_count = 0
        self.call_args = None

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        self.call_args = (args, kwargs)
        if self._side_effect is not None:
            if callable(self._side_effect):
                return self._side_effect(*args, **kwargs)
            raise self._side_effect
        return self._return_value


def _make_fake_litellm(
    completion_return=None,
    cost_return=0.001,
    cost_error=None,
):
    """Build a typed fake litellm module with configurable completion/cost."""
    mod = types.ModuleType("litellm")

    if cost_error is not None:
        mod.completion_cost = _CallTracker(side_effect=cost_error)
    else:
        mod.completion_cost = _CallTracker(return_value=cost_return)

    mod.completion = _CallTracker(return_value=completion_return)
    return mod


def _make_standard_response(
    content="Hello!",
    tool_calls=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
    total_tokens=15,
    model="gpt-4o",
    cost=0.001,
):
    """Build a standard fake LiteLLM response using SimpleNamespace."""
    fake_usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    fake_choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        choices=[fake_choice], usage=fake_usage, model=model,
    )


class TestLiteLLMEngineHealth:
    @pytest.mark.spec("REQ-engine.protocol.health")
    def test_health_importable(self, monkeypatch) -> None:
        fake_litellm = _make_fake_litellm()
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        engine = LiteLLMEngine()
        assert engine.health() is True

    @pytest.mark.spec("REQ-engine.protocol.health")
    def test_health_not_importable(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "litellm", None)
        engine = LiteLLMEngine()
        assert engine.health() is False


class TestLiteLLMEngineGenerate:
    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_generate(self, monkeypatch) -> None:
        fake_resp = _make_standard_response()
        fake_litellm = _make_fake_litellm(
            completion_return=fake_resp, cost_return=0.001,
        )

        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        engine = LiteLLMEngine()
        result = engine.generate(
            [Message(role=Role.USER, content="Hi")], model="gpt-4o"
        )

        assert result["content"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert result["model"] == "gpt-4o"
        assert result["finish_reason"] == "stop"
        assert result["cost_usd"] == 0.001

    @pytest.mark.spec("REQ-engine.litellm")
    def test_generate_with_tools(self, monkeypatch) -> None:
        fake_tool_call = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="calculator",
                arguments='{"expression": "2+2"}',
            ),
        )
        fake_resp = _make_standard_response(
            content="",
            tool_calls=[fake_tool_call],
            finish_reason="tool_calls",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            model="gpt-4o",
        )
        fake_litellm = _make_fake_litellm(
            completion_return=fake_resp, cost_return=0.002,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate math",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        engine = LiteLLMEngine()
        result = engine.generate(
            [Message(role=Role.USER, content="What is 2+2?")],
            model="gpt-4o",
            tools=tools,
        )

        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["name"] == "calculator"
        assert tc["arguments"] == '{"expression": "2+2"}'

    @pytest.mark.spec("REQ-engine.litellm")
    def test_generate_with_api_base(self, monkeypatch) -> None:
        fake_resp = _make_standard_response(
            content="Hi!",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            model="custom-model",
        )
        fake_litellm = _make_fake_litellm(
            completion_return=fake_resp, cost_return=0.0,
        )

        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        engine = LiteLLMEngine(api_base="http://localhost:8080")
        engine.generate(
            [Message(role=Role.USER, content="Hi")], model="custom-model"
        )

        call_kwargs = fake_litellm.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:8080"

    @pytest.mark.spec("REQ-engine.litellm")
    def test_generate_cost_error_fallback(self, monkeypatch) -> None:
        fake_resp = _make_standard_response(model="unknown/model")
        fake_litellm = _make_fake_litellm(
            completion_return=fake_resp,
            cost_error=Exception("Unknown model"),
        )

        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        engine = LiteLLMEngine()
        result = engine.generate(
            [Message(role=Role.USER, content="Hi")], model="unknown/model"
        )

        assert result["cost_usd"] == 0.0


class TestLiteLLMEngineStream:
    @pytest.mark.spec("REQ-engine.protocol.stream")
    def test_stream(self, monkeypatch) -> None:
        chunk1 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]
        )
        chunk2 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo!"))]
        )
        chunk3 = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
        )

        fake_litellm = _make_fake_litellm(
            completion_return=iter([chunk1, chunk2, chunk3]),
        )

        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
        engine = LiteLLMEngine()
        import asyncio

        async def collect() -> list[str]:
            tokens: list[str] = []
            async for token in engine.stream(
                [Message(role=Role.USER, content="Hi")], model="gpt-4o"
            ):
                tokens.append(token)
            return tokens

        tokens = asyncio.run(collect())

        assert tokens == ["Hel", "lo!"]


class TestLiteLLMEngineListModels:
    @pytest.mark.spec("REQ-engine.protocol.list-models")
    def test_list_models_default(self) -> None:
        engine = LiteLLMEngine()
        assert engine.list_models() == []

    @pytest.mark.spec("REQ-engine.protocol.list-models")
    def test_list_models_with_default_model(self) -> None:
        engine = LiteLLMEngine(default_model="anthropic/claude-sonnet-4-20250514")
        assert engine.list_models() == ["anthropic/claude-sonnet-4-20250514"]


class TestLiteLLMEngineRegistry:
    @pytest.mark.spec("REQ-engine.registration")
    def test_registry_key(self) -> None:
        EngineRegistry.register_value("litellm", LiteLLMEngine)
        assert EngineRegistry.contains("litellm")
        cls = EngineRegistry.get("litellm")
        assert cls is LiteLLMEngine
