"""Tests for the Cloud engine backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine.cloud import CloudEngine, estimate_cost

# ---------------------------------------------------------------------------
# Typed fake API clients (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeOpenAIClient:
    """Typed fake for openai.OpenAI() client."""

    def __init__(self, response: SimpleNamespace) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self._response = response

    def _create(self, **kwargs) -> SimpleNamespace:
        return self._response


class _FakeAnthropicClient:
    """Typed fake for anthropic.Anthropic() client."""

    def __init__(self, response: SimpleNamespace) -> None:
        self.messages = SimpleNamespace(create=self._create)
        self._response = response

    def _create(self, **kwargs) -> SimpleNamespace:
        return self._response


class TestEstimateCost:
    @pytest.mark.spec("REQ-engine.cloud")
    def test_known_model(self) -> None:
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50)  # 2.50 + 10.00

    @pytest.mark.spec("REQ-engine.cloud")
    def test_unknown_model(self) -> None:
        assert estimate_cost("unknown-model", 100, 100) == 0.0

    @pytest.mark.spec("REQ-engine.cloud")
    def test_prefix_match(self) -> None:
        cost = estimate_cost("gpt-4o-2024-01-01", 1_000_000, 0)
        assert cost == pytest.approx(2.50)


class TestCloudEngineHealth:
    @pytest.mark.spec("REQ-engine.protocol.health")
    def test_health_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()
        assert engine.health() is False

    @pytest.mark.spec("REQ-engine.protocol.health")
    def test_health_with_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Inject a fake openai module into sys.modules so CloudEngine.__init__
        # can construct a client without the real openai package
        fake_openai = SimpleNamespace(
            OpenAI=lambda api_key=None: SimpleNamespace(),
        )
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()
        assert engine.health() is True


class TestCloudEngineListModels:
    @pytest.mark.spec("REQ-engine.protocol.list-models")
    def test_list_models_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()
        assert engine.list_models() == []


class TestCloudEngineGenerate:
    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_generate_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        fake_usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        fake_choice = SimpleNamespace(
            message=SimpleNamespace(content="Hello!", tool_calls=None),
            finish_reason="stop",
        )
        fake_resp = SimpleNamespace(
            choices=[fake_choice], usage=fake_usage, model="gpt-4o"
        )

        fake_client = _FakeOpenAIClient(fake_resp)

        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()
        engine._openai_client = fake_client

        result = engine.generate(
            [Message(role=Role.USER, content="Hi")], model="gpt-4o"
        )
        assert result["content"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 10

    @pytest.mark.spec("REQ-engine.protocol.generate")
    def test_generate_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        fake_usage = SimpleNamespace(input_tokens=12, output_tokens=8)
        fake_content = SimpleNamespace(text="Greetings!")
        fake_resp = SimpleNamespace(
            content=[fake_content],
            usage=fake_usage,
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
        )

        fake_client = _FakeAnthropicClient(fake_resp)

        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()
        engine._anthropic_client = fake_client

        result = engine.generate(
            [Message(role=Role.USER, content="Hi")],
            model="claude-sonnet-4-20250514",
        )
        assert result["content"] == "Greetings!"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 8
