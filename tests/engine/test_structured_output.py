"""Tests for structured output / JSON mode across engines."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any, List

import httpx
import pytest
import respx

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._stubs import ResponseFormat
from openjarvis.engine.cloud import CloudEngine
from openjarvis.engine.ollama import OllamaEngine

# ---------------------------------------------------------------------------
# ResponseFormat dataclass
# ---------------------------------------------------------------------------


class TestResponseFormat:
    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_default_type(self) -> None:
        rf = ResponseFormat()
        assert rf.type == "json_object"
        assert rf.schema is None

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_schema_type(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", schema=schema)
        assert rf.type == "json_schema"
        assert rf.schema == schema

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_slots(self) -> None:
        rf = ResponseFormat()
        with pytest.raises(AttributeError):
            rf.extra = "nope"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Typed fakes for SDK client boundaries
# ---------------------------------------------------------------------------


class FakeOpenAICompletions:
    """Typed fake for openai chat.completions namespace."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.create_calls: List[dict] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.create_calls.append(kwargs)
        return self._response


class FakeOpenAIChat:
    """Typed fake for openai client.chat namespace."""

    def __init__(self, completions: FakeOpenAICompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """Typed fake for openai.OpenAI() client with chat completions."""

    def __init__(self, response: SimpleNamespace) -> None:
        completions = FakeOpenAICompletions(response)
        self.chat = FakeOpenAIChat(completions)

    @property
    def _completions(self) -> FakeOpenAICompletions:
        return self.chat.completions


class FakeAnthropicMessages:
    """Typed fake for anthropic client.messages namespace."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.create_calls: List[dict] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.create_calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    """Typed fake for anthropic client."""

    def __init__(self, response: SimpleNamespace) -> None:
        self.messages = FakeAnthropicMessages(response)


class FakeGoogleModels:
    """Typed fake for google.genai client.models namespace."""

    def __init__(self, response: SimpleNamespace) -> None:
        self._response = response
        self.generate_content_calls: List[dict] = []

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.generate_content_calls.append(kwargs)
        return self._response


class FakeGoogleClient:
    """Typed fake for google.genai client."""

    def __init__(self, response: SimpleNamespace) -> None:
        self.models = FakeGoogleModels(response)


# ---------------------------------------------------------------------------
# Cloud engine -- OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIStructuredOutput:
    def _make_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[CloudEngine, FakeOpenAIClient]:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()

        fake_usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        fake_choice = SimpleNamespace(
            message=SimpleNamespace(content='{"answer": 42}', tool_calls=None),
            finish_reason="stop",
        )
        fake_resp = SimpleNamespace(
            choices=[fake_choice], usage=fake_usage, model="gpt-4o"
        )
        fake_client = FakeOpenAIClient(fake_resp)
        engine._openai_client = fake_client
        return engine, fake_client

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_object_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        rf = ResponseFormat()
        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="gpt-4o",
            response_format=rf,
        )
        call_kwargs = fake_client._completions.create_calls[0]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_schema_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", schema=schema)
        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="gpt-4o",
            response_format=rf,
        )
        call_kwargs = fake_client._completions.create_calls[0]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "response"
        assert call_kwargs["response_format"]["json_schema"]["schema"] == schema

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_raw_dict_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        raw = {"type": "json_object"}
        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="gpt-4o",
            response_format=raw,
        )
        call_kwargs = fake_client._completions.create_calls[0]
        assert call_kwargs["response_format"] == raw

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_no_response_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        engine.generate(
            [Message(role=Role.USER, content="Hi")],
            model="gpt-4o",
        )
        call_kwargs = fake_client._completions.create_calls[0]
        assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# Cloud engine -- Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicStructuredOutput:
    def _make_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[CloudEngine, FakeAnthropicClient]:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()

        fake_usage = SimpleNamespace(input_tokens=12, output_tokens=8)
        fake_tool_use = SimpleNamespace(
            type="tool_use",
            id="tool_123",
            name="json_output",
            input={"answer": 42},
        )
        fake_resp = SimpleNamespace(
            content=[fake_tool_use],
            usage=fake_usage,
            model="claude-sonnet-4-20250514",
            stop_reason="tool_use",
        )
        fake_client = FakeAnthropicClient(fake_resp)
        engine._anthropic_client = fake_client
        return engine, fake_client

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_mode_uses_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        rf = ResponseFormat()
        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="claude-sonnet-4-20250514",
            response_format=rf,
        )
        call_kwargs = fake_client.messages.create_calls[0]
        # Should have a tools list with the json_output tool
        assert "tools" in call_kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "json_output" in tool_names
        # Should force the tool via tool_choice
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "json_output",
        }

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_schema_uses_custom_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", schema=schema)
        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="claude-sonnet-4-20250514",
            response_format=rf,
        )
        call_kwargs = fake_client.messages.create_calls[0]
        json_tool = [t for t in call_kwargs["tools"] if t["name"] == "json_output"][0]
        assert json_tool["input_schema"] == schema

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_appends_to_existing_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        rf = ResponseFormat()
        existing_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object"},
                },
            }
        ]
        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="claude-sonnet-4-20250514",
            response_format=rf,
            tools=existing_tools,
        )
        call_kwargs = fake_client.messages.create_calls[0]
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search" in tool_names
        assert "json_output" in tool_names


# ---------------------------------------------------------------------------
# Cloud engine -- Google
# ---------------------------------------------------------------------------


class TestGoogleStructuredOutput:
    def _make_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[CloudEngine, FakeGoogleClient]:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        EngineRegistry.register_value("cloud", CloudEngine)
        engine = CloudEngine()

        fake_part = SimpleNamespace(
            text='{"answer": 42}',
            function_call=None,
        )
        fake_candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[fake_part])
        )
        fake_um = SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5
        )
        fake_resp = SimpleNamespace(
            candidates=[fake_candidate],
            usage_metadata=fake_um,
            text='{"answer": 42}',
        )
        fake_client = FakeGoogleClient(fake_resp)
        engine._google_client = fake_client
        return engine, fake_client

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_mode_sets_mime_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        rf = ResponseFormat()

        # Fake the google.genai types module at the module boundary
        # (third-party SDK may not be installed)
        class _FakeGenAIConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        fake_genai_types = SimpleNamespace(
            GenerateContentConfig=_FakeGenAIConfig,
        )

        google_genai_mod = SimpleNamespace(types=fake_genai_types)
        google_mod = SimpleNamespace(genai=google_genai_mod)
        monkeypatch.setitem(sys.modules, "google", google_mod)
        monkeypatch.setitem(sys.modules, "google.genai", google_genai_mod)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai_types)

        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="gemini-2.5-pro",
            response_format=rf,
        )

        call_kwargs = fake_client.models.generate_content_calls[0]
        config_arg = call_kwargs["config"]
        assert config_arg.response_mime_type == "application/json"

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_schema_sets_response_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, fake_client = self._make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", schema=schema)

        # Fake the google.genai types module at the module boundary
        class _FakeGenAIConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        fake_genai_types = SimpleNamespace(
            GenerateContentConfig=_FakeGenAIConfig,
        )

        google_genai_mod = SimpleNamespace(types=fake_genai_types)
        google_mod = SimpleNamespace(genai=google_genai_mod)
        monkeypatch.setitem(sys.modules, "google", google_mod)
        monkeypatch.setitem(sys.modules, "google.genai", google_genai_mod)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_genai_types)

        engine.generate(
            [Message(role=Role.USER, content="Give me JSON")],
            model="gemini-2.5-pro",
            response_format=rf,
        )

        call_kwargs = fake_client.models.generate_content_calls[0]
        config_arg = call_kwargs["config"]
        assert config_arg.response_mime_type == "application/json"
        assert config_arg.response_schema == schema


# ---------------------------------------------------------------------------
# Ollama engine
# ---------------------------------------------------------------------------


class TestOllamaStructuredOutput:
    @pytest.fixture()
    def engine(self) -> OllamaEngine:
        EngineRegistry.register_value("ollama", OllamaEngine)
        return OllamaEngine(host="http://testhost:11434")

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_json_format_in_payload(self, engine: OllamaEngine) -> None:
        rf = ResponseFormat()
        with respx.mock:
            route = respx.post("http://testhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": 42}',
                        },
                        "model": "qwen3:8b",
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                )
            )
            result = engine.generate(
                [Message(role=Role.USER, content="Give me JSON")],
                model="qwen3:8b",
                response_format=rf,
            )
            # Verify the payload sent to Ollama
            sent_payload = json.loads(route.calls[0].request.content)
            assert sent_payload["format"] == "json"

        assert result["content"] == '{"answer": 42}'

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_raw_dict_format_in_payload(self, engine: OllamaEngine) -> None:
        with respx.mock:
            route = respx.post("http://testhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": 42}',
                        },
                        "model": "qwen3:8b",
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                )
            )
            engine.generate(
                [Message(role=Role.USER, content="Give me JSON")],
                model="qwen3:8b",
                response_format={"type": "json_object"},
            )
            sent_payload = json.loads(route.calls[0].request.content)
            assert sent_payload["format"] == "json"

    @pytest.mark.spec("REQ-engine.protocol.response-format")
    def test_no_format_without_response_format(
        self, engine: OllamaEngine
    ) -> None:
        with respx.mock:
            route = respx.post("http://testhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                        },
                        "model": "qwen3:8b",
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                )
            )
            engine.generate(
                [Message(role=Role.USER, content="Hi")],
                model="qwen3:8b",
            )
            sent_payload = json.loads(route.calls[0].request.content)
            assert "format" not in sent_payload
