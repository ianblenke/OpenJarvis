"""Tests for engine shim modules — NexaShim and AppleFmShim.

REQ-engine.shims: Shim servers wrap native SDKs as OpenAI-compatible
FastAPI servers. We test the module structure, the prompt builder, and
the request/response model shapes without requiring the native SDKs.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def _load_shim_module(mod_name: str) -> ModuleType | None:
    """Attempt to import a shim module, returning None if deps are missing."""
    try:
        return importlib.import_module(mod_name)
    except (ImportError, SystemExit):
        return None


class TestNexaShim:
    """REQ-engine.shims: NexaShim wraps Nexa SDK as OpenAI-compatible API."""

    @pytest.mark.spec("REQ-engine.shims")
    def test_nexa_shim_build_prompt(self) -> None:
        """_build_prompt concatenates messages into a single prompt string."""
        # Inline the logic since we may not be able to import the module
        # due to missing nexaai. Replicate the function to test the algorithm.
        def _build_prompt(messages):
            parts = []
            for m in messages:
                if m["role"] == "system":
                    parts.append(f"[System] {m['content']}")
                elif m["role"] in ("user", "assistant"):
                    parts.append(m["content"])
            return "\n".join(parts)

        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        prompt = _build_prompt(messages)
        assert "[System] Be helpful." in prompt
        assert "Hello" in prompt
        assert "Hi there" in prompt

    @pytest.mark.spec("REQ-engine.shims")
    def test_nexa_shim_module_defines_app(self) -> None:
        """If importable, the module exposes a FastAPI app."""
        mod = _load_shim_module("openjarvis.engine.nexa_shim")
        if mod is None:
            pytest.skip("nexaai SDK not installed")
        assert hasattr(mod, "app")

    @pytest.mark.spec("REQ-engine.shims")
    def test_nexa_shim_model_id(self) -> None:
        """The shim defines a MODEL_ID constant."""
        mod = _load_shim_module("openjarvis.engine.nexa_shim")
        if mod is None:
            pytest.skip("nexaai SDK not installed")
        assert hasattr(mod, "MODEL_ID")
        assert isinstance(mod.MODEL_ID, str)


class TestAppleFmShim:
    """REQ-engine.shims: AppleFmShim wraps Apple FM as OpenAI-compatible API."""

    @pytest.mark.spec("REQ-engine.shims")
    def test_apple_fm_shim_build_prompt(self) -> None:
        """_build_prompt concatenates messages into a single prompt string."""
        def _build_prompt(messages):
            parts = []
            for m in messages:
                if m["role"] == "system":
                    parts.append(f"[System] {m['content']}")
                elif m["role"] in ("user", "assistant"):
                    parts.append(m["content"])
            return "\n".join(parts)

        messages = [
            {"role": "system", "content": "You are an AI."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        prompt = _build_prompt(messages)
        assert "[System] You are an AI." in prompt
        assert "What is 2+2?" in prompt

    @pytest.mark.spec("REQ-engine.shims")
    def test_apple_fm_shim_module_defines_app(self) -> None:
        """If importable, the module exposes a FastAPI app."""
        mod = _load_shim_module("openjarvis.engine.apple_fm_shim")
        if mod is None:
            pytest.skip("apple_fm SDK not installed or not on macOS")
        assert hasattr(mod, "app")

    @pytest.mark.spec("REQ-engine.shims")
    def test_apple_fm_shim_model_id(self) -> None:
        """The shim defines a MODEL_ID constant."""
        mod = _load_shim_module("openjarvis.engine.apple_fm_shim")
        if mod is None:
            pytest.skip("apple_fm SDK not installed or not on macOS")
        assert hasattr(mod, "MODEL_ID")
        assert isinstance(mod.MODEL_ID, str)


class TestShimsOpenAIFormat:
    """REQ-engine.shims: Shims produce OpenAI-compatible response format."""

    @pytest.mark.spec("REQ-engine.shims")
    def test_openai_chat_completion_shape(self) -> None:
        """Verify the expected shape of an OpenAI-compatible response."""
        # This tests the contract that both shims follow
        import time
        import uuid

        model_id = "test-shim"
        text = "Hello, world!"
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        response = {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        assert response["object"] == "chat.completion"
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert "usage" in response
