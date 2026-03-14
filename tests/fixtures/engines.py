"""Typed fake engine for anti-mocking tests.

Replaces MagicMock-based mock_engine patterns with a real class that
implements the InferenceEngine protocol. Catches interface drift and
exercises real code paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Sequence

from openjarvis.core.types import Message
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.evals.core.backend import InferenceBackend


class FakeEngine(InferenceEngine):
    """In-process fake engine implementing the real InferenceEngine protocol.

    Use this instead of MagicMock to test agents, tools, and other components
    that depend on an InferenceEngine.

    Parameters
    ----------
    engine_id:
        Engine identifier (default: "fake").
    responses:
        List of response strings to return in order. Cycles if exhausted.
    tool_calls:
        Optional list of tool call dicts to include in responses.
        Each entry is a list of tool calls for that response index.
    models:
        List of model names to report as available.
    healthy:
        Whether health() returns True.
    """

    def __init__(
        self,
        engine_id: str = "fake",
        *,
        responses: Optional[List[str]] = None,
        tool_calls: Optional[List[List[Dict[str, Any]]]] = None,
        models: Optional[List[str]] = None,
        healthy: bool = True,
    ) -> None:
        self.engine_id = engine_id
        self._responses = responses or ["Hello from FakeEngine!"]
        self._tool_calls = tool_calls or []
        self._models = models or ["fake-model"]
        self._healthy = healthy
        self._call_count = 0
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        idx = self._call_count % len(self._responses)
        content = self._responses[idx]
        self._call_count += 1

        self.call_history.append({
            "messages": list(messages),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })

        result: Dict[str, Any] = {
            "content": content,
            "usage": {
                "prompt_tokens": sum(len(m.content) // 4 for m in messages),
                "completion_tokens": len(content) // 4,
                "total_tokens": sum(len(m.content) // 4 for m in messages)
                + len(content) // 4,
            },
            "model": model,
            "finish_reason": "stop",
        }

        if idx < len(self._tool_calls) and self._tool_calls[idx]:
            result["tool_calls"] = self._tool_calls[idx]
            result["finish_reason"] = "tool_calls"

        return result

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        idx = self._call_count % len(self._responses)
        content = self._responses[idx]
        self._call_count += 1

        self.call_history.append({
            "messages": list(messages),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
            "streaming": True,
        })

        for word in content.split():
            yield word + " "

    def list_models(self) -> List[str]:
        return list(self._models)

    def health(self) -> bool:
        return self._healthy

    def close(self) -> None:
        self.call_history.clear()

    def prepare(self, model: str) -> None:
        pass

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        """Reset call count and history for reuse across tests."""
        self._call_count = 0
        self.call_history.clear()


class FakeInferenceBackend(InferenceBackend):
    """Typed fake implementing the InferenceBackend protocol (evals backend).

    Unlike FakeEngine (which returns dicts), this returns plain strings
    from generate() -- matching the InferenceBackend ABC used by
    TraceJudge and LLMOptimizer.
    """

    backend_id = "fake"

    def __init__(
        self,
        *,
        responses: Optional[List[str]] = None,
    ) -> None:
        self._responses = responses or ["0.8\nGood response."]
        self._call_count = 0
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        idx = self._call_count % len(self._responses)
        content = self._responses[idx]
        self._call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "model": model,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return content

    def generate_full(
        self,
        prompt: str,
        *,
        model: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        content = self.generate(
            prompt, model=model, system=system,
            temperature=temperature, max_tokens=max_tokens,
        )
        return {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": model,
            "latency_seconds": 0.1,
            "cost_usd": 0.001,
        }

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        self._call_count = 0
        self.call_history.clear()
