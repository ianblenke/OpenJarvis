"""Tests for engine discovery."""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._base import InferenceEngine
from openjarvis.engine._discovery import (
    discover_engines,
    discover_models,
    get_engine,
)


class _FakeEngine(InferenceEngine):
    engine_id = "fake"

    def __init__(
        self,
        *,
        healthy: bool = True,
        models: list | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        self._healthy = healthy
        self._models = models or []

    def generate(self, messages, *, model, **kwargs):  # noqa: ANN001, ANN003
        return {"content": "ok", "usage": {}}

    async def stream(self, messages, *, model, **kwargs):  # noqa: ANN001, ANN003
        yield "ok"

    def list_models(self) -> list:
        return self._models

    def health(self) -> bool:
        return self._healthy


def _reg(key: str, eid: str) -> None:
    """Register a fake engine type under *key*."""
    cls = type(key.title(), (_FakeEngine,), {"engine_id": eid})
    EngineRegistry.register_value(key, cls)


class TestDiscoverEngines:
    @pytest.mark.spec("REQ-engine.discovery.probe")
    def test_only_healthy_returned(self, monkeypatch) -> None:
        _reg("healthy", "healthy")
        _reg("sick", "sick")

        cfg = JarvisConfig()
        import openjarvis.engine._discovery as _disc_mod
        monkeypatch.setattr(
            _disc_mod, "_make_engine",
            lambda k, c: _FakeEngine(healthy=(k == "healthy")),
        )
        result = discover_engines(cfg)
        assert len(result) == 1
        assert result[0][0] == "healthy"

    @pytest.mark.spec("REQ-engine.discovery.probe")
    def test_default_engine_first(self, monkeypatch) -> None:
        _reg("a", "a")
        _reg("b", "b")

        cfg = JarvisConfig()
        cfg.engine.default = "b"
        import openjarvis.engine._discovery as _disc_mod
        monkeypatch.setattr(
            _disc_mod, "_make_engine",
            lambda k, c: _FakeEngine(healthy=True),
        )
        result = discover_engines(cfg)
        assert result[0][0] == "b"


class TestDiscoverModels:
    @pytest.mark.spec("REQ-engine.discovery.models")
    def test_aggregate_models(self) -> None:
        e1 = _FakeEngine(models=["m1", "m2"])
        e2 = _FakeEngine(models=["m3"])
        result = discover_models([("ollama", e1), ("vllm", e2)])
        assert result == {"ollama": ["m1", "m2"], "vllm": ["m3"]}


class TestGetEngine:
    @pytest.mark.spec("REQ-engine.discovery.get-engine")
    def test_fallback_when_default_unhealthy(self, monkeypatch) -> None:
        _reg("bad", "bad")
        _reg("good", "good")

        cfg = JarvisConfig()
        cfg.engine.default = "bad"

        def _make(k, c):  # noqa: ANN001
            return _FakeEngine(healthy=(k == "good"))

        import openjarvis.engine._discovery as _disc_mod
        monkeypatch.setattr(_disc_mod, "_make_engine", _make)
        result = get_engine(cfg)
        assert result is not None
        assert result[0] == "good"
