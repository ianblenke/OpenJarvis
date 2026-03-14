"""Tests for model resolution fallback chain in jarvis ask."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig

_ask_mod = importlib.import_module("openjarvis.cli.ask")


# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Typed fake engine for ask router tests."""

    def __init__(self) -> None:
        self.engine_id = "mock"

    def health(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["test-model"]

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return {
            "content": "Hello!",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "stop",
        }


def _patch_engine(monkeypatch, engine):
    """Patch engine discovery to use our typed fake via monkeypatch."""
    monkeypatch.setattr(_ask_mod, "get_engine", lambda *a, **kw: ("mock", engine))
    monkeypatch.setattr(_ask_mod, "discover_engines", lambda *a, **kw: {"mock": engine})
    monkeypatch.setattr(
        _ask_mod, "discover_models",
        lambda *a, **kw: {"mock": ["test-model"]},
    )
    monkeypatch.setattr(_ask_mod, "register_builtin_models", lambda *a, **kw: None)
    monkeypatch.setattr(_ask_mod, "merge_discovered_models", lambda *a, **kw: None)


class TestAskModelResolution:
    @pytest.mark.spec("REQ-cli.ask")
    def test_default_model_from_config(self, monkeypatch) -> None:
        """When no -m flag, uses config.intelligence.default_model."""
        engine = _FakeEngine()
        _patch_engine(monkeypatch, engine)
        result = CliRunner().invoke(cli, ["ask", "Hello"])
        assert result.exit_code == 0
        assert "Hello!" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_explicit_model_flag(self, monkeypatch) -> None:
        """The -m flag directly selects a model, bypassing fallback chain."""
        engine = _FakeEngine()
        _patch_engine(monkeypatch, engine)
        result = CliRunner().invoke(
            cli, ["ask", "-m", "test-model", "Hello"],
        )
        assert result.exit_code == 0
        assert "Hello!" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_fallback_to_engine_models(self, monkeypatch) -> None:
        """When default_model is empty, falls back to first engine model."""
        engine = _FakeEngine()
        _patch_engine(monkeypatch, engine)
        cfg = JarvisConfig()
        cfg.telemetry.enabled = False
        cfg.intelligence.default_model = ""
        cfg.intelligence.fallback_model = ""
        cfg.agent.context_from_memory = False
        monkeypatch.setattr(_ask_mod, "load_config", lambda: cfg)
        result = CliRunner().invoke(cli, ["ask", "Hello"])
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_fallback_to_fallback_model(self, monkeypatch) -> None:
        """When default_model is empty and no engine models, uses fallback_model."""
        engine = _FakeEngine()
        _patch_engine(monkeypatch, engine)
        # Override discover_models to return empty list
        monkeypatch.setattr(_ask_mod, "discover_models", lambda *a, **kw: {"mock": []})
        cfg = JarvisConfig()
        cfg.telemetry.enabled = False
        cfg.intelligence.default_model = ""
        cfg.intelligence.fallback_model = "fallback-model"
        cfg.agent.context_from_memory = False
        monkeypatch.setattr(_ask_mod, "load_config", lambda: cfg)
        result = CliRunner().invoke(cli, ["ask", "Hello"])
        assert result.exit_code == 0
