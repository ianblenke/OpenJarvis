"""End-to-end tests for ``jarvis ask``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig

# Import the actual module (not the Click command attribute)
_ask_mod = importlib.import_module("openjarvis.cli.ask")


def _default_response():
    """Return a default engine response dict."""
    return {
        "content": "The answer is 4.",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "model": "test-model",
        "finish_reason": "stop",
    }


class _FakeEngine:
    """Typed fake engine for ask CLI tests."""

    def __init__(self, response: Dict[str, Any] | None = None) -> None:
        self.engine_id = "mock"
        self._response = response or _default_response()

    def health(self) -> bool:
        return True

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return self._response

    def list_models(self) -> List[str]:
        return ["test-model"]


def _patch_ask(
    monkeypatch, tmp_path, *, engine_result=None, no_engine=False
):
    """Set up common patches for ask tests using typed fakes."""
    cfg = JarvisConfig()
    cfg.telemetry.db_path = str(tmp_path / "telemetry.db")

    monkeypatch.setattr(_ask_mod, "load_config", lambda: cfg)

    if no_engine:
        monkeypatch.setattr(
            _ask_mod, "get_engine", lambda *a, **kw: None
        )
    else:
        fake_engine = _FakeEngine(response=engine_result)
        monkeypatch.setattr(
            _ask_mod, "get_engine",
            lambda *a, **kw: ("mock", fake_engine),
        )
        monkeypatch.setattr(
            _ask_mod, "discover_engines",
            lambda c: [("mock", fake_engine)],
        )
        monkeypatch.setattr(
            _ask_mod, "discover_models",
            lambda e: {"mock": ["test-model"]},
        )


class TestAskCommand:
    @pytest.mark.spec("REQ-cli.ask")
    def test_basic_response(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        _patch_ask(monkeypatch, tmp_path)
        result = CliRunner().invoke(
            cli, ["ask", "What is 2+2?"]
        )
        assert result.exit_code == 0
        assert "The answer is 4" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_no_engine_error(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        _patch_ask(monkeypatch, tmp_path, no_engine=True)
        result = CliRunner().invoke(
            cli, ["ask", "Hello"]
        )
        assert result.exit_code != 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_model_override(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        _patch_ask(monkeypatch, tmp_path)
        result = CliRunner().invoke(
            cli, ["ask", "-m", "custom-model", "Hello"]
        )
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_json_output(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        _patch_ask(monkeypatch, tmp_path)
        result = CliRunner().invoke(
            cli, ["ask", "--json", "Hello"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "content" in data

    @pytest.mark.spec("REQ-cli.ask")
    def test_telemetry_recorded(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        _patch_ask(monkeypatch, tmp_path)
        CliRunner().invoke(cli, ["ask", "Hello"])
        db_path = tmp_path / "telemetry.db"
        assert db_path.exists()
