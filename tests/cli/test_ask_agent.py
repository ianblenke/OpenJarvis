"""Tests for ``jarvis ask --agent`` CLI integration."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli

_ask_mod = importlib.import_module("openjarvis.cli.ask")


class _FakeEngine:
    """Typed fake engine for ask --agent CLI tests."""

    def __init__(self, content: str = "Hello from engine") -> None:
        self.engine_id = "mock"
        self._content = content

    def health(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["test-model"]

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return {
            "content": self._content,
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "stop",
        }


def _mock_engine(content="Hello from engine"):
    """Create a typed fake engine that returns content."""
    return _FakeEngine(content=content)


def _register_agents():
    """Re-register agents after registry clear."""
    from openjarvis.agents.orchestrator import OrchestratorAgent
    from openjarvis.agents.simple import SimpleAgent
    from openjarvis.core.registry import AgentRegistry

    for name, cls in [
        ("simple", SimpleAgent),
        ("orchestrator", OrchestratorAgent),
    ]:
        if not AgentRegistry.contains(name):
            AgentRegistry.register_value(name, cls)


def _register_tools():
    """Re-register tools after registry clear."""
    from openjarvis.core.registry import ToolRegistry
    from openjarvis.tools.calculator import CalculatorTool
    from openjarvis.tools.file_read import FileReadTool
    from openjarvis.tools.llm_tool import LLMTool
    from openjarvis.tools.retrieval import RetrievalTool
    from openjarvis.tools.think import ThinkTool

    for name, cls in [
        ("calculator", CalculatorTool),
        ("think", ThinkTool),
        ("retrieval", RetrievalTool),
        ("llm", LLMTool),
        ("file_read", FileReadTool),
    ]:
        if not ToolRegistry.contains(name):
            ToolRegistry.register_value(name, cls)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_setup(monkeypatch):
    """Patch engine discovery to avoid needing a running engine."""
    engine = _mock_engine()
    _register_agents()
    _register_tools()

    from openjarvis.core.config import JarvisConfig

    monkeypatch.setattr(_ask_mod, "load_config", lambda: JarvisConfig())
    monkeypatch.setattr(_ask_mod, "get_engine", lambda *a, **kw: ("mock", engine))
    monkeypatch.setattr(_ask_mod, "discover_engines", lambda c: [("mock", engine)])
    monkeypatch.setattr(_ask_mod, "discover_models", lambda e: {"mock": ["test-model"]})
    monkeypatch.setattr(_ask_mod, "register_builtin_models", lambda *a, **kw: None)
    monkeypatch.setattr(_ask_mod, "merge_discovered_models", lambda *a, **kw: None)
    yield engine


class TestAskAgentOption:
    @pytest.mark.spec("REQ-cli.ask")
    def test_help_shows_agent_option(self, runner):
        result = runner.invoke(cli, ["ask", "--help"])
        assert "--agent" in result.output or "-a" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_help_shows_tools_option(self, runner):
        result = runner.invoke(cli, ["ask", "--help"])
        assert "--tools" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_agent_simple(self, runner, mock_setup):
        result = runner.invoke(cli, ["ask", "--agent", "simple", "Hello"])
        assert result.exit_code == 0
        assert "Hello from engine" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_agent_orchestrator_no_tools(self, runner, mock_setup):
        result = runner.invoke(
            cli, ["ask", "--agent", "orchestrator", "Hello"],
        )
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_agent_orchestrator_with_tools(self, runner, mock_setup):
        result = runner.invoke(
            cli,
            [
                "ask", "--agent", "orchestrator",
                "--tools", "calculator,think",
                "What is 2+2?",
            ],
        )
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_agent_json_output(self, runner, mock_setup):
        result = runner.invoke(
            cli, ["ask", "--agent", "simple", "--json", "Hello"],
        )
        assert result.exit_code == 0
        assert '"content"' in result.output
        assert '"turns"' in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_unknown_agent(self, runner, mock_setup):
        result = runner.invoke(
            cli, ["ask", "--agent", "nonexistent", "Hello"],
        )
        assert result.exit_code != 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_no_agent_uses_direct_mode(self, runner, mock_setup):
        result = runner.invoke(cli, ["ask", "Hello"])
        assert result.exit_code == 0
        assert "Hello from engine" in result.output

    @pytest.mark.spec("REQ-cli.ask")
    def test_agent_simple_with_model(self, runner, mock_setup):
        result = runner.invoke(
            cli, ["ask", "--agent", "simple", "-m", "test-model", "Hello"],
        )
        assert result.exit_code == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_agent_simple_with_temperature(self, runner, mock_setup):
        result = runner.invoke(
            cli, ["ask", "--agent", "simple", "-t", "0.1", "Hello"],
        )
        assert result.exit_code == 0


class TestBuildTools:
    @pytest.mark.spec("REQ-cli.ask")
    def test_build_calculator(self, mock_setup):
        from openjarvis.cli.ask import _build_tools
        from openjarvis.core.config import JarvisConfig

        _register_tools()
        config = JarvisConfig()
        tools = _build_tools(["calculator"], config, mock_setup, "test-model")
        assert len(tools) == 1
        assert tools[0].tool_id == "calculator"

    @pytest.mark.spec("REQ-cli.ask")
    def test_build_think(self, mock_setup):
        from openjarvis.cli.ask import _build_tools
        from openjarvis.core.config import JarvisConfig

        _register_tools()
        config = JarvisConfig()
        tools = _build_tools(["think"], config, mock_setup, "test-model")
        assert len(tools) == 1
        assert tools[0].tool_id == "think"

    @pytest.mark.spec("REQ-cli.ask")
    def test_build_unknown_tool_skipped(self, mock_setup):
        from openjarvis.cli.ask import _build_tools
        from openjarvis.core.config import JarvisConfig

        config = JarvisConfig()
        tools = _build_tools(["nonexistent"], config, mock_setup, "test-model")
        assert len(tools) == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_build_empty_names(self, mock_setup):
        from openjarvis.cli.ask import _build_tools
        from openjarvis.core.config import JarvisConfig

        config = JarvisConfig()
        tools = _build_tools(["", " "], config, mock_setup, "test-model")
        assert len(tools) == 0

    @pytest.mark.spec("REQ-cli.ask")
    def test_build_multiple_tools(self, mock_setup):
        from openjarvis.cli.ask import _build_tools
        from openjarvis.core.config import JarvisConfig

        _register_tools()
        config = JarvisConfig()
        tools = _build_tools(["calculator", "think"], config, mock_setup, "test-model")
        assert len(tools) == 2
