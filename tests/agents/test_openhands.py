"""Tests for OpenHandsAgent (real openhands-sdk wrapper)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from openjarvis.agents._stubs import BaseAgent
from openjarvis.agents.openhands import OpenHandsAgent
from openjarvis.core.registry import AgentRegistry

# ---------------------------------------------------------------------------
# Typed fake (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Typed fake engine for OpenHands agent tests."""

    def __init__(self) -> None:
        self.engine_id = "mock"

    def health(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["test-model"]

    def generate(self, messages, **kwargs) -> Dict[str, Any]:
        return {
            "content": "ok",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
            "finish_reason": "stop",
        }


class TestOpenHandsAgentRegistration:
    @pytest.mark.spec("REQ-agents.openhands")
    def test_registered(self):
        AgentRegistry.register_value("openhands", OpenHandsAgent)
        assert AgentRegistry.contains("openhands")

    def test_agent_id(self):
        engine = _FakeEngine()
        agent = OpenHandsAgent(engine, "test-model")
        assert agent.agent_id == "openhands"

    def test_does_not_accept_tools(self):
        """Real OpenHandsAgent doesn't use ToolUsingAgent base."""
        assert OpenHandsAgent.accepts_tools is False

    def test_is_base_agent(self):
        assert issubclass(OpenHandsAgent, BaseAgent)


class TestOpenHandsAgentImportError:
    def test_run_without_sdk_raises(self):
        """Running without openhands-sdk installed raises ImportError."""
        engine = _FakeEngine()
        agent = OpenHandsAgent(engine, "test-model")
        with pytest.raises(ImportError, match="openhands-sdk"):
            agent.run("Hello")


class TestOpenHandsAgentConstructor:
    def test_default_workspace(self):
        engine = _FakeEngine()
        agent = OpenHandsAgent(engine, "test-model")
        assert agent._workspace  # should be cwd

    def test_custom_workspace(self):
        engine = _FakeEngine()
        agent = OpenHandsAgent(engine, "test-model", workspace="/tmp/test")
        assert agent._workspace == "/tmp/test"

    def test_custom_api_key(self):
        engine = _FakeEngine()
        agent = OpenHandsAgent(engine, "test-model", api_key="sk-test")
        assert agent._api_key == "sk-test"
