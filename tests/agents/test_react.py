"""Backward-compat tests: ensure old import paths still work.

The canonical tests are in test_native_react.py.  This file verifies
that ``from openjarvis.agents.react import ReActAgent`` still works
and produces a working agent.
"""

from __future__ import annotations

from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.agents.react import ReActAgent
from tests.fixtures.engines import FakeEngine


class TestReActShim:
    def test_is_native_react(self):
        """ReActAgent imported from old path is NativeReActAgent."""
        assert ReActAgent is NativeReActAgent

    def test_can_instantiate(self):
        engine = FakeEngine(engine_id="mock")
        agent = ReActAgent(engine, "test-model")
        assert agent.agent_id == "native_react"

    def test_can_run(self):
        engine = FakeEngine(
            engine_id="mock",
            responses=["Thought: Simple.\nFinal Answer: Hello!"],
        )
        agent = ReActAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "Hello!"
