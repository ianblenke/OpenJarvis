"""Tests for A2A protocol types — AgentCard and A2ATask."""

from __future__ import annotations

import pytest

from openjarvis.a2a.protocol import (
    A2ATask,
    AgentCard,
    TaskState,
)


class TestAgentCardSpec:
    """REQ-a2a.protocol.agent-card: Agent discovery card."""

    @pytest.mark.spec("REQ-a2a.protocol.agent-card")
    def test_agent_card_required_fields(self) -> None:
        card = AgentCard(
            name="TestAgent",
            description="A helpful agent",
            url="http://localhost:8000",
            version="1.0.0",
            capabilities=["text-generation"],
            skills=["summarization"],
            authentication={"type": "bearer"},
        )
        assert card.name == "TestAgent"
        assert card.description == "A helpful agent"
        assert card.url == "http://localhost:8000"
        assert card.version == "1.0.0"
        assert card.capabilities == ["text-generation"]
        assert card.skills == ["summarization"]
        assert card.authentication == {"type": "bearer"}

    @pytest.mark.spec("REQ-a2a.protocol.agent-card")
    def test_agent_card_defaults(self) -> None:
        card = AgentCard(name="Minimal")
        assert card.description == ""
        assert card.url == ""
        assert card.version == "0.1.0"
        assert card.capabilities == []
        assert card.skills == []
        assert card.authentication == {}

    @pytest.mark.spec("REQ-a2a.protocol.agent-card")
    def test_agent_card_to_dict(self) -> None:
        card = AgentCard(
            name="Agent",
            description="desc",
            url="http://a.b",
            version="2.0",
            capabilities=["chat"],
            skills=["code"],
            authentication={"scheme": "oauth2"},
        )
        d = card.to_dict()
        assert d["name"] == "Agent"
        assert d["description"] == "desc"
        assert d["url"] == "http://a.b"
        assert d["version"] == "2.0"
        assert d["capabilities"] == ["chat"]
        assert d["skills"] == ["code"]
        assert d["authentication"] == {"scheme": "oauth2"}

    @pytest.mark.spec("REQ-a2a.protocol.agent-card")
    def test_agent_card_list_isolation(self) -> None:
        """Default list fields should be independent between instances."""
        card1 = AgentCard(name="A")
        card2 = AgentCard(name="B")
        card1.capabilities.append("extra")
        assert "extra" not in card2.capabilities


class TestA2ATaskSpec:
    """REQ-a2a.protocol.task: Task lifecycle."""

    @pytest.mark.spec("REQ-a2a.protocol.task")
    def test_task_has_required_fields(self) -> None:
        task = A2ATask(input_text="Hello")
        assert task.task_id  # auto-generated, non-empty
        assert task.state == TaskState.SUBMITTED
        assert task.input_text == "Hello"
        assert task.output_text == ""
        assert task.history == []

    @pytest.mark.spec("REQ-a2a.protocol.task")
    def test_task_state_values(self) -> None:
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"

    @pytest.mark.spec("REQ-a2a.protocol.task")
    def test_task_to_dict(self) -> None:
        task = A2ATask(input_text="test", task_id="abc123")
        d = task.to_dict()
        assert d["id"] == "abc123"
        assert d["state"] == "submitted"
        assert d["input"] == "test"
        assert d["output"] == ""
        assert d["history"] == []

    @pytest.mark.spec("REQ-a2a.protocol.task")
    def test_task_state_transitions(self) -> None:
        task = A2ATask(input_text="query")
        assert task.state == TaskState.SUBMITTED

        task.state = TaskState.WORKING
        assert task.state == TaskState.WORKING

        task.state = TaskState.COMPLETED
        task.output_text = "result"
        assert task.state == TaskState.COMPLETED
        assert task.output_text == "result"

    @pytest.mark.spec("REQ-a2a.protocol.task")
    def test_task_history_tracking(self) -> None:
        task = A2ATask(input_text="hello")
        task.history.append({"role": "agent", "content": "world"})
        assert len(task.history) == 1
        assert task.history[0]["role"] == "agent"

    @pytest.mark.spec("REQ-a2a.protocol.task")
    def test_task_unique_ids(self) -> None:
        task1 = A2ATask(input_text="a")
        task2 = A2ATask(input_text="b")
        assert task1.task_id != task2.task_id
