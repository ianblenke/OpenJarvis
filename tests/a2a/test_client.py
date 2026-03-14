"""Tests for A2A client using respx for HTTP mocking."""

from __future__ import annotations

import httpx
import pytest
import respx

from openjarvis.a2a.client import A2AClient
from openjarvis.a2a.protocol import AgentCard, TaskState


class TestA2AClientDiscover:
    """Test A2A client discovery."""

    @pytest.mark.spec("REQ-a2a.client.discover")
    @respx.mock
    def test_discover_agent(self) -> None:
        card_data = {
            "name": "TestAgent",
            "description": "A test agent",
            "url": "http://localhost:9000",
            "version": "1.0",
            "capabilities": ["text"],
            "skills": [],
        }
        respx.get("http://localhost:9000/.well-known/agent.json").mock(
            return_value=httpx.Response(200, json=card_data)
        )
        client = A2AClient("http://localhost:9000")
        card = client.discover()
        assert card.name == "TestAgent"
        assert card.version == "1.0"

    @pytest.mark.spec("REQ-a2a.client.discover")
    @respx.mock
    def test_discover_returns_agent_card(self) -> None:
        card_data = {
            "name": "CachedAgent",
            "description": "cached",
            "url": "http://localhost:9000",
            "version": "1.0",
        }
        respx.get("http://localhost:9000/.well-known/agent.json").mock(
            return_value=httpx.Response(200, json=card_data)
        )
        client = A2AClient("http://localhost:9000")
        card = client.discover()
        assert isinstance(card, AgentCard)
        assert card.name == "CachedAgent"


class TestA2AClientSendTask:
    """Test A2A client task operations."""

    @pytest.mark.spec("REQ-a2a.client.send-task")
    @respx.mock
    def test_send_task(self) -> None:
        response_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "id": "task-123",
                "state": "working",
                "input": "hello",
                "output": "",
            },
        }
        respx.post("http://localhost:9000/a2a/tasks").mock(
            return_value=httpx.Response(200, json=response_data)
        )
        client = A2AClient("http://localhost:9000")
        task = client.send_task("hello")
        assert task.task_id == "task-123"
        assert task.state == TaskState.WORKING

    @pytest.mark.spec("REQ-a2a.client.get-task")
    @respx.mock
    def test_get_task(self) -> None:
        response_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "id": "task-123",
                "state": "completed",
                "input": "hello",
                "output": "world",
            },
        }
        respx.post("http://localhost:9000/a2a/tasks").mock(
            return_value=httpx.Response(200, json=response_data)
        )
        client = A2AClient("http://localhost:9000")
        task = client.get_task("task-123")
        assert task.state == TaskState.COMPLETED
        assert task.output_text == "world"

    @pytest.mark.spec("REQ-a2a.client.cancel-task")
    @respx.mock
    def test_cancel_task(self) -> None:
        response_data = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "task_id": "task-123",
                "state": "canceled",
                "input_text": "hello",
                "output_text": "",
            },
        }
        respx.post("http://localhost:9000/a2a/tasks").mock(
            return_value=httpx.Response(200, json=response_data)
        )
        client = A2AClient("http://localhost:9000")
        task = client.cancel_task("task-123")
        assert task.state == TaskState.CANCELED
