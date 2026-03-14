"""Tests for A2A server — handler and routes."""

from __future__ import annotations

import pytest

from openjarvis.a2a.protocol import AgentCard
from openjarvis.a2a.server import A2AServer
from openjarvis.core.events import EventBus


class TestA2AServerHandler:
    """REQ-a2a.server.handler: JSON-RPC handler dispatches methods correctly."""

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_handle_task_send(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card, handler=lambda x: f"Echo: {x}")
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"input": "Hello"},
            "id": "req-1",
        })
        assert response["result"]["state"] == "completed"
        assert "Echo: Hello" in response["result"]["output"]
        assert response["id"] == "req-1"

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_handle_task_get(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card, handler=lambda x: x)
        send_resp = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"input": "data"},
            "id": "1",
        })
        task_id = send_resp["result"]["id"]

        get_resp = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": "2",
        })
        assert get_resp["result"]["id"] == task_id
        assert get_resp["result"]["state"] == "completed"

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_handle_task_cancel(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card, handler=lambda x: x)
        send_resp = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"input": "data"},
            "id": "1",
        })
        task_id = send_resp["result"]["id"]

        cancel_resp = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"id": task_id},
            "id": "2",
        })
        assert cancel_resp["result"]["state"] == "canceled"

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_handle_unknown_method_returns_error(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card)
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "unknown/rpc",
            "params": {},
            "id": "1",
        })
        assert "error" in response
        assert response["error"]["code"] == -32601

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_handler_error_produces_failed_task(self) -> None:
        card = AgentCard(name="Test")

        def failing_handler(x: str) -> str:
            raise RuntimeError("handler broke")

        server = A2AServer(card, handler=failing_handler)
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"input": "trigger"},
            "id": "1",
        })
        assert response["result"]["state"] == "failed"
        assert "handler broke" in response["result"]["output"]

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_no_handler_returns_default_message(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card)
        response = server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"input": "hello"},
            "id": "1",
        })
        assert response["result"]["state"] == "completed"
        assert "No handler configured" in response["result"]["output"]

    @pytest.mark.spec("REQ-a2a.server.handler")
    def test_events_emitted_on_task_send(self) -> None:
        from openjarvis.core.events import EventType

        bus = EventBus(record_history=True)
        card = AgentCard(name="Test")
        server = A2AServer(card, handler=lambda x: x, bus=bus)
        server.handle_request({
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"input": "hi"},
            "id": "1",
        })
        event_types = {e.event_type for e in bus.history}
        assert EventType.A2A_TASK_RECEIVED in event_types
        assert EventType.A2A_TASK_COMPLETED in event_types


class TestA2AServerRoutes:
    """REQ-a2a.server.routes: get_routes() returns route definitions."""

    @pytest.mark.spec("REQ-a2a.server.routes")
    def test_get_routes_returns_list(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card)
        routes = server.get_routes()
        assert isinstance(routes, list)
        assert len(routes) >= 2

    @pytest.mark.spec("REQ-a2a.server.routes")
    def test_routes_include_agent_card_endpoint(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card)
        routes = server.get_routes()
        paths = [r["path"] for r in routes]
        assert "/.well-known/agent.json" in paths

    @pytest.mark.spec("REQ-a2a.server.routes")
    def test_routes_include_tasks_endpoint(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card)
        routes = server.get_routes()
        paths = [r["path"] for r in routes]
        assert "/a2a/tasks" in paths

    @pytest.mark.spec("REQ-a2a.server.routes")
    def test_routes_have_method_and_handler(self) -> None:
        card = AgentCard(name="Test")
        server = A2AServer(card)
        routes = server.get_routes()
        for route in routes:
            assert "path" in route
            assert "method" in route
            assert "handler" in route
