"""Tests for A2A tool adapter."""

from __future__ import annotations

import httpx
import pytest
import respx

from openjarvis.a2a.tool import A2AAgentTool


class TestA2AAgentTool:
    """Test wrapping a remote A2A agent as a local tool."""

    @pytest.mark.spec("REQ-a2a.tool.wrapper")
    @respx.mock
    def test_tool_creation(self) -> None:
        card_data = {
            "name": "RemoteAgent",
            "description": "A remote helper agent",
            "url": "http://remote:9000",
            "version": "1.0",
        }
        respx.get("http://remote:9000/.well-known/agent.json").mock(
            return_value=httpx.Response(200, json=card_data)
        )
        from openjarvis.a2a.client import A2AClient

        client = A2AClient("http://remote:9000")
        tool = A2AAgentTool(client)
        assert tool.tool_id is not None
        assert "RemoteAgent" in tool.spec.description or tool.spec.name

    @pytest.mark.spec("REQ-a2a.tool.wrapper")
    @respx.mock
    def test_tool_execute(self) -> None:
        card_data = {
            "name": "RemoteAgent",
            "description": "helper",
            "url": "http://remote:9000",
            "version": "1.0",
        }
        respx.get("http://remote:9000/.well-known/agent.json").mock(
            return_value=httpx.Response(200, json=card_data)
        )
        task_response = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "id": "t-1",
                "state": "completed",
                "input": "what is 2+2?",
                "output": "4",
            },
        }
        respx.post("http://remote:9000/a2a/tasks").mock(
            return_value=httpx.Response(200, json=task_response)
        )
        from openjarvis.a2a.client import A2AClient

        client = A2AClient("http://remote:9000")
        tool = A2AAgentTool(client)
        result = tool.execute(input="what is 2+2?")
        assert result.success
        assert "4" in result.content

    @pytest.mark.spec("REQ-a2a.tool.wrapper")
    @respx.mock
    def test_tool_spec_properties(self) -> None:
        card_data = {
            "name": "SpecAgent",
            "description": "spec test",
            "url": "http://remote:9000",
            "version": "1.0",
        }
        respx.get("http://remote:9000/.well-known/agent.json").mock(
            return_value=httpx.Response(200, json=card_data)
        )
        from openjarvis.a2a.client import A2AClient

        client = A2AClient("http://remote:9000")
        tool = A2AAgentTool(client)
        spec = tool.spec
        assert spec.name is not None
        assert spec.description is not None
        assert "input" in spec.parameters.get("properties", {})
