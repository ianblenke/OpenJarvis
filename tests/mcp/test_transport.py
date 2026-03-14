"""Tests for MCP transport implementations."""

from __future__ import annotations

import json
import sys
import textwrap

import httpx
import pytest
import respx

from openjarvis.mcp.protocol import MCPRequest
from openjarvis.mcp.server import MCPServer
from openjarvis.mcp.transport import InProcessTransport, SSETransport, StdioTransport
from openjarvis.tools.calculator import CalculatorTool
from openjarvis.tools.think import ThinkTool


@pytest.fixture
def server():
    """MCP server with calculator and think tools."""
    return MCPServer([CalculatorTool(), ThinkTool()])


class TestInProcessTransport:
    @pytest.mark.spec("REQ-mcp.transport.in-process-initialize")
    def test_direct_call(self, server):
        transport = InProcessTransport(server)
        req = MCPRequest(method="initialize", id=1)
        resp = transport.send(req)
        assert resp.error is None
        assert "serverInfo" in resp.result

    @pytest.mark.spec("REQ-mcp.transport.in-process-tools-list")
    def test_roundtrip_tools_list(self, server):
        transport = InProcessTransport(server)
        req = MCPRequest(method="tools/list", id=2)
        resp = transport.send(req)
        assert resp.error is None
        tools = resp.result["tools"]
        assert len(tools) == 2

    @pytest.mark.spec("REQ-mcp.transport.in-process-tools-call")
    def test_roundtrip_tools_call(self, server):
        transport = InProcessTransport(server)
        req = MCPRequest(
            method="tools/call",
            params={"name": "think", "arguments": {"thought": "testing transport"}},
            id=3,
        )
        resp = transport.send(req)
        assert resp.error is None
        assert resp.result is not None
        assert "content" in resp.result

    @pytest.mark.spec("REQ-mcp.transport.in-process-multiple")
    def test_multiple_calls(self, server):
        transport = InProcessTransport(server)
        for i in range(5):
            req = MCPRequest(method="tools/list", id=i)
            resp = transport.send(req)
            assert resp.error is None

    @pytest.mark.spec("REQ-mcp.transport.in-process-close")
    def test_close_is_noop(self, server):
        transport = InProcessTransport(server)
        transport.close()  # Should not raise

    @pytest.mark.spec("REQ-mcp.transport.in-process-error-method")
    def test_error_method(self, server):
        transport = InProcessTransport(server)
        req = MCPRequest(method="unknown/method", id=1)
        resp = transport.send(req)
        assert resp.error is not None

    @pytest.mark.spec("REQ-mcp.transport.in-process-tool-name")
    def test_tool_names(self, server):
        transport = InProcessTransport(server)
        req = MCPRequest(method="tools/list", id=1)
        resp = transport.send(req)
        tool_names = {t["name"] for t in resp.result["tools"]}
        assert "calculator" in tool_names
        assert "think" in tool_names

    @pytest.mark.spec("REQ-mcp.transport.in-process-think-tool")
    def test_think_tool_call(self, server):
        transport = InProcessTransport(server)
        req = MCPRequest(
            method="tools/call",
            params={"name": "think", "arguments": {"thought": "deep thought"}},
            id=10,
        )
        resp = transport.send(req)
        assert resp.error is None
        assert resp.result is not None


class TestStdioTransport:
    @pytest.mark.spec("REQ-mcp.transport.stdio-send-receive")
    def test_send_receive(self, tmp_path):
        """Use a simple Python echo script as the subprocess."""
        script = tmp_path / "echo_server.py"
        script.write_text(textwrap.dedent("""\
            import sys
            import json
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                resp = {
                    "jsonrpc": "2.0",
                    "id": req.get("id", 0),
                    "result": {"echo": req.get("method", "")},
                }
                sys.stdout.write(json.dumps(resp) + "\\n")
                sys.stdout.flush()
        """))

        transport = StdioTransport([sys.executable, str(script)])
        try:
            req = MCPRequest(method="test/echo", id=1)
            resp = transport.send(req)
            assert resp.error is None
            assert resp.result["echo"] == "test/echo"
            assert resp.id == 1
        finally:
            transport.close()

    @pytest.mark.spec("REQ-mcp.transport.stdio-multiple")
    def test_multiple_requests(self, tmp_path):
        """Send multiple requests to the subprocess."""
        script = tmp_path / "echo_server.py"
        script.write_text(textwrap.dedent("""\
            import sys
            import json
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                resp = {
                    "jsonrpc": "2.0",
                    "id": req.get("id", 0),
                    "result": {"method": req.get("method", "")},
                }
                sys.stdout.write(json.dumps(resp) + "\\n")
                sys.stdout.flush()
        """))

        transport = StdioTransport([sys.executable, str(script)])
        try:
            for i in range(3):
                req = MCPRequest(method=f"test/{i}", id=i)
                resp = transport.send(req)
                assert resp.result["method"] == f"test/{i}"
        finally:
            transport.close()

    @pytest.mark.spec("REQ-mcp.transport.stdio-close")
    def test_close_terminates_process(self, tmp_path):
        script = tmp_path / "sleep_server.py"
        script.write_text(textwrap.dedent("""\
            import sys
            import time
            time.sleep(300)
        """))

        transport = StdioTransport([sys.executable, str(script)])
        proc = transport._process
        assert proc is not None
        assert proc.poll() is None  # still running
        transport.close()
        assert transport._process is None

    @pytest.mark.spec("REQ-mcp.transport.stdio-close-idempotent")
    def test_close_idempotent(self, tmp_path):
        script = tmp_path / "sleep_server.py"
        script.write_text("import time; time.sleep(300)")
        transport = StdioTransport([sys.executable, str(script)])
        transport.close()
        transport.close()  # Should not raise


class TestSSETransport:
    @pytest.mark.spec("REQ-mcp.transport.sse-send-receive")
    @respx.mock
    def test_send_receive(self):
        """Use respx to mock HTTP response at system boundary."""
        response_body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        )
        respx.post("http://localhost:8080/mcp").mock(
            return_value=httpx.Response(200, text=response_body)
        )

        transport = SSETransport("http://localhost:8080/mcp")
        req = MCPRequest(method="tools/list", id=1)
        resp = transport.send(req)
        assert resp.error is None
        assert resp.result == {"tools": []}

    @pytest.mark.spec("REQ-mcp.transport.sse-json-post")
    @respx.mock
    def test_send_posts_json(self):
        """Verify the HTTP POST includes correct headers and body."""
        response_body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})
        route = respx.post("http://localhost:8080/mcp").mock(
            return_value=httpx.Response(200, text=response_body)
        )

        transport = SSETransport("http://localhost:8080/mcp")
        req = MCPRequest(method="initialize", id=1)
        transport.send(req)

        assert route.called
        call = route.calls.last
        assert call.request.headers["content-type"] == "application/json"
        body = json.loads(call.request.content)
        assert body["method"] == "initialize"
        assert body["jsonrpc"] == "2.0"

    @pytest.mark.spec("REQ-mcp.transport.sse-close")
    def test_close_is_noop(self):
        transport = SSETransport("http://localhost:8080/mcp")
        transport.close()  # Should not raise

    @pytest.mark.spec("REQ-mcp.transport.sse-error-response")
    @respx.mock
    def test_error_response(self):
        """Simulate server returning an error response."""
        response_body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32601, "message": "Not found"},
            }
        )
        respx.post("http://localhost:8080/mcp").mock(
            return_value=httpx.Response(200, text=response_body)
        )

        transport = SSETransport("http://localhost:8080/mcp")
        req = MCPRequest(method="unknown", id=1)
        resp = transport.send(req)
        assert resp.error is not None
        assert resp.error["code"] == -32601

    @pytest.mark.spec("REQ-mcp.transport.sse-success-result")
    @respx.mock
    def test_success_result(self):
        """Verify successful result is correctly parsed."""
        response_body = json.dumps(
            {"jsonrpc": "2.0", "id": 5, "result": {"data": [1, 2, 3]}}
        )
        respx.post("http://localhost:8080/mcp").mock(
            return_value=httpx.Response(200, text=response_body)
        )

        transport = SSETransport("http://localhost:8080/mcp")
        req = MCPRequest(method="test/data", id=5)
        resp = transport.send(req)
        assert resp.id == 5
        assert resp.result == {"data": [1, 2, 3]}
        assert resp.error is None

    @pytest.mark.spec("REQ-mcp.transport.sse-http-error")
    @respx.mock
    def test_http_error_raises(self):
        """HTTP errors should propagate."""
        respx.post("http://localhost:8080/mcp").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        transport = SSETransport("http://localhost:8080/mcp")
        req = MCPRequest(method="test", id=1)
        with pytest.raises(httpx.HTTPStatusError):
            transport.send(req)
