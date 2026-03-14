"""Tests for MCP protocol message types."""

from __future__ import annotations

import json

import pytest

from openjarvis.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    MCPError,
    MCPNotification,
    MCPRequest,
    MCPResponse,
)


class TestMCPProtocolTypes:
    """REQ-mcp.protocol.types: MCPRequest, MCPResponse, MCPNotification types exist."""

    @pytest.mark.spec("REQ-mcp.protocol.types")
    def test_protocol_types_importable(self):
        """All three JSON-RPC 2.0 message types are importable."""
        assert MCPRequest is not None
        assert MCPResponse is not None
        assert MCPNotification is not None

    @pytest.mark.spec("REQ-mcp.protocol.types")
    def test_request_follows_jsonrpc(self):
        req = MCPRequest(method="test", id=1)
        parsed = json.loads(req.to_json())
        assert parsed["jsonrpc"] == "2.0"
        assert "method" in parsed
        assert "id" in parsed
        assert "params" in parsed

    @pytest.mark.spec("REQ-mcp.protocol.types")
    def test_response_follows_jsonrpc(self):
        resp = MCPResponse(result={"ok": True}, id=1)
        parsed = json.loads(resp.to_json())
        assert parsed["jsonrpc"] == "2.0"
        assert "result" in parsed
        assert "id" in parsed

    @pytest.mark.spec("REQ-mcp.protocol.types")
    def test_notification_follows_jsonrpc(self):
        notif = MCPNotification(method="ping", params={})
        parsed = json.loads(notif.to_json())
        assert parsed["jsonrpc"] == "2.0"
        assert "method" in parsed
        assert "id" not in parsed


class TestMCPRequest:
    @pytest.mark.spec("REQ-mcp.protocol.request-serialize")
    def test_serialize_deserialize(self):
        req = MCPRequest(method="tools/list", params={"cursor": None}, id=1)
        data = req.to_json()
        restored = MCPRequest.from_json(data)
        assert restored.method == "tools/list"
        assert restored.id == 1
        assert restored.jsonrpc == "2.0"
        assert restored.params == {"cursor": None}

    @pytest.mark.spec("REQ-mcp.protocol.request-initialize")
    def test_initialize_request(self):
        req = MCPRequest(method="initialize", params={}, id=1)
        parsed = json.loads(req.to_json())
        assert parsed["method"] == "initialize"
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 1

    @pytest.mark.spec("REQ-mcp.protocol.request-tools-list")
    def test_tools_list_request(self):
        req = MCPRequest(method="tools/list", id=2)
        parsed = json.loads(req.to_json())
        assert parsed["method"] == "tools/list"
        assert parsed["id"] == 2

    @pytest.mark.spec("REQ-mcp.protocol.request-tools-call")
    def test_tools_call_request(self):
        req = MCPRequest(
            method="tools/call",
            params={"name": "calculator", "arguments": {"expression": "2+2"}},
            id=3,
        )
        parsed = json.loads(req.to_json())
        assert parsed["method"] == "tools/call"
        assert parsed["params"]["name"] == "calculator"
        assert parsed["params"]["arguments"]["expression"] == "2+2"

    @pytest.mark.spec("REQ-mcp.protocol.request-default-params")
    def test_default_params_empty(self):
        req = MCPRequest(method="test")
        assert req.params == {}

    @pytest.mark.spec("REQ-mcp.protocol.request-string-id")
    def test_string_id(self):
        req = MCPRequest(method="test", id="abc-123")
        data = req.to_json()
        restored = MCPRequest.from_json(data)
        assert restored.id == "abc-123"

    @pytest.mark.spec("REQ-mcp.protocol.request-from-json-missing-params")
    def test_from_json_missing_params(self):
        raw = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
        req = MCPRequest.from_json(raw)
        assert req.params == {}
        assert req.method == "test"

    @pytest.mark.spec("REQ-mcp.protocol.request-from-json-missing-id")
    def test_from_json_missing_id(self):
        raw = json.dumps({"jsonrpc": "2.0", "method": "test"})
        req = MCPRequest.from_json(raw)
        assert req.id == 0  # default

    @pytest.mark.spec("REQ-mcp.protocol.request-to-json-structure")
    def test_to_json_structure(self):
        req = MCPRequest(method="test/method", params={"key": "val"}, id=42)
        parsed = json.loads(req.to_json())
        assert set(parsed.keys()) == {"jsonrpc", "id", "method", "params"}
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 42
        assert parsed["method"] == "test/method"
        assert parsed["params"] == {"key": "val"}


class TestMCPResponse:
    @pytest.mark.spec("REQ-mcp.protocol.response-success-roundtrip")
    def test_serialize_deserialize_success(self):
        resp = MCPResponse(result={"tools": []}, id=1)
        data = resp.to_json()
        restored = MCPResponse.from_json(data)
        assert restored.result == {"tools": []}
        assert restored.error is None
        assert restored.id == 1

    @pytest.mark.spec("REQ-mcp.protocol.response-error-roundtrip")
    def test_serialize_deserialize_error(self):
        resp = MCPResponse.error_response(1, METHOD_NOT_FOUND, "Not found")
        data = resp.to_json()
        restored = MCPResponse.from_json(data)
        assert restored.error is not None
        assert restored.error["code"] == METHOD_NOT_FOUND
        assert restored.error["message"] == "Not found"

    @pytest.mark.spec("REQ-mcp.protocol.response-error-factory")
    def test_error_response_factory(self):
        resp = MCPResponse.error_response(42, INVALID_PARAMS, "Bad params")
        assert resp.error["code"] == INVALID_PARAMS
        assert resp.error["message"] == "Bad params"
        assert resp.id == 42
        assert resp.result is None

    @pytest.mark.spec("REQ-mcp.protocol.response-error-with-data")
    def test_error_response_with_data(self):
        resp = MCPResponse.error_response(
            1, INTERNAL_ERROR, "Oops", data={"detail": "stack"},
        )
        assert resp.error["data"] == {"detail": "stack"}

    @pytest.mark.spec("REQ-mcp.protocol.response-error-without-data")
    def test_error_response_without_data(self):
        resp = MCPResponse.error_response(1, INTERNAL_ERROR, "Oops")
        assert "data" not in resp.error

    @pytest.mark.spec("REQ-mcp.protocol.response-success-json")
    def test_success_response_json(self):
        resp = MCPResponse(result={"value": 42}, id=5)
        parsed = json.loads(resp.to_json())
        assert "result" in parsed
        assert "error" not in parsed
        assert parsed["result"]["value"] == 42

    @pytest.mark.spec("REQ-mcp.protocol.response-error-excludes-result")
    def test_error_excludes_result(self):
        resp = MCPResponse.error_response(1, PARSE_ERROR, "Parse error")
        parsed = json.loads(resp.to_json())
        assert "error" in parsed
        assert "result" not in parsed

    @pytest.mark.spec("REQ-mcp.protocol.response-jsonrpc-version")
    def test_jsonrpc_version(self):
        resp = MCPResponse(result={}, id=1)
        parsed = json.loads(resp.to_json())
        assert parsed["jsonrpc"] == "2.0"

    @pytest.mark.spec("REQ-mcp.protocol.response-from-json-defaults")
    def test_from_json_defaults(self):
        raw = json.dumps({"result": "ok"})
        resp = MCPResponse.from_json(raw)
        assert resp.result == "ok"
        assert resp.id == 0
        assert resp.jsonrpc == "2.0"
        assert resp.error is None


class TestMCPNotification:
    @pytest.mark.spec("REQ-mcp.protocol.notification-format")
    def test_format(self):
        notif = MCPNotification(method="notifications/initialized", params={})
        parsed = json.loads(notif.to_json())
        assert parsed["method"] == "notifications/initialized"
        assert parsed["jsonrpc"] == "2.0"

    @pytest.mark.spec("REQ-mcp.protocol.notification-no-id")
    def test_no_id_field(self):
        notif = MCPNotification(method="test")
        parsed = json.loads(notif.to_json())
        assert "id" not in parsed

    @pytest.mark.spec("REQ-mcp.protocol.notification-with-params")
    def test_with_params(self):
        notif = MCPNotification(method="progress", params={"percent": 50})
        parsed = json.loads(notif.to_json())
        assert parsed["params"]["percent"] == 50

    @pytest.mark.spec("REQ-mcp.protocol.notification-default-params")
    def test_default_params(self):
        notif = MCPNotification(method="test")
        assert notif.params == {}


class TestMCPError:
    @pytest.mark.spec("REQ-mcp.protocol.error-is-exception")
    def test_error_is_exception(self):
        err = MCPError(code=METHOD_NOT_FOUND, message="Not found")
        assert isinstance(err, Exception)

    @pytest.mark.spec("REQ-mcp.protocol.error-str")
    def test_error_str(self):
        err = MCPError(code=-32601, message="Not found")
        assert "-32601" in str(err)
        assert "Not found" in str(err)

    @pytest.mark.spec("REQ-mcp.protocol.error-with-data")
    def test_error_with_data(self):
        err = MCPError(code=INTERNAL_ERROR, message="Oops", data={"trace": "..."})
        assert err.data == {"trace": "..."}

    @pytest.mark.spec("REQ-mcp.protocol.error-default-data")
    def test_error_default_data(self):
        err = MCPError(code=INTERNAL_ERROR, message="Oops")
        assert err.data is None


class TestErrorCodes:
    @pytest.mark.spec("REQ-mcp.protocol.error-codes")
    def test_parse_error(self):
        assert PARSE_ERROR == -32700

    @pytest.mark.spec("REQ-mcp.protocol.error-codes")
    def test_invalid_request(self):
        assert INVALID_REQUEST == -32600

    @pytest.mark.spec("REQ-mcp.protocol.error-codes")
    def test_method_not_found(self):
        assert METHOD_NOT_FOUND == -32601

    @pytest.mark.spec("REQ-mcp.protocol.error-codes")
    def test_invalid_params(self):
        assert INVALID_PARAMS == -32602

    @pytest.mark.spec("REQ-mcp.protocol.error-codes")
    def test_internal_error(self):
        assert INTERNAL_ERROR == -32603
