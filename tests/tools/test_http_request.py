"""Tests for the HTTP request tool with SSRF protection."""

from __future__ import annotations

import types

import httpx
import pytest
import respx

import openjarvis.tools.http_request as _http_mod
from openjarvis.tools.http_request import HttpRequestTool

# ---------------------------------------------------------------------------
# Typed fake for the Rust module (forces httpx fallback)
# ---------------------------------------------------------------------------


class _FakeRustHttpTool:
    """Typed fake Rust HTTP tool that always raises, forcing httpx fallback."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("mocked out")


def _make_fake_rust_module():
    """Build a typed fake Rust module whose HttpRequestTool always raises."""
    mod = types.ModuleType("fake_openjarvis_rust")
    mod.HttpRequestTool = lambda: _FakeRustHttpTool()  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def _force_httpx_fallback(monkeypatch):
    """Patch the Rust HTTP tool so it raises, falling back to httpx.

    The Rust backend makes real HTTP requests that bypass respx mocks.
    By making the Rust HttpRequestTool().execute() raise, the tool falls
    through to the httpx code path where respx interception works.
    """
    fake_mod = _make_fake_rust_module()
    monkeypatch.setattr(
        "openjarvis._rust_bridge.get_rust_module",
        lambda: fake_mod,
    )


class TestHttpRequestTool:
    @pytest.mark.spec("REQ-tools.http-request")
    def test_spec_name_and_category(self):
        tool = HttpRequestTool()
        assert tool.spec.name == "http_request"
        assert tool.spec.category == "network"

    @pytest.mark.spec("REQ-tools.http-request")
    def test_spec_required_capabilities(self):
        tool = HttpRequestTool()
        assert "network:fetch" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.http-request")
    def test_spec_parameters_require_url(self):
        tool = HttpRequestTool()
        assert "url" in tool.spec.parameters["properties"]
        assert "url" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.http-request")
    def test_tool_id(self):
        tool = HttpRequestTool()
        assert tool.tool_id == "http_request"

    @pytest.mark.spec("REQ-tools.http-request")
    def test_to_openai_function(self):
        tool = HttpRequestTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "http_request"
        assert "url" in fn["function"]["parameters"]["properties"]

    @pytest.mark.spec("REQ-tools.http-request")
    def test_no_url(self):
        tool = HttpRequestTool()
        result = tool.execute()
        assert result.success is False
        assert "No URL" in result.content

    @pytest.mark.spec("REQ-tools.http-request")
    def test_empty_url(self):
        tool = HttpRequestTool()
        result = tool.execute(url="")
        assert result.success is False
        assert "No URL" in result.content

    @pytest.mark.spec("REQ-tools.http-request")
    def test_ssrf_blocked_private_ip(self, monkeypatch):
        """Request to private IP should be blocked by SSRF protection."""
        tool = HttpRequestTool()
        monkeypatch.setattr(
            _http_mod, "check_ssrf",
            lambda url: "URL resolves to private IP: 192.168.1.1",
        )
        result = tool.execute(url="http://192.168.1.1/admin")
        assert result.success is False
        assert "SSRF protection" in result.content
        assert "private IP" in result.content

    @pytest.mark.spec("REQ-tools.http-request")
    def test_ssrf_blocked_metadata_endpoint(self, monkeypatch):
        """Request to cloud metadata endpoint should be blocked."""
        tool = HttpRequestTool()
        monkeypatch.setattr(
            _http_mod, "check_ssrf",
            lambda url: "Blocked host: 169.254.169.254 (cloud metadata endpoint)",
        )
        result = tool.execute(url="http://169.254.169.254/latest/meta-data/")
        assert result.success is False
        assert "SSRF protection" in result.content
        assert "metadata" in result.content.lower() or "Blocked host" in result.content

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_successful_get(self, monkeypatch):
        """Successful GET request returns response content and metadata."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(
                200,
                text='{"key": "value"}',
                headers={"content-type": "application/json"},
            )
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(url="https://api.example.com/data")
        assert result.success is True
        assert '"key": "value"' in result.content
        assert result.metadata["status_code"] == 200
        assert "application/json" in result.metadata["content_type"]
        assert "elapsed_ms" in result.metadata

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_post_with_body(self, monkeypatch):
        """POST request with body sends content correctly."""
        respx.post("https://api.example.com/submit").mock(
            return_value=httpx.Response(
                201,
                text='{"id": 42}',
                headers={"content-type": "application/json"},
            )
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(
            url="https://api.example.com/submit",
            method="POST",
            body='{"name": "test"}',
            headers={"Content-Type": "application/json"},
        )
        assert result.success is True
        assert '"id": 42' in result.content
        assert result.metadata["status_code"] == 201

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_put_method(self, monkeypatch):
        """PUT request works correctly."""
        respx.put("https://api.example.com/resource/1").mock(
            return_value=httpx.Response(200, text="updated")
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(
            url="https://api.example.com/resource/1",
            method="PUT",
            body="new data",
        )
        assert result.success is True
        assert "updated" in result.content

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_delete_method(self, monkeypatch):
        """DELETE request works correctly."""
        respx.delete("https://api.example.com/resource/1").mock(
            return_value=httpx.Response(204, text="")
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(
            url="https://api.example.com/resource/1",
            method="DELETE",
        )
        assert result.success is True

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_head_method(self, monkeypatch):
        """HEAD request works correctly."""
        respx.head("https://api.example.com/check").mock(
            return_value=httpx.Response(
                200,
                text="",
                headers={"x-custom": "header-value"},
            )
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(
            url="https://api.example.com/check",
            method="HEAD",
        )
        assert result.success is True
        assert result.metadata["status_code"] == 200

    @pytest.mark.spec("REQ-tools.http-request")
    def test_timeout_handling(self, monkeypatch):
        """Timeout should produce a clear error."""
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        monkeypatch.setattr(
            _http_mod, "httpx",
            type("_httpx", (), {
                "request": staticmethod(
                    lambda *a, **kw: (_ for _ in ()).throw(
                        httpx.TimeoutException("timed out")
                    )
                ),
                "TimeoutException": httpx.TimeoutException,
                "ConnectError": httpx.ConnectError,
            })(),
        )
        result = tool.execute(url="https://slow.example.com", timeout=5)
        assert result.success is False
        assert "timed out" in result.content.lower()

    @pytest.mark.spec("REQ-tools.http-request")
    def test_request_error(self, monkeypatch):
        """Connection error should produce a clear error."""
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)

        def _raise_connect(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(_http_mod.httpx, "request", _raise_connect)
        result = tool.execute(url="https://down.example.com")
        assert result.success is False
        assert "Request error" in result.content

    @pytest.mark.spec("REQ-tools.http-request")
    def test_method_validation(self):
        """Invalid HTTP method should be rejected."""
        tool = HttpRequestTool()
        result = tool.execute(url="https://example.com", method="TRACE")
        assert result.success is False
        assert "Unsupported HTTP method" in result.content
        assert "TRACE" in result.content

    @pytest.mark.spec("REQ-tools.http-request")
    def test_method_case_insensitive(self, monkeypatch):
        """Method should be case-insensitive."""
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        with respx.mock:
            respx.get("https://api.example.com/data").mock(
                return_value=httpx.Response(200, text="ok")
            )
            result = tool.execute(url="https://api.example.com/data", method="get")
        assert result.success is True

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_response_truncation(self, monkeypatch):
        """Response larger than 1 MB should be truncated."""
        large_body = "x" * 2_000_000  # 2 MB
        respx.get("https://api.example.com/large").mock(
            return_value=httpx.Response(200, text=large_body)
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(url="https://api.example.com/large")
        assert result.success is True
        assert "[Response truncated at 1 MB]" in result.content
        assert result.metadata["truncated"] is True

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_response_not_truncated_when_small(self, monkeypatch):
        """Response smaller than 1 MB should not be truncated."""
        small_body = "hello world"
        respx.get("https://api.example.com/small").mock(
            return_value=httpx.Response(200, text=small_body)
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(url="https://api.example.com/small")
        assert result.success is True
        assert result.content == "hello world"
        assert result.metadata["truncated"] is False

    @respx.mock
    @pytest.mark.spec("REQ-tools.http-request")
    def test_metadata_includes_headers(self, monkeypatch):
        """Response metadata should include headers dict."""
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(
                200,
                text="ok",
                headers={
                    "content-type": "text/plain",
                    "x-request-id": "abc123",
                },
            )
        )
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)
        result = tool.execute(url="https://api.example.com/data")
        assert isinstance(result.metadata["headers"], dict)
        assert result.metadata["headers"]["x-request-id"] == "abc123"


    @pytest.mark.spec("REQ-tools.http-request")
    def test_rust_success_path(self, monkeypatch):
        """Exercise the Rust HttpRequestTool success path (line 111).

        Uses a typed fake that returns content instead of raising,
        so the httpx fallback is NOT triggered.
        """

        class _SuccessRustHttpTool:
            def execute(self, url, method, body=None):
                return '{"rust": true}'

        def _make_success_mod():
            mod = types.ModuleType("fake_rust_success")
            mod.HttpRequestTool = lambda: _SuccessRustHttpTool()
            return mod

        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            _make_success_mod,
        )
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)

        tool = HttpRequestTool()
        result = tool.execute(url="https://api.example.com/data")
        assert result.success is True
        assert '{"rust": true}' in result.content
        assert result.metadata["status_code"] == 200
        assert result.metadata["truncated"] is False

    @pytest.mark.spec("REQ-tools.http-request")
    def test_unexpected_exception(self, monkeypatch):
        """Exercise the generic Exception handler (lines 177-178)."""
        tool = HttpRequestTool()
        monkeypatch.setattr(_http_mod, "check_ssrf", lambda url: None)

        def _raise_unexpected(*a, **kw):
            raise RuntimeError("something completely unexpected")

        monkeypatch.setattr(_http_mod.httpx, "request", _raise_unexpected)
        result = tool.execute(url="https://api.example.com/fail")
        assert result.success is False
        assert "Unexpected error" in result.content
        assert "something completely unexpected" in result.content


__all__ = ["TestHttpRequestTool"]
