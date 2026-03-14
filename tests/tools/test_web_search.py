"""Tests for the web search tool."""

from __future__ import annotations

import builtins
import sys
from typing import Any, Dict, List, Optional

import pytest

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools.web_search import WebSearchTool

# ---------------------------------------------------------------------------
# Typed fakes for external dependencies
# ---------------------------------------------------------------------------


class FakeTavilyClient:
    """Typed fake for tavily.TavilyClient, replacing MagicMock.

    Tracks calls to ``search()`` for assertion and returns preconfigured results.
    """

    def __init__(self, results: Optional[List[Dict[str, str]]] = None) -> None:
        self._results = results if results is not None else []
        self.search_calls: List[Dict[str, Any]] = []
        self._error: Optional[Exception] = None

    def set_error(self, error: Exception) -> None:
        """Configure search() to raise on next call."""
        self._error = error

    def search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        self.search_calls.append({"query": query, **kwargs})
        if self._error is not None:
            raise self._error
        return {"results": self._results}


class FakeTavilyModule:
    """Typed fake for the 'tavily' module, replacing MagicMock module injection."""

    def __init__(self, client: FakeTavilyClient) -> None:
        self._client = client

    def TavilyClient(self, *args: Any, **kwargs: Any) -> FakeTavilyClient:
        return self._client


class FakeHttpResponse:
    """Typed fake for httpx.Response, replacing MagicMock."""

    def __init__(
        self,
        text: str = "",
        content_type: str = "text/html",
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.headers = {"content-type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=self,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    @pytest.mark.spec("REQ-tools.web-search")
    def test_spec_name_and_category(self):
        tool = WebSearchTool(api_key="test-key")
        assert tool.spec.name == "web_search"
        assert tool.spec.category == "search"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_spec_requires_api_key_metadata(self):
        tool = WebSearchTool(api_key="test-key")
        assert tool.spec.metadata["requires_api_key"] == "TAVILY_API_KEY"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_spec_parameters_require_query(self):
        tool = WebSearchTool(api_key="test-key")
        assert "query" in tool.spec.parameters["properties"]
        assert "query" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_no_query(self):
        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="")
        assert result.success is False
        assert "No query" in result.content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_no_query_param(self):
        tool = WebSearchTool(api_key="test-key")
        result = tool.execute()
        assert result.success is False
        assert "No query" in result.content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_no_api_key(self, monkeypatch):
        tool = WebSearchTool(api_key=None)
        # Clear env var to ensure no fallback
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tool._api_key = None
        result = tool.execute(query="test query")
        assert result.success is False
        assert "No API key" in result.content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_mocked_tavily(self, monkeypatch):
        client = FakeTavilyClient(results=[
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "content": "Content about test.",
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "content": "More content.",
            },
        ])
        fake_module = FakeTavilyModule(client)
        monkeypatch.setitem(sys.modules, "tavily", fake_module)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is True
        assert "Result 1" in result.content
        assert "Result 2" in result.content
        assert result.metadata["num_results"] == 2

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_tavily_error(self, monkeypatch):
        client = FakeTavilyClient()
        client.set_error(RuntimeError("API rate limit exceeded"))
        fake_module = FakeTavilyModule(client)
        monkeypatch.setitem(sys.modules, "tavily", fake_module)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is False
        assert "Search error" in result.content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_max_results_parameter(self, monkeypatch):
        client = FakeTavilyClient(results=[])
        fake_module = FakeTavilyModule(client)
        monkeypatch.setitem(sys.modules, "tavily", fake_module)

        tool = WebSearchTool(api_key="test-key", max_results=3)
        tool.execute(query="test", max_results=7)
        assert len(client.search_calls) == 1
        assert client.search_calls[0]["query"] == "test"
        assert client.search_calls[0]["max_results"] == 7

    @pytest.mark.spec("REQ-tools.base.openai-format")
    def test_to_openai_function(self):
        tool = WebSearchTool(api_key="test-key")
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "web_search"
        assert "query" in fn["function"]["parameters"]["properties"]

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_import_error(self, monkeypatch):
        """Simulate tavily-python not being installed."""
        monkeypatch.delitem(sys.modules, "tavily", raising=False)

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                raise ImportError("No module named 'tavily'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is False
        assert "tavily-python not installed" in result.content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_empty_results(self, monkeypatch):
        client = FakeTavilyClient(results=[])
        fake_module = FakeTavilyModule(client)
        monkeypatch.setitem(sys.modules, "tavily", fake_module)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="obscure query")
        assert result.success is True
        assert result.content == "No results found."

    @pytest.mark.spec("REQ-tools.web-search")
    def test_tool_id(self):
        tool = WebSearchTool(api_key="test-key")
        assert tool.tool_id == "web_search"

    @pytest.mark.spec("REQ-tools.base.registration")
    def test_registry_registration(self):
        ToolRegistry.register_value("web_search", WebSearchTool)
        assert ToolRegistry.contains("web_search")


# ---------------------------------------------------------------------------
# URL detection and fetching tests
# ---------------------------------------------------------------------------


class TestUrlDetection:
    @pytest.mark.spec("REQ-tools.web-search")
    def test_is_url_https(self):
        assert WebSearchTool._is_url("https://example.com") is True

    @pytest.mark.spec("REQ-tools.web-search")
    def test_is_url_http(self):
        assert WebSearchTool._is_url("http://example.com") is True

    @pytest.mark.spec("REQ-tools.web-search")
    def test_is_url_with_whitespace(self):
        assert WebSearchTool._is_url("  https://example.com  ") is True

    @pytest.mark.spec("REQ-tools.web-search")
    def test_is_url_plain_text(self):
        assert WebSearchTool._is_url("what are punic wars") is False

    @pytest.mark.spec("REQ-tools.web-search")
    def test_is_url_empty(self):
        assert WebSearchTool._is_url("") is False

    @pytest.mark.spec("REQ-tools.web-search")
    def test_extract_url_from_text(self):
        url = WebSearchTool._extract_url(
            "Summarize this: https://example.com/page please"
        )
        assert url == "https://example.com/page"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_extract_url_none_when_absent(self):
        assert WebSearchTool._extract_url("no urls here") is None

    @pytest.mark.spec("REQ-tools.web-search")
    def test_extract_url_strips_trailing_punctuation(self):
        url = WebSearchTool._extract_url("See https://example.com/page.")
        assert url == "https://example.com/page"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_extract_url_from_complex_text(self):
        url = WebSearchTool._extract_url(
            "Read https://arxiv.org/abs/2310.03714 and summarize"
        )
        assert url == "https://arxiv.org/abs/2310.03714"


class TestUrlNormalization:
    @pytest.mark.spec("REQ-tools.web-search")
    def test_arxiv_pdf_to_abs(self):
        url = WebSearchTool._normalize_url("https://arxiv.org/pdf/2310.03714")
        assert url == "https://arxiv.org/abs/2310.03714"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_arxiv_pdf_with_extension(self):
        url = WebSearchTool._normalize_url(
            "https://arxiv.org/pdf/2310.03714.pdf"
        )
        assert url == "https://arxiv.org/abs/2310.03714"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_non_arxiv_unchanged(self):
        url = WebSearchTool._normalize_url("https://example.com/page")
        assert url == "https://example.com/page"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_arxiv_abs_unchanged(self):
        url = WebSearchTool._normalize_url("https://arxiv.org/abs/2310.03714")
        assert url == "https://arxiv.org/abs/2310.03714"


class TestUrlFetching:
    @pytest.mark.spec("REQ-tools.web-search")
    def test_fetch_url_success(self, monkeypatch):
        """Typed fake HTTP GET returns HTML, stripped to text."""
        import httpx

        fake_resp = FakeHttpResponse(
            text="<html><body><p>Hello world</p></body></html>",
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        content = WebSearchTool._fetch_url("https://example.com")
        assert "Hello world" in content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_fetch_url_strips_scripts(self, monkeypatch):
        import httpx

        fake_resp = FakeHttpResponse(
            text="<html><script>var x=1;</script><body>Content</body></html>",
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        content = WebSearchTool._fetch_url("https://example.com")
        assert "var x" not in content
        assert "Content" in content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_fetch_url_truncates_long_content(self, monkeypatch):
        import httpx

        fake_resp = FakeHttpResponse(
            text="<p>" + "x" * 10000 + "</p>",
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        content = WebSearchTool._fetch_url("https://example.com", max_chars=100)
        assert len(content) < 200
        assert "[Content truncated]" in content

    @pytest.mark.spec("REQ-tools.web-search")
    def test_fetch_url_pdf_content_type(self, monkeypatch):
        import httpx

        fake_resp = FakeHttpResponse(
            text="%PDF-1.4 binary data",
            content_type="application/pdf",
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        content = WebSearchTool._fetch_url("https://example.com/file.pdf")
        assert "PDF" in content
        assert "cannot be read" in content


class TestExecuteWithUrl:
    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_with_url_query(self, monkeypatch):
        """When query is a URL, fetch instead of search."""
        import httpx

        fake_resp = FakeHttpResponse(
            text="<html><body>Page content here</body></html>",
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="https://example.com/article")
        assert result.success is True
        assert "Page content here" in result.content
        assert result.metadata.get("mode") == "fetch"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_with_embedded_url(self, monkeypatch):
        """When query contains a URL within text, detect and fetch it."""
        import httpx

        fake_resp = FakeHttpResponse(
            text="<html><body>Article text</body></html>",
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(
            query="Summarize https://example.com/article please"
        )
        assert result.success is True
        assert result.metadata.get("mode") == "fetch"

    @pytest.mark.spec("REQ-tools.web-search")
    def test_execute_url_fetch_failure(self, monkeypatch):
        """URL fetch failure returns error result."""
        import httpx

        def _raise(*a, **kw):
            raise httpx.HTTPError("Connection failed")

        monkeypatch.setattr(httpx, "get", _raise)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="https://example.com/broken")
        assert result.success is False
        assert "Failed to fetch URL" in result.content
