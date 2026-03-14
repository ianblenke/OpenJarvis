"""Tests for browser_axtree tool."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

import openjarvis.tools.browser_axtree as _axtree_mod
from openjarvis.tools.browser_axtree import BrowserAXTreeTool

# ---------------------------------------------------------------------------
# Typed fakes replacing MagicMock for Playwright objects
# ---------------------------------------------------------------------------


class _FakeAccessibility:
    """Typed fake for page.accessibility."""

    def __init__(self, snapshot_data: Optional[Dict[str, Any]] = None,
                 error: Optional[Exception] = None) -> None:
        self._snapshot_data = snapshot_data
        self._error = error

    def snapshot(self) -> Optional[Dict[str, Any]]:
        if self._error is not None:
            raise self._error
        return self._snapshot_data


class _FakePage:
    """Typed fake for a Playwright Page with an accessibility property."""

    def __init__(self, snapshot_data: Optional[Dict[str, Any]] = None,
                 error: Optional[Exception] = None) -> None:
        self.accessibility = _FakeAccessibility(snapshot_data, error)


class _FakeSession:
    """Typed fake for _BrowserSession with a configurable page."""

    def __init__(self, page: Optional[_FakePage] = None,
                 page_error: Optional[Exception] = None) -> None:
        self._page = page
        self._page_error = page_error

    @property
    def page(self):
        if self._page_error is not None:
            raise self._page_error
        return self._page


def _make_default_snapshot() -> Dict[str, Any]:
    return {
        "role": "WebArea",
        "name": "Test Page",
        "children": [
            {"role": "heading", "name": "Welcome", "level": 1},
            {"role": "link", "name": "Click me", "url": "https://example.com"},
            {"role": "textbox", "name": "Search", "value": ""},
        ],
    }


class TestBrowserAXTreeTool:
    @pytest.mark.spec("REQ-tools.browser")
    def test_instantiation(self) -> None:
        tool = BrowserAXTreeTool()
        assert tool.tool_id == "browser_axtree"
        assert tool.spec.name == "browser_axtree"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_returns_tree(self, monkeypatch) -> None:
        page = _FakePage(snapshot_data=_make_default_snapshot())
        session = _FakeSession(page=page)
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute()
        assert result.success is True
        assert "heading" in result.content
        assert "Welcome" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_includes_all_roles(self, monkeypatch) -> None:
        page = _FakePage(snapshot_data=_make_default_snapshot())
        session = _FakeSession(page=page)
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute()
        assert result.success is True
        assert "WebArea" in result.content
        assert "link" in result.content
        assert "textbox" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_includes_node_count_metadata(self, monkeypatch) -> None:
        page = _FakePage(snapshot_data=_make_default_snapshot())
        session = _FakeSession(page=page)
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute()
        assert result.success is True
        # 1 root + 3 children = 4 nodes
        assert result.metadata["node_count"] == 4

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_max_depth(self, monkeypatch) -> None:
        """When max_depth=1 only the root node should appear."""
        page = _FakePage(snapshot_data=_make_default_snapshot())
        session = _FakeSession(page=page)
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute(max_depth=1)
        assert result.success is True
        assert "WebArea" in result.content
        # Children at depth 1 should not be present
        assert "heading" not in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_empty_snapshot(self, monkeypatch) -> None:
        page = _FakePage(snapshot_data=None)
        session = _FakeSession(page=page)
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute()
        assert result.success is False
        assert "No accessibility tree" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_playwright_not_installed(self, monkeypatch) -> None:
        session = _FakeSession(
            page_error=ImportError("playwright not installed"),
        )
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute()
        assert result.success is False
        assert "playwright" in result.content.lower()

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_category_and_capabilities(self) -> None:
        tool = BrowserAXTreeTool()
        assert tool.spec.category == "browser"
        assert "network:fetch" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_has_max_depth_parameter(self) -> None:
        tool = BrowserAXTreeTool()
        props = tool.spec.parameters.get("properties", {})
        assert "max_depth" in props
        assert props["max_depth"]["type"] == "integer"

    @pytest.mark.spec("REQ-tools.browser")
    def test_to_openai_function(self) -> None:
        tool = BrowserAXTreeTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "browser_axtree"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_snapshot_error(self, monkeypatch) -> None:
        page = _FakePage(error=Exception("Browser crashed"))
        session = _FakeSession(page=page)
        monkeypatch.setattr(_axtree_mod, "_session", session)
        tool = BrowserAXTreeTool()
        result = tool.execute()
        assert result.success is False
        assert "AX tree extraction error" in result.content
