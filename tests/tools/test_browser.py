"""Tests for browser automation tools."""

from __future__ import annotations

import base64
import builtins
from typing import Any, Dict, List, Optional

import pytest

from openjarvis.core.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Typed fakes replacing MagicMock for Playwright objects
# ---------------------------------------------------------------------------


class FakePage:
    """Typed fake for a Playwright Page object.

    Tracks method calls and returns configurable values.
    """

    def __init__(self) -> None:
        self._title: str = "Test Page"
        self._inner_text: str = "Hello World"
        self._screenshot_data: bytes = b"\x89PNG\x00fake-screenshot-data"
        self._goto_status: int = 200
        self._goto_error: Optional[Exception] = None
        self._click_error: Optional[Exception] = None
        self._fill_error: Optional[Exception] = None
        self._screenshot_error: Optional[Exception] = None
        self._inner_text_error: Optional[Exception] = None
        self._eval_results: list = []

        # Call tracking
        self.goto_calls: List[Dict[str, Any]] = []
        self.click_calls: List[str] = []
        self.fill_calls: List[tuple[str, str]] = []
        self.type_calls: List[tuple[str, str]] = []
        self.get_by_text_calls: List[str] = []
        self.inner_text_calls: List[str] = []
        self.screenshot_calls: List[Dict[str, Any]] = []

    def title(self) -> str:
        return self._title

    def inner_text(self, selector: str = "body") -> str:
        self.inner_text_calls.append(selector)
        if self._inner_text_error is not None:
            raise self._inner_text_error
        return self._inner_text

    def screenshot(self, **kwargs: Any) -> bytes:
        self.screenshot_calls.append(kwargs)
        if self._screenshot_error is not None:
            raise self._screenshot_error
        return self._screenshot_data

    def goto(self, url: str, **kwargs: Any) -> "FakeResponse":
        self.goto_calls.append({"url": url, **kwargs})
        if self._goto_error is not None:
            raise self._goto_error
        return FakeResponse(status=self._goto_status)

    def click(self, selector: str) -> None:
        self.click_calls.append(selector)
        if self._click_error is not None:
            raise self._click_error

    def get_by_text(self, text: str) -> "FakeLocator":
        self.get_by_text_calls.append(text)
        return FakeLocator()

    def fill(self, selector: str, value: str) -> None:
        self.fill_calls.append((selector, value))
        if self._fill_error is not None:
            raise self._fill_error

    def type(self, selector: str, text: str) -> None:
        self.type_calls.append((selector, text))

    def eval_on_selector_all(self, selector: str, expression: str) -> list:
        return self._eval_results


class FakeResponse:
    """Typed fake for Playwright Response."""

    def __init__(self, status: int = 200) -> None:
        self.status = status


class FakeLocator:
    """Typed fake for Playwright Locator (from get_by_text)."""

    def __init__(self) -> None:
        self.click_count = 0

    def click(self) -> None:
        self.click_count += 1


class FakeBrowserSession:
    """Typed fake for _BrowserSession.

    Replaces the MagicMock session pattern with a real object that
    exposes a configurable FakePage.
    """

    def __init__(self, page: Optional[FakePage] = None) -> None:
        self._page = page or FakePage()

    @property
    def page(self) -> FakePage:
        return self._page


class ImportErrorBrowserSession:
    """Fake session whose .page raises ImportError."""

    @property
    def page(self) -> FakePage:
        raise ImportError(
            "playwright not installed. Install with: uv sync --extra browser"
        )


class SsrfBlockingModule:
    """Typed fake for openjarvis.security.ssrf module."""

    def __init__(self, blocked_message: Optional[str] = None) -> None:
        self._blocked = blocked_message

    def check_ssrf(self, url: str) -> Optional[str]:
        return self._blocked


# ---------------------------------------------------------------------------
# TestBrowserNavigateTool
# ---------------------------------------------------------------------------


class TestBrowserNavigateTool:
    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_name_and_category(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        assert tool.spec.name == "browser_navigate"
        assert tool.spec.category == "browser"

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_requires_url_parameter(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        assert "url" in tool.spec.parameters["properties"]
        assert "url" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_has_wait_for_parameter(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        assert "wait_for" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_required_capabilities(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        assert "network:fetch" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.browser")
    def test_tool_id(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        assert tool.tool_id == "browser_navigate"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_url(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        result = tool.execute(url="")
        assert result.success is False
        assert "No URL" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_url_param(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        result = tool.execute()
        assert result.success is False
        assert "No URL" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_playwright_not_installed(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        session = ImportErrorBrowserSession()
        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserNavigateTool()
        result = tool.execute(url="https://example.com")
        assert result.success is False
        assert "playwright not installed" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_ssrf_blocked(self, monkeypatch):
        from openjarvis.tools.browser import BrowserNavigateTool

        ssrf_mod = SsrfBlockingModule(
            "Blocked host: 169.254.169.254 (cloud metadata endpoint)",
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "openjarvis.security.ssrf",
            ssrf_mod,
        )

        tool = BrowserNavigateTool()
        result = tool.execute(url="http://169.254.169.254/latest/meta-data/")

        assert result.success is False
        assert "SSRF blocked" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_ssrf_module_missing(self, monkeypatch):
        """When ssrf module is not available, skip check and proceed."""
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        page = FakePage()
        session = FakeBrowserSession(page)

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "openjarvis.security.ssrf":
                raise ImportError("No module named 'openjarvis.security.ssrf'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(browser_mod, "_session", session)
        monkeypatch.setattr(builtins, "__import__", _mock_import)
        tool = BrowserNavigateTool()
        result = tool.execute(url="https://example.com")
        # Should succeed since SSRF check is skipped
        assert result.success is True

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_success(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        page = FakePage()
        page._title = "Example Domain"
        page._inner_text = "Example page content"
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserNavigateTool()
        result = tool.execute(url="https://example.com")

        assert result.success is True
        assert "Example Domain" in result.content
        assert "Example page content" in result.content
        assert result.metadata["url"] == "https://example.com"
        assert result.metadata["title"] == "Example Domain"
        assert result.metadata["status"] == 200

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_with_wait_for(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserNavigateTool()
        tool.execute(url="https://example.com", wait_for="networkidle")

        assert len(page.goto_calls) == 1
        assert page.goto_calls[0]["url"] == "https://example.com"
        assert page.goto_calls[0]["wait_until"] == "networkidle"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_invalid_wait_for_defaults_to_load(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserNavigateTool()
        tool.execute(url="https://example.com", wait_for="invalid")

        assert page.goto_calls[0]["wait_until"] == "load"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_content_truncation(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        page = FakePage()
        page._inner_text = "x" * 6000
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserNavigateTool()
        result = tool.execute(url="https://example.com")

        assert result.success is True
        assert "[Content truncated]" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_navigation_error(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserNavigateTool

        page = FakePage()
        page._goto_error = Exception("net::ERR_NAME_NOT_RESOLVED")
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserNavigateTool()
        result = tool.execute(url="https://nonexistent.example")

        assert result.success is False
        assert "Navigation error" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_to_openai_function(self):
        from openjarvis.tools.browser import BrowserNavigateTool

        tool = BrowserNavigateTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "browser_navigate"
        assert "url" in fn["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# TestBrowserClickTool
# ---------------------------------------------------------------------------


class TestBrowserClickTool:
    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_name_and_category(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        assert tool.spec.name == "browser_click"
        assert tool.spec.category == "browser"

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_requires_selector(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        assert "selector" in tool.spec.parameters["properties"]
        assert "selector" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_has_by_text_parameter(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        assert "by_text" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.browser")
    def test_tool_id(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        assert tool.tool_id == "browser_click"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_selector(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        result = tool.execute(selector="")
        assert result.success is False
        assert "No selector" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_selector_param(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        result = tool.execute()
        assert result.success is False
        assert "No selector" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_playwright_not_installed(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserClickTool

        session = ImportErrorBrowserSession()
        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserClickTool()
        result = tool.execute(selector="#btn")
        assert result.success is False
        assert "playwright not installed" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_click_by_css(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserClickTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserClickTool()
        result = tool.execute(selector="#submit-btn")

        assert result.success is True
        assert "Clicked element" in result.content
        assert page.click_calls == ["#submit-btn"]
        assert result.metadata["selector"] == "#submit-btn"
        assert result.metadata["by_text"] is False

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_click_by_text(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserClickTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserClickTool()
        result = tool.execute(selector="Sign In", by_text=True)

        assert result.success is True
        assert page.get_by_text_calls == ["Sign In"]
        assert result.metadata["by_text"] is True

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_click_error(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserClickTool

        page = FakePage()
        page._click_error = Exception("Element not found")
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserClickTool()
        result = tool.execute(selector="#nonexistent")

        assert result.success is False
        assert "Click error" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_to_openai_function(self):
        from openjarvis.tools.browser import BrowserClickTool

        tool = BrowserClickTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "browser_click"


# ---------------------------------------------------------------------------
# TestBrowserTypeTool
# ---------------------------------------------------------------------------


class TestBrowserTypeTool:
    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_name_and_category(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        assert tool.spec.name == "browser_type"
        assert tool.spec.category == "browser"

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_requires_selector_and_text(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        assert "selector" in tool.spec.parameters["properties"]
        assert "text" in tool.spec.parameters["properties"]
        assert "selector" in tool.spec.parameters["required"]
        assert "text" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_has_clear_parameter(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        assert "clear" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.browser")
    def test_tool_id(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        assert tool.tool_id == "browser_type"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_selector(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        result = tool.execute(selector="", text="hello")
        assert result.success is False
        assert "No selector" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_text(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        result = tool.execute(selector="#input", text="")
        assert result.success is False
        assert "No text" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_no_params(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        result = tool.execute()
        assert result.success is False

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_playwright_not_installed(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserTypeTool

        session = ImportErrorBrowserSession()
        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserTypeTool()
        result = tool.execute(selector="#input", text="hello")
        assert result.success is False
        assert "playwright not installed" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_fill_clear_true(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserTypeTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserTypeTool()
        result = tool.execute(selector="#search", text="query", clear=True)

        assert result.success is True
        assert page.fill_calls == [("#search", "query")]
        assert page.type_calls == []
        assert result.metadata["selector"] == "#search"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_fill_default_clear(self, monkeypatch):
        """Default clear=True should use page.fill()."""
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserTypeTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserTypeTool()
        result = tool.execute(selector="#search", text="query")

        assert result.success is True
        assert page.fill_calls == [("#search", "query")]

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_type_clear_false(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserTypeTool

        page = FakePage()
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserTypeTool()
        result = tool.execute(selector="#search", text="query", clear=False)

        assert result.success is True
        assert page.type_calls == [("#search", "query")]
        assert page.fill_calls == []

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_type_error(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserTypeTool

        page = FakePage()
        page._fill_error = Exception("Element not editable")
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserTypeTool()
        result = tool.execute(selector="#readonly", text="hello")

        assert result.success is False
        assert "Type error" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_to_openai_function(self):
        from openjarvis.tools.browser import BrowserTypeTool

        tool = BrowserTypeTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "browser_type"


# ---------------------------------------------------------------------------
# TestBrowserScreenshotTool
# ---------------------------------------------------------------------------


class TestBrowserScreenshotTool:
    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_name_and_category(self):
        from openjarvis.tools.browser import BrowserScreenshotTool

        tool = BrowserScreenshotTool()
        assert tool.spec.name == "browser_screenshot"
        assert tool.spec.category == "browser"

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_has_path_and_full_page(self):
        from openjarvis.tools.browser import BrowserScreenshotTool

        tool = BrowserScreenshotTool()
        props = tool.spec.parameters["properties"]
        assert "path" in props
        assert "full_page" in props

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_no_required_params(self):
        from openjarvis.tools.browser import BrowserScreenshotTool

        tool = BrowserScreenshotTool()
        assert "required" not in tool.spec.parameters

    @pytest.mark.spec("REQ-tools.browser")
    def test_tool_id(self):
        from openjarvis.tools.browser import BrowserScreenshotTool

        tool = BrowserScreenshotTool()
        assert tool.tool_id == "browser_screenshot"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_playwright_not_installed(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserScreenshotTool

        session = ImportErrorBrowserSession()
        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserScreenshotTool()
        result = tool.execute()
        assert result.success is False
        assert "playwright not installed" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_screenshot_basic(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserScreenshotTool

        fake_png = b"\x89PNG\x00screenshot-data"
        page = FakePage()
        page._screenshot_data = fake_png
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserScreenshotTool()
        result = tool.execute()

        assert result.success is True
        assert "Screenshot taken" in result.content
        assert "full page" not in result.content
        expected_b64 = base64.b64encode(fake_png).decode("utf-8")
        assert result.metadata["screenshot_base64"] == expected_b64
        assert page.screenshot_calls == [{"full_page": False}]

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_screenshot_full_page(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserScreenshotTool

        page = FakePage()
        page._screenshot_data = b"png-data"
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserScreenshotTool()
        result = tool.execute(full_page=True)

        assert result.success is True
        assert "full page" in result.content
        assert page.screenshot_calls == [{"full_page": True}]

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_screenshot_save_to_file(self, tmp_path, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserScreenshotTool

        fake_png = b"\x89PNGscreenshot"
        page = FakePage()
        page._screenshot_data = fake_png
        session = FakeBrowserSession(page)

        save_path = str(tmp_path / "test_screenshot.png")

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserScreenshotTool()
        result = tool.execute(path=save_path)

        assert result.success is True
        assert save_path in result.content
        # Verify file was written
        with open(save_path, "rb") as f:
            assert f.read() == fake_png

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_screenshot_error(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserScreenshotTool

        page = FakePage()
        page._screenshot_error = Exception("Browser crashed")
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserScreenshotTool()
        result = tool.execute()

        assert result.success is False
        assert "Screenshot error" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_to_openai_function(self):
        from openjarvis.tools.browser import BrowserScreenshotTool

        tool = BrowserScreenshotTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "browser_screenshot"


# ---------------------------------------------------------------------------
# TestBrowserExtractTool
# ---------------------------------------------------------------------------


class TestBrowserExtractTool:
    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_name_and_category(self):
        from openjarvis.tools.browser import BrowserExtractTool

        tool = BrowserExtractTool()
        assert tool.spec.name == "browser_extract"
        assert tool.spec.category == "browser"

    @pytest.mark.spec("REQ-tools.browser")
    def test_spec_has_selector_and_extract_type(self):
        from openjarvis.tools.browser import BrowserExtractTool

        tool = BrowserExtractTool()
        props = tool.spec.parameters["properties"]
        assert "selector" in props
        assert "extract_type" in props

    @pytest.mark.spec("REQ-tools.browser")
    def test_tool_id(self):
        from openjarvis.tools.browser import BrowserExtractTool

        tool = BrowserExtractTool()
        assert tool.tool_id == "browser_extract"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_playwright_not_installed(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        session = ImportErrorBrowserSession()
        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute()
        assert result.success is False
        assert "playwright not installed" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_invalid_extract_type(self):
        from openjarvis.tools.browser import BrowserExtractTool

        tool = BrowserExtractTool()
        result = tool.execute(extract_type="images")
        assert result.success is False
        assert "Invalid extract_type" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_text(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._inner_text = "Page text content here"
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(extract_type="text")

        assert result.success is True
        assert result.content == "Page text content here"
        assert page.inner_text_calls == ["body"]
        assert result.metadata["extract_type"] == "text"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_text_custom_selector(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._inner_text = "Article content"
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(selector="#article", extract_type="text")

        assert result.success is True
        assert page.inner_text_calls == ["#article"]
        assert result.metadata["selector"] == "#article"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_text_default(self, monkeypatch):
        """Default extract_type should be 'text'."""
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._inner_text = "Default text"
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute()

        assert result.success is True
        assert result.content == "Default text"

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_text_truncation(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._inner_text = "a" * 12000
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(extract_type="text")

        assert result.success is True
        assert "[Content truncated]" in result.content
        # Content should be truncated at 10000 + truncation notice
        assert len(result.content) < 11000

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_links(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._eval_results = [
            {"href": "https://example.com/page1", "text": "Page 1"},
            {"href": "https://example.com/page2", "text": "Page 2"},
        ]
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(extract_type="links")

        assert result.success is True
        assert "[Page 1](https://example.com/page1)" in result.content
        assert "[Page 2](https://example.com/page2)" in result.content
        assert result.metadata["num_links"] == 2

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_links_empty(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._eval_results = []
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(extract_type="links")

        assert result.success is True
        assert result.content == "No links found."
        assert result.metadata["num_links"] == 0

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_tables(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._eval_results = [
            "Name\tAge\nAlice\t30",
            "City\tCountry\nNYC\tUSA",
        ]
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(extract_type="tables")

        assert result.success is True
        assert "Alice" in result.content
        assert "NYC" in result.content
        assert result.metadata["num_tables"] == 2

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_tables_empty(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._eval_results = []
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(extract_type="tables")

        assert result.success is True
        assert result.content == "No tables found."

    @pytest.mark.spec("REQ-tools.browser")
    def test_execute_extract_error(self, monkeypatch):
        import openjarvis.tools.browser as browser_mod
        from openjarvis.tools.browser import BrowserExtractTool

        page = FakePage()
        page._inner_text_error = Exception("Selector not found")
        session = FakeBrowserSession(page)

        monkeypatch.setattr(browser_mod, "_session", session)
        tool = BrowserExtractTool()
        result = tool.execute(selector="#missing", extract_type="text")

        assert result.success is False
        assert "Extract error" in result.content

    @pytest.mark.spec("REQ-tools.browser")
    def test_to_openai_function(self):
        from openjarvis.tools.browser import BrowserExtractTool

        tool = BrowserExtractTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "browser_extract"


# ---------------------------------------------------------------------------
# TestBrowserSession
# ---------------------------------------------------------------------------


class TestBrowserSession:
    @pytest.mark.spec("REQ-tools.browser")
    def test_session_close_resets_state(self):
        from unittest.mock import (
            # MOCK-JUSTIFIED: Playwright lifecycle state reset
            MagicMock,
        )

        from openjarvis.tools.browser import _BrowserSession

        session = _BrowserSession()
        session._playwright = MagicMock()
        session._browser = MagicMock()
        session._page = MagicMock()

        session.close()

        assert session._playwright is None
        assert session._browser is None
        assert session._page is None

    @pytest.mark.spec("REQ-tools.browser")
    def test_session_close_noop_when_not_initialized(self):
        from openjarvis.tools.browser import _BrowserSession

        session = _BrowserSession()
        # Should not raise
        session.close()
        assert session._playwright is None

    @pytest.mark.spec("REQ-tools.browser")
    def test_session_ensure_browser_import_error(self, monkeypatch):
        from openjarvis.tools.browser import _BrowserSession

        session = _BrowserSession()

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if "playwright" in name:
                raise ImportError("No module named 'playwright'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        with pytest.raises(ImportError, match="playwright not installed"):
            session._ensure_browser()

    @pytest.mark.spec("REQ-tools.browser")
    def test_session_page_reuses_existing(self):
        from openjarvis.tools.browser import _BrowserSession

        session = _BrowserSession()
        fake_page = FakePage()
        session._page = fake_page

        # _ensure_browser should not re-create if page exists
        session._ensure_browser()
        assert session._page is fake_page


# ---------------------------------------------------------------------------
# TestRegistryIntegration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    @pytest.mark.spec("REQ-tools.base.registration")
    def test_all_tools_registered(self):
        # Registration happens at import time via @ToolRegistry.register.
        # Other test modules may clear the registry, so re-register if needed.
        from openjarvis.tools.browser import (
            BrowserClickTool,
            BrowserExtractTool,
            BrowserNavigateTool,
            BrowserScreenshotTool,
            BrowserTypeTool,
        )

        tools = {
            "browser_navigate": BrowserNavigateTool,
            "browser_click": BrowserClickTool,
            "browser_type": BrowserTypeTool,
            "browser_screenshot": BrowserScreenshotTool,
            "browser_extract": BrowserExtractTool,
        }
        for key, cls in tools.items():
            if not ToolRegistry.contains(key):
                ToolRegistry.register_value(key, cls)

        assert ToolRegistry.contains("browser_navigate")
        assert ToolRegistry.contains("browser_click")
        assert ToolRegistry.contains("browser_type")
        assert ToolRegistry.contains("browser_screenshot")
        assert ToolRegistry.contains("browser_extract")

    @pytest.mark.spec("REQ-tools.base.registration")
    def test_module_exports(self):
        from openjarvis.tools.browser import __all__

        assert "BrowserNavigateTool" in __all__
        assert "BrowserClickTool" in __all__
        assert "BrowserTypeTool" in __all__
        assert "BrowserScreenshotTool" in __all__
        assert "BrowserExtractTool" in __all__
