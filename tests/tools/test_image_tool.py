"""Tests for the image_generate tool."""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from typing import Any, List

import pytest

from openjarvis.tools.image_tool import ImageGenerateTool

# ---------------------------------------------------------------------------
# Typed fakes for the openai module boundary
# ---------------------------------------------------------------------------


@dataclass
class FakeImageData:
    """Typed fake for openai image response data item."""
    url: str


@dataclass
class FakeImageResponse:
    """Typed fake for openai images.generate() response."""
    data: List[FakeImageData]


class FakeImagesAPI:
    """Typed fake for openai client.images namespace."""

    def __init__(
        self,
        *,
        url: str = "https://example.com/image.png",
        error: Exception | None = None,
    ) -> None:
        self._url = url
        self._error = error
        self.generate_calls: List[dict] = []

    def generate(self, **kwargs: Any) -> FakeImageResponse:
        self.generate_calls.append(kwargs)
        if self._error:
            raise self._error
        return FakeImageResponse(data=[FakeImageData(url=self._url)])


class FakeOpenAIClient:
    """Typed fake for openai.OpenAI() client."""

    def __init__(self, images: FakeImagesAPI | None = None) -> None:
        self.images = images or FakeImagesAPI()


class FakeOpenAIModule:
    """Typed fake for the 'openai' module."""

    def __init__(self, client: FakeOpenAIClient | None = None) -> None:
        self._client = client or FakeOpenAIClient()

    def OpenAI(self, **kwargs: Any) -> FakeOpenAIClient:
        return self._client


@dataclass
class FakeHttpResponse:
    """Typed fake for httpx.get() response."""
    content: bytes
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageGenerateTool:
    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_spec(self):
        tool = ImageGenerateTool()
        assert tool.spec.name == "image_generate"
        assert tool.spec.category == "media"
        assert "prompt" in tool.spec.parameters["properties"]
        assert "prompt" in tool.spec.parameters["required"]
        assert tool.spec.required_capabilities == ["network:fetch"]

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_tool_id(self):
        tool = ImageGenerateTool()
        assert tool.tool_id == "image_generate"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_prompt(self):
        tool = ImageGenerateTool()
        result = tool.execute(prompt="")
        assert result.success is False
        assert "No prompt" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_prompt_param(self):
        tool = ImageGenerateTool()
        result = tool.execute()
        assert result.success is False
        assert "No prompt" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_invalid_size(self):
        tool = ImageGenerateTool()
        result = tool.execute(prompt="a cat", size="999x999")
        assert result.success is False
        assert "Invalid size" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_unsupported_provider(self):
        tool = ImageGenerateTool()
        result = tool.execute(prompt="a cat", provider="midjourney")
        assert result.success is False
        assert "Unsupported provider" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_openai_not_installed(self, monkeypatch):
        """Simulate openai package not being installed."""
        monkeypatch.delitem(sys.modules, "openai", raising=False)
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = ImageGenerateTool()
        result = tool.execute(prompt="a cat")
        assert result.success is False
        assert "openai package not installed" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake_openai = FakeOpenAIModule()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = ImageGenerateTool()
        result = tool.execute(prompt="a cat")
        assert result.success is False
        assert "No API key" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_successful_generation(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        fake_openai = FakeOpenAIModule()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = ImageGenerateTool()
        result = tool.execute(prompt="a cat on a mat")
        assert result.success is True
        assert result.content == "https://example.com/image.png"
        assert result.metadata["url"] == "https://example.com/image.png"
        assert result.metadata["size"] == "1024x1024"
        assert result.metadata["provider"] == "openai"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_save_to_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        fake_openai = FakeOpenAIModule()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        # Fake httpx.get for downloading
        import httpx

        fake_content = b"\x89PNG\r\n\x1a\nfake-image-data"
        fake_resp = FakeHttpResponse(content=fake_content)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_resp)

        output_file = tmp_path / "output.png"
        tool = ImageGenerateTool()
        result = tool.execute(
            prompt="a cat",
            output_path=str(output_file),
        )
        assert result.success is True
        assert output_file.exists()
        assert output_file.read_bytes() == fake_content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_api_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        images_api = FakeImagesAPI(error=RuntimeError("Rate limit exceeded"))
        client = FakeOpenAIClient(images=images_api)
        fake_openai = FakeOpenAIModule(client=client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = ImageGenerateTool()
        result = tool.execute(prompt="a cat")
        assert result.success is False
        assert "Image generation error" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_to_openai_function(self):
        tool = ImageGenerateTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "image_generate"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_save_to_file_download_failure(self, monkeypatch, tmp_path):
        """Exercise lines 137-138: image generated but download/save fails."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        fake_openai = FakeOpenAIModule()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        import httpx

        def _raise_on_get(*a, **kw):
            raise ConnectionError("Download failed")

        monkeypatch.setattr(httpx, "get", _raise_on_get)

        output_file = tmp_path / "output.png"
        tool = ImageGenerateTool()
        result = tool.execute(
            prompt="a cat",
            output_path=str(output_file),
        )
        assert result.success is False
        assert "failed to save" in result.content.lower()
        assert "URL:" in result.content
        assert result.metadata["url"] == "https://example.com/image.png"
