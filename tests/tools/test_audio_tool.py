"""Tests for the audio_transcribe tool."""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from typing import Any, List

import pytest

from openjarvis.tools.audio_tool import AudioTranscribeTool

# ---------------------------------------------------------------------------
# Typed fakes for openai module boundary
# ---------------------------------------------------------------------------


@dataclass
class FakeTranscription:
    """Typed fake for openai transcription response."""
    text: str
    duration: float | None = None


class FakeTranscriptionNoDuration:
    """Typed fake for openai transcription response without duration attr."""

    def __init__(self, text: str) -> None:
        self.text = text
        # Intentionally no 'duration' attribute


class FakeTranscriptionsAPI:
    """Typed fake for openai client.audio.transcriptions namespace."""

    def __init__(
        self,
        *,
        transcription: FakeTranscription | FakeTranscriptionNoDuration | None = None,
        error: Exception | None = None,
    ) -> None:
        self._transcription = transcription or FakeTranscription(
            text="Default transcription.", duration=1.0,
        )
        self._error = error
        self.create_calls: List[dict] = []

    def create(self, **kwargs: Any) -> FakeTranscription | FakeTranscriptionNoDuration:
        self.create_calls.append(kwargs)
        if self._error:
            raise self._error
        return self._transcription


class FakeAudioAPI:
    """Typed fake for openai client.audio namespace."""

    def __init__(self, transcriptions: FakeTranscriptionsAPI | None = None) -> None:
        self.transcriptions = transcriptions or FakeTranscriptionsAPI()


class FakeOpenAIClient:
    """Typed fake for openai.OpenAI() client."""

    def __init__(self, audio: FakeAudioAPI | None = None) -> None:
        self.audio = audio or FakeAudioAPI()


class FakeOpenAIModule:
    """Typed fake for the 'openai' module."""

    def __init__(self, client: FakeOpenAIClient | None = None) -> None:
        self._client = client or FakeOpenAIClient()

    def OpenAI(self, **kwargs: Any) -> FakeOpenAIClient:
        return self._client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioTranscribeTool:
    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_spec(self):
        tool = AudioTranscribeTool()
        assert tool.spec.name == "audio_transcribe"
        assert tool.spec.category == "media"
        assert "file_path" in tool.spec.parameters["properties"]
        assert "file_path" in tool.spec.parameters["required"]
        assert tool.spec.required_capabilities == ["file:read"]

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_tool_id(self):
        tool = AudioTranscribeTool()
        assert tool.tool_id == "audio_transcribe"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_file_path(self):
        tool = AudioTranscribeTool()
        result = tool.execute(file_path="")
        assert result.success is False
        assert "No file_path" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_file_path_param(self):
        tool = AudioTranscribeTool()
        result = tool.execute()
        assert result.success is False
        assert "No file_path" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_file_not_found(self):
        tool = AudioTranscribeTool()
        result = tool.execute(file_path="/nonexistent/audio.mp3")
        assert result.success is False
        assert "File not found" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "audio.xyz"
        f.write_text("not audio", encoding="utf-8")
        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f))
        assert result.success is False
        assert "Unsupported audio format" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_file_too_large(self, tmp_path, monkeypatch):
        f = tmp_path / "large.mp3"
        # Create a small file, then intercept stat on just this path
        f.write_bytes(b"\x00" * 1024)

        tool = AudioTranscribeTool()

        # Patch os.stat (used internally by pathlib.Path.stat) to report
        # a large file size only for our specific test file

        class FakeStatResult:
            """Typed fake stat result that reports 26 MB."""
            st_size = 26 * 1024 * 1024
            st_mode = 0o100644
            st_ino = 0
            st_dev = 0
            st_nlink = 1
            st_uid = 0
            st_gid = 0
            st_atime = 0.0
            st_mtime = 0.0
            st_ctime = 0.0

        original_stat = type(f).stat

        def _patched_stat(self, **kwargs):
            if str(self) == str(f):
                return FakeStatResult()
            return original_stat(self, **kwargs)

        monkeypatch.setattr(type(f), "stat", _patched_stat)

        result = tool.execute(file_path=str(f))
        assert result.success is False
        assert "File too large" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_local_provider_not_implemented(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.write_bytes(b"\x00" * 100)
        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f), provider="local")
        assert result.success is False
        assert "not yet implemented" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_unsupported_provider(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)
        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f), provider="google")
        assert result.success is False
        assert "Unsupported provider" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_openai_not_installed(self, tmp_path, monkeypatch):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)

        monkeypatch.delitem(sys.modules, "openai", raising=False)
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f))
        assert result.success is False
        assert "openai package not installed" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_no_api_key(self, tmp_path, monkeypatch):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fake_openai = FakeOpenAIModule()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f))
        assert result.success is False
        assert "No API key" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_successful_transcription(self, tmp_path, monkeypatch):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        transcription = FakeTranscription(
            text="Hello, this is a transcription.",
            duration=5.5,
        )
        transcriptions_api = FakeTranscriptionsAPI(transcription=transcription)
        audio_api = FakeAudioAPI(transcriptions=transcriptions_api)
        client = FakeOpenAIClient(audio=audio_api)
        fake_openai = FakeOpenAIModule(client=client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f))
        assert result.success is True
        assert result.content == "Hello, this is a transcription."
        assert result.metadata["provider"] == "openai"
        assert result.metadata["duration_ms"] == 5500

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_successful_transcription_with_language(self, tmp_path, monkeypatch):
        f = tmp_path / "audio.wav"
        f.write_bytes(b"\x00" * 100)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        transcription = FakeTranscriptionNoDuration(text="Hola mundo.")
        transcriptions_api = FakeTranscriptionsAPI(transcription=transcription)
        audio_api = FakeAudioAPI(transcriptions=transcriptions_api)
        client = FakeOpenAIClient(audio=audio_api)
        fake_openai = FakeOpenAIModule(client=client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f), language="es")
        assert result.success is True
        assert result.content == "Hola mundo."
        assert result.metadata["language"] == "es"

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_api_error(self, tmp_path, monkeypatch):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        transcriptions_api = FakeTranscriptionsAPI(
            error=RuntimeError("API error"),
        )
        audio_api = FakeAudioAPI(transcriptions=transcriptions_api)
        client = FakeOpenAIClient(audio=audio_api)
        fake_openai = FakeOpenAIModule(client=client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        tool = AudioTranscribeTool()
        result = tool.execute(file_path=str(f))
        assert result.success is False
        assert "Transcription error" in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_supported_formats_accepted(self, tmp_path):
        """All supported formats pass the format check (fail later due to no API)."""
        tool = AudioTranscribeTool()
        for ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"]:
            f = tmp_path / f"audio{ext}"
            f.write_bytes(b"\x00" * 100)
            result = tool.execute(file_path=str(f))
            # Should not fail on format -- will fail on API/import instead
            assert "Unsupported audio format" not in result.content

    @pytest.mark.spec("REQ-tools.base.protocol")
    def test_to_openai_function(self):
        tool = AudioTranscribeTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "audio_transcribe"
