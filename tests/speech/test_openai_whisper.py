"""Tests for OpenAI Whisper API speech backend."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import TranscriptionResult
from openjarvis.speech.openai_whisper import OpenAIWhisperBackend

_whisper_mod = importlib.import_module("openjarvis.speech.openai_whisper")


# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeOpenAIAudioClient:
    """Typed fake for OpenAI client with audio.transcriptions.create()."""

    def __init__(
        self,
        text: str = "Hello from OpenAI",
        language: str = "en",
        duration: float = 2.0,
    ) -> None:
        self._response = SimpleNamespace(
            text=text, language=language, duration=duration,
        )
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        return self._response


@pytest.fixture(autouse=True)
def _register_openai_whisper():
    """Re-register after any registry clear."""
    if not SpeechRegistry.contains("openai"):
        SpeechRegistry.register_value("openai", OpenAIWhisperBackend)


@pytest.mark.spec("REQ-speech.openai-whisper")
def test_openai_whisper_registers():
    assert SpeechRegistry.contains("openai")


@pytest.mark.spec("REQ-speech.openai-whisper")
def test_openai_whisper_transcribe(monkeypatch):
    fake_client = _FakeOpenAIAudioClient()
    monkeypatch.setattr(
        _whisper_mod, "OpenAI", lambda **kwargs: fake_client,
    )
    backend = OpenAIWhisperBackend(api_key="test-key")
    result = backend.transcribe(b"fake audio", format="wav")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello from OpenAI"
    assert result.language == "en"


@pytest.mark.spec("REQ-speech.openai-whisper")
def test_openai_whisper_health(monkeypatch):
    monkeypatch.setattr(
        _whisper_mod, "OpenAI", lambda **kwargs: SimpleNamespace(),
    )
    backend = OpenAIWhisperBackend(api_key="test-key")
    assert backend.health() is True


@pytest.mark.spec("REQ-speech.openai-whisper")
def test_openai_whisper_health_no_key(monkeypatch):
    monkeypatch.setattr(
        _whisper_mod, "OpenAI", lambda **kwargs: SimpleNamespace(),
    )
    backend = OpenAIWhisperBackend.__new__(OpenAIWhisperBackend)
    backend._client = None
    backend._api_key = ""
    assert backend.health() is False
