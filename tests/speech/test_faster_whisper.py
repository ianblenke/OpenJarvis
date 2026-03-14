"""Tests for Faster-Whisper speech backend."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech.faster_whisper import FasterWhisperBackend

_fw_mod = importlib.import_module("openjarvis.speech.faster_whisper")


# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeWhisperModel:
    """Typed fake for faster_whisper.WhisperModel."""

    def __init__(self, *args, **kwargs) -> None:
        self._segments = [
            SimpleNamespace(
                text=" Hello world", start=0.0, end=1.2, avg_logprob=-0.3,
            ),
        ]
        self._info = SimpleNamespace(
            language="en", language_probability=0.95, duration=1.5,
        )

    def transcribe(self, *args, **kwargs):
        return self._segments, self._info


@pytest.fixture(autouse=True)
def _register_faster_whisper():
    """Re-register after any registry clear."""
    if not SpeechRegistry.contains("faster-whisper"):
        SpeechRegistry.register_value("faster-whisper", FasterWhisperBackend)


@pytest.mark.spec("REQ-speech.faster-whisper")
def test_faster_whisper_backend_registers():
    """Backend registers itself in SpeechRegistry."""
    assert SpeechRegistry.contains("faster-whisper")


@pytest.mark.spec("REQ-speech.faster-whisper")
def test_faster_whisper_transcribe(monkeypatch):
    """Transcribe returns a TranscriptionResult."""
    from openjarvis.speech._stubs import TranscriptionResult

    monkeypatch.setattr(_fw_mod, "WhisperModel", _FakeWhisperModel)

    backend = FasterWhisperBackend(model_size="base", device="cpu")
    result = backend.transcribe(b"fake audio bytes")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello world"
    assert result.language == "en"
    assert result.duration_seconds == 1.5


@pytest.mark.spec("REQ-speech.faster-whisper")
def test_faster_whisper_health_no_model(monkeypatch):
    """Health returns False before model is loaded."""
    monkeypatch.setattr(_fw_mod, "WhisperModel", None)

    backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
    backend._model = None
    assert backend.health() is False


@pytest.mark.spec("REQ-speech.faster-whisper")
def test_faster_whisper_supported_formats(monkeypatch):
    """Backend supports standard audio formats."""
    monkeypatch.setattr(_fw_mod, "WhisperModel", _FakeWhisperModel)

    backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
    formats = backend.supported_formats()
    assert "wav" in formats
    assert "mp3" in formats
    assert "webm" in formats
