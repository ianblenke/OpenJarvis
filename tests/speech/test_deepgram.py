"""Tests for Deepgram speech backend."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._stubs import TranscriptionResult
from openjarvis.speech.deepgram import DeepgramSpeechBackend

_dg_mod = importlib.import_module("openjarvis.speech.deepgram")


# ---------------------------------------------------------------------------
# Typed fakes (replacing MagicMock)
# ---------------------------------------------------------------------------


class _FakeDeepgramClient:
    """Typed fake for DeepgramClient with listen.rest.v().transcribe_file()."""

    def __init__(self) -> None:
        alternative = SimpleNamespace(
            transcript="Hello from Deepgram", confidence=0.92,
        )
        channel = SimpleNamespace(
            alternatives=[alternative], detected_language="en",
        )
        result = SimpleNamespace(
            results=SimpleNamespace(channels=[channel]),
            metadata=SimpleNamespace(duration=1.8),
        )
        transcriber = SimpleNamespace(
            transcribe_file=lambda *a, **kw: result,
        )
        self.listen = SimpleNamespace(
            rest=SimpleNamespace(
                v=lambda version: transcriber,
            ),
        )


@pytest.fixture(autouse=True)
def _register_deepgram():
    """Re-register after any registry clear."""
    if not SpeechRegistry.contains("deepgram"):
        SpeechRegistry.register_value("deepgram", DeepgramSpeechBackend)


@pytest.mark.spec("REQ-speech.deepgram")
def test_deepgram_registers():
    assert SpeechRegistry.contains("deepgram")


@pytest.mark.spec("REQ-speech.deepgram")
def test_deepgram_transcribe(monkeypatch):
    monkeypatch.setattr(
        _dg_mod, "DeepgramClient", lambda *a, **kw: _FakeDeepgramClient(),
    )
    backend = DeepgramSpeechBackend(api_key="test-key")
    result = backend.transcribe(b"fake audio", format="wav")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello from Deepgram"


@pytest.mark.spec("REQ-speech.deepgram")
def test_deepgram_health(monkeypatch):
    monkeypatch.setattr(
        _dg_mod, "DeepgramClient", lambda *a, **kw: SimpleNamespace(),
    )
    backend = DeepgramSpeechBackend(api_key="test-key")
    assert backend.health() is True


@pytest.mark.spec("REQ-speech.deepgram")
def test_deepgram_health_no_key(monkeypatch):
    monkeypatch.setattr(
        _dg_mod, "DeepgramClient", lambda *a, **kw: SimpleNamespace(),
    )
    backend = DeepgramSpeechBackend.__new__(DeepgramSpeechBackend)
    backend._client = None
    backend._api_key = ""
    assert backend.health() is False
