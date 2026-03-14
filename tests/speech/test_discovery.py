"""Tests for speech backend auto-discovery -- no MagicMock.

Uses monkeypatch for env vars and real registry manipulation.
"""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import SpeechRegistry
from openjarvis.speech._discovery import (
    DISCOVERY_ORDER,
    _create_backend,
    get_speech_backend,
)
from openjarvis.speech._stubs import SpeechBackend, TranscriptionResult

# ---------------------------------------------------------------------------
# Fake speech backend for testing (no MagicMock)
# ---------------------------------------------------------------------------


class FakeSpeechBackend(SpeechBackend):
    """Minimal concrete SpeechBackend for testing."""

    backend_id = "fake-speech"

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def transcribe(self, audio, *, format="wav", language=None):
        return TranscriptionResult(text="fake transcription")

    def health(self):
        return True

    def supported_formats(self):
        return ["wav", "mp3"]


class FakeCloudBackend(SpeechBackend):
    """Cloud backend requiring an api_key constructor arg."""

    backend_id = "fake-cloud"

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def transcribe(self, audio, *, format="wav", language=None):
        return TranscriptionResult(text="cloud transcription")

    def health(self):
        return bool(self._api_key)

    def supported_formats(self):
        return ["wav"]


# ---------------------------------------------------------------------------
# DISCOVERY_ORDER
# ---------------------------------------------------------------------------


class TestDiscoveryOrder:
    @pytest.mark.spec("REQ-speech.discovery")
    @pytest.mark.spec("REQ-speech.discovery-order-local-first")
    def test_local_backends_before_cloud(self) -> None:
        """faster-whisper (local) should be tried before cloud backends."""
        assert DISCOVERY_ORDER[0] == "faster-whisper"
        fw_idx = DISCOVERY_ORDER.index("faster-whisper")
        assert fw_idx < DISCOVERY_ORDER.index("openai")
        assert fw_idx < DISCOVERY_ORDER.index("deepgram")

    @pytest.mark.spec("REQ-speech.discovery-order-contains-all")
    def test_all_expected_backends_listed(self) -> None:
        assert "faster-whisper" in DISCOVERY_ORDER
        assert "openai" in DISCOVERY_ORDER
        assert "deepgram" in DISCOVERY_ORDER


# ---------------------------------------------------------------------------
# _create_backend
# ---------------------------------------------------------------------------


class TestCreateBackend:
    @pytest.mark.spec("REQ-speech.discovery-create-unregistered")
    def test_returns_none_for_unregistered_key(self) -> None:
        """Unregistered key returns None."""
        config = JarvisConfig()
        result = _create_backend("nonexistent-backend", config)
        assert result is None

    @pytest.mark.spec("REQ-speech.discovery-create-registered")
    @pytest.mark.spec("REQ-speech.protocol.registration")
    def test_returns_backend_for_registered_key(self) -> None:
        """Registered key returns an instance."""
        SpeechRegistry.register_value("test-backend", FakeSpeechBackend)
        config = JarvisConfig()
        result = _create_backend("test-backend", config)
        assert result is not None
        assert isinstance(result, FakeSpeechBackend)

    @pytest.mark.spec("REQ-speech.discovery-create-faster-whisper")
    def test_faster_whisper_passes_config(self) -> None:
        """faster-whisper backend receives model, device, compute_type from config."""

        class CapturingBackend(SpeechBackend):
            backend_id = "capture"

            def __init__(
                self,
                model_size="base",
                device="auto",
                compute_type="float16",
            ):
                self.model_size = model_size
                self.device = device
                self.compute_type = compute_type

            def transcribe(self, audio, *, format="wav", language=None):
                return TranscriptionResult(text="")

            def health(self):
                return True

            def supported_formats(self):
                return ["wav"]

        SpeechRegistry.register_value("faster-whisper", CapturingBackend)
        config = JarvisConfig()
        config.speech.model = "large-v3"
        config.speech.device = "cuda"
        config.speech.compute_type = "int8"
        result = _create_backend("faster-whisper", config)
        assert result is not None
        assert result.model_size == "large-v3"
        assert result.device == "cuda"
        assert result.compute_type == "int8"

    @pytest.mark.spec("REQ-speech.discovery-create-openai-needs-key")
    def test_openai_backend_requires_api_key(self, monkeypatch) -> None:
        """OpenAI backend returns None without OPENAI_API_KEY."""
        SpeechRegistry.register_value("openai", FakeCloudBackend)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = JarvisConfig()
        result = _create_backend("openai", config)
        assert result is None

    @pytest.mark.spec("REQ-speech.discovery-create-openai-with-key")
    def test_openai_backend_with_api_key(self, monkeypatch) -> None:
        """OpenAI backend returns instance when OPENAI_API_KEY is set."""
        SpeechRegistry.register_value("openai", FakeCloudBackend)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        config = JarvisConfig()
        result = _create_backend("openai", config)
        assert result is not None

    @pytest.mark.spec("REQ-speech.discovery-create-deepgram-needs-key")
    def test_deepgram_backend_requires_api_key(self, monkeypatch) -> None:
        """Deepgram backend returns None without DEEPGRAM_API_KEY."""
        SpeechRegistry.register_value("deepgram", FakeCloudBackend)
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        config = JarvisConfig()
        result = _create_backend("deepgram", config)
        assert result is None

    @pytest.mark.spec("REQ-speech.discovery-create-deepgram-with-key")
    def test_deepgram_backend_with_api_key(self, monkeypatch) -> None:
        """Deepgram backend returns instance when DEEPGRAM_API_KEY is set."""
        SpeechRegistry.register_value("deepgram", FakeCloudBackend)
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-456")
        config = JarvisConfig()
        result = _create_backend("deepgram", config)
        assert result is not None

    @pytest.mark.spec("REQ-speech.discovery-create-exception-returns-none")
    def test_exception_during_creation_returns_none(self) -> None:
        """If backend constructor raises, _create_backend returns None."""

        def bad_constructor(**kwargs):
            raise RuntimeError("boom")

        SpeechRegistry.register_value("broken-backend", bad_constructor)
        config = JarvisConfig()
        result = _create_backend("broken-backend", config)
        assert result is None


# ---------------------------------------------------------------------------
# get_speech_backend
# ---------------------------------------------------------------------------


class TestGetSpeechBackend:
    @pytest.mark.spec("REQ-speech.discovery-explicit-backend")
    def test_explicit_backend_selection(self) -> None:
        """Explicit backend='some-key' calls _create_backend with that key."""
        SpeechRegistry.register_value("my-stt", FakeSpeechBackend)
        config = JarvisConfig()
        config.speech.backend = "my-stt"
        result = get_speech_backend(config)
        assert result is not None
        assert isinstance(result, FakeSpeechBackend)

    @pytest.mark.spec("REQ-speech.discovery-explicit-unavailable")
    def test_explicit_backend_not_registered(self) -> None:
        """Explicit backend that is not registered returns None."""
        config = JarvisConfig()
        config.speech.backend = "totally-missing"
        result = get_speech_backend(config)
        assert result is None

    @pytest.mark.spec("REQ-speech.discovery-auto-priority")
    def test_auto_discovery_tries_in_order(self, monkeypatch) -> None:
        """Auto mode tries backends in DISCOVERY_ORDER and returns first success."""
        # Register only "deepgram" so faster-whisper and openai fail
        SpeechRegistry.register_value("deepgram", FakeCloudBackend)
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy")
        config = JarvisConfig()
        config.speech.backend = "auto"
        result = get_speech_backend(config)
        assert result is not None

    @pytest.mark.spec("REQ-speech.discovery-auto-none")
    def test_auto_returns_none_when_nothing_available(self) -> None:
        """Auto mode returns None when no backend can be created."""
        config = JarvisConfig()
        config.speech.backend = "auto"
        result = get_speech_backend(config)
        assert result is None
