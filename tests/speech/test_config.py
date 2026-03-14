"""Tests for speech configuration."""

import pytest

from openjarvis.core.config import JarvisConfig, SpeechConfig


@pytest.mark.spec("REQ-speech.protocol.transcribe")
def test_speech_config_defaults():
    cfg = SpeechConfig()
    assert cfg.backend == "auto"
    assert cfg.model == "base"
    assert cfg.language == ""
    assert cfg.device == "auto"
    assert cfg.compute_type == "float16"


@pytest.mark.spec("REQ-speech.protocol.transcribe")
def test_jarvis_config_has_speech():
    cfg = JarvisConfig()
    assert hasattr(cfg, "speech")
    assert isinstance(cfg.speech, SpeechConfig)
    assert cfg.speech.backend == "auto"
