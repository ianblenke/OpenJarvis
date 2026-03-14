"""Tests for speech integration in SystemBuilder/JarvisSystem."""

import pytest

from openjarvis.system import JarvisSystem


@pytest.mark.spec("REQ-speech.protocol.transcribe")
def test_jarvis_system_has_speech_backend():
    """JarvisSystem has a speech_backend attribute."""
    assert "speech_backend" in JarvisSystem.__dataclass_fields__
