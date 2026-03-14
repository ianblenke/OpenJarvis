"""Tests for speech API endpoints."""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.speech._stubs import TranscriptionResult  # noqa: E402


class _FakeSpeechBackend:
    """Typed fake speech backend implementing the protocol used by routes."""

    backend_id = "mock"

    def health(self) -> bool:
        return True

    def transcribe(self, audio_data: bytes, **kwargs) -> TranscriptionResult:
        return TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration_seconds=1.5,
            segments=[],
        )


@pytest.fixture
def fake_speech_backend():
    return _FakeSpeechBackend()


@pytest.fixture
def app_with_speech(fake_speech_backend):
    from fastapi import FastAPI

    from openjarvis.server.api_routes import speech_router

    app = FastAPI()
    app.state.speech_backend = fake_speech_backend
    app.include_router(speech_router)
    return app


@pytest.fixture
def client(app_with_speech):
    return TestClient(app_with_speech)


@pytest.mark.spec("REQ-server.routes.speech")
def test_transcribe_endpoint(client, fake_speech_backend):
    response = client.post(
        "/v1/speech/transcribe",
        files={"file": ("test.wav", b"fake audio data", "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["language"] == "en"
    assert data["confidence"] == 0.95
    assert data["duration_seconds"] == 1.5


@pytest.mark.spec("REQ-server.routes.speech")
def test_transcribe_no_file(client):
    response = client.post("/v1/speech/transcribe")
    assert response.status_code == 400 or response.status_code == 422


@pytest.mark.spec("REQ-server.routes.speech")
def test_health_endpoint(client):
    response = client.get("/v1/speech/health")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["backend"] == "mock"


@pytest.mark.spec("REQ-server.routes.speech")
def test_health_no_backend():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openjarvis.server.api_routes import speech_router

    app = FastAPI()
    app.state.speech_backend = None
    app.include_router(speech_router)
    client = TestClient(app)

    response = client.get("/v1/speech/health")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
