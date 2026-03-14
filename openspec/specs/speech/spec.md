# Speech Module Spec

Speech-to-text abstraction supporting multiple transcription backends.

## SpeechBackend Protocol (`_stubs.py`)

### REQ-speech.protocol.transcribe: Transcription
`SpeechBackend.transcribe(audio: bytes, *, format="wav", language=None) -> TranscriptionResult`.

### REQ-speech.protocol.health: Health check
`health() -> bool` checks backend availability.

### REQ-speech.protocol.formats: Supported formats
`supported_formats() -> List[str]` returns supported audio formats.

### REQ-speech.protocol.registration: Registry-based registration
All backends use `@SpeechRegistry.register("name")` decorator.

## TranscriptionResult

### REQ-speech.result: Transcription result
`TranscriptionResult` with `text`, `language`, `confidence`, `duration_seconds`, `segments: List[Segment]`.

## Implementations

### REQ-speech.faster-whisper: Faster Whisper
Local transcription via `faster-whisper` library.

### REQ-speech.openai-whisper: OpenAI Whisper API
Cloud transcription via OpenAI Whisper API.

### REQ-speech.deepgram: Deepgram
Cloud transcription via Deepgram SDK.

## Discovery

### REQ-speech.discovery: Backend auto-discovery
Auto-detect available speech backends based on installed dependencies and configuration.

## Discovery (detailed)

### REQ-speech.discovery-order-local-first: Discovery prioritizes local backends
Backend auto-discovery prioritizes local backends (faster-whisper) over cloud backends (OpenAI, Deepgram).

### REQ-speech.discovery-order-contains-all: Discovery includes all backends
Discovery order contains all registered speech backends.

### REQ-speech.discovery-create-unregistered: Create unregistered backend returns None
`create_backend()` returns None for unregistered/unknown backend keys.

### REQ-speech.discovery-create-registered: Create registered backend
`create_backend()` returns a backend instance for registered backend keys.

### REQ-speech.discovery-create-faster-whisper: Create faster-whisper backend
`create_backend("faster_whisper")` creates a FasterWhisperBackend with config-based model settings.

### REQ-speech.discovery-create-openai-needs-key: OpenAI backend requires API key
`create_backend("openai")` returns None when no OpenAI API key is configured.

### REQ-speech.discovery-create-openai-with-key: OpenAI backend with API key
`create_backend("openai")` creates an OpenAIWhisperBackend when an API key is available.

### REQ-speech.discovery-create-deepgram-needs-key: Deepgram backend requires API key
`create_backend("deepgram")` returns None when no Deepgram API key is configured.

### REQ-speech.discovery-create-deepgram-with-key: Deepgram backend with API key
`create_backend("deepgram")` creates a DeepgramBackend when an API key is available.

### REQ-speech.discovery-create-exception-returns-none: Backend creation exception returns None
`create_backend()` returns None when backend instantiation raises an exception.

### REQ-speech.discovery-auto-priority: Auto-discovery priority selection
Auto-discovery selects the highest-priority available backend based on the discovery order.

### REQ-speech.discovery-auto-none: Auto-discovery returns None when no backends available
Auto-discovery returns None when no speech backends are available.

### REQ-speech.discovery-explicit-backend: Explicit backend selection
Discovery supports explicit backend selection by name, bypassing auto-discovery.

### REQ-speech.discovery-explicit-unavailable: Explicit unavailable backend
Discovery returns None when the explicitly requested backend is not available.

## Tests

- `tests/speech/test_*.py` - 7 speech test files
