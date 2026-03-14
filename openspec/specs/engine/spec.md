# Engine Module Spec

The engine module provides inference engine backends with a common protocol, auto-discovery, and registration.

## InferenceEngine Protocol (`_stubs.py`)

### REQ-engine.protocol.generate: Synchronous generation
`generate(messages, *, model, temperature=0.7, max_tokens=1024, **kwargs) -> Dict[str, Any]` returns a dict with at minimum: `content` (str), `usage` (dict with `prompt_tokens`, `completion_tokens`, `total_tokens`), `model` (str), `finish_reason` (str). Optional: `tool_calls`, `cost_usd`, `ttft`, `engine_timing`.

### REQ-engine.protocol.stream: Async streaming
`stream(messages, *, model, temperature=0.7, max_tokens=1024, **kwargs) -> AsyncIterator[str]` yields token strings as generated.

### REQ-engine.protocol.list-models: Model listing
`list_models() -> List[str]` returns identifiers of available models.

### REQ-engine.protocol.health: Health check
`health() -> bool` returns True when engine is reachable and healthy. Uses 2-second timeout.

### REQ-engine.protocol.lifecycle: Resource management
`close()` releases HTTP clients/connections. `prepare(model)` is an optional warm-up hook.

### REQ-engine.protocol.response-format: Structured output
`ResponseFormat` dataclass with `type` ("json_object" or "json_schema") and optional `schema`.

## Discovery (`_discovery.py`)

### REQ-engine.discovery.probe: Engine auto-discovery
`discover_engines(config) -> List[Tuple[str, InferenceEngine]]` probes all registered engines, returns healthy ones sorted with config default first.

### REQ-engine.discovery.models: Model aggregation
`discover_models(engines) -> Dict[str, List[str]]` calls `list_models()` on each engine.

### REQ-engine.discovery.get-engine: Engine lookup
`get_engine(config, engine_key?) -> Tuple[str, InferenceEngine] | None` gets a specific engine or falls back to any healthy engine.

## Implementations

### REQ-engine.ollama: Ollama backend
`OllamaEngine` registered as `"ollama"`. Uses native HTTP API (`/api/chat`, `/api/tags`). Converts tool arguments from JSON strings to dicts. Returns timing metrics.

### REQ-engine.cloud: Multi-provider cloud backend
`CloudEngine` registered as `"cloud"`. Supports OpenAI, Anthropic, Google Gemini. Auto-detects provider from model name. Includes `estimate_cost()`. Handles tool format conversion per provider.

### REQ-engine.litellm: LiteLLM unified router
`LiteLLMEngine` registered as `"litellm"`. Routes to 100+ providers via LiteLLM. Cost tracking.

### REQ-engine.openai-compat: OpenAI-compatible engines
9 data-driven engines from `_OpenAICompatibleEngine` base: vllm, sglang, llamacpp, mlx, lmstudio, exo, nexa, uzu, apple_fm. Each uses a different default port.

### REQ-engine.shims: Shim servers
`NexaShim` and `AppleFmShim` wrap native SDKs as OpenAI-compatible FastAPI servers.

## Registration

### REQ-engine.registration: Registry-based registration
All engines use `@EngineRegistry.register("key")` decorator. Host configuration mapped from `JarvisConfig.engine.*` attributes.

## Utility

### REQ-engine.message-conversion: Message format conversion
`messages_to_dicts(messages)` converts internal `Message` objects to OpenAI-compatible dicts, handling tool_calls, tool_call_id, and name fields.

## Tests

- `tests/engine/test_ollama.py` - Ollama engine behavior
- `tests/engine/test_cloud.py`, `test_cloud_extended.py` - Cloud provider routing, cost estimation
- `tests/engine/test_litellm.py` - LiteLLM routing
- `tests/engine/test_discovery.py` - Engine discovery and fallback
- `tests/engine/test_structured_output.py` - Response format handling
