# Core Module Spec

The core module provides foundational primitives: configuration, events, registry, and canonical types.

## Configuration (`config.py`)

### REQ-core.config.detect-hardware: Hardware auto-detection
`detect_hardware()` returns a `HardwareInfo` with platform, CPU, RAM, and optional GPU info.

### REQ-core.config.recommend-engine: Engine recommendation
`recommend_engine(hw)` returns the best engine string for the given hardware:
- Apple GPU → `"mlx"`
- NVIDIA datacenter → `"vllm"`
- NVIDIA consumer → `"ollama"`
- AMD → `"vllm"`
- CPU-only → `"llamacpp"`

### REQ-core.config.recommend-model: Model recommendation
`recommend_model(hw, engine)` returns the largest Qwen3.5 model fitting available VRAM using Q4_K_M estimation (~0.5 bytes/param with 10% overhead).

### REQ-core.config.load-config: Configuration loading
`load_config(path?)` loads config by: detecting hardware → building defaults → overlaying TOML overrides. TOML paths: explicit path > `OPENJARVIS_CONFIG` env var > `~/.openjarvis/config.toml`.

### REQ-core.config.backward-compat: Backward compatibility
All major config classes maintain property-based backward compatibility (e.g., `EngineConfig.ollama_host` maps to `EngineConfig.ollama.host`).

### REQ-core.config.toml-generation: TOML template generation
`generate_minimal_toml(hw)` and `generate_default_toml(hw)` produce valid TOML configuration templates.

## Event Bus (`events.py`)

### REQ-core.events.pubsub: Synchronous pub/sub
`EventBus` provides `subscribe(type, callback)`, `unsubscribe(type, callback)`, and `publish(type, data)`. Subscribers execute synchronously in registration order.

### REQ-core.events.thread-safety: Thread-safe operations
All EventBus operations are lock-protected. `get_event_bus()` uses thread-safe lazy initialization.

### REQ-core.events.history: Optional history recording
`EventBus(record_history=True)` records all published events. `history` property returns a copy. `clear_history()` discards recorded events.

### REQ-core.events.singleton: Module-level singleton
`get_event_bus()` returns a singleton. `reset_event_bus()` replaces it (for tests).

### REQ-core.events.event-types: Comprehensive event catalog
`EventType` enum covers 60+ event categories across all subsystems (inference, agents, tools, workflow, security, learning, telemetry, etc.).

## Registry (`registry.py`)

### REQ-core.registry.generic-base: Generic registry pattern
`RegistryBase[T]` provides `register(key)` decorator, `register_value(key, value)`, `get(key)`, `create(key, *args)`, `items()`, `keys()`, `contains(key)`, `clear()`.

### REQ-core.registry.isolation: Per-subclass isolation
Each registry subclass has isolated storage. No cross-registry leaks.

### REQ-core.registry.duplicate-prevention: Duplicate key prevention
`register()` and `register_value()` raise `ValueError` on duplicate keys.

### REQ-core.registry.missing-key-error: Missing key errors
`get()` raises `KeyError` with a descriptive message on missing keys.

### REQ-core.registry.typed-subclasses: Typed registry subclasses
11 typed registries: `ModelRegistry`, `EngineRegistry`, `MemoryRegistry`, `AgentRegistry`, `ToolRegistry`, `RouterPolicyRegistry`, `BenchmarkRegistry`, `ChannelRegistry`, `LearningRegistry`, `SkillRegistry`, `SpeechRegistry`.

## Types (`types.py`)

### REQ-core.types.message: OpenAI-compatible messages
`Message` dataclass with `role: Role`, `content`, `name`, `tool_calls`, `tool_call_id`, `metadata`. `Role` enum: SYSTEM, USER, ASSISTANT, TOOL.

### REQ-core.types.conversation: Sliding-window conversation
`Conversation` with `add(message)` (auto-trims if `max_messages` set) and `window(n)` (returns last n messages).

### REQ-core.types.model-spec: Model metadata
`ModelSpec` with model identity, parameter counts, context length, quantization, VRAM requirements, supported engines.

### REQ-core.types.tool-result: Tool execution result
`ToolResult` with `tool_name`, `content`, `success`, `usage`, `cost_usd`, `latency_seconds`, `metadata`.

### REQ-core.types.telemetry-record: Telemetry observation
`TelemetryRecord` with 30+ fields covering tokens, timing, cost, energy, GPU metrics, throughput.

### REQ-core.types.trace: Agent trace
`Trace` with `trace_id`, `query`, `agent`, `model`, `engine`, `steps: List[TraceStep]`, `result`, `outcome`, `feedback`. `add_step()` updates running totals.

### REQ-core.types.routing-context: Query routing metadata
`RoutingContext` with query analysis fields: `has_code`, `has_math`, `language`, `urgency`.

## Tests

- `tests/core/test_config.py` - Configuration loading, hardware detection, engine/model recommendation
- `tests/core/test_events.py` - EventBus pub/sub, history, threading
- `tests/core/test_registry.py` - Registry CRUD, isolation, error cases
- `tests/core/test_types.py` - Message, Conversation, ModelSpec, Trace operations
