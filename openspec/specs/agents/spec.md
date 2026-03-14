# Agents Module Spec

The agents module provides the agent lifecycle, tool-using agent pattern, managed agents, and concrete agent implementations.

## BaseAgent (`_stubs.py`)

### REQ-agents.base.protocol: Agent protocol
`BaseAgent` abstract class with `agent_id: str`, `accepts_tools: bool = False`. Constructor takes `engine: InferenceEngine`, `model: str`, plus optional `bus`, `temperature`, `max_tokens`.

### REQ-agents.base.run: Agent execution
`run(input: str, context?: AgentContext, **kwargs) -> AgentResult` is the core abstract method. Returns `AgentResult` with `content`, `tool_results`, `turns`, `metadata`.

### REQ-agents.base.events: Event emission
`_emit_turn_start(input)` and `_emit_turn_end(**data)` publish `AGENT_TURN_START`/`AGENT_TURN_END` events.

### REQ-agents.base.continuation: Length truncation handling
`_check_continuation(result, messages, max_continuations=2)` handles output truncated by max_tokens by requesting continuation.

### REQ-agents.base.registration: Registry-based registration
All agents use `@AgentRegistry.register("name")` decorator.

## ToolUsingAgent (`_stubs.py`)

### REQ-agents.tool-using.protocol: Tool-using agent protocol
Extends `BaseAgent` with `accepts_tools = True`. Adds `tools`, `max_turns`, `loop_guard_config`, `capability_policy`.

### REQ-agents.tool-using.executor: Tool executor integration
Initializes `ToolExecutor` with provided tools and capability policies.

### REQ-agents.tool-using.loop-guard: Loop detection
Optionally integrates `LoopGuard` for infinite loop detection.

## AgentContext and AgentResult

### REQ-agents.context: Agent context
`AgentContext` with `conversation`, `tools`, `memory_results`, `metadata`.

### REQ-agents.result: Agent result
`AgentResult` with `content`, `tool_results`, `turns`, `metadata`.

## AgentManager (`manager.py`)

### REQ-agents.manager.crud: Agent CRUD
`create_agent()`, `list_agents()`, `get_agent()`, `update_agent()`, `delete_agent()`, `pause_agent()`, `resume_agent()`.

### REQ-agents.manager.concurrency: Tick concurrency
`start_tick(agent_id)` marks as running (raises if already running). `end_tick(agent_id)` marks as idle.

### REQ-agents.manager.checkpoints: State checkpoints
`save_checkpoint()`, `list_checkpoints()`, `get_latest_checkpoint()`, `recover_agent()` for crash recovery.

### REQ-agents.manager.messages: Message queue
`send_message()`, `get_pending_messages()`, `mark_message_delivered()`, `store_agent_response()`, `list_messages()`.

### REQ-agents.manager.learning: Learning log
`update_summary_memory()` (truncated to 2000 chars), `add_learning_log()`, `list_learning_log()`.

### REQ-agents.manager.tasks: Task management
`create_task()`, `list_tasks()`, `update_task()`, `delete_task()`.

### REQ-agents.manager.channels: Channel bindings
`bind_channel()`, `list_channel_bindings()`, `unbind_channel()`, `find_binding_for_channel()`.

### REQ-agents.manager.storage: SQLite with WAL
Uses SQLite with WAL mode, foreign keys, and Merkle hash chain for audit. Schema migrations supported.

## AgentExecutor (`executor.py`)

### REQ-agents.executor.tick: Tick execution
`execute_tick(agent_id)` acquires concurrency guard, invokes agent with retry logic (up to 3 retries), updates stats, releases guard.

### REQ-agents.executor.retry: Error classification
Classifies errors as RetryableError (timeout, connection, rate limit) or FatalError (permission, not found). Exponential backoff: `min(10 * 2^attempt, 300)`.

### REQ-agents.executor.events: Tick events
Publishes `AGENT_TICK_START`, `AGENT_TICK_END`, `AGENT_TICK_ERROR` events.

## Implementations

### REQ-agents.simple: SimpleAgent
Single-turn agent, no tools. Registered as `"simple"`.

### REQ-agents.react: NativeReActAgent
ReAct loop with tools. Registered as `"react"` / `"native_react"`.

### REQ-agents.monitor: MonitorOperative
Persistent agent with message queue. Registered as `"monitor_operative"`.

### REQ-agents.openhands: OpenHands agents
OpenHands integration. Registered as `"openhands"` / `"native_openhands"`.

## Agent Initialization and Configuration

### REQ-agents.init: Agent constructor initialization
BaseAgent constructor stores the engine, model, and optional parameters (bus, temperature, max_tokens) with sensible defaults (temperature=0.7, max_tokens=1024).

### REQ-agents.tools: Tool acceptance configuration
BaseAgent has `accepts_tools = False` by default; ToolUsingAgent sets `accepts_tools = True` to enable tool integration.

### REQ-agents.events: Agent event emission
Agents emit lifecycle events (turn start, turn end) via the EventBus when a bus is provided; no-ops gracefully when no bus is configured.

### REQ-agents.messages: Message building for inference
Agents build message lists from input, optional system prompt, and conversation context for engine inference calls.

### REQ-agents.generate: Engine inference delegation
Agents delegate inference to the underlying engine via `_generate()`, forwarding extra kwargs like temperature and max_tokens.

### REQ-agents.max-turns: Maximum turn enforcement
Tool-using agents enforce a maximum number of turns and produce a configurable message when the limit is reached.

### REQ-agents.strip-think: Think tag stripping
Agents strip `<think>...</think>` blocks and bare closing `</think>` tags from LLM output to clean reasoning traces.

### REQ-agents.tool-agent: Tool-using agent behavior
ToolUsingAgent initializes a ToolExecutor, manages tool call/result cycles, and supports multi-turn tool interactions.

### REQ-agents.truncation: Output truncation handling
OpenHands agents detect and handle output truncated by max_tokens, requesting continuation when needed.

### REQ-agents.url-expansion: URL expansion in messages
OpenHands agents expand URLs in user messages to include fetched content for richer context.

### REQ-agents.error-handling: Agent error handling
Agents handle and surface errors from engine calls and tool execution with appropriate error classification.

### REQ-agents.openhands-run: OpenHands agent execution
OpenHands agent run loop processes user input through the engine with tool calls, manages conversation history, and returns structured AgentResult.

### REQ-agents.react-parsing: ReAct output parsing
NativeReActAgent parses LLM output into thought/action/observation triples following the ReAct format.

### REQ-agents.react-run: ReAct agent execution loop
NativeReActAgent executes the ReAct loop: generate thought+action, execute tool, observe result, repeat until final answer or max turns.

### REQ-agents.registration: Agent registry registration
Concrete agent implementations register themselves via `@AgentRegistry.register("name")` for discovery and instantiation.

## System (SDK)

### REQ-agents.builder: SystemBuilder fluent API
`SystemBuilder` provides a fluent API for configuring engine, model, agent, tools, telemetry, and traces before building a `JarvisSystem`.

### REQ-agents.system-lifecycle: JarvisSystem lifecycle management
`JarvisSystem.close()` cleans up all resources including engine, telemetry store, and trace store.

### REQ-agents.system-ask: JarvisSystem ask interface
`JarvisSystem.ask()` routes queries to agents or directly to the engine, supporting agent override, temperature, and max_tokens parameters.

### REQ-agents.system-tools: System tool building
`JarvisSystem._build_tools()` resolves tool names to tool instances from the registry, skipping unknown tools gracefully.

## Tests

- `tests/agents/test_base_agent.py` - BaseAgent protocol
- `tests/agents/test_simple.py` - SimpleAgent
- `tests/agents/test_native_react.py` - ReAct loop
- `tests/agents/test_orchestrator.py` - Orchestrator
- `tests/agents/test_monitor_operative.py` - MonitorOperative
- `tests/agents/test_executor.py` - AgentExecutor retry logic
- `tests/sdk/test_system.py` - JarvisSystem builder and lifecycle
