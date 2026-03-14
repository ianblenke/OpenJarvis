# Server Module Spec

FastAPI REST API and WebSocket bridge for the OpenJarvis backend.

## Routes

### REQ-server.routes.chat: OpenAI-compatible chat API
`POST /v1/chat/completions` with streaming (SSE) support and optional agent routing.

### REQ-server.routes.models: Model listing
`GET /v1/models` returns available models in OpenAI format.

### REQ-server.routes.health: Health check
`GET /health` returns server health status.

### REQ-server.routes.agents: Managed agent routes
CRUD for managed agents: create, list, get, update, delete, pause, resume, run.

### REQ-server.routes.channels: Channel management
Channel binding and management routes.

### REQ-server.routes.speech: Speech transcription
`POST /v1/speech/transcribe` accepts audio and returns transcription.

### REQ-server.routes.telemetry: Telemetry endpoints
Savings, energy, and telemetry data endpoints.

## Middleware

### REQ-server.middleware.security: Security headers
CSP, X-Frame-Options, HSTS, X-Content-Type-Options, Referrer-Policy headers.

### REQ-server.middleware.cors: CORS configuration
Cross-origin resource sharing for frontend access.

## WebSocket

### REQ-server.websocket: Real-time streaming
WebSocket bridge for real-time agent communication.

## PWA

### REQ-server.pwa: Progressive Web App
Static file serving for the frontend PWA with proper caching.

## Routes (detailed)

### REQ-server.routes-chat-basic: Basic chat completion
Chat endpoint accepts messages and returns a completion response with content.

### REQ-server.routes-chat-id: Chat completion ID
Chat completion responses include a unique response ID.

### REQ-server.routes-chat-finish-reason: Chat finish reason
Chat completion responses include a finish_reason field (stop, length, tool_calls).

### REQ-server.routes-chat-system-message: System message support
Chat endpoint supports system messages for setting agent behavior.

### REQ-server.routes-chat-temperature: Temperature parameter
Chat endpoint respects the temperature parameter for controlling response randomness.

### REQ-server.routes-chat-tools: Tool call support
Chat endpoint supports tool definitions and returns tool call responses when appropriate.

### REQ-server.routes-chat-usage: Usage statistics
Chat completion responses include token usage statistics (prompt_tokens, completion_tokens, total_tokens).

### REQ-server.routes-health-ok: Health check success
`GET /health` returns 200 OK when the server is healthy.

### REQ-server.routes-health-503: Health check failure
`GET /health` returns 503 when the server is unhealthy (e.g., no engine configured).

### REQ-server.routes-info: Server info endpoint
Server info endpoint returns version, capabilities, and configuration summary.

### REQ-server.routes-info-agent: Server info with agent
Server info endpoint includes active agent information when an agent is configured.

### REQ-server.routes-list-models: Model listing
`GET /v1/models` returns available models in OpenAI-compatible format.

### REQ-server.routes-model-object-format: Model object format
Model listing returns objects with id, object="model", created, and owned_by fields.

### REQ-server.routes-multiple-models: Multiple model listing
Model listing returns all configured models, not just the default.

### REQ-server.routes-streaming-sse: Streaming SSE format
Streaming chat responses use Server-Sent Events (SSE) format with `data:` prefixed lines.

### REQ-server.routes-streaming-content: Streaming content chunks
Streaming responses deliver content in incremental delta chunks.

### REQ-server.routes-agent-completion: Agent completion routing
Chat endpoint routes requests to the configured agent for multi-turn tool-using completion.

### REQ-server.routes-agent-conversation: Agent conversation
Chat endpoint supports multi-message conversations with agent state management.

### REQ-server.routes-app-state: Application state endpoint
App state endpoint returns the current server configuration and status.

### REQ-server.routes-app-state-agent: App state with agent
App state includes agent type, tools, and configuration when an agent is active.

### REQ-server.routes-app-state-no-agent: App state without agent
App state returns minimal configuration when no agent is configured.

### REQ-server.routes-channels-no-bridge: Channels without bridge
Channel endpoints return appropriate responses when no channel bridge is configured.

### REQ-server.routes-channel-send-no-bridge: Channel send without bridge
Channel send endpoint returns an error when no bridge is configured.

### REQ-server.routes-channel-status-not-configured: Channel status not configured
Channel status returns "not configured" for channels without active connections.

## Cost Estimation

### REQ-server.cost-pricing-defined: Cost pricing data defined
Server defines pricing data for supported inference providers.

### REQ-server.cost-pricing-fields: Cost pricing fields
Pricing entries include input_cost_per_1k and output_cost_per_1k token rates.

### REQ-server.cost-scenarios-defined: Cost scenarios defined
Server defines representative usage scenarios for cost estimation.

### REQ-server.cost-scenarios-fields: Cost scenario fields
Cost scenarios include name, queries_per_day, avg_input_tokens, and avg_output_tokens.

### REQ-server.cost-monthly-basic: Monthly cost calculation
Monthly cost is computed as daily token usage times 30 times per-token pricing.

### REQ-server.cost-monthly-math: Monthly cost math accuracy
Monthly cost calculation produces mathematically correct results for known inputs.

### REQ-server.cost-monthly-label: Monthly cost provider label
Monthly cost results include the provider label for display purposes.

### REQ-server.cost-monthly-unknown-provider: Unknown provider cost
Monthly cost returns zero for unknown/unsupported providers.

### REQ-server.cost-monthly-zero-calls: Zero calls cost
Monthly cost is zero when queries_per_day is zero.

### REQ-server.cost-all-scenarios: All scenario cost calculation
Cost estimation runs across all defined scenarios and providers.

### REQ-server.cost-scenario-all-providers: Scenario across all providers
Each scenario is evaluated against all supported providers.

### REQ-server.cost-scenario-unknown: Unknown scenario handling
Cost estimation handles unknown scenarios gracefully.

## Savings Estimation

### REQ-server.savings-basic: Basic savings calculation
Savings estimation compares local inference cost to cloud provider costs.

### REQ-server.savings-provider-costs: Provider cost comparison
Savings includes per-provider cost estimates for comparison.

### REQ-server.savings-avg-cost-per-query: Average cost per query
Savings estimation computes average cost per query across all usage.

### REQ-server.savings-avg-cost-zero-calls: Average cost with zero calls
Savings estimation handles zero-call scenarios without division errors.

### REQ-server.savings-zero-tokens: Zero token savings
Savings estimation handles zero token usage correctly.

### REQ-server.savings-cloud-agent-equivalent: Cloud agent equivalent cost
Savings estimation includes equivalent cloud agent service costs.

### REQ-server.savings-json-serializable: Savings JSON serializable
Savings estimation results are fully JSON serializable.

### REQ-server.savings-serialization: Savings serialization
Savings dataclass supports dict/JSON conversion for API responses.

## Tests

- `tests/server/test_routes.py` - API route tests
- `tests/server/test_websocket.py` - WebSocket tests
- `tests/server/test_middleware.py` - Middleware tests
- `tests/server/test_speech_routes.py` - Speech API tests
- `tests/server/test_openai_routes.py` - OpenAI-compatible route tests
- `tests/server/test_cost.py` - Cost estimation tests
- `tests/server/test_savings.py` - Savings estimation tests
