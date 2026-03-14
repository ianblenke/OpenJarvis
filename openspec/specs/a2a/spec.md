# A2A Module Spec

Agent-to-Agent communication protocol following Google's A2A specification (JSON-RPC 2.0).

## Protocol (`protocol.py`)

### REQ-a2a.protocol.task: Task lifecycle
`A2ATask` with `task_id`, `state` (SUBMITTED|WORKING|COMPLETED|FAILED|CANCELED), `input_text`, `output_text`, `history`.

### REQ-a2a.protocol.agent-card: Agent discovery
`AgentCard` with `name`, `description`, `url`, `version`, `capabilities`, `skills`, `authentication`.

## Client (`client.py`)

### REQ-a2a.client.discover: Agent discovery
`A2AClient.discover() -> AgentCard` fetches `/.well-known/agent.json`.

### REQ-a2a.client.send-task: Task submission
`send_task(input_text) -> A2ATask` submits work to remote agent.

### REQ-a2a.client.get-task: Task polling
`get_task(task_id) -> A2ATask` retrieves task status.

### REQ-a2a.client.cancel-task: Task cancellation
`cancel_task(task_id) -> A2ATask` cancels a running task.

## Server (`server.py`)

### REQ-a2a.server.handler: JSON-RPC handler
`A2AServer.handle_request(request_data) -> Dict` handles JSON-RPC requests.

### REQ-a2a.server.routes: FastAPI routes
`get_routes()` returns route definitions for FastAPI mounting.

## Tool Wrapper (`tool.py`)

### REQ-a2a.tool.wrapper: Remote agent as tool
`A2AAgentTool(BaseTool)` wraps a remote A2A agent as a local tool that agents can invoke.

## Tests

- `tests/a2a/test_a2a.py` - Existing A2A tests
