# MCP Module Spec

Model Context Protocol implementation (JSON-RPC 2.0) for tool interoperability.

## Protocol (`protocol.py`)

### REQ-mcp.protocol.types: Protocol types
`MCPRequest`, `MCPResponse`, `MCPNotification` following JSON-RPC 2.0 format.

## Client (`client.py`)

### REQ-mcp.client.initialize: Session initialization
`MCPClient.initialize() -> Dict[str, Any]` negotiates capabilities with server.

### REQ-mcp.client.list-tools: Tool discovery
`list_tools() -> List[ToolSpec]` discovers tools from MCP server.

### REQ-mcp.client.call-tool: Tool invocation
`call_tool(name, arguments) -> Dict[str, Any]` invokes tool on server.

### REQ-mcp.client.close: Connection cleanup
`close()` releases connection resources.

## Server (`server.py`)

### REQ-mcp.server.handler: Request handling
`MCPServer` auto-discovers built-in tools and handles `initialize`, `tools/list`, `tools/call` methods.

### REQ-mcp.server.discovery: Tool auto-discovery
Automatically registers all `ToolRegistry` entries as MCP tools.

## Protocol (detailed)

### REQ-mcp.protocol.request-serialize: Request serialization roundtrip
MCPRequest supports JSON serialization and deserialization with all fields preserved.

### REQ-mcp.protocol.request-initialize: Initialize request
MCPRequest can construct an `initialize` method request with capabilities negotiation parameters.

### REQ-mcp.protocol.request-tools-list: Tools list request
MCPRequest can construct a `tools/list` method request for tool discovery.

### REQ-mcp.protocol.request-tools-call: Tools call request
MCPRequest can construct a `tools/call` method request with tool name and arguments.

### REQ-mcp.protocol.request-default-params: Default request params
MCPRequest defaults to empty params dict when no parameters are provided.

### REQ-mcp.protocol.request-string-id: String request ID support
MCPRequest supports string-type request IDs in addition to integer IDs.

### REQ-mcp.protocol.request-from-json-missing-params: Missing params handling
MCPRequest.from_json handles missing params field by defaulting to empty dict.

### REQ-mcp.protocol.request-from-json-missing-id: Missing ID handling
MCPRequest.from_json handles missing id field gracefully.

### REQ-mcp.protocol.request-to-json-structure: Request JSON structure
MCPRequest.to_json produces valid JSON-RPC 2.0 structure with jsonrpc, method, params, and id fields.

### REQ-mcp.protocol.response-success-roundtrip: Success response roundtrip
MCPResponse success responses survive JSON serialization and deserialization.

### REQ-mcp.protocol.response-error-roundtrip: Error response roundtrip
MCPResponse error responses survive JSON serialization and deserialization with error details preserved.

### REQ-mcp.protocol.response-error-factory: Error response factory method
MCPResponse provides a factory method to create error responses from error codes and messages.

### REQ-mcp.protocol.response-error-with-data: Error response with data
MCPResponse error responses can carry additional error data beyond code and message.

### REQ-mcp.protocol.response-error-without-data: Error response without data
MCPResponse error responses work correctly without optional error data.

### REQ-mcp.protocol.response-success-json: Success response JSON format
MCPResponse success responses produce JSON with result field and no error field.

### REQ-mcp.protocol.response-error-excludes-result: Error excludes result
MCPResponse error responses exclude the result field per JSON-RPC 2.0 spec.

### REQ-mcp.protocol.response-jsonrpc-version: JSON-RPC version field
MCPResponse always includes `"jsonrpc": "2.0"` in serialized output.

### REQ-mcp.protocol.response-from-json-defaults: Response deserialization defaults
MCPResponse.from_json applies sensible defaults for missing optional fields.

### REQ-mcp.protocol.notification-format: Notification JSON format
MCPNotification serializes to valid JSON-RPC 2.0 notification format with method and params.

### REQ-mcp.protocol.notification-no-id: Notification has no ID
MCPNotification does not include an id field, distinguishing it from requests per JSON-RPC 2.0.

### REQ-mcp.protocol.notification-with-params: Notification with params
MCPNotification supports arbitrary params for event data.

### REQ-mcp.protocol.notification-default-params: Notification default params
MCPNotification defaults to empty params when none are provided.

### REQ-mcp.protocol.error-is-exception: MCPError is an exception
MCPError inherits from Exception for use in try/except error handling.

### REQ-mcp.protocol.error-str: MCPError string representation
MCPError provides a human-readable string representation including code and message.

### REQ-mcp.protocol.error-with-data: MCPError with additional data
MCPError can carry additional structured data beyond the error code and message.

### REQ-mcp.protocol.error-default-data: MCPError default data
MCPError defaults data to None when not provided.

### REQ-mcp.protocol.error-codes: Standard MCP error codes
MCP defines standard error codes (ParseError, InvalidRequest, MethodNotFound, etc.) per JSON-RPC 2.0.

## Transport Layer

### REQ-mcp.transport.in-process-initialize: In-process transport initialization
In-process transport directly calls the MCP server handler without network overhead.

### REQ-mcp.transport.in-process-tools-list: In-process tools/list
In-process transport supports tools/list requests returning all registered tools.

### REQ-mcp.transport.in-process-tools-call: In-process tools/call
In-process transport supports tools/call requests executing tools and returning results.

### REQ-mcp.transport.in-process-multiple: In-process multiple calls
In-process transport handles multiple sequential requests correctly.

### REQ-mcp.transport.in-process-close: In-process close
In-process transport close is a no-op since there are no network resources to release.

### REQ-mcp.transport.in-process-error-method: In-process error for unknown method
In-process transport returns a MethodNotFound error for unrecognized request methods.

### REQ-mcp.transport.in-process-tool-name: In-process tool name resolution
In-process transport resolves tool names from the registry for tools/call requests.

### REQ-mcp.transport.in-process-think-tool: In-process think tool
In-process transport supports the built-in think tool for agent reasoning steps.

### REQ-mcp.transport.stdio-send-receive: Stdio transport send/receive
Stdio transport communicates with MCP servers via stdin/stdout JSON-RPC messages.

### REQ-mcp.transport.stdio-multiple: Stdio multiple requests
Stdio transport handles multiple sequential requests over the same process connection.

### REQ-mcp.transport.stdio-close: Stdio transport close
Stdio transport terminates the subprocess on close.

### REQ-mcp.transport.stdio-close-idempotent: Stdio close idempotent
Stdio transport close can be called multiple times without error.

### REQ-mcp.transport.sse-send-receive: SSE transport send/receive
SSE transport communicates with MCP servers via HTTP POST and Server-Sent Events.

### REQ-mcp.transport.sse-json-post: SSE transport JSON POST
SSE transport sends JSON-RPC requests via HTTP POST to the server endpoint.

### REQ-mcp.transport.sse-close: SSE transport close
SSE transport close is a no-op since each request is an independent HTTP call.

### REQ-mcp.transport.sse-error-response: SSE transport error response
SSE transport correctly propagates error responses from the server.

### REQ-mcp.transport.sse-success-result: SSE transport success result
SSE transport correctly extracts result data from successful server responses.

### REQ-mcp.transport.sse-http-error: SSE transport HTTP error
SSE transport raises appropriate errors for HTTP-level failures (4xx, 5xx).

## Tests

- `tests/mcp/test_*.py` - MCP protocol and transport tests
