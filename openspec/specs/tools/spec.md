# Tools Module Spec

The tools module provides the tool protocol, executor, and built-in tool implementations.

## ToolSpec (`_stubs.py`)

### REQ-tools.spec: Tool specification
`ToolSpec` dataclass with `name`, `description`, `parameters` (JSON schema), `category`, `cost_estimate`, `latency_estimate`, `requires_confirmation`, `timeout_seconds`, `required_capabilities`, `metadata`.

## BaseTool (`_stubs.py`)

### REQ-tools.base.protocol: Tool protocol
`BaseTool` abstract class with `tool_id: str`, abstract `spec` property returning `ToolSpec`, abstract `execute(**params) -> ToolResult`.

### REQ-tools.base.openai-format: OpenAI function format
`to_openai_function()` converts tool spec to OpenAI function-calling format.

### REQ-tools.base.registration: Registry-based registration
Tools use `@ToolRegistry.register("name")` decorator.

## ToolExecutor (`_stubs.py`)

### REQ-tools.executor.dispatch: Tool dispatch
`execute(tool_call: ToolCall) -> ToolResult` parses JSON arguments, dispatches to tool, measures latency.

### REQ-tools.executor.pipeline: Execution pipeline
Pipeline: tool lookup → JSON parse → capability check → taint check → confirmation → event emission → execute with timeout → taint auto-detect → event emission.

### REQ-tools.executor.timeout: Execution timeout
Uses ThreadPoolExecutor with configurable timeout. Timeout → `ToolResult(success=False)`.

### REQ-tools.executor.security: Security integration
Checks capability policy (RBAC), taint violations, and requires_confirmation flag before execution.

### REQ-tools.executor.events: Event emission
Publishes `TOOL_CALL_START` and `TOOL_CALL_END` events.

### REQ-tools.executor.listing: Tool listing
`available_tools()` returns specs. `get_openai_tools()` returns OpenAI format.

## Built-in Tools

### REQ-tools.calculator: Calculator
Safe math evaluation via AST whitelisting.

### REQ-tools.file-read: File read
Read files with sensitivity checking and allowed-dir restrictions.

### REQ-tools.web-search: Web search
Search the web via Tavily or similar backend.

### REQ-tools.code-interpreter: Code interpreter
Execute Python code in sandboxed environment.

### REQ-tools.retrieval: Memory retrieval
RAG/memory retrieval from configured backends.

### REQ-tools.shell-exec: Shell execution
Execute shell commands with safety restrictions.

### REQ-tools.browser: Browser automation
Web browsing via Playwright with accessibility tree extraction.

### REQ-tools.git: Git operations
Git repository operations.

### REQ-tools.git.timeout: Git command timeout handling
Git tools return a failure result when a subprocess command exceeds the configured timeout.

### REQ-tools.git.truncate: Git output truncation
Git tools truncate command output that exceeds `_MAX_OUTPUT_BYTES` and append a truncation notice.

### REQ-tools.http-request: HTTP requests
Make HTTP requests with SSRF protection.

### REQ-tools.apply-patch: Patch application
Apply unified diffs to files.

## Utility

### REQ-tools.descriptions: Tool description builder
`build_tool_descriptions(tools, *, include_category, include_cost)` builds rich text for agent system prompts.

## Calculator (detailed)

### REQ-tools.calculator.spec: Calculator tool specification
Calculator tool provides a ToolSpec with name, description, and parameter schema for math expressions.

### REQ-tools.calculator.execute: Calculator execution
Calculator evaluates mathematical expressions and returns the numeric result.

### REQ-tools.calculator.safety: Calculator safety
Calculator uses AST whitelisting to prevent code execution; rejects dangerous expressions.

### REQ-tools.calculator.eval: Calculator evaluation
Calculator correctly evaluates arithmetic, trigonometric, and algebraic expressions.

### REQ-tools.calculator.openai: Calculator OpenAI format
Calculator tool converts to OpenAI function-calling format with proper parameter schema.

## File Read (detailed)

### REQ-tools.file-read.spec: File read tool specification
File read tool provides a ToolSpec with name, description, and parameter schema.

### REQ-tools.file-read.execute: File read execution
File read tool reads file contents and returns them as a string result.

### REQ-tools.file-read.security: File read security
File read tool checks file sensitivity (secrets, credentials) before reading.

### REQ-tools.file-read.allowed-dirs: File read allowed directories
File read tool restricts reads to configured allowed directories.

### REQ-tools.file-read.openai: File read OpenAI format
File read tool converts to OpenAI function-calling format.

## File Write

### REQ-tools.file-write.spec: File write tool specification
File write tool provides a ToolSpec with name, description, and parameter schema.

### REQ-tools.file-write.execute: File write execution
File write tool writes content to a file and returns success/failure result.

### REQ-tools.file-write.security: File write security
File write tool checks file sensitivity and validates target path before writing.

### REQ-tools.file-write.allowed-dirs: File write allowed directories
File write tool restricts writes to configured allowed directories.

### REQ-tools.file-write.openai: File write OpenAI format
File write tool converts to OpenAI function-calling format.

## Think Tool

### REQ-tools.think.spec: Think tool specification
Think tool provides a ToolSpec for agent internal reasoning steps.

### REQ-tools.think.execute: Think tool execution
Think tool accepts a thought string and returns it as the result (passthrough for reasoning).

### REQ-tools.think.openai: Think tool OpenAI format
Think tool converts to OpenAI function-calling format.

## Git (detailed)

### REQ-tools.git.status.spec: Git status tool specification
Git status tool provides a ToolSpec for checking repository status.

### REQ-tools.git.status.execute: Git status execution
Git status tool executes `git status` and returns the output.

### REQ-tools.git.status.openai: Git status OpenAI format
Git status tool converts to OpenAI function-calling format.

### REQ-tools.git.diff.spec: Git diff tool specification
Git diff tool provides a ToolSpec for showing repository changes.

### REQ-tools.git.diff.execute: Git diff execution
Git diff tool executes `git diff` and returns the output.

### REQ-tools.git.log.spec: Git log tool specification
Git log tool provides a ToolSpec for showing commit history.

### REQ-tools.git.log.execute: Git log execution
Git log tool executes `git log` and returns the formatted output.

### REQ-tools.git.log.openai: Git log OpenAI format
Git log tool converts to OpenAI function-calling format.

### REQ-tools.git.commit.spec: Git commit tool specification
Git commit tool provides a ToolSpec for creating commits.

### REQ-tools.git.commit.execute: Git commit execution
Git commit tool executes `git commit` with the provided message.

## Patch Application (detailed)

### REQ-tools.patch.spec: Patch tool specification
Patch tool provides a ToolSpec for applying unified diffs to files.

### REQ-tools.patch.execute: Patch execution
Patch tool applies a unified diff to the target file and returns success/failure.

### REQ-tools.patch.parse: Patch parsing
Patch tool parses unified diff format into hunk objects with line ranges and content.

### REQ-tools.patch.apply-hunks: Hunk application
Patch tool applies individual hunks to file content, handling context lines and offsets.

### REQ-tools.patch.security: Patch security
Patch tool validates target paths against allowed directories and prevents path traversal.

## Knowledge Graph Tools

### REQ-tools.knowledge-graph.add-entity: Add entity tool
`KGAddEntityTool` provides a tool interface for adding entities to the knowledge graph, with spec validation, backend availability checks, and support for optional properties.

### REQ-tools.knowledge-graph.add-relation: Add relation tool
`KGAddRelationTool` provides a tool interface for adding relations between entities in the knowledge graph, with spec validation, backend availability checks, and configurable weight.

### REQ-tools.knowledge-graph.query: Query tool
`KGQueryTool` provides a tool interface for querying entities and relations in the knowledge graph by type, with optional limit and backend availability checks.

### REQ-tools.knowledge-graph.neighbors: Neighbors tool
`KGNeighborsTool` provides a tool interface for finding neighboring entities in the knowledge graph, with optional relation type and direction filtering.

## Tests

- `tests/tools/test_calculator.py` - Calculator safety and accuracy
- `tests/tools/test_file_read.py` - File read with sensitivity
- `tests/tools/test_web_search.py` - Web search
- `tests/tools/test_shell_exec.py` - Shell execution
- `tests/tools/test_browser.py` - Browser automation
- `tests/tools/test_git_tool.py` - Git operations
- `tests/tools/test_http_request.py` - HTTP with SSRF
- `tests/tools/test_file_write.py` - File write tool
- `tests/tools/test_think.py` - Think tool
- `tests/tools/test_patch.py` - Patch application
