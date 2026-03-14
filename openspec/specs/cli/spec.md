# CLI Module Spec

Command-line interface for OpenJarvis built with Click.

## Commands

### REQ-cli.ask: Ask command
`jarvis ask` sends a query to the AI with optional model/agent/tool selection.

### REQ-cli.chat: Interactive chat
`jarvis chat` starts an interactive chat session.

### REQ-cli.serve: Server
`jarvis serve` starts the FastAPI server.

### REQ-cli.model: Model management
`jarvis model list|pull|info` manages available models.

### REQ-cli.bench: Benchmarking
`jarvis bench` runs performance benchmarks.

### REQ-cli.doctor: System diagnostics
`jarvis doctor` checks system health (hardware, engines, dependencies).

### REQ-cli.init: Project initialization
`jarvis init` creates configuration files.

### REQ-cli.add: Add content to memory
`jarvis add` ingests files/directories into memory.

### REQ-cli.vault: Secret management
`jarvis vault` manages encrypted secrets.

### REQ-cli.channel: Channel management
`jarvis channel` manages messaging channel connections.

### REQ-cli.telemetry: Telemetry commands
`jarvis telemetry` views energy and performance metrics.

### REQ-cli.daemon: Background daemon
`jarvis daemon` manages the background agent daemon.

### REQ-cli.quickstart: Quick setup
`jarvis quickstart` guides new users through initial setup.

## Tests

- `tests/cli/test_*.py` - 25+ CLI test files
