# Sandbox Module Spec

Containerized execution environment for secure code execution.

## ContainerRunner

### REQ-sandbox.runner.run: Container execution
`ContainerRunner.run(input_data, *, workspace, mounts, secrets, env) -> Dict[str, Any]` runs code in Docker/Podman container.

### REQ-sandbox.runner.stop: Container stop
`stop(container_name)` stops a running container.

### REQ-sandbox.runner.cleanup: Orphan cleanup
`cleanup_orphans()` removes stale containers.

## SandboxedAgent

### REQ-sandbox.agent: Transparent agent wrapper
`SandboxedAgent(BaseAgent)` wraps any agent to run inside a container. Transparent to callers.

## Security

### REQ-sandbox.security.mounts: Mount allowlist
Only whitelisted paths can be mounted into the container.

### REQ-sandbox.security.secrets: Secret injection
Secrets are injected as files, not environment variables.

### REQ-sandbox.security.limits: Resource limits
Configurable timeout, WASM memory limits, max concurrent containers.

## Mount Security

### REQ-sandbox.mount.allowed-root: Allowed mount root validation
Mount entries validate that source paths are within allowed root directories and default to read-only mode.

### REQ-sandbox.mount.allowlist: Mount allowlist configuration
MountAllowlist defines configurable allowed root directories and blocked path patterns with independent defaults.

### REQ-sandbox.mount.blocked-patterns: Blocked mount patterns
Default blocked patterns prevent mounting sensitive paths (.ssh, .env, .pem, .key, etc.).

### REQ-sandbox.mount.load-config: Mount config loading
Mount allowlist can be loaded from configuration files with custom roots and patterns.

### REQ-sandbox.mount.traversal: Path traversal prevention
Mount validation prevents path traversal attacks (e.g., `../`) that would escape allowed roots.

### REQ-sandbox.mount.validate: Single mount validation
`validate_mount()` checks a single mount entry against the allowlist and returns pass/fail with reason.

### REQ-sandbox.mount.validate-list: Mount list validation
`validate_mounts()` checks a list of mount entries and returns all validation results.

## WASM Sandbox

### REQ-sandbox.wasm.factory: WASM sandbox factory
Factory method creates WASM sandbox instances with configured memory limits and permissions.

### REQ-sandbox.wasm.result: WASM execution result
WASM sandbox returns structured execution results with stdout, stderr, exit code, and resource usage.

### REQ-sandbox.wasm.runner: WASM sandbox runner
WASM sandbox runner executes code within WebAssembly memory and instruction limits.

## Tests

- `tests/sandbox/test_*.py` - 3 sandbox test files
