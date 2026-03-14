# Security Module Spec

The security module provides guardrails, scanning, audit logging, RBAC capabilities, taint tracking, SSRF protection, rate limiting, and file sensitivity policies.

## Types (`types.py`)

### REQ-security.types.threat-level: Threat classification
`ThreatLevel` enum: LOW, MEDIUM, HIGH, CRITICAL.

### REQ-security.types.redaction-mode: Action modes
`RedactionMode` enum: WARN (log), REDACT (replace), BLOCK (raise).

### REQ-security.types.scan-result: Scan result
`ScanResult` with `findings: List[ScanFinding]`, `clean` property (True if no findings), `highest_threat` property.

## Scanners (`scanner.py`)

### REQ-security.scanner.protocol: Scanner protocol
`BaseScanner` abstract class with `scanner_id`, `scan(text) -> ScanResult`, `redact(text) -> str`.

### REQ-security.scanner.secrets: Secret detection
`SecretScanner` detects API keys (OpenAI, Anthropic, AWS, GitHub), tokens, passwords via regex patterns. All CRITICAL threat level.

### REQ-security.scanner.pii: PII detection
`PIIScanner` detects email, SSN, credit cards, phone numbers, public IPs. Uses Rust backend.

## GuardrailsEngine (`guardrails.py`)

### REQ-security.guardrails.wrapping: Engine wrapping
`GuardrailsEngine` wraps any `InferenceEngine` with input/output scanning. Not registered in EngineRegistry.

### REQ-security.guardrails.modes: Redaction modes
WARN mode logs findings. REDACT mode replaces matches with `[REDACTED:pattern_name]`. BLOCK mode raises `SecurityBlockError`.

### REQ-security.guardrails.events: Security events
Publishes `SECURITY_ALERT` or `SECURITY_BLOCK` events to EventBus.

## Audit (`audit.py`)

### REQ-security.audit.append-only: Append-only log
`AuditLogger` uses SQLite with append-only policy. No updates allowed.

### REQ-security.audit.merkle-chain: Merkle hash chain
Each row's `row_hash` is SHA256 of row data + `prev_hash`. `verify_chain()` detects tampering.

### REQ-security.audit.query: Event querying
`query(event_type?, since?, limit=100)` filters logged events.

### REQ-security.audit.auto-subscribe: EventBus integration
Subscribes to SECURITY_SCAN, SECURITY_ALERT, SECURITY_BLOCK events automatically.

## Capabilities (`capabilities.py`)

### REQ-security.capabilities.rbac: Role-based access control
`CapabilityPolicy` with `grant()`, `deny()`, `check()` methods. 10 capability types: file:read, file:write, network:fetch, code:execute, memory:read, memory:write, channel:send, tool:invoke, schedule:create, system:admin.

### REQ-security.capabilities.deny-precedence: Denial precedence
Explicit denials take precedence over grants.

### REQ-security.capabilities.glob-patterns: Glob pattern matching
Capabilities and resources support glob patterns via `fnmatch`.

## Taint Tracking (`taint.py`)

### REQ-security.taint.labels: Taint labels
`TaintLabel` enum: PII, SECRET, USER_PRIVATE, EXTERNAL.

### REQ-security.taint.propagation: Taint propagation
`propagate_taint(input_taint, output_text)` unions input taint with auto-detected output taint.

### REQ-security.taint.sink-policy: Sink policies
`SINK_POLICY` maps tool names to forbidden taint labels. `check_taint()` validates before tool execution.

### REQ-security.taint.auto-detect: Auto-detection
`auto_detect_taint(text)` detects PII and secrets in text using regex patterns.

## SSRF Protection (`ssrf.py`)

### REQ-security.ssrf.private-ip: Private IP blocking
`is_private_ip()` blocks 10.x, 172.16.x, 192.168.x, 127.x, ::1, link-local.

### REQ-security.ssrf.metadata: Cloud metadata blocking
Blocks cloud metadata endpoints (169.254.169.254, metadata.google.internal, etc.).

### REQ-security.ssrf.dns-rebind: DNS rebind protection
`check_ssrf()` resolves hostname and checks if it points to private IP.

## Rate Limiting (`rate_limiter.py`)

### REQ-security.rate-limiter.token-bucket: Token bucket algorithm
`TokenBucket` with `consume(tokens) -> (allowed, wait_seconds)`. Thread-safe.

### REQ-security.rate-limiter.per-key: Per-key rate limiting
`RateLimiter` maintains separate buckets per key (e.g., `agent_id:tool_name`).

## File Policy (`file_policy.py`)

### REQ-security.file-policy.sensitive: Sensitive file detection
`is_sensitive_file(path)` checks against patterns: .env, .secret, *.pem, *.key, id_rsa, .htpasswd, etc.

### REQ-security.file-policy.filter: Path filtering
`filter_sensitive_paths(paths)` returns only non-sensitive paths.

## Tests

- `tests/security/test_guardrails.py` - Engine wrapping and scanning
- `tests/security/test_ssrf.py` - SSRF protection
- `tests/security/test_audit.py` - Audit logging and Merkle chain
- `tests/security/test_capabilities.py` - RBAC policy
- `tests/security/test_taint.py` - Taint tracking
- `tests/security/test_scanner.py` - Secret/PII detection
- `tests/security/test_rate_limiter.py` - Rate limiting
- `tests/security/test_file_policy.py` - File sensitivity
