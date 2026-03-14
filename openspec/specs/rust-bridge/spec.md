# Rust Bridge Module Spec

PyO3 Python bindings for performance-critical Rust implementations.

## Crates

### REQ-rust.core: Core types
`openjarvis-core` crate providing Rust implementations of config, events, registry, types, hardware detection.

### REQ-rust.engine: Engine implementations
`openjarvis-engine` crate with Rust engine backends (Ollama, OpenAI-compat, discovery).

### REQ-rust.security: Security backends
`openjarvis-security` crate with Rust implementations of scanner, SSRF, capabilities, taint, rate limiter, file policy, audit.

### REQ-rust.tools: Tool implementations
`openjarvis-tools` crate with Rust tool and storage backends (SQLite, BM25, FAISS, ColBERT bindings).

### REQ-rust.learning: Learning components
`openjarvis-learning` crate with optimization engine, search space, trial runner.

### REQ-rust.telemetry: Telemetry
`openjarvis-telemetry` crate with telemetry store and energy monitoring.

### REQ-rust.python: Python bindings
`openjarvis-python` crate with PyO3 bindings exposing Rust implementations to Python. 18 binding files.

## Testing

### REQ-rust.testing: Inline tests
All Rust crates include `#[cfg(test)]` modules with inline tests. ~74 out of 124 files have tests (~60%).

## Tests

- Rust tests via `cargo test --workspace`
- Coverage via `cargo tarpaulin --workspace`
