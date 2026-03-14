# Telemetry Module Spec

Performance and energy monitoring with platform-specific GPU/CPU metrics.

## TelemetryStore

### REQ-telemetry.store: Persistent telemetry storage
SQLite-backed store with 40+ columns tracking tokens, latency, TTFT, cost, energy (CPU/GPU/DRAM), GPU metrics (utilization, memory, temperature), ITL stats (mean, median, p90-p99).

### REQ-telemetry.store.record: Record telemetry
Store `TelemetryRecord` observations from inference calls.

### REQ-telemetry.store.query: Query telemetry
Query telemetry with filters on model, engine, agent, time range.

## Energy Monitoring

### REQ-telemetry.energy.nvidia: NVIDIA GPU energy
Monitor NVIDIA GPU energy via `pynvml`. Measures power, temperature, utilization.

### REQ-telemetry.energy.amd: AMD GPU energy
Monitor AMD GPU energy via `amdsmi`.

### REQ-telemetry.energy.apple: Apple Silicon energy
Monitor Apple Silicon energy metrics.

### REQ-telemetry.energy.rapl: CPU RAPL energy
Monitor CPU energy via Intel RAPL (Running Average Power Limit).

### REQ-telemetry.energy.monitor: Unified energy monitor
`EnergyMonitor` that auto-detects available backends and provides unified `sample()` interface.

## GPU Monitoring

### REQ-telemetry.gpu-monitor: GPU metrics collection
Collect GPU utilization, memory usage, temperature at configurable intervals.

## Derived Metrics

### REQ-telemetry.derived: Computed metrics
Derived metrics: tokens/joule, energy/output-token, throughput/watt, efficiency ratios.

## ITL Metrics

### REQ-telemetry.itl: Inter-token latency
Track inter-token latency statistics (mean, median, p90, p95, p99).

## vLLM Metrics

### REQ-telemetry.vllm: vLLM-specific metrics
Scrape and store vLLM Prometheus metrics.

## Instrumented Engine

### REQ-telemetry.instrumented-engine: Engine wrapper
Wraps any `InferenceEngine` to automatically record `TelemetryRecord` on each inference call.

## TelemetryStore (detailed)

### REQ-telemetry.store-create-table: Store table creation
TelemetryStore creates the telemetry table with all required columns on initialization.

### REQ-telemetry.store-record-insert: Record insertion
TelemetryStore inserts TelemetryRecord observations with all fields mapped to columns.

### REQ-telemetry.store-record-multiple: Multiple record insertion
TelemetryStore handles multiple sequential record insertions correctly.

### REQ-telemetry.store-record-energy-fields: Energy field recording
TelemetryStore records energy-specific fields (cpu_energy, gpu_energy, dram_energy).

### REQ-telemetry.store-record-streaming-flag: Streaming flag recording
TelemetryStore records whether the inference call used streaming mode.

### REQ-telemetry.store-record-warmup-flag: Warmup flag recording
TelemetryStore records whether the inference call was a warmup call (excluded from stats).

### REQ-telemetry.store-fetchall: Fetch all records
TelemetryStore fetches all records with default SQL query.

### REQ-telemetry.store-fetchall-custom-sql: Fetch with custom SQL
TelemetryStore supports custom SQL queries for filtered record retrieval.

### REQ-telemetry.store-metadata-json: Metadata JSON storage
TelemetryStore stores arbitrary metadata as JSON in a metadata column.

### REQ-telemetry.store-metadata-empty: Empty metadata handling
TelemetryStore handles records with no metadata gracefully.

### REQ-telemetry.store-persistence: Store persistence across sessions
TelemetryStore data persists across store close/reopen cycles.

### REQ-telemetry.store-reopen-existing: Reopen existing store
TelemetryStore reopens an existing database file and reads previously stored records.

### REQ-telemetry.store-schema-migration: Schema migration
TelemetryStore handles schema migrations when opening databases with older schemas.

### REQ-telemetry.store-bus-subscribe: EventBus subscription
TelemetryStore can subscribe to an EventBus to automatically record telemetry from events.

### REQ-telemetry.store-bus-ignores-non-record: Bus ignores non-record events
TelemetryStore bus handler ignores events that are not TelemetryRecord instances.

### REQ-telemetry.store-bus-ignores-non-telemetry-record: Bus ignores non-telemetry record types
TelemetryStore bus handler ignores record types that are not TelemetryRecord.

## Derived Metrics (detailed)

### REQ-telemetry.derived-flops-dense: Dense model FLOPS calculation
Derived metrics compute FLOPS for dense (non-MoE) models based on parameter count and tokens.

### REQ-telemetry.derived-flops-moe: MoE model FLOPS calculation
Derived metrics compute FLOPS for Mixture-of-Experts models using active parameter count.

### REQ-telemetry.derived-flops-none-active: FLOPS with no active params
Derived metrics handle models with no active parameter count gracefully.

### REQ-telemetry.derived-bytes-fp16: FP16 memory bytes calculation
Derived metrics compute memory bytes for FP16/BF16 models (2 bytes per parameter).

### REQ-telemetry.derived-bytes-int8: INT8 memory bytes calculation
Derived metrics compute memory bytes for INT8 quantized models (1 byte per parameter).

### REQ-telemetry.derived-mfu: Model FLOPS utilization
Derived metrics compute MFU (Model FLOPS Utilization) as actual vs. theoretical peak FLOPS.

### REQ-telemetry.derived-mbu: Memory bandwidth utilization
Derived metrics compute MBU (Memory Bandwidth Utilization) for memory-bound inference.

### REQ-telemetry.derived-ipj: Inferences per joule
Derived metrics compute inferences per joule as an energy efficiency measure.

### REQ-telemetry.derived-ipj-zero-energy: IPJ with zero energy
Derived metrics return zero IPJ when energy consumption is zero.

### REQ-telemetry.derived-zero-peak: Zero peak FLOPS handling
Derived metrics handle zero peak FLOPS gracefully (avoids division by zero).

### REQ-telemetry.derived-multi-gpu: Multi-GPU metrics
Derived metrics aggregate across multiple GPUs for multi-GPU inference setups.

### REQ-telemetry.derived-efficiency-returns-dataclass: Efficiency returns dataclass
Derived efficiency computation returns a structured dataclass with all efficiency metrics.

### REQ-telemetry.derived-store-roundtrip: Derived metrics store roundtrip
Derived metrics can be stored and retrieved from the telemetry store.

### REQ-telemetry.derived-summary-weighted: Weighted summary metrics
Derived summary metrics use token-count weighted averages for aggregation.

### REQ-telemetry.derived-itl-basic: Basic ITL computation
Derived metrics compute basic inter-token latency statistics from token timestamps.

### REQ-telemetry.derived-itl-two-timestamps: ITL with two timestamps
Derived metrics compute ITL from exactly two timestamps (single interval).

### REQ-telemetry.derived-itl-too-few: ITL with too few timestamps
Derived metrics return None/empty ITL when fewer than two timestamps are available.

### REQ-telemetry.derived-itl-varying: ITL with varying intervals
Derived metrics correctly compute ITL statistics from timestamps with varying intervals.

### REQ-telemetry.derived-itl-percentiles-ordered: ITL percentiles ordering
Derived ITL percentiles (p90, p95, p99) are correctly ordered (p90 <= p95 <= p99).

## Tests

- `tests/telemetry/test_*.py` - 24+ telemetry test files
