# Evals Module Spec

Evaluation framework with 30+ benchmark datasets, 35+ scorers, and multiple execution backends.

## Runner

### REQ-evals.runner: Evaluation runner
Orchestrates dataset loading, model inference, scoring, and result tracking.

### REQ-evals.runner.agentic: Agentic evaluation
Evaluates agent pipelines (not just single-turn inference).

## Datasets (30+)

### REQ-evals.datasets: Benchmark dataset loading
Each dataset class wraps a HuggingFace dataset or custom data source. Returns standardized fields for evaluation.

Key datasets: MMLU, GSM8K, HumanEval, HellaSwag, ARC, TruthfulQA, MATH, WinoGrande, BigBenchHard, IFEval, LiveCodeBench, SWE-bench, GPQA, MuSR, MGSM, SimpleQA, BrowseComp, etc.

## Scorers (35+)

### REQ-evals.scorers: Scoring implementations
Deterministic scoring logic for each benchmark type. Categories: exact match, MCQ, code execution, structural checks, LLM-as-judge.

## Backends

### REQ-evals.backends: Execution backends
Backend implementations for running evaluations against different engines.

## Trackers

### REQ-evals.trackers.wandb: Weights & Biases tracking
Log evaluation results to W&B.

### REQ-evals.trackers.sheets: Google Sheets tracking
Log evaluation results to Google Sheets.

## Environments

### REQ-evals.environments: Execution environments
Docker/cloud execution environments for sandboxed evaluation.

## Configuration

### REQ-evals.config.dataclasses: Eval configuration dataclasses
Eval config uses structured dataclasses for benchmark, model, run, and suite configuration with typed fields.

### REQ-evals.config.defaults: Default configuration values
Eval config provides sensible defaults for all optional fields including temperature, max_tokens, and output paths.

### REQ-evals.config.load: Configuration loading
`load_eval_config(path)` loads eval configuration from TOML files with section-based parsing.

### REQ-evals.config.validation: Configuration validation
Eval config validates required fields and raises errors for invalid combinations (e.g., missing model, empty benchmarks).

### REQ-evals.config.expand: Configuration expansion
Config expansion resolves template variables and expands shorthand notation into full configuration objects.

### REQ-evals.config.expand-metadata: Metadata expansion
Config expansion populates metadata fields from context (timestamps, run IDs, environment info).

### REQ-evals.config.expand-output-path: Output path expansion
Config expansion resolves output path templates with model name, timestamp, and benchmark identifiers.

### REQ-evals.config.expand-precedence: Expansion precedence rules
Config expansion follows precedence: explicit values override suite defaults, which override global defaults.

### REQ-evals.config.expand-run-fields: Run field expansion
Config expansion populates run-level fields like run_id, timestamp, and agent configuration.

### REQ-evals.config.benchmark-fields: Benchmark configuration fields
Benchmark config includes name, dataset, scorer, max_samples, and optional overrides for model parameters.

### REQ-evals.config.model-fields: Model configuration fields
Model config includes provider, model name, temperature, max_tokens, and API endpoint settings.

### REQ-evals.config.constants: Eval configuration constants
Eval module defines constants for default values, supported formats, and standard field names.

## Datasets (detailed)

### REQ-evals.dataset.base-protocol: Dataset base protocol
All datasets implement a common protocol with `load()`, iteration, length, and field access methods.

### REQ-evals.dataset.load: Dataset loading
`load()` method fetches and prepares dataset records from the configured source (HuggingFace, file, etc.).

### REQ-evals.dataset.file-load: File-based dataset loading
Datasets can be loaded from local JSON/JSONL files with field mapping to the standard eval record format.

### REQ-evals.dataset.iteration: Dataset iteration
Datasets support iteration via `__iter__` and `__len__`, yielding standardized eval records.

### REQ-evals.dataset.episodes: Episode-based datasets
Some datasets support multi-turn episodes where each record contains a sequence of messages rather than a single prompt.

### REQ-evals.dataset.eval-record: Eval record structure
`EvalRecord` provides a standardized structure with prompt, expected answer, metadata, and optional episode data.

## Runner (detailed)

### REQ-evals.runner.execution: Runner execution pipeline
The eval runner executes the full pipeline: load dataset, warm up, run inference, score, aggregate, and track results.

### REQ-evals.runner.dataset-loading: Runner dataset loading
The runner loads and validates the configured dataset before starting evaluation, raising on missing/empty datasets.

### REQ-evals.runner.warmup: Runner warmup phase
The runner optionally performs warmup inference calls before the main evaluation to stabilize latency measurements.

### REQ-evals.runner.format-messages: Message formatting for inference
The runner formats eval records into the message format expected by the inference engine (system/user/assistant).

### REQ-evals.runner.think-tag-stripping: Think tag stripping in output
The runner strips `<think>...</think>` blocks from model output before scoring to isolate the final answer.

### REQ-evals.runner.episode-mode: Episode evaluation mode
The runner supports multi-turn episode evaluation where each episode involves multiple inference calls with conversation history.

### REQ-evals.runner.result-aggregation: Result aggregation
The runner aggregates per-record scores into summary statistics (mean, median, min, max, std) across the evaluation.

### REQ-evals.runner.metric-stats: Metric statistics computation
The runner computes detailed metric statistics including per-benchmark breakdowns and confidence intervals.

### REQ-evals.runner.output: Runner output persistence
The runner writes evaluation results to the configured output path in JSON format with full metadata.

### REQ-evals.runner.tracker-lifecycle: Tracker lifecycle management
The runner manages tracker lifecycle: initialize before eval, log during eval, finalize after eval completes.

### REQ-evals.runner.progress-callback: Progress callback support
The runner supports an optional progress callback invoked after each record for real-time progress reporting.

## Scorers (detailed)

### REQ-evals.scorer.base-protocol: Scorer base protocol
All scorers implement a common protocol with a `score(prediction, reference, **kwargs) -> float` method.

### REQ-evals.scorer.gaia-exact: GAIA exact match scorer
Scores GAIA benchmark responses using exact string matching with normalization (case, whitespace, articles).

### REQ-evals.scorer.gaia-scorer: GAIA composite scorer
GAIA scorer combines exact match with numeric tolerance and list comparison for varied answer types.

### REQ-evals.scorer.gpqa-mcq: GPQA multiple-choice scorer
Scores GPQA responses by extracting the selected option letter and comparing to the correct answer.

### REQ-evals.scorer.mmlu-pro-mcq: MMLU-Pro multiple-choice scorer
Scores MMLU-Pro responses by extracting the selected option from model output with flexible parsing.

### REQ-evals.scorer.supergpqa-mcq: SuperGPQA multiple-choice scorer
Scores SuperGPQA responses using multiple-choice answer extraction with domain-specific normalization.

### REQ-evals.scorer.contains-phrases: Contains-phrases scorer
Scores responses by checking whether required phrases appear in the model output (case-insensitive).

### REQ-evals.scorer.email-triage: Email triage scorer
Scores email triage benchmark responses by comparing predicted labels/actions to expected classifications.

### REQ-evals.scorer.normalize-number: Number normalization
Normalizes numeric strings for comparison (strips commas, handles percentages, converts words to digits).

### REQ-evals.scorer.normalize-str: String normalization
Normalizes strings for comparison by lowercasing, stripping whitespace, removing articles and punctuation.

### REQ-evals.scorer.llm-judge-base: LLM-as-judge base scorer
Base scorer that uses an LLM to evaluate response quality against a rubric or reference answer.

## Trackers (detailed)

### REQ-evals.tracker.base-protocol: Tracker base protocol
All trackers implement initialize/log/finalize lifecycle methods for recording evaluation results.

### REQ-evals.tracker.lifecycle: Tracker lifecycle
Trackers are initialized before evaluation, receive log calls during evaluation, and finalize after completion.

### REQ-evals.tracker.stat-val: Tracker stat value formatting
Trackers format metric statistics into display-ready values with appropriate precision and units.

### REQ-evals.tracker.flatten-metric-stats: Metric stats flattening
Trackers flatten nested metric statistics into a flat key-value map for logging backends.

### REQ-evals.tracker.wandb-init: W&B tracker initialization
W&B tracker initializes a wandb run with project name, config, and tags from the eval configuration.

### REQ-evals.tracker.wandb-log: W&B tracker logging
W&B tracker logs per-record metrics and aggregated results to the wandb run.

### REQ-evals.tracker.wandb-finish: W&B tracker finalization
W&B tracker calls `wandb.finish()` to complete the run and upload remaining data.

### REQ-evals.tracker.wandb-import: W&B import handling
W&B tracker handles import of the wandb library gracefully, with clear errors when not installed.

### REQ-evals.tracker.wandb-summary: W&B summary metrics
W&B tracker writes summary metrics to `wandb.summary` for dashboard display.

### REQ-evals.tracker.sheets-lifecycle: Google Sheets tracker lifecycle
Sheets tracker connects to the spreadsheet on init, appends rows during eval, and disconnects on finalize.

### REQ-evals.tracker.sheets-row: Google Sheets row formatting
Sheets tracker formats evaluation results into spreadsheet rows with consistent column ordering.

### REQ-evals.tracker.sheets-import: Google Sheets import handling
Sheets tracker handles import of gspread/google-auth libraries gracefully, with clear errors when not installed.

### REQ-evals.tracker.sheets-noop: Google Sheets no-op behavior
Sheets tracker gracefully no-ops when credentials are missing or the spreadsheet is not configured.

## Tests

- `tests/evals/test_*.py` - 18+ evaluation test files
- `src/openjarvis/evals/tests/` - Internal evaluation tests
