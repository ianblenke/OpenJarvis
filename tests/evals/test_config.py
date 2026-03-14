"""Tests for eval configuration loading, validation, and matrix expansion."""

from __future__ import annotations

import pytest

from openjarvis.evals.core.config import (
    KNOWN_BENCHMARKS,
    VALID_BACKENDS,
    EvalConfigError,
    expand_suite,
    load_eval_config,
)
from openjarvis.evals.core.types import (
    BenchmarkConfig,
    DefaultsConfig,
    EvalSuiteConfig,
    ExecutionConfig,
    JudgeConfig,
    ModelConfig,
    RunConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path, content: str, filename: str = "eval.toml"):
    path = tmp_path / filename
    path.write_text(content)
    return path


def _minimal_toml() -> str:
    return (
        '[[models]]\n'
        'name = "qwen3:8b"\n'
        '\n'
        '[[benchmarks]]\n'
        'name = "supergpqa"\n'
    )


def _full_toml() -> str:
    return (
        '[meta]\n'
        'name = "test-suite"\n'
        'description = "A test eval suite"\n'
        '\n'
        '[defaults]\n'
        'temperature = 0.5\n'
        'max_tokens = 4096\n'
        '\n'
        '[judge]\n'
        'model = "gpt-4o"\n'
        'engine = "cloud"\n'
        'temperature = 0.0\n'
        'max_tokens = 1024\n'
        '\n'
        '[run]\n'
        'max_workers = 8\n'
        'output_dir = "output/"\n'
        'seed = 123\n'
        'telemetry = true\n'
        'gpu_metrics = true\n'
        'warmup_samples = 2\n'
        'wandb_project = "my-project"\n'
        'wandb_entity = "my-team"\n'
        '\n'
        '[[models]]\n'
        'name = "gpt-4o"\n'
        'engine = "openai"\n'
        'temperature = 0.3\n'
        'max_tokens = 2048\n'
        'param_count_b = 1.8\n'
        'gpu_peak_tflops = 312.0\n'
        '\n'
        '[[models]]\n'
        'name = "qwen3:8b"\n'
        '\n'
        '[[benchmarks]]\n'
        'name = "supergpqa"\n'
        'backend = "jarvis-direct"\n'
        'max_samples = 50\n'
        'split = "test"\n'
        '\n'
        '[[benchmarks]]\n'
        'name = "gaia"\n'
        'backend = "jarvis-agent"\n'
        'judge_model = "claude-3-opus"\n'
        'temperature = 0.7\n'
        'max_tokens = 8192\n'
    )


# ---------------------------------------------------------------------------
# Tests: load_eval_config
# ---------------------------------------------------------------------------


class TestLoadEvalConfig:
    """Tests for TOML config file loading."""

    @pytest.mark.spec("REQ-evals.config.load")
    def test_load_minimal_config(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)

        assert len(config.models) == 1
        assert config.models[0].name == "qwen3:8b"
        assert len(config.benchmarks) == 1
        assert config.benchmarks[0].name == "supergpqa"

    @pytest.mark.spec("REQ-evals.config.load")
    def test_load_full_config(self, tmp_path):
        path = _write_toml(tmp_path, _full_toml())
        config = load_eval_config(path)

        assert config.meta.name == "test-suite"
        assert config.meta.description == "A test eval suite"
        assert config.defaults.temperature == 0.5
        assert config.defaults.max_tokens == 4096
        assert config.judge.model == "gpt-4o"
        assert config.run.max_workers == 8
        assert config.run.seed == 123
        assert config.run.telemetry is True
        assert config.run.warmup_samples == 2
        assert len(config.models) == 2
        assert len(config.benchmarks) == 2

    @pytest.mark.spec("REQ-evals.config.load")
    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_config(tmp_path / "nonexistent.toml")


class TestLoadEvalConfigValidation:
    """Tests for config validation rules."""

    @pytest.mark.spec("REQ-evals.config.validation")
    def test_no_models_raises(self, tmp_path):
        content = '[[benchmarks]]\nname = "supergpqa"\n'
        path = _write_toml(tmp_path, content)
        with pytest.raises(EvalConfigError, match="models"):
            load_eval_config(path)

    @pytest.mark.spec("REQ-evals.config.validation")
    def test_no_benchmarks_raises(self, tmp_path):
        content = '[[models]]\nname = "test-model"\n'
        path = _write_toml(tmp_path, content)
        with pytest.raises(EvalConfigError, match="benchmarks"):
            load_eval_config(path)

    @pytest.mark.spec("REQ-evals.config.validation")
    def test_model_without_name_raises(self, tmp_path):
        content = (
            '[[models]]\n'
            'engine = "openai"\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "supergpqa"\n'
        )
        path = _write_toml(tmp_path, content)
        with pytest.raises(EvalConfigError, match="name"):
            load_eval_config(path)

    @pytest.mark.spec("REQ-evals.config.validation")
    def test_benchmark_without_name_raises(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test-model"\n'
            '\n'
            '[[benchmarks]]\n'
            'backend = "jarvis-direct"\n'
        )
        path = _write_toml(tmp_path, content)
        with pytest.raises(EvalConfigError, match="name"):
            load_eval_config(path)

    @pytest.mark.spec("REQ-evals.config.validation")
    def test_invalid_backend_raises(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test-model"\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "supergpqa"\n'
            'backend = "invalid-backend"\n'
        )
        path = _write_toml(tmp_path, content)
        with pytest.raises(EvalConfigError, match="invalid-backend"):
            load_eval_config(path)


class TestLoadEvalConfigDefaults:
    """Tests for default values when config sections are omitted."""

    @pytest.mark.spec("REQ-evals.config.defaults")
    def test_default_temperature(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)
        assert config.defaults.temperature == 0.0

    @pytest.mark.spec("REQ-evals.config.defaults")
    def test_default_max_tokens(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)
        assert config.defaults.max_tokens == 2048

    @pytest.mark.spec("REQ-evals.config.defaults")
    def test_default_max_workers(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)
        assert config.run.max_workers == 4

    @pytest.mark.spec("REQ-evals.config.defaults")
    def test_default_seed(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)
        assert config.run.seed == 42

    @pytest.mark.spec("REQ-evals.config.defaults")
    def test_default_benchmark_backend(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)
        assert config.benchmarks[0].backend == "jarvis-direct"

    @pytest.mark.spec("REQ-evals.config.defaults")
    def test_default_meta_empty(self, tmp_path):
        path = _write_toml(tmp_path, _minimal_toml())
        config = load_eval_config(path)
        assert config.meta.name == ""
        assert config.meta.description == ""


class TestLoadEvalConfigModelFields:
    """Tests for model-level config fields."""

    @pytest.mark.spec("REQ-evals.config.model-fields")
    def test_model_engine_and_provider(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test-model"\n'
            'engine = "vllm"\n'
            'provider = "local"\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "supergpqa"\n'
        )
        path = _write_toml(tmp_path, content)
        config = load_eval_config(path)
        assert config.models[0].engine == "vllm"
        assert config.models[0].provider == "local"

    @pytest.mark.spec("REQ-evals.config.model-fields")
    def test_model_hardware_params(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test-model"\n'
            'param_count_b = 7.0\n'
            'active_params_b = 3.5\n'
            'gpu_peak_tflops = 312.0\n'
            'gpu_peak_bandwidth_gb_s = 2048.0\n'
            'num_gpus = 4\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "supergpqa"\n'
        )
        path = _write_toml(tmp_path, content)
        config = load_eval_config(path)
        m = config.models[0]
        assert m.param_count_b == 7.0
        assert m.active_params_b == 3.5
        assert m.gpu_peak_tflops == 312.0
        assert m.gpu_peak_bandwidth_gb_s == 2048.0
        assert m.num_gpus == 4


class TestLoadEvalConfigBenchmarkFields:
    """Tests for benchmark-level config fields."""

    @pytest.mark.spec("REQ-evals.config.benchmark-fields")
    def test_benchmark_max_samples_and_split(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test"\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "gaia"\n'
            'max_samples = 100\n'
            'split = "validation"\n'
        )
        path = _write_toml(tmp_path, content)
        config = load_eval_config(path)
        b = config.benchmarks[0]
        assert b.max_samples == 100
        assert b.split == "validation"

    @pytest.mark.spec("REQ-evals.config.benchmark-fields")
    def test_benchmark_tools(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test"\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "gaia"\n'
            'tools = ["web_search", "code_exec"]\n'
        )
        path = _write_toml(tmp_path, content)
        config = load_eval_config(path)
        assert config.benchmarks[0].tools == ["web_search", "code_exec"]

    @pytest.mark.spec("REQ-evals.config.benchmark-fields")
    def test_benchmark_judge_model_override(self, tmp_path):
        content = (
            '[[models]]\n'
            'name = "test"\n'
            '\n'
            '[[benchmarks]]\n'
            'name = "gaia"\n'
            'judge_model = "claude-3-opus"\n'
        )
        path = _write_toml(tmp_path, content)
        config = load_eval_config(path)
        assert config.benchmarks[0].judge_model == "claude-3-opus"


# ---------------------------------------------------------------------------
# Tests: expand_suite
# ---------------------------------------------------------------------------


class TestExpandSuite:
    """Tests for expanding EvalSuiteConfig into RunConfig list."""

    @pytest.mark.spec("REQ-evals.config.expand")
    def test_single_model_single_benchmark(self):
        suite = EvalSuiteConfig(
            models=[ModelConfig(name="test-model")],
            benchmarks=[BenchmarkConfig(name="supergpqa")],
        )
        configs = expand_suite(suite)

        assert len(configs) == 1
        assert configs[0].benchmark == "supergpqa"
        assert configs[0].model == "test-model"

    @pytest.mark.spec("REQ-evals.config.expand")
    def test_cross_product(self):
        suite = EvalSuiteConfig(
            models=[
                ModelConfig(name="model-a"),
                ModelConfig(name="model-b"),
            ],
            benchmarks=[
                BenchmarkConfig(name="bench-x"),
                BenchmarkConfig(name="bench-y"),
                BenchmarkConfig(name="bench-z"),
            ],
        )
        configs = expand_suite(suite)

        assert len(configs) == 6
        pairs = [(c.model, c.benchmark) for c in configs]
        assert ("model-a", "bench-x") in pairs
        assert ("model-a", "bench-y") in pairs
        assert ("model-a", "bench-z") in pairs
        assert ("model-b", "bench-x") in pairs
        assert ("model-b", "bench-y") in pairs
        assert ("model-b", "bench-z") in pairs


class TestExpandSuitePrecedence:
    """Tests for parameter merge precedence: benchmark > model > defaults."""

    @pytest.mark.spec("REQ-evals.config.expand-precedence")
    def test_defaults_used_when_no_overrides(self):
        suite = EvalSuiteConfig(
            defaults=DefaultsConfig(temperature=0.5, max_tokens=4096),
            models=[ModelConfig(name="model-a")],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert configs[0].temperature == 0.5
        assert configs[0].max_tokens == 4096

    @pytest.mark.spec("REQ-evals.config.expand-precedence")
    def test_model_overrides_defaults(self):
        suite = EvalSuiteConfig(
            defaults=DefaultsConfig(temperature=0.5, max_tokens=4096),
            models=[ModelConfig(name="model-a", temperature=0.3, max_tokens=2048)],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert configs[0].temperature == 0.3
        assert configs[0].max_tokens == 2048

    @pytest.mark.spec("REQ-evals.config.expand-precedence")
    def test_benchmark_overrides_model(self):
        suite = EvalSuiteConfig(
            defaults=DefaultsConfig(temperature=0.5),
            models=[ModelConfig(name="model-a", temperature=0.3)],
            benchmarks=[BenchmarkConfig(name="bench", temperature=0.9)],
        )
        configs = expand_suite(suite)

        assert configs[0].temperature == 0.9

    @pytest.mark.spec("REQ-evals.config.expand-precedence")
    def test_benchmark_judge_model_overrides_suite_judge(self):
        suite = EvalSuiteConfig(
            judge=JudgeConfig(model="default-judge"),
            models=[ModelConfig(name="model-a")],
            benchmarks=[BenchmarkConfig(name="bench", judge_model="special-judge")],
        )
        configs = expand_suite(suite)

        assert configs[0].judge_model == "special-judge"

    @pytest.mark.spec("REQ-evals.config.expand-precedence")
    def test_suite_judge_used_when_benchmark_has_none(self):
        suite = EvalSuiteConfig(
            judge=JudgeConfig(model="default-judge"),
            models=[ModelConfig(name="model-a")],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert configs[0].judge_model == "default-judge"


class TestExpandSuiteOutputPath:
    """Tests for output path generation."""

    @pytest.mark.spec("REQ-evals.config.expand-output-path")
    def test_auto_generated_output_path(self):
        suite = EvalSuiteConfig(
            run=ExecutionConfig(output_dir="results/"),
            models=[ModelConfig(name="gpt-4o")],
            benchmarks=[BenchmarkConfig(name="supergpqa")],
        )
        configs = expand_suite(suite)

        assert configs[0].output_path == "results/supergpqa_gpt-4o.jsonl"

    @pytest.mark.spec("REQ-evals.config.expand-output-path")
    def test_model_slug_replaces_special_chars(self):
        suite = EvalSuiteConfig(
            run=ExecutionConfig(output_dir="out/"),
            models=[ModelConfig(name="org/model:v2")],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert "org-model-v2" in configs[0].output_path

    @pytest.mark.spec("REQ-evals.config.expand-output-path")
    def test_output_dir_trailing_slash_stripped(self):
        suite = EvalSuiteConfig(
            run=ExecutionConfig(output_dir="results///"),
            models=[ModelConfig(name="m")],
            benchmarks=[BenchmarkConfig(name="b")],
        )
        configs = expand_suite(suite)

        # Should not have double slashes
        assert "///" not in configs[0].output_path


class TestExpandSuiteMetadata:
    """Tests for model metadata propagation."""

    @pytest.mark.spec("REQ-evals.config.expand-metadata")
    def test_model_meta_propagated(self):
        suite = EvalSuiteConfig(
            models=[ModelConfig(
                name="model",
                param_count_b=7.0,
                gpu_peak_tflops=312.0,
            )],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert configs[0].metadata["param_count_b"] == 7.0
        assert configs[0].metadata["gpu_peak_tflops"] == 312.0

    @pytest.mark.spec("REQ-evals.config.expand-metadata")
    def test_zero_params_not_in_metadata(self):
        suite = EvalSuiteConfig(
            models=[ModelConfig(name="model", param_count_b=0.0)],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert "param_count_b" not in configs[0].metadata

    @pytest.mark.spec("REQ-evals.config.expand-metadata")
    def test_multi_gpu_in_metadata(self):
        suite = EvalSuiteConfig(
            models=[ModelConfig(name="model", num_gpus=4)],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert configs[0].metadata["num_gpus"] == 4

    @pytest.mark.spec("REQ-evals.config.expand-metadata")
    def test_single_gpu_not_in_metadata(self):
        suite = EvalSuiteConfig(
            models=[ModelConfig(name="model", num_gpus=1)],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        assert "num_gpus" not in configs[0].metadata


class TestExpandSuiteRunFields:
    """Tests for execution-level fields propagation."""

    @pytest.mark.spec("REQ-evals.config.expand-run-fields")
    def test_run_fields_propagated(self):
        suite = EvalSuiteConfig(
            run=ExecutionConfig(
                max_workers=16,
                seed=99,
                telemetry=True,
                gpu_metrics=True,
                warmup_samples=5,
                wandb_project="proj",
                wandb_entity="team",
            ),
            models=[ModelConfig(name="model")],
            benchmarks=[BenchmarkConfig(name="bench")],
        )
        configs = expand_suite(suite)

        c = configs[0]
        assert c.max_workers == 16
        assert c.seed == 99
        assert c.telemetry is True
        assert c.gpu_metrics is True
        assert c.warmup_samples == 5
        assert c.wandb_project == "proj"
        assert c.wandb_entity == "team"


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestConfigConstants:
    """Tests for module-level constants."""

    @pytest.mark.spec("REQ-evals.config.constants")
    def test_valid_backends(self):
        assert "jarvis-direct" in VALID_BACKENDS
        assert "jarvis-agent" in VALID_BACKENDS

    @pytest.mark.spec("REQ-evals.config.constants")
    def test_known_benchmarks_non_empty(self):
        assert len(KNOWN_BENCHMARKS) > 0
        assert "supergpqa" in KNOWN_BENCHMARKS
        assert "gaia" in KNOWN_BENCHMARKS


# ---------------------------------------------------------------------------
# Tests: Data classes
# ---------------------------------------------------------------------------


class TestConfigDataClasses:
    """Tests for config-related dataclasses."""

    @pytest.mark.spec("REQ-evals.config.dataclasses")
    def test_run_config_defaults(self):
        rc = RunConfig(benchmark="b", backend="jarvis-direct", model="m")
        assert rc.max_workers == 4
        assert rc.temperature == 0.0
        assert rc.max_tokens == 2048
        assert rc.seed == 42
        assert rc.episode_mode is False
        assert rc.warmup_samples == 0

    @pytest.mark.spec("REQ-evals.config.dataclasses")
    def test_model_config_defaults(self):
        mc = ModelConfig(name="test")
        assert mc.engine is None
        assert mc.temperature is None
        assert mc.param_count_b == 0.0
        assert mc.num_gpus == 1

    @pytest.mark.spec("REQ-evals.config.dataclasses")
    def test_benchmark_config_defaults(self):
        bc = BenchmarkConfig(name="test")
        assert bc.backend == "jarvis-direct"
        assert bc.max_samples is None
        assert bc.split is None
        assert bc.tools == []

    @pytest.mark.spec("REQ-evals.config.dataclasses")
    def test_eval_suite_config_defaults(self):
        esc = EvalSuiteConfig()
        assert esc.meta.name == ""
        assert esc.defaults.temperature == 0.0
        assert esc.models == []
        assert esc.benchmarks == []
