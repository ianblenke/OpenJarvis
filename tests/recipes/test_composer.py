"""Tests for recipe composer bridges -- recipe_to_eval_suite and recipe_to_operator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openjarvis.recipes.composer import recipe_to_eval_suite, recipe_to_operator
from openjarvis.recipes.loader import Recipe, load_recipe

DISCRETE_TOML = textwrap.dedent("""\
    [recipe]
    name = "bench-agent"
    kind = "discrete"
    description = "A discrete agent for benchmarking"
    version = "0.1.0"

    [intelligence]
    model = "qwen3:8b"
    quantization = "q4_K_M"
    provider = "ollama"

    [engine]
    key = "ollama"

    [agent]
    type = "native_react"
    max_turns = 20
    temperature = 0.2
    tools = ["shell_exec", "file_read", "think"]
    system_prompt = "You are a benchmark agent."

    [learning]
    routing = "heuristic"
    agent = "none"

    [eval]
    benchmarks = ["terminalbench", "gaia"]
    backend = "jarvis-agent"
    max_samples = 50
    judge_model = "gpt-4o"
""")

OPERATOR_TOML = textwrap.dedent("""\
    [recipe]
    name = "my-operator"
    kind = "operator"
    description = "A test operator"
    version = "2.0.0"

    [intelligence]
    model = "qwen3:8b"

    [engine]
    key = "ollama"

    [agent]
    type = "orchestrator"
    max_turns = 15
    temperature = 0.3
    tools = ["web_search", "memory_store", "think"]
    system_prompt = "You are a monitoring agent."

    [schedule]
    type = "interval"
    value = "600"

    [channels]
    output = ["slack", "telegram"]

    [learning]
    routing = "heuristic"
""")


class TestRecipeToEvalSuite:
    @pytest.mark.spec("REQ-recipes.compose-eval")
    @pytest.mark.spec("REQ-recipes.composer.eval-suite-basic")
    def test_basic_eval_suite(self, tmp_path: Path) -> None:
        p = tmp_path / "bench.toml"
        p.write_text(DISCRETE_TOML)
        r = load_recipe(p)

        suite = recipe_to_eval_suite(r)

        assert suite.meta.name == "bench-agent-eval"
        assert "Auto-generated" in suite.meta.description
        assert len(suite.models) == 1
        assert suite.models[0].name == "qwen3:8b"
        assert suite.models[0].engine == "ollama"
        assert suite.models[0].provider == "ollama"
        assert len(suite.benchmarks) == 2
        bench_names = {b.name for b in suite.benchmarks}
        assert bench_names == {"terminalbench", "gaia"}
        for b in suite.benchmarks:
            assert b.backend == "jarvis-agent"
            assert b.agent == "native_react"
            assert b.tools == ["shell_exec", "file_read", "think"]
            assert b.max_samples == 50
            assert b.judge_model == "gpt-4o"

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-benchmark-override")
    def test_eval_suite_benchmark_override(self, tmp_path: Path) -> None:
        p = tmp_path / "bench.toml"
        p.write_text(DISCRETE_TOML)
        r = load_recipe(p)

        suite = recipe_to_eval_suite(r, benchmarks=["supergpqa"])

        assert len(suite.benchmarks) == 1
        assert suite.benchmarks[0].name == "supergpqa"

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-max-samples-override")
    def test_eval_suite_max_samples_override(self, tmp_path: Path) -> None:
        p = tmp_path / "bench.toml"
        p.write_text(DISCRETE_TOML)
        r = load_recipe(p)

        suite = recipe_to_eval_suite(r, max_samples=10)

        for b in suite.benchmarks:
            assert b.max_samples == 10

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-judge-override")
    def test_eval_suite_judge_override(self, tmp_path: Path) -> None:
        p = tmp_path / "bench.toml"
        p.write_text(DISCRETE_TOML)
        r = load_recipe(p)

        suite = recipe_to_eval_suite(r, judge_model="gpt-5")

        assert suite.judge.model == "gpt-5"
        for b in suite.benchmarks:
            assert b.judge_model == "gpt-5"

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-no-model-error")
    def test_eval_suite_no_model_raises(self) -> None:
        r = Recipe(name="no-model", eval_benchmarks=["gaia"])
        with pytest.raises(ValueError, match="no model"):
            recipe_to_eval_suite(r)

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-no-benchmarks-error")
    def test_eval_suite_no_benchmarks_raises(self) -> None:
        r = Recipe(name="no-bench", model="qwen3:8b")
        with pytest.raises(ValueError, match="no benchmarks"):
            recipe_to_eval_suite(r)

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-suites-fallback")
    def test_eval_suite_falls_back_to_suites(self) -> None:
        r = Recipe(
            name="suites-fallback",
            model="qwen3:8b",
            eval_suites=["coding"],
        )
        suite = recipe_to_eval_suite(r)
        assert len(suite.benchmarks) == 1
        assert suite.benchmarks[0].name == "coding"

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-direct-backend")
    def test_eval_suite_direct_backend_when_no_agent(self) -> None:
        r = Recipe(
            name="direct",
            model="qwen3:8b",
            eval_benchmarks=["supergpqa"],
        )
        suite = recipe_to_eval_suite(r)
        assert suite.benchmarks[0].backend == "jarvis-direct"
        assert suite.benchmarks[0].agent is None
        assert suite.benchmarks[0].tools == []

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-defaults")
    def test_eval_suite_defaults_config(self, tmp_path: Path) -> None:
        p = tmp_path / "bench.toml"
        p.write_text(DISCRETE_TOML)
        r = load_recipe(p)

        suite = recipe_to_eval_suite(r)

        assert suite.defaults.temperature == pytest.approx(0.2)
        assert suite.defaults.max_tokens == 2048

    @pytest.mark.spec("REQ-recipes.composer.eval-suite-model-temperature")
    def test_eval_suite_inherits_model_temperature(self, tmp_path: Path) -> None:
        p = tmp_path / "bench.toml"
        p.write_text(DISCRETE_TOML)
        r = load_recipe(p)

        suite = recipe_to_eval_suite(r)
        assert suite.models[0].temperature == pytest.approx(0.2)


class TestRecipeToOperator:
    @pytest.mark.spec("REQ-recipes.compose-operator")
    @pytest.mark.spec("REQ-recipes.composer.operator-basic")
    def test_basic_operator_manifest(self, tmp_path: Path) -> None:
        p = tmp_path / "op.toml"
        p.write_text(OPERATOR_TOML)
        r = load_recipe(p)

        m = recipe_to_operator(r)

        assert m.id == "my-operator"
        assert m.name == "my-operator"
        assert m.version == "2.0.0"
        assert m.description == "A test operator"
        assert m.tools == ["web_search", "memory_store", "think"]
        assert m.system_prompt == "You are a monitoring agent."
        assert m.max_turns == 15
        assert m.temperature == pytest.approx(0.3)
        assert m.schedule_type == "interval"
        assert m.schedule_value == "600"

    @pytest.mark.spec("REQ-recipes.composer.operator-no-schedule-error")
    def test_operator_no_schedule_raises(self) -> None:
        r = Recipe(name="no-sched", kind="operator")
        with pytest.raises(ValueError, match="no \\[schedule\\]"):
            recipe_to_operator(r)

    @pytest.mark.spec("REQ-recipes.composer.operator-defaults")
    def test_operator_defaults(self) -> None:
        r = Recipe(
            name="minimal-op",
            kind="operator",
            schedule_type="interval",
            schedule_value="300",
        )
        m = recipe_to_operator(r)
        assert m.max_turns == 20
        assert m.temperature == pytest.approx(0.3)
        assert m.schedule_value == "300"
        assert m.system_prompt == ""
        assert m.system_prompt_path == ""
        assert m.tools == []
        assert m.required_capabilities == []

    @pytest.mark.spec("REQ-recipes.composer.operator-required-capabilities")
    def test_operator_required_capabilities(self) -> None:
        r = Recipe(
            name="cap-op",
            kind="operator",
            schedule_type="cron",
            schedule_value="0 * * * *",
            required_capabilities=["network", "filesystem"],
        )
        m = recipe_to_operator(r)
        assert m.required_capabilities == ["network", "filesystem"]

    @pytest.mark.spec("REQ-recipes.composer.operator-prompt-fields")
    def test_operator_prompt_fields(self) -> None:
        r = Recipe(
            name="prompt-op",
            kind="operator",
            schedule_type="interval",
            schedule_value="60",
            system_prompt="Custom prompt",
            system_prompt_path="/path/to/prompt.md",
        )
        m = recipe_to_operator(r)
        assert m.system_prompt == "Custom prompt"
        assert m.system_prompt_path == "/path/to/prompt.md"

    @pytest.mark.spec("REQ-recipes.composer.operator-schedule-value-default")
    def test_operator_schedule_value_default(self) -> None:
        """When schedule_value is None, defaults to '300'."""
        r = Recipe(
            name="default-sched",
            kind="operator",
            schedule_type="interval",
            schedule_value=None,
        )
        m = recipe_to_operator(r)
        assert m.schedule_value == "300"
