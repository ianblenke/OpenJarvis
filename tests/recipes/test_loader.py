"""Tests for recipe system -- loader, discovery, and resolution."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openjarvis.recipes.loader import (
    Recipe,
    discover_recipes,
    load_recipe,
    resolve_recipe,
)

SAMPLE_TOML = textwrap.dedent("""\
    [recipe]
    name = "test_recipe"
    description = "A test recipe"
    version = "2.0.0"

    [intelligence]
    model = "llama3:8b"
    quantization = "q4_K_M"

    [engine]
    key = "ollama"

    [agent]
    type = "native_react"
    max_turns = 12
    temperature = 0.4
    tools = ["calculator", "think"]
    system_prompt = "You are a test assistant."

    [learning]
    routing = "grpo"
    agent = "icl_updater"

    [eval]
    suites = ["reasoning", "coding"]
""")


class TestRecipeDataclass:
    @pytest.mark.spec("REQ-recipes.recipe")
    @pytest.mark.spec("REQ-recipes.dataclass.defaults")
    def test_recipe_defaults(self) -> None:
        r = Recipe(name="test")
        assert r.name == "test"
        assert r.description == ""
        assert r.version == "0.1.0"
        assert r.kind == "discrete"
        assert r.model is None
        assert r.tools == []
        assert r.eval_suites == []
        assert r.eval_benchmarks == []
        assert r.channels == []
        assert r.required_capabilities == []
        assert r.raw == {}

    @pytest.mark.spec("REQ-recipes.recipe")
    @pytest.mark.spec("REQ-recipes.dataclass.fields")
    def test_recipe_fields(self) -> None:
        r = Recipe(
            name="full",
            description="desc",
            version="1.0.0",
            kind="operator",
            model="llama3:8b",
            quantization="q4_K_M",
            provider="ollama",
            engine_key="ollama",
            agent_type="orchestrator",
            max_turns=20,
            temperature=0.5,
            max_tokens=2048,
            tools=["calculator"],
            system_prompt="hello",
            routing_policy="grpo",
            agent_policy="icl",
            schedule_type="interval",
            schedule_value="300",
            channels=["slack"],
        )
        assert r.name == "full"
        assert r.kind == "operator"
        assert r.model == "llama3:8b"
        assert r.max_turns == 20
        assert r.schedule_type == "interval"
        assert r.channels == ["slack"]


class TestLoadRecipe:
    @pytest.mark.spec("REQ-recipes.load")
    @pytest.mark.spec("REQ-recipes.loader.toml")
    def test_load_recipe_from_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(SAMPLE_TOML)

        recipe = load_recipe(toml_file)

        assert recipe.name == "test_recipe"
        assert recipe.description == "A test recipe"
        assert recipe.version == "2.0.0"
        assert recipe.model == "llama3:8b"
        assert recipe.quantization == "q4_K_M"
        assert recipe.engine_key == "ollama"
        assert recipe.agent_type == "native_react"
        assert recipe.max_turns == 12
        assert recipe.temperature == pytest.approx(0.4)
        assert recipe.tools == ["calculator", "think"]
        assert recipe.system_prompt == "You are a test assistant."
        assert recipe.routing_policy == "grpo"
        assert recipe.agent_policy == "icl_updater"
        assert recipe.eval_suites == ["reasoning", "coding"]
        assert isinstance(recipe.raw, dict)
        assert "recipe" in recipe.raw

    @pytest.mark.spec("REQ-recipes.loader.missing-file")
    def test_load_recipe_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_recipe("/nonexistent/path/recipe.toml")

    @pytest.mark.spec("REQ-recipes.loader.defaults")
    def test_load_recipe_defaults(self, tmp_path: Path) -> None:
        """Minimal TOML should yield sensible defaults."""
        toml_file = tmp_path / "minimal.toml"
        toml_file.write_text("[recipe]\nname = \"minimal\"\n")

        recipe = load_recipe(toml_file)

        assert recipe.name == "minimal"
        assert recipe.version == "0.1.0"
        assert recipe.kind == "discrete"
        assert recipe.model is None
        assert recipe.tools == []
        assert recipe.eval_suites == []

    @pytest.mark.spec("REQ-recipes.loader.name-from-filename")
    def test_load_recipe_name_from_filename(self, tmp_path: Path) -> None:
        """When [recipe] has no name, use the file stem."""
        toml_file = tmp_path / "my_recipe.toml"
        toml_file.write_text("[recipe]\ndescription = \"no name\"\n")

        recipe = load_recipe(toml_file)
        assert recipe.name == "my_recipe"

    @pytest.mark.spec("REQ-recipes.loader.operator-kind")
    def test_load_operator_recipe(self, tmp_path: Path) -> None:
        toml_text = textwrap.dedent("""\
            [recipe]
            name = "my-operator"
            kind = "operator"
            description = "A test operator"

            [agent]
            type = "orchestrator"
            tools = ["web_search", "think"]
            system_prompt = "You are an operator."

            [schedule]
            type = "interval"
            value = "600"

            [channels]
            output = ["slack", "telegram"]
        """)
        p = tmp_path / "op.toml"
        p.write_text(toml_text)
        r = load_recipe(p)

        assert r.kind == "operator"
        assert r.name == "my-operator"
        assert r.schedule_type == "interval"
        assert r.schedule_value == "600"
        assert r.channels == ["slack", "telegram"]

    @pytest.mark.spec("REQ-recipes.loader.schedule-implies-operator")
    def test_schedule_implies_operator_kind(self, tmp_path: Path) -> None:
        toml_text = textwrap.dedent("""\
            [recipe]
            name = "auto-op"

            [schedule]
            type = "cron"
            value = "0 8 * * *"
        """)
        p = tmp_path / "auto.toml"
        p.write_text(toml_text)
        r = load_recipe(p)
        assert r.kind == "operator"

    @pytest.mark.spec("REQ-recipes.loader.external-prompt")
    def test_external_prompt_file(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("External prompt content.")

        # Use relative path
        toml_text = textwrap.dedent("""\
            [recipe]
            name = "ext-prompt"

            [agent]
            type = "simple"
            system_prompt_path = "prompt.md"
        """)
        p = tmp_path / "ext.toml"
        p.write_text(toml_text)
        r = load_recipe(p)
        assert r.system_prompt == "External prompt content."

    @pytest.mark.spec("REQ-recipes.loader.legacy-operator")
    def test_legacy_operator_format(self, tmp_path: Path) -> None:
        toml_text = textwrap.dedent("""\
            [operator]
            name = "legacy-op"
            description = "A legacy operator manifest"
            version = "0.1.0"

            [operator.agent]
            max_turns = 10
            temperature = 0.4
            tools = ["think", "web_search"]
            system_prompt = "Legacy prompt."

            [operator.schedule]
            type = "cron"
            value = "0 */2 * * *"
        """)
        p = tmp_path / "legacy.toml"
        p.write_text(toml_text)
        r = load_recipe(p)

        assert r.kind == "operator"
        assert r.name == "legacy-op"
        assert r.max_turns == 10
        assert r.temperature == pytest.approx(0.4)
        assert r.tools == ["think", "web_search"]
        assert r.system_prompt == "Legacy prompt."
        assert r.schedule_type == "cron"
        assert r.schedule_value == "0 */2 * * *"

    @pytest.mark.spec("REQ-recipes.loader.eval-fields")
    def test_load_eval_fields(self, tmp_path: Path) -> None:
        toml_text = textwrap.dedent("""\
            [recipe]
            name = "bench"

            [intelligence]
            model = "qwen3:8b"

            [eval]
            benchmarks = ["terminalbench", "gaia"]
            backend = "jarvis-agent"
            max_samples = 50
            judge_model = "gpt-4o"
        """)
        p = tmp_path / "bench.toml"
        p.write_text(toml_text)
        r = load_recipe(p)

        assert r.eval_benchmarks == ["terminalbench", "gaia"]
        assert r.eval_backend == "jarvis-agent"
        assert r.eval_max_samples == 50
        assert r.eval_judge_model == "gpt-4o"

    @pytest.mark.spec("REQ-recipes.loader.provider-from-engine")
    def test_provider_from_engine_section(self, tmp_path: Path) -> None:
        toml_text = textwrap.dedent("""\
            [recipe]
            name = "cloud"

            [engine]
            key = "cloud"
            provider = "openai"
        """)
        p = tmp_path / "cloud.toml"
        p.write_text(toml_text)
        r = load_recipe(p)
        assert r.provider == "openai"


class TestDiscoverRecipes:
    @pytest.mark.spec("REQ-recipes.discover")
    @pytest.mark.spec("REQ-recipes.discover.builtin")
    def test_discover_builtin_recipes(self) -> None:
        recipes = discover_recipes()
        names = {r.name for r in recipes}
        assert "coding_assistant" in names
        assert "research_assistant" in names
        assert "general_assistant" in names
        assert len(recipes) >= 3

    @pytest.mark.spec("REQ-recipes.discover.extra-dirs")
    def test_discover_extra_dirs(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "custom.toml"
        toml_file.write_text(
            '[recipe]\nname = "custom"\ndescription = "extra"\n'
        )
        recipes = discover_recipes(extra_dirs=[tmp_path])
        names = {r.name for r in recipes}
        assert "custom" in names

    @pytest.mark.spec("REQ-recipes.discover.skip-malformed")
    def test_discover_skips_malformed(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not valid toml {{{{")
        recipes = discover_recipes(extra_dirs=[tmp_path])
        # Should not raise; malformed files are silently skipped
        names = {r.name for r in recipes}
        assert "bad" not in names

    @pytest.mark.spec("REQ-recipes.discover.kind-filter")
    def test_discover_kind_filter(self, tmp_path: Path) -> None:
        disc_toml = tmp_path / "disc.toml"
        disc_toml.write_text('[recipe]\nname = "disc"\nkind = "discrete"\n')
        op_toml = tmp_path / "op.toml"
        op_toml.write_text(textwrap.dedent("""\
            [recipe]
            name = "op"
            kind = "operator"

            [schedule]
            type = "interval"
            value = "300"
        """))

        discrete = discover_recipes(extra_dirs=[tmp_path], kind="discrete")
        discrete_names = {r.name for r in discrete}
        assert "disc" in discrete_names
        # The operator one should not appear in discrete results
        assert "op" not in discrete_names

    @pytest.mark.spec("REQ-recipes.discover.name-override")
    def test_discover_later_dir_overrides(self, tmp_path: Path) -> None:
        """Later directories override earlier ones with the same recipe name."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "r.toml").write_text('[recipe]\nname = "r"\ndescription = "first"\n')
        (dir2 / "r.toml").write_text('[recipe]\nname = "r"\ndescription = "second"\n')

        recipes = discover_recipes(extra_dirs=[dir1, dir2])
        r = [x for x in recipes if x.name == "r"]
        assert len(r) == 1
        assert r[0].description == "second"


class TestResolveRecipe:
    @pytest.mark.spec("REQ-recipes.resolve")
    @pytest.mark.spec("REQ-recipes.resolve.found")
    def test_resolve_recipe_found(self) -> None:
        recipe = resolve_recipe("coding_assistant")
        assert recipe is not None
        assert recipe.name == "coding_assistant"

    @pytest.mark.spec("REQ-recipes.resolve.not-found")
    def test_resolve_recipe_not_found(self) -> None:
        result = resolve_recipe("nonexistent_recipe_xyz")
        assert result is None


class TestRecipeToBuilderKwargs:
    @pytest.mark.spec("REQ-recipes.builders")
    @pytest.mark.spec("REQ-recipes.builder-kwargs.full")
    def test_recipe_to_builder_kwargs(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(SAMPLE_TOML)

        recipe = load_recipe(toml_file)
        kwargs = recipe.to_builder_kwargs()

        assert kwargs["model"] == "llama3:8b"
        assert kwargs["engine_key"] == "ollama"
        assert kwargs["agent"] == "native_react"
        assert kwargs["tools"] == ["calculator", "think"]
        assert kwargs["temperature"] == pytest.approx(0.4)
        assert kwargs["max_turns"] == 12
        assert kwargs["system_prompt"] == "You are a test assistant."
        assert kwargs["routing_policy"] == "grpo"
        assert kwargs["agent_policy"] == "icl_updater"
        assert kwargs["quantization"] == "q4_K_M"
        assert kwargs["eval_suites"] == ["reasoning", "coding"]

    @pytest.mark.spec("REQ-recipes.builder-kwargs.omit-none")
    def test_kwargs_omit_none_fields(self) -> None:
        recipe = Recipe(name="sparse")
        kwargs = recipe.to_builder_kwargs()
        assert "model" not in kwargs
        assert "engine_key" not in kwargs
        assert "agent" not in kwargs
        assert "tools" not in kwargs
        assert "temperature" not in kwargs

    @pytest.mark.spec("REQ-recipes.builder-kwargs.prompt-from-path")
    def test_system_prompt_path_resolved(self, tmp_path: Path) -> None:
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("Hello from file.")
        r = Recipe(name="ext", system_prompt_path=str(prompt))
        kw = r.to_builder_kwargs()
        assert kw["system_prompt"] == "Hello from file."

    @pytest.mark.spec("REQ-recipes.builder-kwargs.no-schedule-channels")
    def test_schedule_and_channels_not_in_kwargs(self) -> None:
        """Schedule/channel fields are operator-specific and not builder kwargs."""
        r = Recipe(
            name="op",
            kind="operator",
            schedule_type="cron",
            schedule_value="0 * * * *",
            channels=["slack"],
        )
        kw = r.to_builder_kwargs()
        assert "schedule_type" not in kw
        assert "channels" not in kw
