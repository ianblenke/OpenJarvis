"""Tests for the agent template loader using tmp_path for isolation."""

from __future__ import annotations

import pytest

from openjarvis.templates.agent_templates import (
    AgentTemplate,
    discover_templates,
    load_template,
)

# ---------------------------------------------------------------------------
# AgentTemplate dataclass
# ---------------------------------------------------------------------------


class TestAgentTemplate:
    @pytest.mark.spec("REQ-templates.agent.create")
    @pytest.mark.spec("REQ-templates.template")
    def test_create_with_all_fields(self) -> None:
        tpl = AgentTemplate(
            name="test-agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            agent_type="native_react",
            tools=["calculator", "think"],
            max_turns=5,
            temperature=0.3,
        )
        assert tpl.name == "test-agent"
        assert tpl.description == "A test agent"
        assert tpl.system_prompt == "You are a test agent."
        assert tpl.agent_type == "native_react"
        assert tpl.tools == ["calculator", "think"]
        assert tpl.max_turns == 5
        assert tpl.temperature == 0.3

    @pytest.mark.spec("REQ-templates.agent.defaults")
    def test_defaults(self) -> None:
        tpl = AgentTemplate(name="minimal")
        assert tpl.description == ""
        assert tpl.system_prompt == ""
        assert tpl.agent_type == "simple"
        assert tpl.tools == []
        assert tpl.max_turns == 10
        assert tpl.temperature == 0.7


# ---------------------------------------------------------------------------
# load_template()
# ---------------------------------------------------------------------------


class TestLoadTemplate:
    @pytest.mark.spec("REQ-templates.loader.toml")
    @pytest.mark.spec("REQ-templates.load")
    def test_load_full_template(self, tmp_path) -> None:
        toml_content = b"""\
[template]
name = "code-reviewer"
description = "Reviews code for bugs and style"

[agent]
type = "native_react"
max_turns = 8
temperature = 0.3
tools = ["file_read", "think"]
system_prompt = "You are a code reviewer."
"""
        path = tmp_path / "code-reviewer.toml"
        path.write_bytes(toml_content)

        tpl = load_template(path)
        assert isinstance(tpl, AgentTemplate)
        assert tpl.name == "code-reviewer"
        assert tpl.description == "Reviews code for bugs and style"
        assert tpl.system_prompt == "You are a code reviewer."
        assert tpl.agent_type == "native_react"
        assert tpl.tools == ["file_read", "think"]
        assert tpl.max_turns == 8
        assert tpl.temperature == 0.3

    @pytest.mark.spec("REQ-templates.loader.toml")
    def test_load_template_uses_stem_as_default_name(self, tmp_path) -> None:
        toml_content = b"""\
[template]
description = "No name specified"

[agent]
system_prompt = "You are helpful."
"""
        path = tmp_path / "my-agent.toml"
        path.write_bytes(toml_content)

        tpl = load_template(path)
        assert tpl.name == "my-agent"  # Falls back to file stem

    @pytest.mark.spec("REQ-templates.loader.toml")
    def test_load_template_defaults_when_sections_missing(self, tmp_path) -> None:
        toml_content = b"""\
[template]
name = "bare"
"""
        path = tmp_path / "bare.toml"
        path.write_bytes(toml_content)

        tpl = load_template(path)
        assert tpl.name == "bare"
        assert tpl.description == ""
        assert tpl.system_prompt == ""
        assert tpl.agent_type == "simple"
        assert tpl.tools == []
        assert tpl.max_turns == 10
        assert tpl.temperature == 0.7

    @pytest.mark.spec("REQ-templates.loader.toml")
    def test_load_template_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_template("/tmp/nonexistent_template_12345.toml")

    @pytest.mark.spec("REQ-templates.loader.toml")
    def test_load_template_empty_file(self, tmp_path) -> None:
        path = tmp_path / "empty.toml"
        path.write_bytes(b"")

        tpl = load_template(path)
        assert tpl.name == "empty"  # Falls back to stem
        assert tpl.agent_type == "simple"

    @pytest.mark.spec("REQ-templates.loader.toml")
    def test_load_template_only_agent_section(self, tmp_path) -> None:
        toml_content = b"""\
[agent]
type = "orchestrator"
max_turns = 20
temperature = 0.5
tools = ["calculator"]
system_prompt = "You orchestrate."
"""
        path = tmp_path / "orchestrator.toml"
        path.write_bytes(toml_content)

        tpl = load_template(path)
        assert tpl.name == "orchestrator"  # From stem
        assert tpl.agent_type == "orchestrator"
        assert tpl.max_turns == 20
        assert tpl.temperature == 0.5
        assert tpl.tools == ["calculator"]
        assert tpl.system_prompt == "You orchestrate."


# ---------------------------------------------------------------------------
# discover_templates()
# ---------------------------------------------------------------------------


class TestDiscoverTemplates:
    @pytest.mark.spec("REQ-templates.discover.extra-dirs")
    @pytest.mark.spec("REQ-templates.discover")
    def test_discover_from_extra_dirs(self, tmp_path) -> None:
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()

        (tpl_dir / "alpha.toml").write_bytes(b"""\
[template]
name = "alpha"
[agent]
system_prompt = "Alpha agent."
""")
        (tpl_dir / "beta.toml").write_bytes(b"""\
[template]
name = "beta"
[agent]
system_prompt = "Beta agent."
""")

        templates = discover_templates(extra_dirs=[tpl_dir])
        names = {t.name for t in templates}
        assert "alpha" in names
        assert "beta" in names

    @pytest.mark.spec("REQ-templates.discover.sort")
    def test_discover_returns_sorted_by_name(self, tmp_path) -> None:
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()

        for name in ["zebra", "alpha", "middle"]:
            (tpl_dir / f"{name}.toml").write_bytes(
                f'[template]\nname = "{name}"\n[agent]\nsystem_prompt = "x"'.encode()
            )

        templates = discover_templates(extra_dirs=[tpl_dir])
        match_set = {"zebra", "alpha", "middle"}
        names_from_extra = [
            t.name for t in templates if t.name in match_set
        ]
        assert names_from_extra == sorted(names_from_extra)

    @pytest.mark.spec("REQ-templates.discover.override")
    def test_later_dirs_override_earlier(self, tmp_path) -> None:
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "shared.toml").write_bytes(b"""\
[template]
name = "shared"
description = "from dir_a"
[agent]
system_prompt = "A"
""")
        (dir_b / "shared.toml").write_bytes(b"""\
[template]
name = "shared"
description = "from dir_b"
[agent]
system_prompt = "B"
""")

        templates = discover_templates(extra_dirs=[dir_a, dir_b])
        shared = [t for t in templates if t.name == "shared"]
        assert len(shared) == 1
        assert shared[0].description == "from dir_b"

    @pytest.mark.spec("REQ-templates.discover.empty-dir")
    def test_discover_skips_missing_dirs(self, tmp_path) -> None:
        nonexistent = tmp_path / "nonexistent"
        # Should not raise, just skip
        templates = discover_templates(extra_dirs=[nonexistent])
        # Returns at least whatever builtin templates exist (possibly none in test env)
        assert isinstance(templates, list)

    @pytest.mark.spec("REQ-templates.discover.extra-dirs")
    def test_discover_with_no_extra_dirs(self) -> None:
        # Should not raise; returns builtins + user templates (if any exist)
        templates = discover_templates()
        assert isinstance(templates, list)

    @pytest.mark.spec("REQ-templates.discover.extra-dirs")
    def test_discover_ignores_non_toml_files(self, tmp_path) -> None:
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()

        (tpl_dir / "valid.toml").write_bytes(b"""\
[template]
name = "valid"
[agent]
system_prompt = "ok"
""")
        (tpl_dir / "not-a-template.txt").write_text("this is not a template")
        (tpl_dir / "readme.md").write_text("# Templates")

        templates = discover_templates(extra_dirs=[tpl_dir])
        names_from_extra = [t.name for t in templates if t.name == "valid"]
        assert len(names_from_extra) == 1
