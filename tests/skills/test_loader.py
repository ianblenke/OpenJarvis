"""Tests for skill TOML loader."""

from __future__ import annotations

import pytest

from openjarvis.skills.loader import load_skill
from openjarvis.skills.types import SkillManifest


class TestLoadSkill:
    """Test skill loading from TOML files."""

    @pytest.mark.spec("REQ-skills.loader.toml")
    def test_load_valid_skill(self, tmp_path) -> None:
        toml_content = """
[skill]
name = "test-skill"
version = "1.0"
description = "A test skill"
author = "test"

[[skill.steps]]
tool_name = "calculator"
arguments_template = '{"expression": "{input}"}'
output_key = "result"
"""
        path = tmp_path / "skill.toml"
        path.write_text(toml_content)
        manifest = load_skill(str(path))
        assert manifest.name == "test-skill"
        assert manifest.version == "1.0"
        assert len(manifest.steps) == 1
        assert manifest.steps[0].tool_name == "calculator"

    @pytest.mark.spec("REQ-skills.loader.toml")
    def test_load_multi_step_skill(self, tmp_path) -> None:
        toml_content = """
[skill]
name = "pipeline"
version = "1.0"
description = "Multi-step skill"
author = "test"

[[skill.steps]]
tool_name = "echo"
arguments_template = '{"text": "{input}"}'
output_key = "step1"

[[skill.steps]]
tool_name = "calculator"
arguments_template = '{"expression": "{step1}"}'
output_key = "step2"
"""
        path = tmp_path / "skill.toml"
        path.write_text(toml_content)
        manifest = load_skill(str(path))
        assert len(manifest.steps) == 2
        assert manifest.steps[0].output_key == "step1"
        assert manifest.steps[1].output_key == "step2"

    @pytest.mark.spec("REQ-skills.loader.toml")
    def test_load_skill_file_not_found(self) -> None:
        with pytest.raises((FileNotFoundError, ValueError)):
            load_skill("/nonexistent/skill.toml")

    @pytest.mark.spec("REQ-skills.loader.toml")
    def test_load_skill_invalid_toml(self, tmp_path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text("not valid toml {{{{")
        with pytest.raises(Exception):
            load_skill(str(path))


class TestSkillManifestBytes:
    """Test manifest serialization for signing."""

    @pytest.mark.spec("REQ-skills.loader.security")
    @pytest.mark.spec("REQ-skills.types.manifest")
    def test_manifest_bytes_deterministic(self) -> None:
        from openjarvis.skills.types import SkillStep

        manifest = SkillManifest(
            name="test",
            version="1.0",
            description="test",
            author="test",
            steps=[
                SkillStep(
                    tool_name="echo",
                    arguments_template='{"text": "{input}"}',
                    output_key="result",
                ),
            ],
        )
        bytes1 = manifest.manifest_bytes()
        bytes2 = manifest.manifest_bytes()
        assert bytes1 == bytes2

    @pytest.mark.spec("REQ-skills.types.manifest")
    def test_manifest_bytes_excludes_signature(self) -> None:

        manifest = SkillManifest(
            name="test",
            version="1.0",
            description="test",
            author="test",
            steps=[],
            signature="fake-sig",
        )
        data = manifest.manifest_bytes()
        assert b"fake-sig" not in data
