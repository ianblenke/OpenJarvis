"""Tests for the apply_patch tool.

Covers _parse_patch(), _apply_hunks(), and ApplyPatchTool.execute()
with real file system operations via tmp_path.
"""

from __future__ import annotations

import pytest

from openjarvis.tools.apply_patch import (
    ApplyPatchTool,
    _apply_hunks,
    _parse_patch,
)

# ---------------------------------------------------------------------------
# _parse_patch
# ---------------------------------------------------------------------------


class TestParsePatch:
    @pytest.mark.spec("REQ-tools.patch.parse")
    @pytest.mark.spec("REQ-tools.apply-patch")
    def test_simple_patch(self):
        patch = (
            "--- a/file.txt\n"
            "+++ b/file.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+line2_modified\n"
            " line3\n"
        )
        path, hunks = _parse_patch(patch)
        assert path == "file.txt"
        assert len(hunks) == 1
        assert hunks[0].old_start == 1
        assert hunks[0].old_count == 3
        assert hunks[0].new_start == 1
        assert hunks[0].new_count == 3

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_strips_b_prefix(self):
        patch = (
            "--- a/path/to/file.py\n"
            "+++ b/path/to/file.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        path, hunks = _parse_patch(patch)
        assert path == "path/to/file.py"

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_no_b_prefix(self):
        patch = (
            "--- a/file.txt\n"
            "+++ file.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        path, hunks = _parse_patch(patch)
        assert path == "file.txt"

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_dev_null_target(self):
        patch = (
            "--- /dev/null\n"
            "+++ /dev/null\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        path, hunks = _parse_patch(patch)
        assert path is None

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_multi_hunk(self):
        patch = (
            "--- a/file.txt\n"
            "+++ b/file.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-a\n"
            "+A\n"
            " b\n"
            "@@ -10,2 +10,2 @@\n"
            " y\n"
            "-z\n"
            "+Z\n"
        )
        path, hunks = _parse_patch(patch)
        assert len(hunks) == 2
        assert hunks[0].old_start == 1
        assert hunks[1].old_start == 10

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_no_hunks_raises(self):
        with pytest.raises(ValueError, match="No hunks"):
            _parse_patch("this is not a patch")

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_empty_patch_raises(self):
        with pytest.raises(ValueError, match="No hunks"):
            _parse_patch("")

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_hunk_header_single_line(self):
        """Hunk header without count implies count=1."""
        patch = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -5 +5 @@\n"
            "-old\n"
            "+new\n"
        )
        _, hunks = _parse_patch(patch)
        assert hunks[0].old_count == 1
        assert hunks[0].new_count == 1

    @pytest.mark.spec("REQ-tools.patch.parse")
    def test_hunk_lines_captured(self):
        patch = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " ctx\n"
            "-removed\n"
            "+added\n"
            " ctx2\n"
        )
        _, hunks = _parse_patch(patch)
        assert len(hunks[0].lines) == 4
        assert hunks[0].lines[0] == " ctx"
        assert hunks[0].lines[1] == "-removed"
        assert hunks[0].lines[2] == "+added"
        assert hunks[0].lines[3] == " ctx2"


# ---------------------------------------------------------------------------
# _apply_hunks
# ---------------------------------------------------------------------------


class TestApplyHunks:
    @pytest.mark.spec("REQ-tools.patch.apply-hunks")
    def test_simple_replacement(self):
        original = "line1\nline2\nline3\n"
        _, hunks = _parse_patch(
            "--- a/f\n+++ b/f\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+LINE2\n"
            " line3\n"
        )
        result = _apply_hunks(original, hunks)
        assert "LINE2" in result
        assert "line2\n" not in result

    @pytest.mark.spec("REQ-tools.patch.apply-hunks")
    def test_addition(self):
        original = "first\nsecond\n"
        _, hunks = _parse_patch(
            "--- a/f\n+++ b/f\n"
            "@@ -1,2 +1,3 @@\n"
            " first\n"
            "+inserted\n"
            " second\n"
        )
        result = _apply_hunks(original, hunks)
        assert result == "first\ninserted\nsecond\n"

    @pytest.mark.spec("REQ-tools.patch.apply-hunks")
    def test_removal(self):
        original = "keep\nremove_me\nkeep_too\n"
        _, hunks = _parse_patch(
            "--- a/f\n+++ b/f\n"
            "@@ -1,3 +1,2 @@\n"
            " keep\n"
            "-remove_me\n"
            " keep_too\n"
        )
        result = _apply_hunks(original, hunks)
        assert result == "keep\nkeep_too\n"

    @pytest.mark.spec("REQ-tools.patch.apply-hunks")
    def test_context_mismatch_raises(self):
        original = "aaa\nbbb\nccc\n"
        _, hunks = _parse_patch(
            "--- a/f\n+++ b/f\n"
            "@@ -1,3 +1,3 @@\n"
            " aaa\n"
            "-WRONG\n"
            "+replaced\n"
            " ccc\n"
        )
        with pytest.raises(ValueError, match="mismatch"):
            _apply_hunks(original, hunks)

    @pytest.mark.spec("REQ-tools.patch.apply-hunks")
    def test_multi_hunk_application(self):
        original = "a\nb\nc\nd\ne\nf\ng\nh\n"
        _, hunks = _parse_patch(
            "--- a/f\n+++ b/f\n"
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            " c\n"
            "@@ -6,3 +6,3 @@\n"
            " f\n"
            "-g\n"
            "+G\n"
            " h\n"
        )
        result = _apply_hunks(original, hunks)
        lines = result.splitlines()
        assert "B" in lines
        assert "G" in lines
        assert "b" not in lines
        assert "g" not in lines


# ---------------------------------------------------------------------------
# ApplyPatchTool
# ---------------------------------------------------------------------------


class TestApplyPatchTool:
    @pytest.mark.spec("REQ-tools.patch.spec")
    def test_spec_name(self):
        tool = ApplyPatchTool()
        assert tool.spec.name == "apply_patch"

    @pytest.mark.spec("REQ-tools.patch.spec")
    def test_spec_category(self):
        tool = ApplyPatchTool()
        assert tool.spec.category == "filesystem"

    @pytest.mark.spec("REQ-tools.patch.spec")
    def test_spec_required_capabilities(self):
        tool = ApplyPatchTool()
        assert "file:write" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.patch.spec")
    def test_tool_id(self):
        tool = ApplyPatchTool()
        assert tool.tool_id == "apply_patch"

    @pytest.mark.spec("REQ-tools.patch.spec")
    def test_spec_patch_required(self):
        tool = ApplyPatchTool()
        assert "patch" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_no_patch_provided(self):
        tool = ApplyPatchTool()
        result = tool.execute(patch="")
        assert result.success is False
        assert "No patch provided" in result.content

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_simple_one_hunk_patch(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        patch = (
            "--- a/hello.txt\n"
            "+++ b/hello.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+line2_modified\n"
            " line3\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is True
        assert result.metadata["hunks_applied"] == 1
        content = f.read_text(encoding="utf-8")
        assert "line2_modified" in content
        assert "line2\n" not in content

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_multi_hunk_patch(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text(
            "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta\n",
            encoding="utf-8",
        )
        patch = (
            "--- a/multi.txt\n"
            "+++ b/multi.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " alpha\n"
            "-beta\n"
            "+BETA\n"
            " gamma\n"
            "@@ -6,3 +6,3 @@\n"
            " zeta\n"
            "-eta\n"
            "+ETA\n"
            " theta\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is True
        assert result.metadata["hunks_applied"] == 2
        content = f.read_text(encoding="utf-8")
        assert "BETA" in content
        assert "ETA" in content
        assert "beta" not in content
        lines = content.splitlines()
        assert "eta" not in lines

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_context_mismatch_error(self, tmp_path):
        f = tmp_path / "mismatch.txt"
        f.write_text("aaa\nbbb\nccc\n", encoding="utf-8")
        patch = (
            "--- a/mismatch.txt\n"
            "+++ b/mismatch.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " aaa\n"
            "-WRONG_CONTENT\n"
            "+replaced\n"
            " ccc\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is False
        assert "mismatch" in result.content.lower()

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_backup_creation(self, tmp_path):
        f = tmp_path / "backup_me.txt"
        f.write_text("original\ncontent\n", encoding="utf-8")
        patch = (
            "--- a/backup_me.txt\n"
            "+++ b/backup_me.txt\n"
            "@@ -1,2 +1,2 @@\n"
            " original\n"
            "-content\n"
            "+new_content\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=True)
        assert result.success is True
        assert "backup_path" in result.metadata
        bak = tmp_path / "backup_me.txt.bak"
        assert bak.exists()
        assert bak.read_text(encoding="utf-8") == "original\ncontent\n"
        assert "new_content" in f.read_text(encoding="utf-8")

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_backup_disabled(self, tmp_path):
        f = tmp_path / "no_bak.txt"
        f.write_text("hello\nworld\n", encoding="utf-8")
        patch = (
            "--- a/no_bak.txt\n"
            "+++ b/no_bak.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-hello\n"
            "+goodbye\n"
            " world\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is True
        assert "backup_path" not in result.metadata
        bak = tmp_path / "no_bak.txt.bak"
        assert not bak.exists()

    @pytest.mark.spec("REQ-tools.patch.security")
    def test_blocks_sensitive_files(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SECRET=foo\n", encoding="utf-8")
        patch = (
            "--- a/.env\n"
            "+++ b/.env\n"
            "@@ -1 +1 @@\n"
            "-SECRET=foo\n"
            "+SECRET=bar\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f))
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_auto_detect_path_from_patch_header(self, tmp_path):
        f = tmp_path / "auto.txt"
        f.write_text("one\ntwo\nthree\n", encoding="utf-8")
        patch = (
            "--- a/auto.txt\n"
            f"+++ b/{f}\n"
            "@@ -1,3 +1,3 @@\n"
            " one\n"
            "-two\n"
            "+TWO\n"
            " three\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, backup=False)
        assert result.success is True
        assert "TWO" in f.read_text(encoding="utf-8")

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_malformed_patch(self):
        tool = ApplyPatchTool()
        result = tool.execute(patch="this is not a patch at all")
        assert result.success is False
        lower = result.content.lower()
        assert "malformed" in lower or "no hunks" in lower

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_file_not_found(self):
        tool = ApplyPatchTool()
        patch = (
            "--- a/nonexistent.txt\n"
            "+++ b/nonexistent.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = tool.execute(patch=patch, path="/nonexistent/path/file.txt")
        assert result.success is False
        assert "not found" in result.content.lower()

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_addition_only_hunk(self, tmp_path):
        f = tmp_path / "add_only.txt"
        f.write_text("first\nsecond\n", encoding="utf-8")
        patch = (
            "--- a/add_only.txt\n"
            "+++ b/add_only.txt\n"
            "@@ -1,2 +1,3 @@\n"
            " first\n"
            "+inserted\n"
            " second\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is True
        content = f.read_text(encoding="utf-8")
        assert content == "first\ninserted\nsecond\n"

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_removal_only_hunk(self, tmp_path):
        f = tmp_path / "del_only.txt"
        f.write_text("keep\nremove_me\nkeep_too\n", encoding="utf-8")
        patch = (
            "--- a/del_only.txt\n"
            "+++ b/del_only.txt\n"
            "@@ -1,3 +1,2 @@\n"
            " keep\n"
            "-remove_me\n"
            " keep_too\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is True
        content = f.read_text(encoding="utf-8")
        assert content == "keep\nkeep_too\n"

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_no_target_path_no_header(self):
        """No explicit path and no +++ header path -> error."""
        tool = ApplyPatchTool()
        patch = (
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = tool.execute(patch=patch)
        assert result.success is False
        lower = result.content.lower()
        assert "no target path" in lower or "auto-detect" in lower

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_directory_not_file(self, tmp_path):
        """Trying to patch a directory should fail."""
        d = tmp_path / "a_dir"
        d.mkdir()
        patch = (
            "--- a/x\n"
            "+++ b/x\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(d))
        assert result.success is False
        assert "not a file" in result.content.lower()

    @pytest.mark.spec("REQ-tools.patch.execute")
    def test_result_metadata_contains_path(self, tmp_path):
        f = tmp_path / "meta.txt"
        f.write_text("x\ny\n", encoding="utf-8")
        patch = (
            "--- a/meta.txt\n"
            "+++ b/meta.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-x\n"
            "+X\n"
            " y\n"
        )
        tool = ApplyPatchTool()
        result = tool.execute(patch=patch, path=str(f), backup=False)
        assert result.success is True
        assert "path" in result.metadata
