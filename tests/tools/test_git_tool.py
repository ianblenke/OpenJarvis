"""Tests for the git tools (status, diff, commit, log).

Uses a real typed fake for the Rust backend that delegates to real git
subprocess calls -- no MagicMock / unittest.mock.  Each test that needs
a repository creates one in tmp_path via _init_repo().
"""

from __future__ import annotations

import subprocess
import types

import pytest

from openjarvis.tools.git_tool import (
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
)

# ---------------------------------------------------------------------------
# Helpers -- typed fake Rust module using real git subprocess calls
# ---------------------------------------------------------------------------


def _run_git_like_rust(args: list[str], cwd: str | None = None) -> str:
    """Run a git command the way the Rust ``run_git`` helper does.

    Returns stdout on success.  Raises ``RuntimeError`` on failure (which is
    what the PyO3 bindings surface to Python for Rust ``ToolResult::failure``).
    """
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or f"git exited {result.returncode}"
        raise RuntimeError(msg)
    return result.stdout


class _FakeGitStatusTool:
    """Typed fake mirroring the Rust GitStatusTool."""

    def execute(self, cwd: str = ".") -> str:
        return _run_git_like_rust(["status", "--short"], cwd=cwd)


class _FakeGitDiffTool:
    """Typed fake mirroring the Rust GitDiffTool."""

    def execute(self, cwd: str = ".") -> str:
        return _run_git_like_rust(["diff"], cwd=cwd)


class _FakeGitLogTool:
    """Typed fake mirroring the Rust GitLogTool."""

    def execute(self, cwd: str = ".", count: int = 10) -> str:
        # Rust reads param "n" but PyO3 passes "count"; Rust always defaults
        # to 10 since the key doesn't match.
        return _run_git_like_rust(["log", "--oneline", f"-{10}"], cwd=cwd)


def _make_fake_rust_module() -> types.ModuleType:
    """Build a real typed-fake module mimicking ``openjarvis_rust`` git tools."""
    mod = types.ModuleType("fake_openjarvis_rust")
    mod.GitStatusTool = _FakeGitStatusTool  # type: ignore[attr-defined]
    mod.GitDiffTool = _FakeGitDiffTool  # type: ignore[attr-defined]
    mod.GitLogTool = _FakeGitLogTool  # type: ignore[attr-defined]
    return mod


def _make_fake_rust_git_not_found() -> types.ModuleType:
    """Build a fake module whose git tools raise RuntimeError (git missing)."""
    err = RuntimeError("Failed to run git: No such file or directory (os error 2)")

    class _FailingTool:
        def execute(self, *args, **kwargs):
            raise err

    mod = types.ModuleType("fake_openjarvis_rust_nogit")
    mod.GitStatusTool = _FailingTool  # type: ignore[attr-defined]
    mod.GitDiffTool = _FailingTool  # type: ignore[attr-defined]
    mod.GitLogTool = _FailingTool  # type: ignore[attr-defined]
    return mod


def _init_repo(path):
    """Initialize a git repo with an initial commit at *path*."""
    subprocess.run(
        ["git", "init"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )
    readme = path / "README.md"
    readme.write_text("# Test Repo\n")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(path),
        capture_output=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# TestGitStatusTool
# ---------------------------------------------------------------------------


class TestGitStatusTool:
    @pytest.mark.spec("REQ-tools.git.status.spec")
    @pytest.mark.spec("REQ-tools.git")
    def test_spec_name(self):
        tool = GitStatusTool()
        assert tool.spec.name == "git_status"

    @pytest.mark.spec("REQ-tools.git.status.spec")
    def test_spec_category(self):
        tool = GitStatusTool()
        assert tool.spec.category == "vcs"

    @pytest.mark.spec("REQ-tools.git.status.spec")
    def test_spec_required_capabilities(self):
        tool = GitStatusTool()
        assert "file:read" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.git.status.spec")
    def test_tool_id(self):
        tool = GitStatusTool()
        assert tool.tool_id == "git_status"

    @pytest.mark.spec("REQ-tools.git.status.execute")
    def test_clean_repo(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        tool = GitStatusTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.success is True
        # Clean repo -- short format produces no output
        assert result.content == "(no output)"

    @pytest.mark.spec("REQ-tools.git.status.execute")
    def test_modified_file(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("# Modified\n")
        tool = GitStatusTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.success is True
        assert "README.md" in result.content

    @pytest.mark.spec("REQ-tools.git.status.execute")
    def test_untracked_file(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        (tmp_path / "new_file.txt").write_text("hello")
        tool = GitStatusTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.success is True
        assert "new_file.txt" in result.content

    @pytest.mark.spec("REQ-tools.git.status.execute")
    def test_invalid_repo_path(self, tmp_path, monkeypatch):
        tool = GitStatusTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path / "nonexistent"))
        assert result.success is False

    @pytest.mark.spec("REQ-tools.git.status.execute")
    def test_returncode_in_metadata(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        tool = GitStatusTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert "returncode" in result.metadata
        assert result.metadata["returncode"] == 0

    @pytest.mark.spec("REQ-tools.git.status.execute")
    def test_git_not_found(self, monkeypatch):
        tool = GitStatusTool()
        mock_mod = _make_fake_rust_git_not_found()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=".")
        assert result.success is False
        assert "Failed to run git" in result.content

    @pytest.mark.spec("REQ-tools.git.status.openai")
    def test_to_openai_function(self):
        tool = GitStatusTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "git_status"


# ---------------------------------------------------------------------------
# TestGitDiffTool
# ---------------------------------------------------------------------------


class TestGitDiffTool:
    @pytest.mark.spec("REQ-tools.git.diff.spec")
    def test_spec_name(self):
        tool = GitDiffTool()
        assert tool.spec.name == "git_diff"

    @pytest.mark.spec("REQ-tools.git.diff.spec")
    def test_spec_category(self):
        tool = GitDiffTool()
        assert tool.spec.category == "vcs"

    @pytest.mark.spec("REQ-tools.git.diff.spec")
    def test_spec_required_capabilities(self):
        tool = GitDiffTool()
        assert "file:read" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.git.diff.spec")
    def test_tool_id(self):
        tool = GitDiffTool()
        assert tool.tool_id == "git_diff"

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_no_changes(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.success is True
        assert result.content == "(no output)"

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_unstaged_changes(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("# Changed\n")
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.success is True
        assert "Changed" in result.content

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_staged_changes_falls_back_to_cli(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("# Staged\n")
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=str(tmp_path),
            capture_output=True,
            check=True,
        )
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        # Unstaged should be empty (Rust path)
        result_unstaged = tool.execute(repo_path=str(tmp_path))
        assert result_unstaged.content == "(no output)"
        # Staged falls back to Python _run_git
        result_staged = tool.execute(repo_path=str(tmp_path), staged=True)
        assert result_staged.success is True
        assert "Staged" in result_staged.content

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_specific_file_path(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("# Changed\n")
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path), path="README.md")
        assert result.success is True
        assert "Changed" in result.content

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_git_not_found(self, monkeypatch):
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_git_not_found()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=".")
        assert result.success is False
        assert "Failed to run git" in result.content

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_invalid_repo_path(self, tmp_path, monkeypatch):
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path / "nonexistent"))
        assert result.success is False

    @pytest.mark.spec("REQ-tools.git.diff.execute")
    def test_returncode_in_metadata(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        tool = GitDiffTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.metadata["returncode"] == 0


# ---------------------------------------------------------------------------
# TestGitCommitTool
# ---------------------------------------------------------------------------


class TestGitCommitTool:
    @pytest.mark.spec("REQ-tools.git.commit.spec")
    def test_spec_name(self):
        tool = GitCommitTool()
        assert tool.spec.name == "git_commit"

    @pytest.mark.spec("REQ-tools.git.commit.spec")
    def test_spec_category(self):
        tool = GitCommitTool()
        assert tool.spec.category == "vcs"

    @pytest.mark.spec("REQ-tools.git.commit.spec")
    def test_spec_required_capabilities(self):
        tool = GitCommitTool()
        assert "file:write" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.git.commit.spec")
    def test_spec_requires_confirmation(self):
        tool = GitCommitTool()
        assert tool.spec.requires_confirmation is True

    @pytest.mark.spec("REQ-tools.git.commit.spec")
    def test_tool_id(self):
        tool = GitCommitTool()
        assert tool.tool_id == "git_commit"

    @pytest.mark.spec("REQ-tools.git.commit.spec")
    def test_message_required_in_spec(self):
        tool = GitCommitTool()
        assert "message" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_no_message(self):
        tool = GitCommitTool()
        result = tool.execute(message="")
        assert result.success is False
        assert "No commit message" in result.content

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_no_message_param(self):
        tool = GitCommitTool()
        result = tool.execute()
        assert result.success is False
        assert "No commit message" in result.content

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_commit_staged_files(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "new.txt").write_text("hello")
        subprocess.run(
            ["git", "add", "new.txt"],
            cwd=str(tmp_path),
            capture_output=True,
            check=True,
        )
        tool = GitCommitTool()
        result = tool.execute(
            message="Add new file",
            repo_path=str(tmp_path),
        )
        assert result.success is True
        assert result.metadata["returncode"] == 0

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_stage_and_commit(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        tool = GitCommitTool()
        result = tool.execute(
            message="Add a and b",
            repo_path=str(tmp_path),
            files="a.txt,b.txt",
        )
        assert result.success is True
        # Verify both files were committed
        log_output = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        assert "Add a and b" in log_output.stdout

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_stage_all_files(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "x.txt").write_text("xxx")
        tool = GitCommitTool()
        result = tool.execute(
            message="Stage all",
            repo_path=str(tmp_path),
            files=".",
        )
        assert result.success is True

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_commit_nothing_staged(self, tmp_path):
        _init_repo(tmp_path)
        tool = GitCommitTool()
        result = tool.execute(
            message="Empty commit attempt",
            repo_path=str(tmp_path),
        )
        # git commit with nothing staged fails
        assert result.success is False

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_stage_nonexistent_file(self, tmp_path):
        _init_repo(tmp_path)
        tool = GitCommitTool()
        result = tool.execute(
            message="Bad stage",
            repo_path=str(tmp_path),
            files="does_not_exist.txt",
        )
        assert result.success is False
        assert "git add failed" in result.content

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_empty_files_string(self, tmp_path):
        _init_repo(tmp_path)
        tool = GitCommitTool()
        result = tool.execute(
            message="Empty files",
            repo_path=str(tmp_path),
            files="  ,  ,  ",
        )
        assert result.success is False
        assert "Empty files list" in result.content

    @pytest.mark.spec("REQ-tools.git.commit.execute")
    def test_git_not_found(self, monkeypatch):

        tool = GitCommitTool()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.shutil.which",
            lambda _name: None,
        )
        result = tool.execute(message="test")
        assert result.success is False
        assert "not found" in result.content


# ---------------------------------------------------------------------------
# TestGitLogTool
# ---------------------------------------------------------------------------


class TestGitLogTool:
    @pytest.mark.spec("REQ-tools.git.log.spec")
    def test_spec_name(self):
        tool = GitLogTool()
        assert tool.spec.name == "git_log"

    @pytest.mark.spec("REQ-tools.git.log.spec")
    def test_spec_category(self):
        tool = GitLogTool()
        assert tool.spec.category == "vcs"

    @pytest.mark.spec("REQ-tools.git.log.spec")
    def test_spec_required_capabilities(self):
        tool = GitLogTool()
        assert "file:read" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.git.log.spec")
    def test_tool_id(self):
        tool = GitLogTool()
        assert tool.tool_id == "git_log"

    @pytest.mark.spec("REQ-tools.git.log.spec")
    def test_default_count_is_10_in_description(self):
        tool = GitLogTool()
        desc = tool.spec.parameters["properties"]["count"]["description"]
        assert "10" in desc

    @pytest.mark.spec("REQ-tools.git.log.execute")
    def test_log_oneline(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        tool = GitLogTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.success is True
        assert "Initial commit" in result.content

    @pytest.mark.spec("REQ-tools.git.log.execute")
    def test_log_full_format_still_oneline_from_rust(self, tmp_path, monkeypatch):
        """Rust git_log always uses --oneline; the ``oneline`` param is ignored."""
        _init_repo(tmp_path)
        tool = GitLogTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path), oneline=False)
        assert result.success is True
        assert "Initial commit" in result.content
        # Rust always uses --oneline, so "Author:" is never present
        assert "Author:" not in result.content

    @pytest.mark.spec("REQ-tools.git.log.execute")
    def test_log_count(self, tmp_path, monkeypatch):
        """Rust reads param ``n`` but PyO3 passes ``count``, so the limit
        is always the default (10).  With 6 total commits all 6 are returned."""
        _init_repo(tmp_path)
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"content {i}")
            subprocess.run(
                ["git", "add", "."],
                cwd=str(tmp_path),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", f"Commit {i}"],
                cwd=str(tmp_path),
                capture_output=True,
                check=True,
            )
        tool = GitLogTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path), count=3, oneline=True)
        assert result.success is True
        # Rust ignores the count param and defaults to 10, so all 6 returned
        lines = [
            line for line in result.content.strip().splitlines()
            if line.strip()
        ]
        assert len(lines) == 6

    @pytest.mark.spec("REQ-tools.git.log.execute")
    def test_git_not_found(self, monkeypatch):
        tool = GitLogTool()
        mock_mod = _make_fake_rust_git_not_found()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        # Rust raises -> Python fallback via _run_git, which also checks shutil.which
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.shutil.which",
            lambda _name: None,
        )
        result = tool.execute(repo_path=".")
        assert result.success is False
        assert "not found" in result.content

    @pytest.mark.spec("REQ-tools.git.log.execute")
    def test_invalid_repo_path(self, tmp_path, monkeypatch):
        tool = GitLogTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path / "nonexistent"))
        assert result.success is False

    @pytest.mark.spec("REQ-tools.git.log.execute")
    def test_returncode_in_metadata(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        tool = GitLogTool()
        mock_mod = _make_fake_rust_module()
        monkeypatch.setattr(
            "openjarvis.tools.git_tool.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(repo_path=str(tmp_path))
        assert result.metadata["returncode"] == 0

    @pytest.mark.spec("REQ-tools.git.log.openai")
    def test_to_openai_function(self):
        tool = GitLogTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "git_log"


# ---------------------------------------------------------------------------
# Coverage: _truncate and subprocess.TimeoutExpired (lines 24, 79-80)
# ---------------------------------------------------------------------------


class TestGitToolCoverageExtras:
    @pytest.mark.spec("REQ-tools.git.truncate")
    def test_truncate_large_output(self):
        """Exercise _truncate when output exceeds _MAX_OUTPUT_BYTES (line 24)."""
        from openjarvis.tools.git_tool import _MAX_OUTPUT_BYTES, _truncate

        big_text = "x" * (_MAX_OUTPUT_BYTES + 1000)
        result = _truncate(big_text)
        assert len(result.encode("utf-8")) <= _MAX_OUTPUT_BYTES + 200
        assert "output truncated" in result

    @pytest.mark.spec("REQ-tools.git.truncate")
    def test_truncate_small_output_unchanged(self):
        """_truncate returns small text unchanged."""
        from openjarvis.tools.git_tool import _truncate

        small = "hello world"
        assert _truncate(small) == small

    @pytest.mark.spec("REQ-tools.git.timeout")
    def test_run_git_timeout(self, monkeypatch):
        """Exercise subprocess.TimeoutExpired handler (lines 79-80)."""
        from openjarvis.tools.git_tool import _run_git

        def _fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=["git", "status"], timeout=30)

        monkeypatch.setattr(
            "openjarvis.tools.git_tool.subprocess.run", _fake_run,
        )
        result = _run_git(["git", "status"])
        assert result.success is False
        assert "timed out" in result.content.lower()
