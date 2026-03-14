"""Tests for the shell_exec tool.

Tests use typed fakes for the Rust backend to verify the Python wrapper
handles the Rust output format correctly:
    "Exit code: {code}\\n--- stdout ---\\n{stdout}\\n--- stderr ---\\n{stderr}"
"""

from __future__ import annotations

import os
import types

import pytest

from openjarvis.tools.shell_exec import ShellExecTool

# ---------------------------------------------------------------------------
# Typed fake for the Rust module
# ---------------------------------------------------------------------------


def _rust_output(stdout: str = "", stderr: str = "", code: int = 0) -> str:
    """Build the Rust shell_exec output format."""
    return f"Exit code: {code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"


class _FakeRustShellExecTool:
    """Typed fake for the Rust ShellExecTool class."""

    def __init__(self, return_value: str | None = None,
                 side_effect: Exception | None = None) -> None:
        self._return_value = return_value
        self._side_effect = side_effect

    def execute(self, *args, **kwargs) -> str:
        if self._side_effect is not None:
            raise self._side_effect
        return self._return_value or ""


def _make_fake_rust_mod(
    return_value: str | None = None,
    side_effect: Exception | None = None,
) -> types.ModuleType:
    """Create a typed fake Rust module with a ShellExecTool."""
    mod = types.ModuleType("fake_openjarvis_rust")
    tool_instance = _FakeRustShellExecTool(
        return_value=return_value, side_effect=side_effect,
    )
    # ShellExecTool() should return our fake instance
    mod.ShellExecTool = lambda: tool_instance  # type: ignore[attr-defined]
    return mod


class TestShellExecTool:
    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_spec(self):
        tool = ShellExecTool()
        assert tool.spec.name == "shell_exec"
        assert tool.spec.category == "system"
        assert tool.spec.requires_confirmation is True
        assert tool.spec.timeout_seconds == 60.0
        assert "code:execute" in tool.spec.required_capabilities
        assert "command" in tool.spec.parameters["properties"]
        assert "command" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_no_command(self):
        tool = ShellExecTool()
        result = tool.execute(command="")
        assert result.success is False
        assert "No command" in result.content

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_no_command_param(self):
        tool = ShellExecTool()
        result = tool.execute()
        assert result.success is False
        assert "No command" in result.content

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_simple_echo(self, monkeypatch):
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(stdout="hello\n"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.content
        assert "--- stdout ---" in result.content

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_capture_stderr(self, monkeypatch):
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(stderr="error_msg\n"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="echo error_msg >&2")
        assert "error_msg" in result.content
        assert "--- stderr ---" in result.content

    @pytest.mark.skip(
        reason="Rust backend has no timeout -- Command::output() blocks",
    )
    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_timeout_exceeded(self):
        tool = ShellExecTool()
        result = tool.execute(command="sleep 60", timeout=1)
        assert result.success is False
        assert "timed out" in result.content
        assert result.metadata["returncode"] == -1
        assert result.metadata["timeout_used"] == 1

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_timeout_capped_at_max(self, monkeypatch):
        """timeout param is still capped in Python; Rust ignores it."""
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(stdout="ok\n"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="echo ok", timeout=999)
        assert result.success is True
        assert result.metadata["timeout_used"] == 300

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_working_dir(self, tmp_path, monkeypatch):
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(stdout=str(tmp_path) + "\n"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="pwd", working_dir=str(tmp_path))
        assert result.success is True
        assert str(tmp_path) in result.content
        assert result.metadata["working_dir"] == str(tmp_path)

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_working_dir_not_exists(self):
        tool = ShellExecTool()
        result = tool.execute(command="echo hi", working_dir="/nonexistent/path")
        assert result.success is False
        assert "does not exist" in result.content

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_working_dir_not_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("data", encoding="utf-8")
        tool = ShellExecTool()
        result = tool.execute(command="echo hi", working_dir=str(f))
        assert result.success is False
        assert "not a directory" in result.content

    @pytest.mark.skip(reason="Rust backend inherits parent env -- no env isolation")
    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_env_clearing(self):
        """Verify that arbitrary env vars are NOT passed through."""
        marker = "OPENJARVIS_TEST_SECRET_12345"
        os.environ[marker] = "leaked"
        try:
            tool = ShellExecTool()
            result = tool.execute(command=f"echo ${marker}")
            assert result.success is True
            assert "leaked" not in result.content
        finally:
            os.environ.pop(marker, None)

    @pytest.mark.skip(
        reason="Rust backend inherits parent env -- no env_passthrough",
    )
    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_env_passthrough(self):
        """Verify that explicitly listed env vars ARE passed through."""
        marker = "OPENJARVIS_TEST_PASSTHROUGH_67890"
        os.environ[marker] = "allowed_value"
        try:
            tool = ShellExecTool()
            result = tool.execute(
                command=f"echo ${marker}",
                env_passthrough=[marker],
            )
            assert result.success is True
            assert "allowed_value" in result.content
        finally:
            os.environ.pop(marker, None)

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_returncode_in_metadata(self, monkeypatch):
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(stdout="ok\n"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="echo ok")
        assert result.success is True
        assert result.metadata["returncode"] == 0

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_nonzero_returncode(self, monkeypatch):
        """Non-zero exit in Rust returns ToolResult::failure() but PyO3 binding
        returns Ok(content).  The Python wrapper currently treats that as
        success=True (it only sets success=False on exception).  The Rust
        output still contains the exit code in the formatted string."""
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(code=42),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="exit 42")
        # PyO3 binding returns content for both success/failure ToolResults,
        # so Python wrapper sets success=True and returncode=0.
        assert result.success is True
        assert "Exit code: 42" in result.content

    @pytest.mark.skip(reason="Rust backend has no output truncation")
    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_max_output_truncation(self, tmp_path):
        """Stdout exceeding 100 KB is truncated."""
        tool = ShellExecTool()
        result = tool.execute(
            command="python3 -c \"print('A' * 200000)\"",
        )
        assert "truncated" in result.content
        assert len(result.content) < 200_000

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_no_output(self, monkeypatch):
        """Rust always returns the format string even when stdout/stderr are empty."""
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="true")
        assert result.success is True
        assert "Exit code: 0" in result.content
        assert "--- stdout ---" in result.content
        assert "--- stderr ---" in result.content

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_tool_id(self):
        tool = ShellExecTool()
        assert tool.tool_id == "shell_exec"

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_to_openai_function(self):
        tool = ShellExecTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "shell_exec"
        assert "command" in fn["function"]["parameters"]["properties"]

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_default_timeout_metadata(self, monkeypatch):
        mock_mod = _make_fake_rust_mod(
            return_value=_rust_output(stdout="ok\n"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="echo ok")
        assert result.metadata["timeout_used"] == 30

    @pytest.mark.spec("REQ-tools.shell-exec")
    def test_rust_exception_sets_failure(self, monkeypatch):
        """When the Rust backend raises an exception, Python sets success=False."""
        mock_mod = _make_fake_rust_mod(
            side_effect=RuntimeError("Failed to execute: No such file or directory"),
        )
        tool = ShellExecTool()
        monkeypatch.setattr(
            "openjarvis._rust_bridge.get_rust_module",
            lambda: mock_mod,
        )
        result = tool.execute(command="/nonexistent_binary")
        assert result.success is False
        assert result.metadata["returncode"] == -1
