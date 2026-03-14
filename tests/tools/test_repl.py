"""Tests for the persistent REPL tool."""

from __future__ import annotations

import time

import pytest

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools.repl import ReplTool


class TestReplSpec:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_spec_name(self):
        tool = ReplTool()
        assert tool.spec.name == "repl"

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_spec_category(self):
        tool = ReplTool()
        assert tool.spec.category == "code"

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_spec_parameters(self):
        tool = ReplTool()
        params = tool.spec.parameters
        assert params["type"] == "object"
        assert "code" in params["properties"]
        assert "session_id" in params["properties"]
        assert "reset" in params["properties"]
        assert params["required"] == ["code"]

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_tool_id(self):
        tool = ReplTool()
        assert tool.tool_id == "repl"

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_to_openai_function(self):
        tool = ReplTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "repl"


class TestReplBasicExecution:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_expression(self):
        tool = ReplTool()
        result = tool.execute(code="2 + 2")
        assert result.success
        assert "4" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_print(self):
        tool = ReplTool()
        result = tool.execute(code="print('hello')")
        assert result.success
        assert "hello" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_multiline(self):
        tool = ReplTool()
        result = tool.execute(code="x = 5\nprint(x * 2)")
        assert result.success
        assert "10" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_no_output(self):
        tool = ReplTool()
        result = tool.execute(code="x = 42")
        assert result.success
        assert result.content == "(no output)"


class TestReplStatePersistence:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_variable_persists(self):
        tool = ReplTool()
        r1 = tool.execute(code="x = 42")
        sid = r1.metadata["session_id"]
        r2 = tool.execute(code="print(x)", session_id=sid)
        assert r2.success
        assert "42" in r2.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_function_persists(self):
        tool = ReplTool()
        r1 = tool.execute(code="def square(n): return n * n")
        sid = r1.metadata["session_id"]
        r2 = tool.execute(code="square(7)", session_id=sid)
        assert r2.success
        assert "49" in r2.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_import_persists(self):
        tool = ReplTool()
        r1 = tool.execute(code="import math")
        sid = r1.metadata["session_id"]
        r2 = tool.execute(code="math.sqrt(16)", session_id=sid)
        assert r2.success
        assert "4.0" in r2.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_class_persists(self):
        tool = ReplTool()
        r1 = tool.execute(code="class Foo:\n    val = 99")
        sid = r1.metadata["session_id"]
        r2 = tool.execute(code="Foo.val", session_id=sid)
        assert r2.success
        assert "99" in r2.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_mutable_state_across_calls(self):
        tool = ReplTool()
        r1 = tool.execute(code="data = []")
        sid = r1.metadata["session_id"]
        tool.execute(code="data.append(1)", session_id=sid)
        tool.execute(code="data.append(2)", session_id=sid)
        r4 = tool.execute(code="print(data)", session_id=sid)
        assert "[1, 2]" in r4.content


class TestReplSessionManagement:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_auto_create_session(self):
        tool = ReplTool()
        result = tool.execute(code="x = 1")
        assert "session_id" in result.metadata
        assert result.metadata["session_id"]

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_explicit_session_id(self):
        tool = ReplTool()
        result = tool.execute(code="x = 1", session_id="my-session")
        assert result.metadata["session_id"] == "my-session"

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_session_isolation(self):
        tool = ReplTool()
        tool.execute(code="x = 'session_a'", session_id="a")
        tool.execute(code="x = 'session_b'", session_id="b")
        r3 = tool.execute(code="print(x)", session_id="a")
        assert "session_a" in r3.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_session_reset(self):
        tool = ReplTool()
        tool.execute(code="x = 42", session_id="s1")
        tool.execute(code="print('reset')", session_id="s1", reset=True)
        result = tool.execute(code="print(x)", session_id="s1")
        assert not result.success
        assert "NameError" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_execution_count(self):
        tool = ReplTool()
        r1 = tool.execute(code="x = 1", session_id="cnt")
        assert r1.metadata["execution_count"] == 1
        r2 = tool.execute(code="x += 1", session_id="cnt")
        assert r2.metadata["execution_count"] == 2

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_lru_eviction(self):
        tool = ReplTool(max_sessions=2)
        tool.execute(code="x = 'first'", session_id="s1")
        time.sleep(0.01)
        tool.execute(code="x = 'second'", session_id="s2")
        time.sleep(0.01)
        # s1 is oldest; creating s3 should evict s1
        tool.execute(code="x = 'third'", session_id="s3")
        # s1 should be gone — new session with no x
        result = tool.execute(code="print(x)", session_id="s1")
        assert "NameError" in result.content


class TestReplErrorHandling:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_syntax_error(self):
        tool = ReplTool()
        result = tool.execute(code="def foo(")
        assert not result.success
        assert "SyntaxError" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_runtime_error(self):
        tool = ReplTool()
        result = tool.execute(code="1 / 0")
        assert not result.success
        assert "ZeroDivisionError" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_name_error(self):
        tool = ReplTool()
        result = tool.execute(code="print(undefined)")
        assert not result.success
        assert "NameError" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_error_doesnt_corrupt_session(self):
        tool = ReplTool()
        tool.execute(code="x = 10", session_id="err")
        tool.execute(code="1 / 0", session_id="err")  # Error
        r3 = tool.execute(code="print(x)", session_id="err")
        assert r3.success
        assert "10" in r3.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_no_code(self):
        tool = ReplTool()
        result = tool.execute(code="")
        assert not result.success
        assert "No code" in result.content


class TestReplSecurity:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_blocked_os_system(self):
        tool = ReplTool()
        result = tool.execute(code="os.system('ls')")
        assert not result.success
        assert "Blocked" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_blocked_subprocess(self):
        tool = ReplTool()
        result = tool.execute(code="import subprocess")
        assert not result.success
        assert "Blocked" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_blocked_open(self):
        tool = ReplTool()
        result = tool.execute(code="f = open('file.txt')")
        assert not result.success
        assert "Blocked" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_safe_imports_allowed(self):
        tool = ReplTool()
        result = tool.execute(code="import math\nprint(math.pi)")
        assert result.success
        assert "3.14" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_safe_json_import(self):
        tool = ReplTool()
        result = tool.execute(code="import json\nprint(json.dumps({'a': 1}))")
        assert result.success
        assert '"a"' in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_unsafe_import_blocked(self):
        tool = ReplTool()
        result = tool.execute(code="import os")
        assert not result.success
        assert "not allowed" in result.content


class TestReplTimeout:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_sleep_timeout(self):
        tool = ReplTool(timeout=1)
        result = tool.execute(code="import time\ntime.sleep(10)")
        assert not result.success
        assert "timed out" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_infinite_loop_timeout(self):
        tool = ReplTool(timeout=1)
        result = tool.execute(code="while True: pass")
        assert not result.success
        assert "timed out" in result.content


class TestReplOutput:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_truncation(self):
        tool = ReplTool(max_output=50)
        result = tool.execute(code="print('x' * 200)")
        assert "truncated" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_expression_display(self):
        """Expressions should show their repr (REPL-like behavior)."""
        tool = ReplTool()
        result = tool.execute(code="2 + 2")
        assert "4" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_string_expression_display(self):
        tool = ReplTool()
        result = tool.execute(code="'hello'")
        assert "hello" in result.content


class TestReplStderrCapture:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_stderr_output_captured(self):
        """Exercise line 287: stderr is appended to output.

        Since 'sys' is not in the safe import list, we inject it
        into the session namespace directly before executing.
        """
        import sys as _sys


        tool = ReplTool()
        # Create a session with sys pre-injected
        r1 = tool.execute(code="x = 1", session_id="stderr-test")
        sid = r1.metadata["session_id"]
        # Inject sys module into the session namespace
        tool._sessions[sid].namespace["sys"] = _sys
        # Now use sys.stderr.write
        result = tool.execute(
            code="sys.stderr.write('warning message')",
            session_id=sid,
        )
        assert result.success
        assert "warning message" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_builtins_as_module(self, monkeypatch):
        """Exercise line 61: when __builtins__ is a module, not a dict.

        The _make_safe_import function has two branches depending on
        whether __builtins__ is a dict or module. Force the module path
        by temporarily replacing __builtins__ with the builtins module.
        """
        import builtins

        import openjarvis.tools.repl as repl_mod

        # Temporarily set __builtins__ to the module (not the dict)
        # to ensure the `else` branch (line 61) is exercised
        original = repl_mod.__builtins__
        monkeypatch.setattr(repl_mod, "__builtins__", builtins)
        try:
            safe_import = repl_mod._make_safe_import()
            # Should allow safe module
            math_mod = safe_import("math")
            assert hasattr(math_mod, "sqrt")
            # Should block unsafe module
            with pytest.raises(ImportError, match="not allowed"):
                safe_import("os")
        finally:
            repl_mod.__builtins__ = original


class TestReplRegistration:
    @pytest.mark.spec("REQ-tools.base.registration")
    def test_registered(self):
        # Re-register after conftest clears all registries
        ToolRegistry.register_value("repl", ReplTool)
        assert ToolRegistry.contains("repl")
