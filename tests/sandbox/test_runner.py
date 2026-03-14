"""Tests for ContainerRunner and SandboxedAgent."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Optional

import pytest

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.sandbox.runner import (
    _OUTPUT_END,
    _OUTPUT_START,
    ContainerRunner,
    SandboxedAgent,
)
from tests.fixtures.engines import FakeEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_output(payload: dict) -> str:
    return f"{_OUTPUT_START}\n{json.dumps(payload)}\n{_OUTPUT_END}"


def _mock_proc(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class StubAgent(BaseAgent):
    """Minimal concrete BaseAgent for testing SandboxedAgent wrapping."""

    agent_id = "stub"
    accepts_tools = False

    def __init__(
        self,
        engine: Any = None,
        model: str = "test-model",
        **kw: Any,
    ) -> None:
        _engine = engine or FakeEngine()
        super().__init__(_engine, model, **kw)

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        return AgentResult(content="stub result", turns=1)


class FakeContainerRunner:
    """Typed fake for ContainerRunner — avoids subprocess calls."""

    def __init__(
        self,
        result: dict[str, Any] | None = None,
    ) -> None:
        self._result = result or {"content": "fake result"}
        self.run_calls: list[dict[str, Any]] = []

    def run(
        self,
        input_data: dict[str, Any],
        *,
        workspace: str = "",
        mounts: list[str] | None = None,
        secrets: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.run_calls.append({
            "input_data": input_data,
            "workspace": workspace,
            "mounts": mounts,
            "secrets": secrets,
        })
        return self._result


# ---------------------------------------------------------------------------
# ContainerRunner tests
# ---------------------------------------------------------------------------


class TestContainerRunnerInit:
    def test_defaults(self):
        runner = ContainerRunner()
        assert runner._image == ContainerRunner.DEFAULT_IMAGE
        assert runner._timeout == ContainerRunner.DEFAULT_TIMEOUT
        assert runner._runtime == "docker"
        assert runner._max_concurrent == 5

    def test_custom_values(self):
        runner = ContainerRunner(
            image="custom:latest",
            timeout=600,
            max_concurrent=10,
            runtime="podman",
        )
        assert runner._image == "custom:latest"
        assert runner._timeout == 600
        assert runner._runtime == "podman"
        assert runner._max_concurrent == 10


class TestBuildDockerArgs:
    def test_basic_args(self, monkeypatch):
        runner = ContainerRunner()
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        args = runner._build_docker_args(
            "test-container", [], None,
        )
        assert "/usr/bin/docker" in args
        assert "run" in args
        assert "--rm" in args
        assert "--name" in args
        assert "test-container" in args
        assert "--network" in args
        assert "none" in args
        assert runner._image in args

    def test_with_mounts(self, monkeypatch):
        runner = ContainerRunner()
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        args = runner._build_docker_args(
            "test-container", ["/data/project"], None,
        )
        assert "-v" in args
        idx = args.index("-v")
        assert args[idx + 1] == "/data/project:/data/project:ro"

    def test_with_env(self, monkeypatch):
        runner = ContainerRunner()
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        args = runner._build_docker_args(
            "test-container", [], {"FOO": "bar"},
        )
        assert "-e" in args
        idx = args.index("-e")
        assert args[idx + 1] == "FOO=bar"


class TestContainerRunnerRun:
    @pytest.mark.spec("REQ-sandbox.runner.run")
    def test_successful_run(self, monkeypatch):
        runner = ContainerRunner()
        output = _wrap_output({"content": "Hello!"})
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: subprocess runs external container
            "subprocess.run",
            lambda *a, **kw: _mock_proc(stdout=output),
        )
        result = runner.run({"prompt": "test"})
        assert result["content"] == "Hello!"

    def test_timeout(self, monkeypatch):
        runner = ContainerRunner(timeout=5)
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )

        def _raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["docker"], timeout=5)

        monkeypatch.setattr(  # MOCK-JUSTIFIED: subprocess runs external container
            "subprocess.run", _raise_timeout,
        )
        # Also patch stop() to avoid subprocess calls during cleanup
        monkeypatch.setattr(runner, "stop", lambda name: None)
        result = runner.run({"prompt": "test"})
        assert result["error"] is True
        assert result["error_type"] == "timeout"

    def test_nonzero_exit(self, monkeypatch):
        runner = ContainerRunner()
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: subprocess runs external container
            "subprocess.run",
            lambda *a, **kw: _mock_proc(returncode=1, stderr="OOM killed"),
        )
        result = runner.run({"prompt": "test"})
        assert result["error"] is True
        assert "OOM killed" in result["content"]

    def test_no_sentinel_output(self, monkeypatch):
        runner = ContainerRunner()
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: subprocess runs external container
            "subprocess.run",
            lambda *a, **kw: _mock_proc(stdout="plain text output"),
        )
        result = runner.run({"prompt": "test"})
        assert result["content"] == "plain text output"


class TestContainerRunnerRuntimeCheck:
    @pytest.mark.spec("REQ-sandbox.runner.stop")
    def test_raises_when_runtime_not_found(self, monkeypatch):
        runner = ContainerRunner()
        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: None,
        )
        with pytest.raises(RuntimeError, match="not found"):
            runner._check_runtime()


class TestCleanupOrphans:
    @pytest.mark.spec("REQ-sandbox.runner.cleanup")
    def test_cleanup_orphans(self, monkeypatch):
        runner = ContainerRunner()
        call_log: list[list[str]] = []

        def _fake_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            call_log.append(cmd)
            return _mock_proc(stdout="abc123\ndef456")

        monkeypatch.setattr(  # MOCK-JUSTIFIED: shutil.which probes filesystem
            "shutil.which", lambda cmd: "/usr/bin/docker",
        )
        monkeypatch.setattr(  # MOCK-JUSTIFIED: subprocess runs external container
            "subprocess.run", _fake_run,
        )
        runner.cleanup_orphans()
        assert len(call_log) == 2  # ps + rm


class TestContainerRunnerParseOutput:
    def test_parse_valid_json(self):
        output = _wrap_output({"content": "result", "metadata": {}})
        result = ContainerRunner._parse_output(output)
        assert result["content"] == "result"

    def test_parse_no_sentinels(self):
        result = ContainerRunner._parse_output("plain output")
        assert result["content"] == "plain output"

    def test_parse_invalid_json(self):
        output = f"{_OUTPUT_START}\nnot-json\n{_OUTPUT_END}"
        result = ContainerRunner._parse_output(output)
        assert result.get("parse_error") is True


# ---------------------------------------------------------------------------
# SandboxedAgent tests
# ---------------------------------------------------------------------------


class TestSandboxedAgentInit:
    @pytest.mark.spec("REQ-sandbox.agent")
    def test_accepts_tools_false(self):
        assert SandboxedAgent.accepts_tools is False

    def test_agent_id(self):
        assert SandboxedAgent.agent_id == "sandboxed"

    def test_wraps_agent(self):
        engine = FakeEngine()
        inner = StubAgent(engine=engine, model="test-model")
        runner = FakeContainerRunner()

        agent = SandboxedAgent(
            inner, runner,
            engine=engine, model="test-model",
        )
        assert agent._wrapped_agent is inner
        assert agent._runner is runner


class TestSandboxedAgentRun:
    def test_delegates_to_runner(self):
        engine = FakeEngine()
        inner = StubAgent(engine=engine, model="test-model")
        runner = FakeContainerRunner(result={
            "content": "sandbox result",
            "tool_results": [],
            "metadata": {},
        })

        agent = SandboxedAgent(
            inner, runner, engine=engine, model="test-model",
        )
        result = agent.run("hello")

        assert isinstance(result, AgentResult)
        assert result.content == "sandbox result"
        assert result.turns == 1
        assert len(runner.run_calls) == 1

    def test_parses_tool_results(self):
        engine = FakeEngine()
        inner = StubAgent(engine=engine, model="m")
        runner = FakeContainerRunner(result={
            "content": "done",
            "tool_results": [
                {
                    "tool_name": "calc",
                    "content": "42",
                    "success": True,
                },
            ],
        })

        agent = SandboxedAgent(
            inner, runner, engine=engine, model="m",
        )
        result = agent.run("compute")

        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calc"
        assert result.tool_results[0].content == "42"

    def test_emits_events(self):
        engine = FakeEngine()
        inner = StubAgent(engine=engine, model="m")
        runner = FakeContainerRunner(result={"content": "ok"})

        bus = EventBus(record_history=True)
        agent = SandboxedAgent(
            inner, runner, engine=engine, model="m", bus=bus,
        )
        agent.run("test")

        types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in types
        assert EventType.AGENT_TURN_END in types


# ---------------------------------------------------------------------------
# Sandbox security tests
# ---------------------------------------------------------------------------


class TestSandboxSecuritySecrets:
    @pytest.mark.spec("REQ-sandbox.security.secrets")
    def test_runner_accepts_secrets(self):
        """ContainerRunner.run() accepts a secrets parameter for injection."""
        runner = FakeContainerRunner(result={"content": "ok"})
        result = runner.run(
            {"prompt": "test"},
            secrets={"API_KEY": "sk-test"},
        )
        assert result["content"] == "ok"
        assert runner.run_calls[0]["secrets"] == {"API_KEY": "sk-test"}


class TestSandboxSecurityLimits:
    @pytest.mark.spec("REQ-sandbox.security.limits")
    def test_runner_has_timeout(self):
        """ContainerRunner enforces timeout as a resource limit."""
        runner = ContainerRunner(timeout=30)
        assert runner._timeout == 30

    @pytest.mark.spec("REQ-sandbox.security.limits")
    def test_runner_has_max_concurrent(self):
        """ContainerRunner limits concurrent containers."""
        runner = ContainerRunner(max_concurrent=3)
        assert runner._max_concurrent == 3
