"""Tests for the Docker-sandboxed code interpreter tool."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Typed fakes for the docker module
# ---------------------------------------------------------------------------


class _FakeContainer:
    """Typed fake for docker.models.containers.Container."""

    def __init__(
        self,
        wait_return: Dict[str, int] | None = None,
        logs_outputs: List[bytes] | None = None,
        wait_error: Exception | None = None,
    ) -> None:
        self._wait_return = wait_return or {"StatusCode": 0}
        self._logs_outputs = list(logs_outputs or [b"", b""])
        self._wait_error = wait_error
        self._logs_call_count = 0
        self.remove_calls: List[Dict[str, Any]] = []

    def wait(self, **kwargs) -> Dict[str, int]:
        if self._wait_error is not None:
            raise self._wait_error
        return self._wait_return

    def logs(self, **kwargs) -> bytes:
        idx = min(self._logs_call_count, len(self._logs_outputs) - 1)
        result = self._logs_outputs[idx]
        self._logs_call_count += 1
        return result

    def remove(self, **kwargs) -> None:
        self.remove_calls.append(kwargs)


class _FakeContainersAPI:
    """Typed fake for docker.client.containers."""

    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.run_calls: List[Dict[str, Any]] = []

    def run(self, *args, **kwargs) -> _FakeContainer:
        self.run_calls.append({"args": args, "kwargs": kwargs})
        return self._container


class _FakeDockerClient:
    """Typed fake for docker.DockerClient."""

    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainersAPI(container)


def _make_fake_docker_module(container: _FakeContainer) -> types.ModuleType:
    """Build a typed fake docker module."""
    mod = types.ModuleType("docker")
    client = _FakeDockerClient(container)
    mod.from_env = lambda: client  # type: ignore[attr-defined]
    return mod


class TestDockerCodeInterpreterTool:
    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_spec(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        tool = DockerCodeInterpreterTool()
        spec = tool.spec
        assert spec.name == "code_interpreter_docker"
        assert "code" in spec.parameters["properties"]
        assert spec.category == "code"

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_empty_code(self):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        tool = DockerCodeInterpreterTool()
        result = tool.execute(code="")
        assert not result.success
        assert "No code" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_successful_execution(self, monkeypatch):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        container = _FakeContainer(
            wait_return={"StatusCode": 0},
            logs_outputs=[b"Hello World\n", b""],
        )
        fake_docker = _make_fake_docker_module(container)

        monkeypatch.setitem(sys.modules, "docker", fake_docker)
        tool = DockerCodeInterpreterTool()
        result = tool.execute(code="print('Hello World')")

        assert result.success
        assert "Hello World" in result.content
        assert len(container.remove_calls) == 1
        assert container.remove_calls[0] == {"force": True}

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_execution_error(self, monkeypatch):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        container = _FakeContainer(
            wait_return={"StatusCode": 1},
            logs_outputs=[b"", b"NameError: name 'foo' is not defined\n"],
        )
        fake_docker = _make_fake_docker_module(container)

        monkeypatch.setitem(sys.modules, "docker", fake_docker)
        tool = DockerCodeInterpreterTool()
        result = tool.execute(code="print(foo)")

        assert not result.success
        assert "NameError" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_container_resource_limits(self, monkeypatch):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        container = _FakeContainer(
            wait_return={"StatusCode": 0},
            logs_outputs=[b"ok\n", b""],
        )
        fake_docker = _make_fake_docker_module(container)

        tool = DockerCodeInterpreterTool(
            memory_limit="256m",
            cpu_count=2,
            network_disabled=True,
            pids_limit=50,
        )

        monkeypatch.setitem(sys.modules, "docker", fake_docker)
        tool.execute(code="print('ok')")

        # Get the client to check run call args
        client = fake_docker.from_env()
        call_kwargs = client.containers.run_calls[0]["kwargs"]
        assert call_kwargs["mem_limit"] == "256m"
        assert call_kwargs["nano_cpus"] == 2 * 10**9
        assert call_kwargs["network_disabled"] is True
        assert call_kwargs["pids_limit"] == 50
        assert call_kwargs["read_only"] is True

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_output_truncation(self, monkeypatch):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        container = _FakeContainer(
            wait_return={"StatusCode": 0},
            logs_outputs=[b"x" * 20000, b""],
        )
        fake_docker = _make_fake_docker_module(container)

        monkeypatch.setitem(sys.modules, "docker", fake_docker)
        tool = DockerCodeInterpreterTool(max_output=100)
        result = tool.execute(code="print('x' * 20000)")

        assert result.success
        assert len(result.content) < 200
        assert "truncated" in result.content

    @pytest.mark.spec("REQ-tools.code-interpreter")
    def test_container_cleanup_on_error(self, monkeypatch):
        from openjarvis.tools.code_interpreter_docker import (
            DockerCodeInterpreterTool,
        )

        container = _FakeContainer(
            wait_error=Exception("timeout"),
        )
        fake_docker = _make_fake_docker_module(container)

        monkeypatch.setitem(sys.modules, "docker", fake_docker)
        tool = DockerCodeInterpreterTool()
        result = tool.execute(code="import time; time.sleep(999)")

        assert not result.success
        assert len(container.remove_calls) == 1
        assert container.remove_calls[0] == {"force": True}
