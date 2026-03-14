"""Tests for WASM sandbox runner.

Covers WasmResult dataclass, WasmRunner.available(),
WasmRunner construction, and create_sandbox_runner() factory.
"""

from __future__ import annotations

import pytest

from openjarvis.sandbox.wasm_runner import WasmResult, WasmRunner, create_sandbox_runner

# ---------------------------------------------------------------------------
# WasmResult dataclass
# ---------------------------------------------------------------------------


class TestWasmResult:
    @pytest.mark.spec("REQ-sandbox.wasm.result")
    def test_default_values(self):
        result = WasmResult()
        assert result.success is True
        assert result.output == ""
        assert result.duration_seconds == 0.0
        assert result.fuel_consumed == 0
        assert result.memory_used_bytes == 0

    @pytest.mark.spec("REQ-sandbox.wasm.result")
    def test_custom_values(self):
        result = WasmResult(
            success=False,
            output="error occurred",
            duration_seconds=1.5,
            fuel_consumed=500,
            memory_used_bytes=1024,
        )
        assert result.success is False
        assert result.output == "error occurred"
        assert result.duration_seconds == 1.5
        assert result.fuel_consumed == 500
        assert result.memory_used_bytes == 1024

    @pytest.mark.spec("REQ-sandbox.wasm.result")
    def test_partial_construction(self):
        result = WasmResult(success=True, output="done", duration_seconds=0.1)
        assert result.success is True
        assert result.output == "done"
        assert result.duration_seconds == 0.1
        assert result.fuel_consumed == 0
        assert result.memory_used_bytes == 0

    @pytest.mark.spec("REQ-sandbox.wasm.result")
    def test_slots_prevents_arbitrary_attrs(self):
        result = WasmResult()
        with pytest.raises(AttributeError):
            result.extra = "nope"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# WasmRunner
# ---------------------------------------------------------------------------


class TestWasmRunner:
    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    def test_runner_creation_defaults(self):
        runner = WasmRunner()
        assert runner._fuel_limit == 1_000_000
        assert runner._memory_limit_mb == 256
        assert runner._timeout == 30.0

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    def test_runner_creation_custom(self):
        runner = WasmRunner(fuel_limit=500_000, memory_limit_mb=128, timeout=10.0)
        assert runner._fuel_limit == 500_000
        assert runner._memory_limit_mb == 128
        assert runner._timeout == 10.0

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    def test_available_returns_bool(self):
        result = WasmRunner.available()
        assert isinstance(result, bool)

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    def test_available_is_static_method(self):
        # Can be called on the class without instantiation
        result = WasmRunner.available()
        assert isinstance(result, bool)

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    def test_run_without_wasmtime_returns_failure(self):
        """If wasmtime is not installed, run() should return a failure result."""
        runner = WasmRunner()
        if not runner.available():
            result = runner.run(b"fake wasm bytes")
            assert result.success is False
            assert "wasmtime" in result.output.lower()

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    @pytest.mark.skipif(
        not WasmRunner.available(),
        reason="wasmtime not installed",
    )
    def test_validate_invalid_bytes(self):
        runner = WasmRunner()
        assert runner.validate(b"not valid wasm") is False

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    @pytest.mark.skipif(
        not WasmRunner.available(),
        reason="wasmtime not installed",
    )
    def test_validate_empty_bytes(self):
        runner = WasmRunner()
        assert runner.validate(b"") is False

    @pytest.mark.spec("REQ-sandbox.wasm.runner")
    @pytest.mark.skipif(
        not WasmRunner.available(),
        reason="wasmtime not installed",
    )
    def test_run_invalid_wasm_returns_error(self):
        runner = WasmRunner()
        result = runner.run(b"not valid wasm")
        assert result.success is False
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# create_sandbox_runner factory
# ---------------------------------------------------------------------------


class TestCreateSandboxRunner:
    @pytest.mark.spec("REQ-sandbox.wasm.factory")
    def test_factory_does_not_crash_with_none_config(self):
        # Should return either a WasmRunner, ContainerRunner, or None
        result = create_sandbox_runner()
        # Just verify it does not crash
        assert result is None or result is not None

    @pytest.mark.spec("REQ-sandbox.wasm.factory")
    def test_factory_with_wasm_config(self):
        class WasmConfig:
            runtime = "wasm"
            wasm_fuel_limit = 100_000
            wasm_memory_limit_mb = 64
            timeout = 10
            image = "test"

        runner = create_sandbox_runner(WasmConfig())
        if WasmRunner.available():
            assert isinstance(runner, WasmRunner)
            assert runner._fuel_limit == 100_000
            assert runner._memory_limit_mb == 64
            assert runner._timeout == 10

    @pytest.mark.spec("REQ-sandbox.wasm.factory")
    def test_factory_with_non_wasm_config(self):
        class DockerConfig:
            runtime = "docker"
            image = "test-image:latest"
            timeout = 60

        result = create_sandbox_runner(DockerConfig())
        # Should not return a WasmRunner since runtime != "wasm"
        assert not isinstance(result, WasmRunner) or result is None

    @pytest.mark.spec("REQ-sandbox.wasm.factory")
    def test_factory_config_without_runtime(self):
        class EmptyConfig:
            pass

        result = create_sandbox_runner(EmptyConfig())
        # Should not return a WasmRunner since runtime is ""
        if result is not None:
            # If Docker is available it returns a ContainerRunner,
            # otherwise None
            assert not isinstance(result, WasmRunner)

    @pytest.mark.spec("REQ-sandbox.wasm.factory")
    def test_factory_with_wasm_config_defaults(self):
        """Config with wasm runtime but no sub-fields uses defaults."""
        class MinimalWasmConfig:
            runtime = "wasm"

        runner = create_sandbox_runner(MinimalWasmConfig())
        if WasmRunner.available():
            assert isinstance(runner, WasmRunner)
            assert runner._fuel_limit == 1_000_000
            assert runner._memory_limit_mb == 256
            assert runner._timeout == 30
