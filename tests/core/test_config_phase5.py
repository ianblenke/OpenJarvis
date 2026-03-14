"""Phase 5 foundation tests — BenchmarkRegistry and config."""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig, load_config
from openjarvis.core.registry import BenchmarkRegistry


class TestBenchmarkRegistry:
    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_register_and_get(self):
        BenchmarkRegistry.register_value("test-bench", "dummy")
        assert BenchmarkRegistry.get("test-bench") == "dummy"

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_keys(self):
        BenchmarkRegistry.register_value("a", 1)
        BenchmarkRegistry.register_value("b", 2)
        assert set(BenchmarkRegistry.keys()) == {"a", "b"}

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_contains(self):
        BenchmarkRegistry.register_value("present", True)
        assert BenchmarkRegistry.contains("present")
        assert not BenchmarkRegistry.contains("absent")

    @pytest.mark.spec("REQ-core.registry.duplicate-prevention")
    def test_duplicate_raises(self):
        BenchmarkRegistry.register_value("dup", 1)
        with pytest.raises(ValueError, match="already has an entry"):
            BenchmarkRegistry.register_value("dup", 2)


class TestConfigPhase5:
    @pytest.mark.spec("REQ-core.config.load-config")
    def test_jarvis_config_loads(self):
        cfg = JarvisConfig()
        assert cfg.engine is not None
        assert cfg.learning is not None

    @pytest.mark.spec("REQ-core.registry.typed-subclasses")
    def test_benchmark_registry_importable(self):
        from openjarvis.core.registry import BenchmarkRegistry

        assert BenchmarkRegistry is not None

    @pytest.mark.spec("REQ-core.registry.isolation")
    def test_registry_isolation(self):
        """BenchmarkRegistry entries don't leak into other registries."""
        from openjarvis.core.registry import ModelRegistry

        BenchmarkRegistry.register_value("iso-test", "bench-value")
        with pytest.raises(KeyError):
            ModelRegistry.get("iso-test")

    @pytest.mark.spec("REQ-core.config.load-config")
    def test_load_config_default(self):
        cfg = load_config()
        assert isinstance(cfg, JarvisConfig)
