"""Tests for the decorator-based registry system."""

from __future__ import annotations

import pytest

from openjarvis.core.registry import (
    EngineRegistry,
    ModelRegistry,
    RouterPolicyRegistry,
)


class TestRegistryBase:
    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_register_and_get(self) -> None:
        @ModelRegistry.register("test-model")
        class _Dummy:
            pass

        assert ModelRegistry.get("test-model") is _Dummy

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_register_value(self) -> None:
        ModelRegistry.register_value("val", 42)
        assert ModelRegistry.get("val") == 42

    @pytest.mark.spec("REQ-core.registry.duplicate-prevention")
    def test_duplicate_raises(self) -> None:
        ModelRegistry.register_value("dup", 1)
        with pytest.raises(ValueError, match="already has an entry"):
            ModelRegistry.register_value("dup", 2)

    @pytest.mark.spec("REQ-core.registry.missing-key-error")
    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="does not have an entry"):
            ModelRegistry.get("nonexistent")

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_create_instantiates(self) -> None:
        @ModelRegistry.register("factory")
        class _Cls:
            def __init__(self, x: int) -> None:
                self.x = x

        obj = ModelRegistry.create("factory", 7)
        assert obj.x == 7

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_create_non_callable_raises(self) -> None:
        ModelRegistry.register_value("plain", "hello")
        with pytest.raises(TypeError, match="not callable"):
            ModelRegistry.create("plain")

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_items(self) -> None:
        ModelRegistry.register_value("a", 1)
        ModelRegistry.register_value("b", 2)
        assert dict(ModelRegistry.items()) == {"a": 1, "b": 2}

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_keys(self) -> None:
        ModelRegistry.register_value("x", 10)
        ModelRegistry.register_value("y", 20)
        assert set(ModelRegistry.keys()) == {"x", "y"}

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_contains(self) -> None:
        ModelRegistry.register_value("present", True)
        assert ModelRegistry.contains("present")
        assert not ModelRegistry.contains("absent")

    @pytest.mark.spec("REQ-core.registry.generic-base")
    def test_clear(self) -> None:
        ModelRegistry.register_value("temp", 0)
        ModelRegistry.clear()
        assert ModelRegistry.keys() == ()

    @pytest.mark.spec("REQ-core.registry.isolation")
    def test_isolation_between_registries(self) -> None:
        """Entries in ModelRegistry must not leak into EngineRegistry."""
        ModelRegistry.register_value("shared-key", "model")
        with pytest.raises(KeyError):
            EngineRegistry.get("shared-key")


class TestRouterPolicyRegistry:
    def test_register_and_get(self) -> None:
        RouterPolicyRegistry.register_value("test-policy", "dummy")
        assert RouterPolicyRegistry.get("test-policy") == "dummy"

    def test_keys(self) -> None:
        RouterPolicyRegistry.register_value("a", 1)
        RouterPolicyRegistry.register_value("b", 2)
        assert set(RouterPolicyRegistry.keys()) == {"a", "b"}

    def test_contains(self) -> None:
        RouterPolicyRegistry.register_value("present", True)
        assert RouterPolicyRegistry.contains("present")
        assert not RouterPolicyRegistry.contains("absent")

    def test_duplicate_raises(self) -> None:
        RouterPolicyRegistry.register_value("dup", 1)
        with pytest.raises(ValueError, match="already has an entry"):
            RouterPolicyRegistry.register_value("dup", 2)
