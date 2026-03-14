"""Tests for heuristic policy registration."""

from __future__ import annotations

import pytest

from openjarvis.core.registry import RouterPolicyRegistry
from openjarvis.learning.routing.heuristic_policy import ensure_registered
from openjarvis.learning.routing.router import HeuristicRouter


class TestHeuristicPolicy:
    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_registered_as_heuristic(self) -> None:
        ensure_registered()
        assert RouterPolicyRegistry.contains("heuristic")

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_value_is_heuristic_router(self) -> None:
        ensure_registered()
        assert RouterPolicyRegistry.get("heuristic") is HeuristicRouter

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_can_instantiate(self) -> None:
        ensure_registered()
        cls = RouterPolicyRegistry.get("heuristic")
        router = cls(available_models=["model-a"])
        assert router.available_models == ["model-a"]
