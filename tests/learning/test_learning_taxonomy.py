"""Tests for the learning policy ABC taxonomy."""

from __future__ import annotations

import pytest

from openjarvis.learning._stubs import (
    AgentLearningPolicy,
    IntelligenceLearningPolicy,
    LearningPolicy,
)


class TestLearningPolicyABC:
    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            LearningPolicy()

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_cannot_instantiate_intelligence(self):
        with pytest.raises(TypeError):
            IntelligenceLearningPolicy()

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_cannot_instantiate_agent(self):
        with pytest.raises(TypeError):
            AgentLearningPolicy()

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_target_intelligence(self):
        assert IntelligenceLearningPolicy.target == "intelligence"

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_target_agent(self):
        assert AgentLearningPolicy.target == "agent"

    @pytest.mark.spec("REQ-learning.learning-policy")
    def test_hierarchy(self):
        assert issubclass(IntelligenceLearningPolicy, LearningPolicy)
        assert issubclass(AgentLearningPolicy, LearningPolicy)
