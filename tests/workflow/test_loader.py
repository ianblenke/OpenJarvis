"""Tests for workflow TOML loader."""

from __future__ import annotations

import pytest

from openjarvis.workflow.loader import load_workflow
from openjarvis.workflow.types import NodeType


class TestLoadWorkflow:
    """Test workflow loading from TOML files."""

    @pytest.mark.spec("REQ-workflow.loader.toml")
    def test_load_simple_workflow(self, tmp_path) -> None:
        toml_content = """
[workflow]
name = "test-workflow"

[[workflow.nodes]]
id = "start"
type = "transform"
transform = "concatenate"

[[workflow.nodes]]
id = "end"
type = "transform"
transform = "concatenate"

[[workflow.edges]]
source = "start"
target = "end"
"""
        path = tmp_path / "workflow.toml"
        path.write_text(toml_content)
        graph = load_workflow(str(path))
        assert graph is not None
        assert len(list(graph.nodes)) >= 2

    @pytest.mark.spec("REQ-workflow.loader.toml")
    def test_load_workflow_with_agent_node(self, tmp_path) -> None:
        toml_content = """
[workflow]
name = "agent-workflow"

[[workflow.nodes]]
id = "agent1"
type = "agent"
agent = "simple"
tools = ["calculator"]

[[workflow.nodes]]
id = "output"
type = "transform"
transform = "concatenate"

[[workflow.edges]]
source = "agent1"
target = "output"
"""
        path = tmp_path / "workflow.toml"
        path.write_text(toml_content)
        graph = load_workflow(str(path))
        node = graph.get_node("agent1")
        assert node.node_type == NodeType.AGENT
        assert node.agent == "simple"

    @pytest.mark.spec("REQ-workflow.loader.toml")
    def test_load_workflow_file_not_found(self) -> None:
        with pytest.raises((FileNotFoundError, ValueError)):
            load_workflow("/nonexistent/workflow.toml")

    @pytest.mark.spec("REQ-workflow.loader.toml")
    def test_load_workflow_invalid_toml(self, tmp_path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text("not valid toml {{{{")
        with pytest.raises(Exception):
            load_workflow(str(path))


    @pytest.mark.spec("REQ-workflow.loader.toml")
    def test_load_workflow_with_cycle_raises(self, tmp_path) -> None:
        """Exercise line 64: invalid workflow raises ValueError."""
        toml_content = """
[workflow]
name = "cyclic-workflow"

[[workflow.nodes]]
id = "a"
type = "agent"
agent = "simple"

[[workflow.nodes]]
id = "b"
type = "agent"
agent = "simple"

[[workflow.edges]]
source = "a"
target = "b"

[[workflow.edges]]
source = "b"
target = "a"
"""
        path = tmp_path / "workflow.toml"
        path.write_text(toml_content)
        with pytest.raises(ValueError, match="Invalid workflow"):
            load_workflow(str(path))

    @pytest.mark.spec("REQ-workflow.loader.toml")
    def test_load_workflow_default_name_from_stem(self, tmp_path) -> None:
        """When no name is provided, graph uses file stem."""
        toml_content = """
[workflow]

[[workflow.nodes]]
id = "only"
type = "transform"
"""
        path = tmp_path / "my_pipeline.toml"
        path.write_text(toml_content)
        graph = load_workflow(str(path))
        assert graph.name == "my_pipeline"


class TestLoadWorkflowConditionNode:
    """Test condition node loading."""

    @pytest.mark.spec("REQ-workflow.types.node-types")
    def test_load_condition_node(self, tmp_path) -> None:
        toml_content = """
[workflow]
name = "conditional"

[[workflow.nodes]]
id = "check"
type = "condition"
condition_expr = "len(input) > 10"

[[workflow.nodes]]
id = "long"
type = "transform"
transform = "concatenate"

[[workflow.nodes]]
id = "short"
type = "transform"
transform = "concatenate"

[[workflow.edges]]
source = "check"
target = "long"
condition = "true"

[[workflow.edges]]
source = "check"
target = "short"
condition = "false"
"""
        path = tmp_path / "workflow.toml"
        path.write_text(toml_content)
        graph = load_workflow(str(path))
        node = graph.get_node("check")
        assert node.node_type == NodeType.CONDITION
