"""Tests for individual workflow engine node types."""

from __future__ import annotations

import pytest

from openjarvis.core.events import EventBus
from openjarvis.workflow.builder import WorkflowBuilder
from openjarvis.workflow.engine import WorkflowEngine
from openjarvis.workflow.types import NodeType, WorkflowNode


class TestTransformNodes:
    """Test transform node execution."""

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_concatenate_transform(self) -> None:
        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="concatenate")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="hello world")
        assert result.success

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_first_line_transform(self) -> None:
        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="first_line")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="first\nsecond\nthird")
        assert result.success


class TestWorkflowEvents:
    """Test event emission during workflow execution."""

    @pytest.mark.spec("REQ-workflow.engine.events")
    def test_workflow_start_end_events(self) -> None:
        from openjarvis.core.events import EventType

        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="concatenate")
            .build()
        )
        engine.run(graph, system=None, initial_input="test")
        event_types = [e.event_type for e in bus.history]
        assert EventType.WORKFLOW_START in event_types
        assert EventType.WORKFLOW_END in event_types

    @pytest.mark.spec("REQ-workflow.engine.events")
    def test_node_start_end_events(self) -> None:
        from openjarvis.core.events import EventType

        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="concatenate")
            .build()
        )
        engine.run(graph, system=None, initial_input="test")
        event_types = [e.event_type for e in bus.history]
        assert EventType.WORKFLOW_NODE_START in event_types
        assert EventType.WORKFLOW_NODE_END in event_types


class TestWorkflowBuilder:
    """Test builder methods individually."""

    @pytest.mark.spec("REQ-workflow.builder.fluent")
    def test_add_tool_node(self) -> None:
        graph = (
            WorkflowBuilder()
            .add_tool("t1", tool_name="calculator", tool_args={"expression": "1+1"})
            .build()
        )
        node = graph.get_node("t1")
        assert node.node_type == NodeType.TOOL

    @pytest.mark.spec("REQ-workflow.builder.fluent")
    def test_add_condition_node(self) -> None:
        builder = WorkflowBuilder()
        builder.add_condition("c1", expr="len(input) > 5")
        builder.add_transform("yes", transform="concatenate")
        builder.add_transform("no", transform="concatenate")
        builder.connect("c1", "yes", condition="true")
        builder.connect("c1", "no", condition="false")
        graph = builder.build()
        node = graph.get_node("c1")
        assert node.node_type == NodeType.CONDITION

    @pytest.mark.spec("REQ-workflow.builder.fluent")
    def test_add_loop_node(self) -> None:
        graph = (
            WorkflowBuilder()
            .add_loop("l1", agent="simple", max_iterations=3, exit_condition="done")
            .build()
        )
        node = graph.get_node("l1")
        assert node.node_type == NodeType.LOOP
        assert node.max_iterations == 3

    @pytest.mark.spec("REQ-workflow.builder.fluent")
    def test_connect_edges(self) -> None:
        graph = (
            WorkflowBuilder()
            .add_transform("a", transform="concatenate")
            .add_transform("b", transform="concatenate")
            .connect("a", "b")
            .build()
        )
        successors = graph.successors("a")
        assert "b" in successors


class TestWorkflowTypes:
    """Test workflow type dataclasses."""

    @pytest.mark.spec("REQ-workflow.types.workflow-node")
    def test_workflow_node_creation(self) -> None:
        node = WorkflowNode(id="test", node_type=NodeType.AGENT, agent="simple")
        assert node.id == "test"
        assert node.node_type == NodeType.AGENT

    @pytest.mark.spec("REQ-workflow.types.workflow-result")
    def test_workflow_result_creation(self) -> None:
        from openjarvis.workflow.types import WorkflowResult

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            steps=[],
            final_output="done",
            total_duration_seconds=1.5,
        )
        assert result.success
        assert result.total_duration_seconds == 1.5

    @pytest.mark.spec("REQ-workflow.types.node-types")
    def test_node_types_enum(self) -> None:
        assert NodeType.AGENT.value == "agent"
        assert NodeType.TOOL.value == "tool"
        assert NodeType.CONDITION.value == "condition"
        assert NodeType.LOOP.value == "loop"
        assert NodeType.TRANSFORM.value == "transform"
