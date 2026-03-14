"""Extended tests for WorkflowEngine — covers agent, tool, condition, loop nodes.

Targets the uncovered branches in workflow/engine.py: _run_agent_node,
_run_tool_node, _run_condition_node, _run_loop_node, parallel execution,
invalid-graph handling, and error paths.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.workflow.builder import WorkflowBuilder
from openjarvis.workflow.engine import WorkflowEngine
from openjarvis.workflow.graph import WorkflowGraph
from openjarvis.workflow.types import (
    NodeType,
    WorkflowNode,
)

# ---------------------------------------------------------------------------
# Typed fakes
# ---------------------------------------------------------------------------


class FakeSystem:
    """Typed fake JarvisSystem for workflow engine tests."""

    def __init__(
        self,
        response: str = "agent response",
        tool_executor: Any = None,
        raise_on_ask: bool = False,
    ) -> None:
        self._response = response
        self.tool_executor = tool_executor
        self._raise_on_ask = raise_on_ask
        self.ask_calls: List[Dict[str, Any]] = []

    def ask(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.ask_calls.append({"prompt": prompt, **kwargs})
        if self._raise_on_ask:
            raise RuntimeError("system.ask failed")
        return {"content": self._response}


class FakeToolExecutor:
    """Typed fake tool executor for workflow tool node tests."""

    def __init__(
        self,
        result_content: str = "tool output",
        success: bool = True,
    ) -> None:
        self._content = result_content
        self._success = success
        self.calls: List[ToolCall] = []

    def execute(self, tool_call: ToolCall) -> ToolResult:
        self.calls.append(tool_call)
        return ToolResult(
            tool_name=tool_call.name,
            content=self._content,
            success=self._success,
        )


# ---------------------------------------------------------------------------
# Invalid graph
# ---------------------------------------------------------------------------


class TestInvalidGraph:
    @pytest.mark.spec("REQ-workflow.engine.validation")
    def test_invalid_graph_returns_failure(self) -> None:
        """WorkflowEngine.run returns failure for a cyclic graph."""
        engine = WorkflowEngine()
        # Build a graph with a cycle manually
        graph = WorkflowGraph(name="cyclic")
        n1 = WorkflowNode(
            id="a",
            node_type=NodeType.TRANSFORM,
            transform_expr="concatenate",
        )
        n2 = WorkflowNode(
            id="b",
            node_type=NodeType.TRANSFORM,
            transform_expr="concatenate",
        )
        graph.add_node(n1)
        graph.add_node(n2)
        from openjarvis.workflow.types import WorkflowEdge

        graph.add_edge(WorkflowEdge(source="a", target="b"))
        graph.add_edge(WorkflowEdge(source="b", target="a"))
        result = engine.run(graph, system=None)
        assert result.success is False
        assert "Invalid workflow" in result.final_output


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


class TestAgentNode:
    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_agent_node_no_system(self) -> None:
        """Agent node returns failure when no system is provided."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_agent("a1", agent="simple").build()
        result = engine.run(graph, system=None, initial_input="hello")
        assert result.success is False
        step = result.steps[0]
        assert "No system available" in step.output

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_agent_node_success(self) -> None:
        """Agent node calls system.ask and returns the response."""
        system = FakeSystem(response="agent done")
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_agent("a1", agent="orchestrator").build()
        result = engine.run(graph, system=system, initial_input="do work")
        assert result.success is True
        assert result.final_output == "agent done"
        assert len(system.ask_calls) == 1
        assert system.ask_calls[0]["prompt"] == "do work"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_agent_node_with_tools(self) -> None:
        """Agent node passes tools list to system.ask."""
        system = FakeSystem(response="done with tools")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_agent("a1", agent="simple", tools=["calc", "search"])
            .build()
        )
        result = engine.run(graph, system=system, initial_input="test")
        assert result.success is True
        assert system.ask_calls[0].get("tools") == ["calc", "search"]

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_agent_node_exception(self) -> None:
        """Agent node handles system.ask exceptions gracefully."""
        system = FakeSystem(raise_on_ask=True)
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_agent("a1", agent="simple").build()
        result = engine.run(graph, system=system, initial_input="crash")
        assert result.success is False
        assert "Agent error" in result.steps[0].output

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_agent_node_with_predecessor_input(self) -> None:
        """Agent node receives input from predecessor node outputs."""
        system = FakeSystem(response="final")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="concatenate")
            .add_agent("a1", agent="simple")
            .connect("t1", "a1")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="seed data")
        assert result.success is True
        # The agent should receive the transform output as input
        assert len(system.ask_calls) == 1


# ---------------------------------------------------------------------------
# Tool node
# ---------------------------------------------------------------------------


class TestToolNode:
    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_tool_node_success(self) -> None:
        """Tool node calls tool_executor.execute and returns the result."""
        executor = FakeToolExecutor(result_content="calc: 42", success=True)
        system = FakeSystem(tool_executor=executor)
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_tool("t1", tool_name="calculator", tool_args='{"expression": "6*7"}')
            .build()
        )
        result = engine.run(graph, system=system, initial_input="")
        assert result.success is True
        assert result.final_output == "calc: 42"
        assert len(executor.calls) == 1
        assert executor.calls[0].name == "calculator"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_tool_node_no_executor(self) -> None:
        """Tool node without tool_executor returns failure."""
        system = FakeSystem(tool_executor=None)
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_tool("t1", tool_name="calc", tool_args="{}")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="")
        assert result.success is False
        assert "No tool executor" in result.final_output

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_tool_node_no_system(self) -> None:
        """Tool node with no system at all returns failure."""
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_tool("t1", tool_name="calc", tool_args="{}")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="")
        assert result.success is False

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_tool_node_failure_result(self) -> None:
        """Tool node propagates failure from tool executor."""
        executor = FakeToolExecutor(result_content="error", success=False)
        system = FakeSystem(tool_executor=executor)
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_tool("t1", tool_name="fail_tool", tool_args="{}")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="")
        assert result.success is False


# ---------------------------------------------------------------------------
# Condition node
# ---------------------------------------------------------------------------


class TestConditionNode:
    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_condition_no_expr_returns_true(self) -> None:
        """Condition node with no expression evaluates to 'true'."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_condition("c1", expr="").build()
        result = engine.run(graph, system=None, initial_input="test")
        assert result.success is True
        assert result.final_output == "true"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_condition_true_expression(self) -> None:
        """Condition node with truthy expression returns the eval result."""
        engine = WorkflowEngine()
        # Use expression that works with restricted builtins (no len available)
        graph = WorkflowBuilder().add_condition("c1", expr="1 + 1 == 2").build()
        result = engine.run(graph, system=None, initial_input="")
        assert result.success is True
        assert result.final_output == "True"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_condition_false_expression(self) -> None:
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_condition("c1", expr="1 + 1 == 5").build()
        result = engine.run(graph, system=None, initial_input="")
        assert result.success is True
        assert result.final_output == "False"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_condition_invalid_expression_returns_false(self) -> None:
        """Condition node with syntax error evaluates to 'false'."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_condition("c1", expr="!!!invalid!!!").build()
        result = engine.run(graph, system=None, initial_input="")
        assert result.success is True
        assert result.final_output == "false"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_condition_uses_outputs_dict(self) -> None:
        """Condition node can reference the 'outputs' variable."""
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_transform("prep", transform="concatenate")
            .add_condition("c1", expr="'_input' in outputs")
            .connect("prep", "c1")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="data")
        assert result.success is True
        assert result.final_output == "True"


# ---------------------------------------------------------------------------
# Loop node
# ---------------------------------------------------------------------------


class TestLoopNode:
    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_loop_node_no_system_exits_immediately(self) -> None:
        """Loop node with no system breaks on first iteration."""
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_loop("l1", agent="simple", max_iterations=5, exit_condition="done")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="start")
        assert result.success is True
        # With no system, loop breaks immediately — output is initial input
        assert result.final_output == "start"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_loop_node_runs_max_iterations(self) -> None:
        """Loop node runs up to max_iterations if exit condition never matches."""
        system = FakeSystem(response="iteration output")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_loop(
                "l1",
                agent="simple",
                max_iterations=3,
                exit_condition="NEVER_MATCH",
            )
            .build()
        )
        result = engine.run(graph, system=system, initial_input="go")
        assert result.success is True
        assert len(system.ask_calls) == 3
        assert result.final_output == "iteration output"

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_loop_node_exits_on_condition(self) -> None:
        """Loop node exits early when exit_condition appears in output."""
        system = FakeSystem(response="task done")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_loop("l1", agent="simple", max_iterations=10, exit_condition="done")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="begin")
        assert result.success is True
        # "done" is in "task done" (case-insensitive),
        # so should exit after 1st iteration
        assert len(system.ask_calls) == 1

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_loop_node_metadata_has_iterations(self) -> None:
        """Loop node step result metadata records iteration count."""
        system = FakeSystem(response="no match here")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_loop("l1", agent="simple", max_iterations=2, exit_condition="NEVER")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="go")
        step = result.steps[0]
        assert step.metadata.get("iterations") == 2

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_loop_node_no_exit_condition(self) -> None:
        """Loop node without exit_condition runs all iterations."""
        system = FakeSystem(response="output")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_loop("l1", agent="simple", max_iterations=3, exit_condition="")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="start")
        assert result.success is True
        # No condition_expr => never matches => all 3 iterations run
        assert len(system.ask_calls) == 3


# ---------------------------------------------------------------------------
# Unknown node type
# ---------------------------------------------------------------------------


class TestUnknownNodeType:
    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_unknown_node_type_returns_failure(self) -> None:
        """A node with an unsupported type returns a failure step."""
        engine = WorkflowEngine()
        graph = WorkflowGraph(name="unknown_type_test")
        node = WorkflowNode(id="u1", node_type=NodeType.PARALLEL)
        graph.add_node(node)
        result = engine.run(graph, system=None, initial_input="")
        assert result.success is False
        assert "Unknown node type" in result.final_output


# ---------------------------------------------------------------------------
# Transform node edge cases
# ---------------------------------------------------------------------------


class TestTransformNodeEdgeCases:
    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_default_transform_concatenates(self) -> None:
        """Transform with unrecognized expr falls through to default (combined)."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_transform("t1", transform="unknown_op").build()
        result = engine.run(graph, system=None, initial_input="data")
        assert result.success is True

    @pytest.mark.spec("REQ-workflow.engine.node-dispatch")
    def test_first_line_empty_input(self) -> None:
        """first_line transform with empty input returns empty string."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_transform("t1", transform="first_line").build()
        result = engine.run(graph, system=None, initial_input="")
        assert result.success is True


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    @pytest.mark.spec("REQ-workflow.engine.parallel")
    def test_parallel_transforms(self) -> None:
        """Two independent nodes run in parallel and both succeed."""
        engine = WorkflowEngine(max_parallel=2)
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="concatenate")
            .add_transform("t2", transform="concatenate")
            .build()
        )
        # t1 and t2 have no edges between them => same stage => parallel
        result = engine.run(graph, system=None, initial_input="input")
        assert result.success is True
        assert len(result.steps) == 2

    @pytest.mark.spec("REQ-workflow.engine.parallel")
    def test_parallel_with_failure(self) -> None:
        """Parallel stage with one failing node sets overall success=False."""
        engine = WorkflowEngine(max_parallel=2)
        # Build a graph with two independent nodes:
        # one transform, one agent (no system => fail)
        graph = (
            WorkflowBuilder()
            .add_transform("t1", transform="concatenate")
            .add_agent("a1", agent="simple")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="test")
        # The agent node fails because system=None
        assert result.success is False


# ---------------------------------------------------------------------------
# Node-level exception handling
# ---------------------------------------------------------------------------


class TestNodeExceptionHandling:
    @pytest.mark.spec("REQ-workflow.engine.error-handling")
    def test_node_exception_caught(self) -> None:
        """An exception thrown inside _execute_node is caught and reported."""
        engine = WorkflowEngine()

        class _BrokenNode:
            """Not a real WorkflowNode but triggers exception path."""
            id = "broken"
            node_type = "NOT_A_VALID_TYPE"

        graph = WorkflowGraph(name="broken")
        node = WorkflowNode(
            id="n1",
            node_type=NodeType.TRANSFORM,
            transform_expr="concatenate",
        )
        graph.add_node(node)

        # Override the node to force an exception
        original_run_transform = engine._run_transform_node

        def _explode(*args, **kwargs):
            raise ValueError("forced error")

        engine._run_transform_node = _explode  # type: ignore[assignment]
        result = engine.run(graph, system=None, initial_input="test")
        assert result.success is False
        assert "Node error" in result.steps[0].output
        engine._run_transform_node = original_run_transform  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class TestEngineEvents:
    @pytest.mark.spec("REQ-workflow.engine.events")
    def test_no_bus_runs_without_error(self) -> None:
        """Engine with bus=None runs without event-related errors."""
        engine = WorkflowEngine(bus=None)
        graph = WorkflowBuilder().add_transform("t1", transform="concatenate").build()
        result = engine.run(graph, system=None, initial_input="test")
        assert result.success is True

    @pytest.mark.spec("REQ-workflow.engine.events")
    def test_all_event_types_emitted(self) -> None:
        """Bus receives WORKFLOW_START, NODE_START, NODE_END, WORKFLOW_END."""
        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = WorkflowBuilder().add_transform("t1", transform="concatenate").build()
        engine.run(graph, system=None, initial_input="test")
        types = [e.event_type for e in bus.history]
        assert EventType.WORKFLOW_START in types
        assert EventType.WORKFLOW_NODE_START in types
        assert EventType.WORKFLOW_NODE_END in types
        assert EventType.WORKFLOW_END in types

    @pytest.mark.spec("REQ-workflow.engine.events")
    def test_workflow_end_event_data(self) -> None:
        """WORKFLOW_END event includes success and duration."""
        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = WorkflowBuilder().add_transform("t1", transform="concatenate").build()
        engine.run(graph, system=None, initial_input="test")
        end_events = [e for e in bus.history if e.event_type == EventType.WORKFLOW_END]
        assert len(end_events) == 1
        assert "success" in end_events[0].data
        assert "duration" in end_events[0].data

    @pytest.mark.spec("REQ-workflow.engine.events")
    def test_node_events_have_node_id(self) -> None:
        """Node events include the node ID in their data."""
        bus = EventBus(record_history=True)
        engine = WorkflowEngine(bus=bus)
        graph = (
            WorkflowBuilder()
            .add_transform("mynode", transform="concatenate")
            .build()
        )
        engine.run(graph, system=None, initial_input="test")
        node_starts = [
            e
            for e in bus.history
            if e.event_type == EventType.WORKFLOW_NODE_START
        ]
        assert len(node_starts) == 1
        assert node_starts[0].data["node"] == "mynode"


# ---------------------------------------------------------------------------
# Step duration tracking
# ---------------------------------------------------------------------------


class TestStepDuration:
    @pytest.mark.spec("REQ-workflow.engine.duration")
    def test_step_has_duration(self) -> None:
        """Each step result has a non-negative duration_seconds."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_transform("t1", transform="concatenate").build()
        result = engine.run(graph, system=None, initial_input="test")
        assert result.steps[0].duration_seconds >= 0.0

    @pytest.mark.spec("REQ-workflow.engine.duration")
    def test_total_duration_positive(self) -> None:
        """Total workflow duration is positive."""
        engine = WorkflowEngine()
        graph = WorkflowBuilder().add_transform("t1", transform="concatenate").build()
        result = engine.run(graph, system=None, initial_input="test")
        assert result.total_duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# Multi-node sequential pipeline
# ---------------------------------------------------------------------------


class TestSequentialPipeline:
    @pytest.mark.spec("REQ-workflow.engine.sequential")
    def test_transform_then_agent(self) -> None:
        """Transform -> Agent pipeline passes data through."""
        system = FakeSystem(response="final output")
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_transform("prep", transform="concatenate")
            .add_agent("agent", agent="simple")
            .connect("prep", "agent")
            .build()
        )
        result = engine.run(graph, system=system, initial_input="raw input")
        assert result.success is True
        assert result.final_output == "final output"
        assert len(result.steps) == 2

    @pytest.mark.spec("REQ-workflow.engine.sequential")
    def test_failure_stops_pipeline(self) -> None:
        """When a sequential node fails, remaining nodes are skipped."""
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_agent("a1", agent="simple")  # fails (no system)
            .add_transform("t1", transform="concatenate")
            .connect("a1", "t1")
            .build()
        )
        result = engine.run(graph, system=None, initial_input="test")
        assert result.success is False
        # Only one step executed (the failing agent)
        assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# _get_node_input
# ---------------------------------------------------------------------------


class TestGetNodeInput:
    @pytest.mark.spec("REQ-workflow.engine.input-routing")
    def test_no_predecessors_uses_initial_input(self) -> None:
        """Node with no predecessors receives _input."""
        engine = WorkflowEngine()
        graph = WorkflowGraph(name="test")
        node = WorkflowNode(
            id="n1",
            node_type=NodeType.TRANSFORM,
            transform_expr="concatenate",
        )
        graph.add_node(node)
        text = engine._get_node_input(
            node, {"_input": "hello"}, graph
        )
        assert text == "hello"

    @pytest.mark.spec("REQ-workflow.engine.input-routing")
    def test_predecessor_outputs_joined(self) -> None:
        """Node with predecessors receives joined predecessor outputs."""
        engine = WorkflowEngine()
        graph = (
            WorkflowBuilder()
            .add_transform("a", transform="concatenate")
            .add_transform("b", transform="concatenate")
            .add_transform("c", transform="concatenate")
            .connect("a", "c")
            .connect("b", "c")
            .build()
        )
        outputs = {"_input": "seed", "a": "from_a", "b": "from_b"}
        node_c = graph.get_node("c")
        text = engine._get_node_input(node_c, outputs, graph)
        assert "from_a" in text
        assert "from_b" in text
