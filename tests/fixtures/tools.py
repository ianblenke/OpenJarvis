"""Typed fake tools for anti-mocking tests.

Real BaseTool implementations for testing agents and tool executors
without mocking.
"""

from __future__ import annotations

from typing import Any

from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


class CalculatorStub(BaseTool):
    """Real calculator tool for testing. Evaluates simple math expressions."""

    tool_id = "calculator"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Evaluate a mathematical expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
            category="math",
        )

    def execute(self, **params: Any) -> ToolResult:
        expr = params.get("expression", "")
        try:
            result = str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
            return ToolResult(tool_name="calculator", content=result, success=True)
        except Exception as exc:
            return ToolResult(
                tool_name="calculator",
                content=f"Error: {exc}",
                success=False,
            )


class ThinkStub(BaseTool):
    """Think tool that just echoes input. For testing agent reasoning loops."""

    tool_id = "think"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="think",
            description="Think through a problem step by step.",
            parameters={
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your thought process.",
                    },
                },
                "required": ["thought"],
            },
            category="reasoning",
        )

    def execute(self, **params: Any) -> ToolResult:
        thought = params.get("thought", "")
        return ToolResult(
            tool_name="think",
            content=f"Thought recorded: {thought}",
            success=True,
        )


class FailingTool(BaseTool):
    """Tool that always fails. For testing error handling paths."""

    tool_id = "failing_tool"

    def __init__(self, error_message: str = "Tool execution failed") -> None:
        self._error_message = error_message

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="failing_tool",
            description="A tool that always fails (for testing).",
            parameters={"type": "object", "properties": {}},
            category="test",
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(
            tool_name="failing_tool",
            content=self._error_message,
            success=False,
        )


class SlowTool(BaseTool):
    """Tool with configurable delay. For testing timeout behavior."""

    tool_id = "slow_tool"

    def __init__(self, delay_seconds: float = 5.0) -> None:
        self._delay = delay_seconds

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow_tool",
            description="A tool that takes time to execute (for testing timeouts).",
            parameters={"type": "object", "properties": {}},
            category="test",
            timeout_seconds=1.0,
        )

    def execute(self, **params: Any) -> ToolResult:
        import time

        time.sleep(self._delay)
        return ToolResult(
            tool_name="slow_tool",
            content="Done after delay",
            success=True,
        )


class ConfirmationTool(BaseTool):
    """Tool that requires user confirmation. For testing confirmation flow."""

    tool_id = "confirmation_tool"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="confirmation_tool",
            description="A tool requiring confirmation (for testing).",
            parameters={"type": "object", "properties": {}},
            category="test",
            requires_confirmation=True,
        )

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(
            tool_name="confirmation_tool",
            content="Confirmed and executed",
            success=True,
        )


class EchoTool(BaseTool):
    """Tool that echoes its input. Useful as a general-purpose test tool."""

    tool_id = "echo"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo the input text back.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo.",
                    },
                },
                "required": ["text"],
            },
            category="test",
        )

    def execute(self, **params: Any) -> ToolResult:
        text = params.get("text", "")
        return ToolResult(tool_name="echo", content=text, success=True)


def make_tool_set() -> list[BaseTool]:
    """Return a standard set of test tools for agent testing."""
    return [CalculatorStub(), ThinkStub(), EchoTool()]
