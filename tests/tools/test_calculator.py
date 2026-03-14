"""Tests for the calculator tool.

Covers safe_eval() and CalculatorTool (spec properties, execute with
valid expressions, error cases). Uses the real Rust backend.
"""

from __future__ import annotations

import math

import pytest

from openjarvis.tools.calculator import CalculatorTool, safe_eval

# ---------------------------------------------------------------------------
# safe_eval — Rust-backed math expression evaluator
# ---------------------------------------------------------------------------


class TestSafeEval:
    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_addition(self):
        assert safe_eval("2 + 3") == 5

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_subtraction(self):
        assert safe_eval("10 - 3") == 7

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_multiplication(self):
        assert safe_eval("4 * 5") == 20

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_division(self):
        assert safe_eval("10 / 4") == 2.5

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_floor_division(self):
        assert safe_eval("floor(10/3)") == 3

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_modulo(self):
        assert safe_eval("10 % 3") == 1

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_power(self):
        assert safe_eval("2^10") == 1024

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_negative(self):
        assert safe_eval("-5 + 3") == -2

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_nested_expressions(self):
        assert safe_eval("(2 + 3) * (4 - 1)") == 15

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_sqrt(self):
        assert safe_eval("sqrt(16)") == 4.0

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_log(self):
        assert abs(safe_eval("ln(e)") - 1.0) < 1e-10

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_pi_constant(self):
        assert abs(safe_eval("pi") - math.pi) < 1e-10

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_e_constant(self):
        assert abs(safe_eval("e") - math.e) < 1e-10

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_sin_zero(self):
        assert abs(safe_eval("sin(0)")) < 1e-10

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_cos_zero(self):
        assert abs(safe_eval("cos(0)") - 1.0) < 1e-10

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_large_expression(self):
        result = safe_eval("2 * 3 + 4 * 5 - 1")
        assert result == 25

    @pytest.mark.spec("REQ-tools.calculator.eval")
    def test_division_by_zero_returns_inf(self):
        # meval returns infinity for division by zero
        assert safe_eval("1 / 0") == math.inf

    @pytest.mark.spec("REQ-tools.calculator.safety")
    def test_syntax_error(self):
        with pytest.raises(ValueError):
            safe_eval("2 +")

    @pytest.mark.spec("REQ-tools.calculator.safety")
    def test_unsupported_string_constant(self):
        with pytest.raises(ValueError):
            safe_eval("'hello'")

    @pytest.mark.spec("REQ-tools.calculator.safety")
    def test_unknown_function(self):
        with pytest.raises(ValueError, match="Unknown function"):
            safe_eval("exec(1)")

    @pytest.mark.spec("REQ-tools.calculator.safety")
    def test_unknown_variable(self):
        with pytest.raises(ValueError, match="unknown variable"):
            safe_eval("x + 1")

    @pytest.mark.spec("REQ-tools.calculator.safety")
    def test_empty_expression_raises(self):
        with pytest.raises(ValueError):
            safe_eval("")

    @pytest.mark.spec("REQ-tools.calculator.safety")
    def test_import_statement_rejected(self):
        with pytest.raises(ValueError):
            safe_eval("import os")


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    @pytest.mark.spec("REQ-tools.calculator.spec")
    def test_spec_name(self):
        tool = CalculatorTool()
        assert tool.spec.name == "calculator"

    @pytest.mark.spec("REQ-tools.calculator.spec")
    def test_spec_category(self):
        tool = CalculatorTool()
        assert tool.spec.category == "math"

    @pytest.mark.spec("REQ-tools.calculator.spec")
    def test_spec_has_expression_param(self):
        tool = CalculatorTool()
        props = tool.spec.parameters["properties"]
        assert "expression" in props

    @pytest.mark.spec("REQ-tools.calculator.spec")
    def test_spec_expression_required(self):
        tool = CalculatorTool()
        assert "expression" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.calculator.spec")
    def test_tool_id(self):
        tool = CalculatorTool()
        assert tool.tool_id == "calculator"

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_basic_math(self):
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 3 * 4")
        assert result.success is True
        assert result.content == "14.0"

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_sqrt_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="sqrt(144)")
        assert result.success is True
        assert result.content == "12.0"

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_empty_expression(self):
        tool = CalculatorTool()
        result = tool.execute(expression="")
        assert result.success is False
        assert "No expression" in result.content

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_no_expression(self):
        tool = CalculatorTool()
        result = tool.execute()
        assert result.success is False
        assert "No expression" in result.content

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_division_by_zero_not_error(self):
        tool = CalculatorTool()
        result = tool.execute(expression="1/0")
        # meval returns infinity for division by zero (not an error)
        assert result.success is True
        assert result.content == "inf"

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_invalid_expression_error(self):
        tool = CalculatorTool()
        result = tool.execute(expression="import os")
        assert result.success is False

    @pytest.mark.spec("REQ-tools.calculator.execute")
    def test_tool_name_in_result(self):
        tool = CalculatorTool()
        result = tool.execute(expression="1+1")
        assert result.tool_name == "calculator"

    @pytest.mark.spec("REQ-tools.calculator.openai")
    def test_openai_function_format(self):
        tool = CalculatorTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "calculator"
        assert "expression" in fn["function"]["parameters"]["properties"]

    @pytest.mark.spec("REQ-tools.calculator.spec")
    def test_spec_description_non_empty(self):
        tool = CalculatorTool()
        assert len(tool.spec.description) > 0
