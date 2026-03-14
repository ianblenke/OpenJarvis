"""Tests for server cost calculator and savings modules.

Tests estimate_monthly_cost, estimate_scenario, compute_savings, and
savings_to_dict -- pure computation, no mocks needed.
"""

from __future__ import annotations

import pytest

from openjarvis.server.cost_calculator import (
    SCENARIOS,
    CostEstimate,
    Scenario,
    estimate_all_scenarios,
    estimate_monthly_cost,
    estimate_scenario,
)
from openjarvis.server.savings import (
    CLOUD_PRICING,
    ProviderSavings,
    SavingsSummary,
    compute_savings,
    savings_to_dict,
)

# ---------------------------------------------------------------------------
# SCENARIOS and CLOUD_PRICING constants
# ---------------------------------------------------------------------------


class TestConstants:
    @pytest.mark.spec("REQ-server.cost-scenarios-defined")
    def test_scenarios_not_empty(self) -> None:
        assert len(SCENARIOS) > 0

    @pytest.mark.spec("REQ-server.cost-scenarios-fields")
    def test_scenario_has_required_fields(self) -> None:
        for name, scenario in SCENARIOS.items():
            assert isinstance(scenario, Scenario)
            assert scenario.name == name
            assert len(scenario.label) > 0
            assert scenario.calls_per_month > 0
            assert scenario.avg_input_tokens > 0
            assert scenario.avg_output_tokens > 0

    @pytest.mark.spec("REQ-server.cost-pricing-defined")
    def test_cloud_pricing_not_empty(self) -> None:
        assert len(CLOUD_PRICING) > 0

    @pytest.mark.spec("REQ-server.cost-pricing-fields")
    def test_cloud_pricing_has_required_keys(self) -> None:
        for key, pricing in CLOUD_PRICING.items():
            assert "input_per_1m" in pricing
            assert "output_per_1m" in pricing
            assert "label" in pricing
            assert pricing["input_per_1m"] > 0
            assert pricing["output_per_1m"] > 0


# ---------------------------------------------------------------------------
# estimate_monthly_cost
# ---------------------------------------------------------------------------


class TestEstimateMonthlyCost:
    @pytest.mark.spec("REQ-server.cost-monthly-basic")
    def test_basic_cost_estimate(self) -> None:
        """Estimate for a known provider should return positive costs."""
        provider_key = next(iter(CLOUD_PRICING))
        result = estimate_monthly_cost(
            calls_per_month=1000,
            avg_input_tokens=500,
            avg_output_tokens=200,
            provider_key=provider_key,
        )
        assert isinstance(result, CostEstimate)
        assert result.monthly_cost > 0
        assert result.annual_cost == pytest.approx(result.monthly_cost * 12)
        assert result.total_calls_per_month == 1000

    @pytest.mark.spec("REQ-server.cost-monthly-math")
    def test_cost_calculation_math(self) -> None:
        """Verify the cost math: (tokens / 1M) * price_per_1M."""
        provider_key = next(iter(CLOUD_PRICING))
        pricing = CLOUD_PRICING[provider_key]

        calls = 1000
        avg_in = 500
        avg_out = 200

        result = estimate_monthly_cost(calls, avg_in, avg_out, provider_key)

        expected_input_cost = (calls * avg_in / 1_000_000) * pricing["input_per_1m"]
        expected_output_cost = (calls * avg_out / 1_000_000) * pricing["output_per_1m"]
        assert result.input_cost == pytest.approx(expected_input_cost)
        assert result.output_cost == pytest.approx(expected_output_cost)
        expected_total = expected_input_cost + expected_output_cost
        assert result.monthly_cost == pytest.approx(expected_total)

    @pytest.mark.spec("REQ-server.cost-monthly-unknown-provider")
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            estimate_monthly_cost(100, 100, 100, "nonexistent-provider-xyz")

    @pytest.mark.spec("REQ-server.cost-monthly-zero-calls")
    def test_zero_calls_returns_zero_cost(self) -> None:
        provider_key = next(iter(CLOUD_PRICING))
        result = estimate_monthly_cost(0, 500, 200, provider_key)
        assert result.monthly_cost == 0.0
        assert result.annual_cost == 0.0

    @pytest.mark.spec("REQ-server.cost-monthly-label")
    def test_estimate_includes_label(self) -> None:
        provider_key = next(iter(CLOUD_PRICING))
        result = estimate_monthly_cost(100, 100, 100, provider_key)
        assert result.label == str(CLOUD_PRICING[provider_key]["label"])
        assert result.provider == provider_key


# ---------------------------------------------------------------------------
# estimate_scenario
# ---------------------------------------------------------------------------


class TestEstimateScenario:
    @pytest.mark.spec("REQ-server.cost-scenario-all-providers")
    def test_returns_one_estimate_per_provider(self) -> None:
        scenario_name = next(iter(SCENARIOS))
        results = estimate_scenario(scenario_name)
        assert len(results) == len(CLOUD_PRICING)
        for est in results:
            assert isinstance(est, CostEstimate)
            assert est.monthly_cost >= 0

    @pytest.mark.spec("REQ-server.cost-scenario-unknown")
    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            estimate_scenario("nonexistent-scenario-xyz")


# ---------------------------------------------------------------------------
# estimate_all_scenarios
# ---------------------------------------------------------------------------


class TestEstimateAllScenarios:
    @pytest.mark.spec("REQ-server.cost-all-scenarios")
    def test_returns_all_scenarios(self) -> None:
        result = estimate_all_scenarios()
        assert len(result) == len(SCENARIOS)
        for name, estimates in result.items():
            assert name in SCENARIOS
            assert len(estimates) == len(CLOUD_PRICING)


# ---------------------------------------------------------------------------
# compute_savings
# ---------------------------------------------------------------------------


class TestComputeSavings:
    @pytest.mark.spec("REQ-server.savings-basic")
    def test_basic_savings(self) -> None:
        summary = compute_savings(
            prompt_tokens=10_000,
            completion_tokens=5_000,
            total_calls=100,
        )
        assert isinstance(summary, SavingsSummary)
        assert summary.total_calls == 100
        assert summary.total_prompt_tokens == 10_000
        assert summary.total_completion_tokens == 5_000
        assert summary.total_tokens == 15_000
        assert summary.local_cost == 0.0
        assert len(summary.per_provider) == len(CLOUD_PRICING)

    @pytest.mark.spec("REQ-server.savings-provider-costs")
    def test_provider_costs_positive(self) -> None:
        summary = compute_savings(
            prompt_tokens=10_000,
            completion_tokens=5_000,
            total_calls=100,
        )
        for p in summary.per_provider:
            assert isinstance(p, ProviderSavings)
            assert p.total_cost > 0
            assert p.input_cost > 0
            assert p.output_cost > 0
            assert p.energy_wh > 0
            assert p.energy_joules == pytest.approx(p.energy_wh * 3600)
            assert p.flops > 0

    @pytest.mark.spec("REQ-server.savings-zero-tokens")
    def test_zero_tokens_zero_costs(self) -> None:
        summary = compute_savings(0, 0, 0)
        for p in summary.per_provider:
            assert p.total_cost == 0.0

    @pytest.mark.spec("REQ-server.savings-avg-cost-per-query")
    def test_avg_cost_per_query(self) -> None:
        summary = compute_savings(
            prompt_tokens=10_000,
            completion_tokens=5_000,
            total_calls=100,
        )
        for key in CLOUD_PRICING:
            assert key in summary.avg_cost_per_query
            provider = next(p for p in summary.per_provider if p.provider == key)
            assert summary.avg_cost_per_query[key] == pytest.approx(
                provider.total_cost / 100
            )

    @pytest.mark.spec("REQ-server.savings-avg-cost-zero-calls")
    def test_avg_cost_zero_calls(self) -> None:
        summary = compute_savings(1000, 500, total_calls=0)
        for key in CLOUD_PRICING:
            assert summary.avg_cost_per_query[key] == 0.0

    @pytest.mark.spec("REQ-server.savings-cloud-agent-equivalent")
    def test_cloud_agent_equivalent(self) -> None:
        summary = compute_savings(1000, 500)
        assert "moderate_low" in summary.cloud_agent_equivalent
        assert "heavy_high" in summary.cloud_agent_equivalent


# ---------------------------------------------------------------------------
# savings_to_dict
# ---------------------------------------------------------------------------


class TestSavingsToDict:
    @pytest.mark.spec("REQ-server.savings-serialization")
    def test_serializes_to_dict(self) -> None:
        summary = compute_savings(10_000, 5_000, 100)
        d = savings_to_dict(summary)
        assert isinstance(d, dict)
        assert d["total_calls"] == 100
        assert d["total_tokens"] == 15_000
        assert d["local_cost"] == 0.0
        assert "per_provider" in d
        assert len(d["per_provider"]) == len(CLOUD_PRICING)

    @pytest.mark.spec("REQ-server.savings-json-serializable")
    def test_dict_is_json_serializable(self) -> None:
        import json
        summary = compute_savings(10_000, 5_000, 100)
        d = savings_to_dict(summary)
        # Should not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0
