"""Tests for derived telemetry metrics -- energy efficiency, throughput, ITL.

Covers:
- efficiency.py: MFU/MBU, estimate_model_flops_per_token, estimate_model_bytes_per_token
- itl.py: compute_itl_stats percentile computation
- Derived fields on TelemetryRecord (tokens_per_joule, energy_per_output_token, etc.)
- Round-trip through TelemetryStore + TelemetryAggregator
"""

from __future__ import annotations

import time

import pytest

from openjarvis.core.types import TelemetryRecord
from openjarvis.telemetry.efficiency import (
    EfficiencyMetrics,
    compute_efficiency,
    estimate_model_bytes_per_token,
    estimate_model_flops_per_token,
)
from openjarvis.telemetry.itl import compute_itl_stats
from openjarvis.telemetry.store import TelemetryStore

# ---------------------------------------------------------------------------
# estimate_model_flops_per_token
# ---------------------------------------------------------------------------


class TestEstimateFlopsPerToken:
    @pytest.mark.spec("REQ-telemetry.derived-flops-dense")
    def test_dense_model_flops(self) -> None:
        """Dense model: FLOPs per token = 2 * params_b * 1e9."""
        result = estimate_model_flops_per_token(7.0)
        assert result == pytest.approx(2.0 * 7.0 * 1e9)

    @pytest.mark.spec("REQ-telemetry.derived-flops-moe")
    def test_moe_model_uses_active_params(self) -> None:
        """MoE: uses active_params_b, not total."""
        result = estimate_model_flops_per_token(47.0, active_params_b=12.9)
        assert result == pytest.approx(2.0 * 12.9 * 1e9)

    @pytest.mark.spec("REQ-telemetry.derived-flops-none-active")
    def test_none_active_defaults_to_total(self) -> None:
        """active_params_b=None defaults to param_count_b."""
        result = estimate_model_flops_per_token(3.0, active_params_b=None)
        assert result == pytest.approx(2.0 * 3.0 * 1e9)


# ---------------------------------------------------------------------------
# estimate_model_bytes_per_token
# ---------------------------------------------------------------------------


class TestEstimateBytesPerToken:
    @pytest.mark.spec("REQ-telemetry.derived-bytes-fp16")
    def test_default_fp16(self) -> None:
        """Default bytes_per_param=2.0 (FP16)."""
        result = estimate_model_bytes_per_token(7.0)
        assert result == pytest.approx(7.0 * 1e9 * 2.0)

    @pytest.mark.spec("REQ-telemetry.derived-bytes-int8")
    def test_int8_quantization(self) -> None:
        """INT8 → bytes_per_param=1.0."""
        result = estimate_model_bytes_per_token(7.0, bytes_per_param=1.0)
        assert result == pytest.approx(7.0 * 1e9 * 1.0)


# ---------------------------------------------------------------------------
# compute_efficiency
# ---------------------------------------------------------------------------


class TestComputeEfficiency:
    @pytest.mark.spec("REQ-telemetry.derived-mfu")
    def test_mfu_calculation(self) -> None:
        """MFU = actual_flops / peak_flops * 100."""
        metrics = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
        )
        # actual_flops = 2 * 7e9 * 100 = 1.4e12
        # peak_flops   = 312e12
        expected_mfu = (1.4e12 / 312e12) * 100.0
        assert metrics.mfu_pct == pytest.approx(expected_mfu)

    @pytest.mark.spec("REQ-telemetry.derived-mbu")
    def test_mbu_calculation(self) -> None:
        """MBU = actual_bandwidth / peak_bandwidth * 100."""
        metrics = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
        )
        actual_bw = (7.0 * 1e9 * 2.0 * 100.0) / 1e9
        expected_mbu = (actual_bw / 2039.0) * 100.0
        assert metrics.mbu_pct == pytest.approx(expected_mbu)

    @pytest.mark.spec("REQ-telemetry.derived-ipj")
    def test_ipj_with_energy(self) -> None:
        """IPJ = accuracy / energy_joules."""
        metrics = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
            energy_joules=50.0,
            accuracy=0.85,
        )
        assert metrics.ipj == pytest.approx(0.85 / 50.0)

    @pytest.mark.spec("REQ-telemetry.derived-ipj-zero-energy")
    def test_ipj_zero_energy(self) -> None:
        """IPJ = 0 when energy_joules = 0."""
        metrics = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
            energy_joules=0.0,
            accuracy=0.9,
        )
        assert metrics.ipj == 0.0

    @pytest.mark.spec("REQ-telemetry.derived-multi-gpu")
    def test_multi_gpu_scaling(self) -> None:
        """Peak FLOPS and bandwidth scale with num_gpus."""
        m1 = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
            num_gpus=1,
        )
        m4 = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
            num_gpus=4,
        )
        # 4 GPUs should have 4x peak -> MFU should be 1/4
        assert m4.mfu_pct == pytest.approx(m1.mfu_pct / 4.0)
        assert m4.peak_flops == pytest.approx(m1.peak_flops * 4.0)

    @pytest.mark.spec("REQ-telemetry.derived-zero-peak")
    def test_zero_peak_flops_no_division_error(self) -> None:
        """MFU is 0 when peak FLOPS is 0."""
        metrics = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=0.0,
            gpu_peak_bandwidth_gb_s=0.0,
            tokens_per_sec=100.0,
        )
        assert metrics.mfu_pct == 0.0
        assert metrics.mbu_pct == 0.0

    @pytest.mark.spec("REQ-telemetry.derived-efficiency-returns-dataclass")
    def test_returns_efficiency_metrics(self) -> None:
        metrics = compute_efficiency(
            param_count_b=7.0,
            active_params_b=None,
            gpu_peak_tflops=312.0,
            gpu_peak_bandwidth_gb_s=2039.0,
            tokens_per_sec=100.0,
        )
        assert isinstance(metrics, EfficiencyMetrics)
        assert metrics.actual_flops > 0
        assert metrics.actual_bandwidth_gb_s > 0


# ---------------------------------------------------------------------------
# compute_itl_stats
# ---------------------------------------------------------------------------


class TestComputeItlStats:
    @pytest.mark.spec("REQ-telemetry.derived-itl-basic")
    def test_basic_itl_computation(self) -> None:
        """ITL stats from evenly-spaced timestamps."""
        timestamps = [0.0, 10.0, 20.0, 30.0, 40.0]
        stats = compute_itl_stats(timestamps)
        # All inter-token latencies are 10.0ms
        assert stats["mean_ms"] == pytest.approx(10.0)
        assert stats["p50_ms"] == pytest.approx(10.0)
        assert stats["min_ms"] == pytest.approx(10.0)
        assert stats["max_ms"] == pytest.approx(10.0)

    @pytest.mark.spec("REQ-telemetry.derived-itl-varying")
    def test_varying_itl(self) -> None:
        """ITL stats from varying-interval timestamps."""
        timestamps = [0.0, 5.0, 15.0, 30.0, 50.0]
        stats = compute_itl_stats(timestamps)
        # ITLs: 5, 10, 15, 20
        assert stats["min_ms"] == pytest.approx(5.0)
        assert stats["max_ms"] == pytest.approx(20.0)
        assert stats["mean_ms"] == pytest.approx(12.5)

    @pytest.mark.spec("REQ-telemetry.derived-itl-too-few")
    def test_too_few_timestamps_returns_zeros(self) -> None:
        """Fewer than 2 timestamps -> all zeros."""
        assert compute_itl_stats([])["mean_ms"] == 0
        assert compute_itl_stats([5.0])["p50_ms"] == 0

    @pytest.mark.spec("REQ-telemetry.derived-itl-two-timestamps")
    def test_two_timestamps(self) -> None:
        """Exactly 2 timestamps -> one ITL value."""
        stats = compute_itl_stats([0.0, 42.0])
        assert stats["mean_ms"] == pytest.approx(42.0)
        assert stats["p99_ms"] == pytest.approx(42.0)

    @pytest.mark.spec("REQ-telemetry.derived-itl-percentiles-ordered")
    def test_percentiles_ordered(self) -> None:
        """p50 <= p90 <= p95 <= p99."""
        timestamps = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0]
        stats = compute_itl_stats(timestamps)
        assert stats["p50_ms"] <= stats["p90_ms"]
        assert stats["p90_ms"] <= stats["p95_ms"]
        assert stats["p95_ms"] <= stats["p99_ms"]


# ---------------------------------------------------------------------------
# Derived metrics stored and queryable via aggregator
# ---------------------------------------------------------------------------


class TestDerivedMetricsInStore:
    @pytest.mark.spec("REQ-telemetry.derived-store-roundtrip")
    def test_store_and_query_derived_fields(self, tmp_path) -> None:
        """Derived metric fields survive store -> aggregator round-trip."""
        from openjarvis.telemetry.aggregator import TelemetryAggregator

        store = TelemetryStore(tmp_path / "test.db")
        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="test-model",
            engine="mock",
            completion_tokens=50,
            energy_joules=10.0,
            energy_per_output_token_joules=0.2,
            throughput_per_watt=0.5,
            tokens_per_joule=5.0,
        )
        store.record(rec)

        agg = TelemetryAggregator(tmp_path / "test.db")
        stats = agg.per_model_stats()
        assert len(stats) == 1
        assert stats[0].avg_energy_per_output_token_joules == pytest.approx(0.2)
        assert stats[0].avg_throughput_per_watt == pytest.approx(0.5)
        assert stats[0].avg_tokens_per_joule == pytest.approx(5.0)
        agg.close()
        store.close()

    @pytest.mark.spec("REQ-telemetry.derived-summary-weighted")
    def test_summary_weighted_averages(self, tmp_path) -> None:
        """Summary aggregation computes weighted averages correctly."""
        from openjarvis.telemetry.aggregator import TelemetryAggregator

        store = TelemetryStore(tmp_path / "test.db")
        for i in range(3):
            store.record(TelemetryRecord(
                timestamp=time.time() + i,
                model_id="m1",
                engine="e1",
                energy_per_output_token_joules=0.1 * (i + 1),
                throughput_per_watt=1.0 * (i + 1),
            ))
        agg = TelemetryAggregator(tmp_path / "test.db")
        summary = agg.summary()
        assert summary.avg_energy_per_output_token_joules > 0
        assert summary.avg_throughput_per_watt > 0
        assert summary.total_calls == 3
        agg.close()
        store.close()
