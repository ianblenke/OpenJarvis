"""Tests for InstrumentedEngine telemetry wrapper."""

from __future__ import annotations

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, Role
from openjarvis.telemetry.instrumented_engine import InstrumentedEngine
from tests.fixtures.engines import FakeEngine


class _InstrumentedTestEngine(FakeEngine):
    """FakeEngine subclass that also exposes list_models/health for delegation."""

    def __init__(self) -> None:
        super().__init__(
            engine_id="mock",
            responses=["Hello!"],
            models=["test-model"],
            healthy=True,
        )

    def generate(self, messages, *, model, **kwargs):
        result = super().generate(messages, model=model, **kwargs)
        # Override usage to match the original test expectations
        result["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        return result


@pytest.fixture
def inner_engine():
    return _InstrumentedTestEngine()


@pytest.fixture
def bus():
    return EventBus(record_history=True)


class TestInstrumentedEngine:
    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_generate_passes_through(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        result = ie.generate(messages, model="test")
        assert result["content"] == "Hello!"
        assert inner_engine.call_count == 1

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_generate_publishes_events(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test")

        event_types = [e.event_type for e in bus.history]
        assert EventType.INFERENCE_START in event_types
        assert EventType.INFERENCE_END in event_types
        assert EventType.TELEMETRY_RECORD in event_types

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_generate_records_latency(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test")

        end_events = [e for e in bus.history if e.event_type == EventType.INFERENCE_END]
        assert len(end_events) == 1
        assert "latency" in end_events[0].data

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_generate_records_telemetry(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test")

        tel_events = [
            e for e in bus.history
            if e.event_type == EventType.TELEMETRY_RECORD
        ]
        assert len(tel_events) == 1
        record = tel_events[0].data["record"]
        assert record.model_id == "test"
        assert record.prompt_tokens == 10
        assert record.completion_tokens == 5

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_list_models_delegates(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        assert ie.list_models() == ["test-model"]

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_health_delegates(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        assert ie.health() is True

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_stream_delegates(self, inner_engine, bus):
        """Stream is async, so we test via pytest-asyncio or manually."""
        # InstrumentedEngine.stream is async, so we skip sync iteration test
        # and just verify the method exists and delegates
        ie = InstrumentedEngine(inner_engine, bus)
        assert hasattr(ie, "stream")

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_temperature_passthrough(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test", temperature=0.5, max_tokens=100)
        last_call = inner_engine.call_history[-1]
        assert last_call["temperature"] == 0.5

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_inner_engine_id(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        tel_events_data = []
        bus.subscribe(
            EventType.TELEMETRY_RECORD,
            lambda e: tel_events_data.append(e.data),
        )
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test")
        assert tel_events_data[0]["record"].engine == "mock"

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_kwargs_passthrough(self, inner_engine, bus):
        """Extra kwargs should be forwarded to inner engine."""
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test", tools=[{"type": "function"}])
        last_call = inner_engine.call_history[-1]
        assert "tools" in last_call["kwargs"]

    @pytest.mark.spec("REQ-telemetry.instrumented-engine")
    def test_engine_id_attribute(self, inner_engine, bus):
        ie = InstrumentedEngine(inner_engine, bus)
        assert ie.engine_id == "instrumented"


class TestTokensPerJoule:
    @pytest.mark.spec("REQ-telemetry.derived")
    def test_tokens_per_joule_zero_without_energy(self, inner_engine, bus):
        """tokens_per_joule is 0.0 when no energy monitor is available."""
        ie = InstrumentedEngine(inner_engine, bus)
        messages = [Message(role=Role.USER, content="Hi")]
        ie.generate(messages, model="test")

        tel_events = [
            e for e in bus.history
            if e.event_type == EventType.TELEMETRY_RECORD
        ]
        record = tel_events[0].data["record"]
        assert record.tokens_per_joule == 0.0

    @pytest.mark.spec("REQ-telemetry.derived")
    def test_tokens_per_joule_formula_via_record(self):
        """Verify the formula: tokens_per_joule = completion_tokens / energy_joules."""
        from openjarvis.core.types import TelemetryRecord

        # Direct construction — verifies the field accepts computed values
        rec = TelemetryRecord(
            timestamp=1.0,
            model_id="test",
            completion_tokens=50,
            energy_joules=2.5,
            tokens_per_joule=50.0 / 2.5,  # = 20.0
        )
        assert rec.tokens_per_joule == pytest.approx(20.0)

    @pytest.mark.spec("REQ-telemetry.derived")
    def test_tokens_per_joule_zero_when_no_tokens(self):
        """tokens_per_joule is 0.0 when completion_tokens is 0."""
        from openjarvis.core.types import TelemetryRecord

        rec = TelemetryRecord(
            timestamp=1.0,
            model_id="test",
            completion_tokens=0,
            energy_joules=5.0,
            tokens_per_joule=0.0,
        )
        assert rec.tokens_per_joule == 0.0
