"""Tests for GuardrailsEngine."""

from __future__ import annotations

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, Role
from openjarvis.security.guardrails import GuardrailsEngine, SecurityBlockError
from openjarvis.security.types import RedactionMode
from tests.fixtures.engines import FakeEngine


def _make_engine(response_content: str = "Hello!") -> FakeEngine:
    """Create a typed fake InferenceEngine."""
    return FakeEngine(
        engine_id="mock",
        responses=[response_content],
        models=["model-a", "model-b"],
    )


class TestGuardrailsEngineWarnMode:
    @pytest.mark.spec("REQ-security.guardrails.modes")
    @pytest.mark.spec("REQ-security.guardrails.events")
    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_warn_mode_passes_through(self) -> None:
        """WARN mode passes through content but publishes event."""
        bus = EventBus(record_history=True)
        mock = _make_engine("The key is sk-abc123def456ghi789jkl012")
        ge = GuardrailsEngine(mock, mode=RedactionMode.WARN, bus=bus)

        messages = [Message(role=Role.USER, content="tell me something")]
        result = ge.generate(messages, model="test")

        # Content should pass through unchanged
        assert result["content"] == "The key is sk-abc123def456ghi789jkl012"
        # Event should be published
        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        assert len(alerts) >= 1

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_warn_mode_no_findings(self) -> None:
        """WARN mode with clean content — no events."""
        bus = EventBus(record_history=True)
        mock = _make_engine("Just a normal response")
        ge = GuardrailsEngine(mock, mode=RedactionMode.WARN, bus=bus)

        messages = [Message(role=Role.USER, content="hello")]
        result = ge.generate(messages, model="test")

        assert result["content"] == "Just a normal response"
        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        assert len(alerts) == 0


class TestGuardrailsEngineRedactMode:
    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_redact_mode_redacts_output(self) -> None:
        """REDACT mode replaces sensitive content in output."""
        bus = EventBus(record_history=True)
        mock = _make_engine("The key is sk-abc123def456ghi789jkl012")
        ge = GuardrailsEngine(mock, mode=RedactionMode.REDACT, bus=bus)

        messages = [Message(role=Role.USER, content="tell me")]
        result = ge.generate(messages, model="test")

        assert "sk-abc123" not in result["content"]
        assert "[REDACTED:" in result["content"]

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_redact_mode_clean_passthrough(self) -> None:
        """REDACT mode with clean content — no changes."""
        mock = _make_engine("Hello there!")
        ge = GuardrailsEngine(mock, mode=RedactionMode.REDACT)

        messages = [Message(role=Role.USER, content="hi")]
        result = ge.generate(messages, model="test")

        assert result["content"] == "Hello there!"


class TestGuardrailsEngineBlockMode:
    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_block_mode_raises(self) -> None:
        """BLOCK mode raises SecurityBlockError when findings in output."""
        bus = EventBus(record_history=True)
        mock = _make_engine("The key is sk-abc123def456ghi789jkl012")
        ge = GuardrailsEngine(mock, mode=RedactionMode.BLOCK, bus=bus)

        messages = [Message(role=Role.USER, content="tell me")]
        with pytest.raises(SecurityBlockError):
            ge.generate(messages, model="test")

        blocks = [e for e in bus.history if e.event_type == EventType.SECURITY_BLOCK]
        assert len(blocks) >= 1

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_block_mode_clean_passthrough(self) -> None:
        """BLOCK mode with clean content — no exception."""
        mock = _make_engine("All good!")
        ge = GuardrailsEngine(mock, mode=RedactionMode.BLOCK)

        messages = [Message(role=Role.USER, content="hi")]
        result = ge.generate(messages, model="test")
        assert result["content"] == "All good!"


class TestGuardrailsEngineInputScanning:
    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_scan_input(self) -> None:
        """Input messages are scanned when scan_input=True."""
        bus = EventBus(record_history=True)
        mock = _make_engine("OK")
        ge = GuardrailsEngine(
            mock, mode=RedactionMode.WARN, scan_input=True, bus=bus,
        )

        secret = "my key sk-abc123def456ghi789jkl012"
        messages = [Message(role=Role.USER, content=secret)]
        ge.generate(messages, model="test")

        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        # Should detect the secret in input
        assert len(alerts) >= 1
        assert any(a.data.get("direction") == "input" for a in alerts)

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_redact_input_modifies_messages_sent_to_engine(
        self,
    ) -> None:
        """REDACT mode on input must send redacted messages."""
        engine = _make_engine("OK")
        ge = GuardrailsEngine(
            engine, mode=RedactionMode.REDACT, scan_input=True,
        )

        secret = "my key sk-abc123def456ghi789jkl012"
        messages = [Message(role=Role.USER, content=secret)]
        ge.generate(messages, model="test")

        # Engine should receive redacted content
        sent_messages = engine.call_history[-1]["messages"]
        assert "sk-abc123" not in sent_messages[0].content
        assert "[REDACTED:" in sent_messages[0].content

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_scan_input_disabled(self) -> None:
        """Input messages are not scanned when scan_input=False."""
        bus = EventBus(record_history=True)
        mock = _make_engine("OK")
        ge = GuardrailsEngine(
            mock, mode=RedactionMode.WARN, scan_input=False,
            bus=bus,
        )

        secret = "my key sk-abc123def456ghi789jkl012"
        messages = [Message(role=Role.USER, content=secret)]
        ge.generate(messages, model="test")

        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        # No input scanning, so no alerts about input
        input_alerts = [a for a in alerts if a.data.get("direction") == "input"]
        assert len(input_alerts) == 0


class TestGuardrailsEngineDelegation:
    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_delegates_list_models(self) -> None:
        """list_models() delegates to wrapped engine."""
        engine = _make_engine()
        ge = GuardrailsEngine(engine)

        models = ge.list_models()
        assert models == ["model-a", "model-b"]

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_delegates_health(self) -> None:
        """health() delegates to wrapped engine."""
        engine = _make_engine()
        ge = GuardrailsEngine(engine)

        assert ge.health() is True

    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_engine_id_delegates(self) -> None:
        """engine_id property delegates to wrapped engine."""
        engine = _make_engine()
        ge = GuardrailsEngine(engine)

        assert ge.engine_id == "mock"


class TestGuardrailsEngineCleanPassthrough:
    @pytest.mark.spec("REQ-security.types.redaction-mode")
    @pytest.mark.spec("REQ-security.guardrails.wrapping")
    def test_clean_passthrough(self) -> None:
        """No findings → content passes through unchanged in all modes."""
        for mode in RedactionMode:
            mock = _make_engine("Nothing special here")
            ge = GuardrailsEngine(mock, mode=mode)

            messages = [Message(role=Role.USER, content="hello")]
            result = ge.generate(messages, model="test")
            expected = "Nothing special here"
            assert result["content"] == expected, f"mode={mode}"


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


async def _async_token_iter(tokens):
    for t in tokens:
        yield t


@pytest.mark.asyncio
class TestGuardrailsEngineStream:
    async def test_stream_yields_tokens(self) -> None:
        """stream() yields all tokens from the wrapped engine."""
        mock = _make_engine()
        mock.stream = lambda messages, **kw: _async_token_iter(
            ["Hello", " ", "world"],
        )
        ge = GuardrailsEngine(mock)

        messages = [Message(role=Role.USER, content="hi")]
        tokens = [t async for t in ge.stream(messages, model="test")]
        assert tokens == ["Hello", " ", "world"]

    async def test_stream_scans_output_post_hoc(self) -> None:
        """stream() publishes SECURITY_ALERT after yielding sensitive tokens."""
        bus = EventBus(record_history=True)
        mock = _make_engine()
        mock.stream = lambda messages, **kw: _async_token_iter(
            ["The key is ", "sk-abc123def456ghi789jkl012"],
        )
        ge = GuardrailsEngine(mock, bus=bus)

        messages = [Message(role=Role.USER, content="show key")]
        _ = [t async for t in ge.stream(messages, model="test")]

        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        assert len(alerts) >= 1
        assert alerts[0].data["direction"] == "output"
        assert alerts[0].data["mode"] == "stream_post_hoc"

    async def test_stream_publishes_alert_with_findings(self) -> None:
        """Alert event contains a non-empty findings list with 'pattern' key."""
        bus = EventBus(record_history=True)
        mock = _make_engine()
        mock.stream = lambda messages, **kw: _async_token_iter(
            ["The key is ", "sk-abc123def456ghi789jkl012"],
        )
        ge = GuardrailsEngine(mock, bus=bus)

        messages = [Message(role=Role.USER, content="show key")]
        _ = [t async for t in ge.stream(messages, model="test")]

        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        assert len(alerts) >= 1
        findings = alerts[0].data["findings"]
        assert isinstance(findings, list) and len(findings) > 0
        assert "pattern" in findings[0]

    async def test_stream_skips_scan_when_disabled(self) -> None:
        """No alert events when scan_output=False, even with sensitive content."""
        bus = EventBus(record_history=True)
        mock = _make_engine()
        mock.stream = lambda messages, **kw: _async_token_iter(
            ["The key is ", "sk-abc123def456ghi789jkl012"],
        )
        ge = GuardrailsEngine(mock, scan_output=False, bus=bus)

        messages = [Message(role=Role.USER, content="show key")]
        _ = [t async for t in ge.stream(messages, model="test")]

        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        assert len(alerts) == 0

    async def test_stream_clean_content_no_events(self) -> None:
        """Clean tokens produce no SECURITY_ALERT events."""
        bus = EventBus(record_history=True)
        mock = _make_engine()
        mock.stream = lambda messages, **kw: _async_token_iter(
            ["Just", " a", " normal", " response"],
        )
        ge = GuardrailsEngine(mock, bus=bus)

        messages = [Message(role=Role.USER, content="hello")]
        _ = [t async for t in ge.stream(messages, model="test")]

        alerts = [e for e in bus.history if e.event_type == EventType.SECURITY_ALERT]
        assert len(alerts) == 0
