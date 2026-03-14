"""Tests for the composition layer."""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.core.events import EventBus
from openjarvis.system import JarvisSystem, SystemBuilder
from tests.fixtures.engines import FakeEngine

# ---------------------------------------------------------------------------
# Lightweight typed fakes for subsystem close/stop verification
# ---------------------------------------------------------------------------


class _CloseableStub:
    """Tracks whether close() was called. Replaces MagicMock for resources."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _StoppableStub:
    """Tracks whether stop() was called. Replaces MagicMock for schedulers."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


# ---------------------------------------------------------------------------
# JarvisSystem tests
# ---------------------------------------------------------------------------


class TestJarvisSystem:
    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_direct_mode(self):
        engine = FakeEngine(responses=["Hello!"])
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test-model",
        )
        result = system.ask("Hi")
        assert result["content"] == "Hello!"
        assert result["model"] == "test-model"
        assert result["engine"] == "fake"

    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_returns_usage(self):
        engine = FakeEngine(responses=["OK"])
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        result = system.ask("Hi")
        assert "prompt_tokens" in result["usage"]

    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_no_agent_direct_mode(self):
        """When agent_name is empty and no agent param, use direct engine mode."""
        engine = FakeEngine(responses=["Direct response"])
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            agent_name="",
        )
        result = system.ask("Hi")
        assert result["content"] == "Direct response"
        assert engine.call_count == 1

    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_with_agent_none_uses_direct(self):
        """Passing agent_name='none' should use direct engine mode."""
        engine = FakeEngine(responses=["Direct response"])
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            agent_name="none",
        )
        result = system.ask("Hi")
        assert result["content"] == "Direct response"

    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_with_agent_override(self):
        """Passing agent= param should use that agent even if system has a default."""
        from openjarvis.agents._stubs import AgentResult
        from openjarvis.core.registry import AgentRegistry

        class TestAgent:
            agent_id = "test-system-agent"

            def __init__(self, eng, model, **kwargs):
                pass

            def run(self, input, context=None, **kwargs):
                return AgentResult(content="From test agent", turns=1)

        # Register (or re-register) the agent
        if not AgentRegistry.contains("test-system-agent"):
            AgentRegistry.register_value("test-system-agent", TestAgent)

        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            agent_name="",
        )
        result = system.ask("Hi", agent="test-system-agent")
        assert result["content"] == "From test agent"

    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_unknown_agent(self):
        """Unknown agent should return an error dict."""
        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        result = system.ask("Hi", agent="nonexistent-agent-xyz")
        assert "Unknown agent" in result["content"]
        assert result.get("error") is True

    @pytest.mark.spec("REQ-agents.system-ask")
    def test_ask_passes_temperature_and_max_tokens(self):
        engine = FakeEngine(responses=["OK"])
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        system.ask("Hi", temperature=0.3, max_tokens=512)
        call = engine.call_history[0]
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 512

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close(self):
        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        system.close()  # Should not raise

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_telemetry(self):
        engine = FakeEngine()
        telem = _CloseableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            telemetry_store=telem,
        )
        system.close()
        assert telem.closed is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_trace_store(self):
        engine = FakeEngine()
        trace = _CloseableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            trace_store=trace,
        )
        system.close()
        assert trace.closed is True

    @pytest.mark.spec("REQ-agents.system-tools")
    def test_build_tools_empty(self):
        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        tools = system._build_tools([])
        assert tools == []

    @pytest.mark.spec("REQ-agents.system-tools")
    def test_build_tools_unknown_tool(self):
        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        tools = system._build_tools(["nonexistent_tool_xyz"])
        assert tools == []


class TestSystemBuilder:
    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_fluent_api(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        result = builder.engine("ollama").model("test").agent("simple")
        assert result is builder  # fluent

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_stores_config(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        assert builder._config is config

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_engine_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        builder.engine("vllm")
        assert builder._engine_key == "vllm"

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_model_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        builder.model("my-model")
        assert builder._model == "my-model"

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_agent_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        builder.agent("orchestrator")
        assert builder._agent_name == "orchestrator"

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_tools_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        builder.tools(["calculator", "think"])
        assert builder._tool_names == ["calculator", "think"]

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_telemetry_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        builder.telemetry(False)
        assert builder._telemetry is False

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_traces_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        builder.traces(True)
        assert builder._traces is True

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_event_bus_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        bus = EventBus()
        builder.event_bus(bus)
        assert builder._bus is bus

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_chaining(self):
        config = JarvisConfig()
        builder = (
            SystemBuilder(config)
            .engine("ollama")
            .model("test-model")
            .agent("simple")
            .tools(["calculator"])
            .telemetry(True)
            .traces(False)
        )
        assert builder._engine_key == "ollama"
        assert builder._model == "test-model"
        assert builder._agent_name == "simple"
        assert builder._tool_names == ["calculator"]
        assert builder._telemetry is True
        assert builder._traces is False

    @pytest.mark.spec("REQ-agents.builder")
    def test_import_works(self):
        from openjarvis.system import JarvisSystem, SystemBuilder

        assert JarvisSystem is not None
        assert SystemBuilder is not None

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_default_config(self):
        """SystemBuilder with no config should load defaults."""
        builder = SystemBuilder()
        assert builder._config is not None
        assert isinstance(builder._config, JarvisConfig)

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_build_raises_without_engine(self):
        """build() should raise RuntimeError when no engine is available."""
        config = JarvisConfig()
        # Use a nonsense engine key to ensure no engine is found
        builder = SystemBuilder(config).engine("nonexistent_engine_xyz_123")
        with pytest.raises(RuntimeError, match="No inference engine"):
            builder.build()

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_sandbox_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        result = builder.sandbox(True)
        assert result is builder  # fluent
        assert builder._sandbox is True

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_scheduler_setter(self):
        config = JarvisConfig()
        builder = SystemBuilder(config)
        result = builder.scheduler(True)
        assert result is builder  # fluent
        assert builder._scheduler is True

    @pytest.mark.spec("REQ-agents.builder")
    def test_builder_sandbox_scheduler_chaining(self):
        config = JarvisConfig()
        builder = (
            SystemBuilder(config)
            .engine("ollama")
            .model("test")
            .sandbox(True)
            .scheduler(True)
        )
        assert builder._sandbox is True
        assert builder._scheduler is True
        assert builder._engine_key == "ollama"


class TestJarvisSystemClose:
    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_scheduler_store(self):
        engine = FakeEngine()
        sched_store = _CloseableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            scheduler_store=sched_store,
        )
        system.close()
        assert sched_store.closed is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_scheduler(self):
        engine = FakeEngine()
        scheduler = _StoppableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            scheduler=scheduler,
        )
        system.close()
        assert scheduler.stopped is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_memory_backend(self):
        engine = FakeEngine()
        mem = _CloseableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            memory_backend=mem,
        )
        system.close()
        assert mem.closed is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_session_store(self):
        engine = FakeEngine()
        sess = _CloseableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            session_store=sess,
        )
        system.close()
        assert sess.closed is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_workflow_engine(self):
        engine = FakeEngine()
        wf = _CloseableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            workflow_engine=wf,
        )
        system.close()
        assert wf.closed is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_system_fields_default_none(self):
        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        assert system.scheduler_store is None
        assert system.scheduler is None
        assert system.container_runner is None

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_close_with_agent_scheduler(self):
        engine = FakeEngine()
        agent_scheduler = _StoppableStub()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
            agent_scheduler=agent_scheduler,
        )
        system.close()
        assert agent_scheduler.stopped is True

    @pytest.mark.spec("REQ-agents.system-lifecycle")
    def test_system_agent_fields_default_none(self):
        engine = FakeEngine()
        system = JarvisSystem(
            config=JarvisConfig(),
            bus=EventBus(),
            engine=engine,
            engine_key="fake",
            model="test",
        )
        assert system.agent_scheduler is None
        assert system.agent_executor is None
