"""Tests for the telemetry SQLite store -- CRUD operations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import TelemetryRecord
from openjarvis.telemetry.store import TelemetryStore

# ---------------------------------------------------------------------------
# Table creation & schema
# ---------------------------------------------------------------------------


class TestTableCreation:
    @pytest.mark.spec("REQ-telemetry.store-create-table")
    def test_creates_table_on_init(self, tmp_path: Path) -> None:
        """Opening a TelemetryStore on a fresh db should create the table."""
        store = TelemetryStore(tmp_path / "test.db")
        rows = store._fetchall()
        assert rows == []
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-reopen-existing")
    def test_opening_existing_db_is_idempotent(self, tmp_path: Path) -> None:
        """Opening a TelemetryStore on an existing db should not fail."""
        db = tmp_path / "test.db"
        store1 = TelemetryStore(db)
        store1.close()
        store2 = TelemetryStore(db)
        rows = store2._fetchall()
        assert rows == []
        store2.close()


# ---------------------------------------------------------------------------
# Record insertion (Create)
# ---------------------------------------------------------------------------


class TestRecordInsert:
    @pytest.mark.spec("REQ-telemetry.store-record-insert")
    def test_record_basic_fields(self, tmp_path: Path) -> None:
        """Inserting a TelemetryRecord persists model_id, engine, tokens."""
        store = TelemetryStore(tmp_path / "test.db")
        ts = time.time()
        rec = TelemetryRecord(
            timestamp=ts,
            model_id="qwen3:8b",
            engine="ollama",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.5,
            cost_usd=0.001,
        )
        store.record(rec)
        rows = store._fetchall()
        assert len(rows) == 1
        # model_id is column index 2 (id=0, timestamp=1, model_id=2)
        assert rows[0][2] == "qwen3:8b"
        # engine is column index 3
        assert rows[0][3] == "ollama"
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-record-energy-fields")
    def test_record_energy_fields(self, tmp_path: Path) -> None:
        """Energy-related fields are persisted correctly."""
        store = TelemetryStore(tmp_path / "test.db")
        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="test",
            energy_joules=42.5,
            power_watts=300.0,
            cpu_energy_joules=5.0,
            gpu_energy_joules=35.0,
            dram_energy_joules=2.5,
            tokens_per_joule=10.0,
            energy_per_output_token_joules=0.1,
            throughput_per_watt=3.3,
            energy_method="hw_counter",
            energy_vendor="nvidia",
        )
        store.record(rec)
        rows = store._fetchall()
        assert len(rows) == 1
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-record-warmup-flag")
    def test_warmup_flag_stored_as_int(self, tmp_path: Path) -> None:
        """is_warmup=True should be stored as integer 1."""
        store = TelemetryStore(tmp_path / "test.db")
        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="test",
            is_warmup=True,
        )
        store.record(rec)
        store._fetchall()
        # is_warmup is in the schema -- find it via SQL
        result = store._fetchall(
            "SELECT is_warmup FROM telemetry WHERE id=1"
        )
        assert result[0][0] == 1
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-record-streaming-flag")
    def test_streaming_flag_stored_as_int(self, tmp_path: Path) -> None:
        """is_streaming=True should be stored as integer 1."""
        store = TelemetryStore(tmp_path / "test.db")
        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="test",
            is_streaming=True,
        )
        store.record(rec)
        result = store._fetchall(
            "SELECT is_streaming FROM telemetry WHERE id=1"
        )
        assert result[0][0] == 1
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-record-multiple")
    def test_multiple_records(self, tmp_path: Path) -> None:
        """Multiple records are appended, not overwritten."""
        store = TelemetryStore(tmp_path / "test.db")
        for i in range(5):
            store.record(TelemetryRecord(
                timestamp=time.time() + i,
                model_id=f"model-{i}",
            ))
        rows = store._fetchall()
        assert len(rows) == 5
        store.close()


# ---------------------------------------------------------------------------
# Read (fetchall helper)
# ---------------------------------------------------------------------------


class TestFetchAll:
    @pytest.mark.spec("REQ-telemetry.store-fetchall")
    def test_fetchall_returns_all_rows(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        for i in range(3):
            store.record(TelemetryRecord(
                timestamp=time.time() + i, model_id=f"m{i}"
            ))
        rows = store._fetchall()
        assert len(rows) == 3
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-fetchall-custom-sql")
    def test_fetchall_custom_sql(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        store.record(TelemetryRecord(
            timestamp=time.time(), model_id="alpha", engine="vllm"
        ))
        store.record(TelemetryRecord(
            timestamp=time.time(), model_id="beta", engine="ollama"
        ))
        rows = store._fetchall(
            "SELECT model_id FROM telemetry WHERE engine='vllm'"
        )
        assert len(rows) == 1
        assert rows[0][0] == "alpha"
        store.close()


# ---------------------------------------------------------------------------
# Metadata JSON round-trip
# ---------------------------------------------------------------------------


class TestMetadataRoundtrip:
    @pytest.mark.spec("REQ-telemetry.store-metadata-json")
    def test_metadata_dict_roundtrips_through_json(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        meta = {"key": "value", "nested": [1, 2, 3], "flag": True}
        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="m1",
            metadata=meta,
        )
        store.record(rec)
        rows = store._fetchall()
        stored_meta = json.loads(rows[0][-1])  # metadata is last column
        assert stored_meta["key"] == "value"
        assert stored_meta["nested"] == [1, 2, 3]
        assert stored_meta["flag"] is True
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-metadata-empty")
    def test_empty_metadata_stores_as_empty_dict(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        rec = TelemetryRecord(timestamp=time.time(), model_id="m1")
        store.record(rec)
        rows = store._fetchall()
        stored_meta = json.loads(rows[0][-1])
        assert stored_meta == {}
        store.close()


# ---------------------------------------------------------------------------
# EventBus subscription
# ---------------------------------------------------------------------------


class TestBusSubscription:
    @pytest.mark.spec("REQ-telemetry.store-bus-subscribe")
    def test_subscribe_to_bus_records_on_event(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        bus = EventBus()
        store.subscribe_to_bus(bus)

        rec = TelemetryRecord(
            timestamp=time.time(),
            model_id="bus-model",
            engine="vllm",
        )
        bus.publish(EventType.TELEMETRY_RECORD, {"record": rec})

        rows = store._fetchall()
        assert len(rows) == 1
        assert rows[0][2] == "bus-model"
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-bus-ignores-non-record")
    def test_bus_ignores_event_without_record(self, tmp_path: Path) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        bus = EventBus()
        store.subscribe_to_bus(bus)

        # Publish event with no "record" key
        bus.publish(EventType.TELEMETRY_RECORD, {"other": "data"})
        rows = store._fetchall()
        assert len(rows) == 0
        store.close()

    @pytest.mark.spec("REQ-telemetry.store-bus-ignores-non-telemetry-record")
    def test_bus_ignores_event_with_non_telemetry_record(
        self, tmp_path: Path
    ) -> None:
        store = TelemetryStore(tmp_path / "test.db")
        bus = EventBus()
        store.subscribe_to_bus(bus)

        # Publish event with a non-TelemetryRecord "record" value
        bus.publish(EventType.TELEMETRY_RECORD, {"record": "not-a-record"})
        rows = store._fetchall()
        assert len(rows) == 0
        store.close()


# ---------------------------------------------------------------------------
# Close and reopen (persistence)
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.spec("REQ-telemetry.store-persistence")
    def test_close_and_reopen_preserves_data(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = TelemetryStore(db_path)
        store.record(TelemetryRecord(
            timestamp=time.time(), model_id="persist-test", engine="e1"
        ))
        store.close()

        store2 = TelemetryStore(db_path)
        rows = store2._fetchall()
        assert len(rows) == 1
        assert rows[0][2] == "persist-test"
        store2.close()


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    @pytest.mark.spec("REQ-telemetry.store-schema-migration")
    def test_migrate_schema_is_idempotent(self, tmp_path: Path) -> None:
        """Calling _migrate_schema multiple times should not raise."""
        store = TelemetryStore(tmp_path / "test.db")
        # _migrate_schema runs in __init__, call it again to verify idempotency
        store._migrate_schema()
        store._migrate_schema()
        # Should still work
        store.record(TelemetryRecord(
            timestamp=time.time(), model_id="migrate-test"
        ))
        rows = store._fetchall()
        assert len(rows) == 1
        store.close()
