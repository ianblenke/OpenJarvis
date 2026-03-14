"""Tests for dataset loading — base class, iteration, sample extraction."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pytest

from openjarvis.evals.core.dataset import DatasetProvider
from openjarvis.evals.core.types import EvalRecord

# ---------------------------------------------------------------------------
# Typed fakes
# ---------------------------------------------------------------------------


class InMemoryDataset(DatasetProvider):
    """Concrete DatasetProvider backed by a list of dicts.

    Simulates loading from an in-memory source (like parsed JSON/CSV rows)
    with support for max_samples, split, and seed.
    """

    dataset_id = "in-memory"
    dataset_name = "InMemory"

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows
        self._records: List[EvalRecord] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        import random

        selected = list(self._rows)

        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(selected)

        if max_samples is not None:
            selected = selected[:max_samples]

        self._records = []
        for idx, row in enumerate(selected):
            self._records.append(EvalRecord(
                record_id=f"mem-{idx}",
                problem=row.get("question", ""),
                reference=row.get("answer", ""),
                category=row.get("category", "chat"),
                subject=row.get("subject", ""),
                metadata=row.get("metadata", {}),
            ))

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)


class EpisodeDataset(DatasetProvider):
    """Dataset with episode structure for sequential processing."""

    dataset_id = "episode"
    dataset_name = "Episode"

    def __init__(self, episodes: List[List[Dict[str, Any]]]) -> None:
        self._episodes_raw = episodes
        self._records: List[EvalRecord] = []
        self._episodes: List[List[EvalRecord]] = []

    def load(
        self,
        *,
        max_samples: Optional[int] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._records = []
        self._episodes = []
        idx = 0
        for ep_raw in self._episodes_raw:
            episode_records: List[EvalRecord] = []
            for row in ep_raw:
                rec = EvalRecord(
                    record_id=f"ep-{idx}",
                    problem=row.get("question", ""),
                    reference=row.get("answer", ""),
                    category="chat",
                )
                self._records.append(rec)
                episode_records.append(rec)
                idx += 1
            self._episodes.append(episode_records)

        if max_samples is not None:
            self._records = self._records[:max_samples]

    def iter_records(self) -> Iterable[EvalRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def iter_episodes(self) -> Iterable[List[EvalRecord]]:
        return iter(self._episodes)


# ---------------------------------------------------------------------------
# Tests: DatasetProvider base class
# ---------------------------------------------------------------------------


class TestDatasetProviderBase:
    """Tests for DatasetProvider ABC contract."""

    @pytest.mark.spec("REQ-evals.dataset.base-protocol")
    def test_base_class_requires_load(self):
        """Cannot instantiate DatasetProvider without implementing load."""
        with pytest.raises(TypeError):
            DatasetProvider()  # type: ignore[abstract]

    @pytest.mark.spec("REQ-evals.dataset.base-protocol")
    def test_base_class_requires_iter_records(self):
        """DatasetProvider requires iter_records to be implemented."""

        class MissingIterRecords(DatasetProvider):
            dataset_id = "x"
            dataset_name = "X"

            def load(self, **kw: Any) -> None:
                pass

            def size(self) -> int:
                return 0

        with pytest.raises(TypeError):
            MissingIterRecords()  # type: ignore[abstract]

    @pytest.mark.spec("REQ-evals.dataset.base-protocol")
    def test_base_class_requires_size(self):
        """DatasetProvider requires size to be implemented."""

        class MissingSize(DatasetProvider):
            dataset_id = "x"
            dataset_name = "X"

            def load(self, **kw: Any) -> None:
                pass

            def iter_records(self) -> Iterable[EvalRecord]:
                return iter([])

        with pytest.raises(TypeError):
            MissingSize()  # type: ignore[abstract]

    @pytest.mark.spec("REQ-evals.dataset.base-protocol")
    def test_create_task_env_default_returns_none(self):
        """Default create_task_env returns None."""
        ds = InMemoryDataset([])
        ds.load()
        record = EvalRecord(
            record_id="t1", problem="test", reference="ref", category="chat",
        )
        assert ds.create_task_env(record) is None

    @pytest.mark.spec("REQ-evals.dataset.base-protocol")
    def test_verify_requirements_default_returns_empty(self):
        """Default verify_requirements returns empty list."""
        ds = InMemoryDataset([])
        assert ds.verify_requirements() == []

    @pytest.mark.spec("REQ-evals.dataset.base-protocol")
    def test_default_iter_episodes_yields_single_record_episodes(self):
        """Default iter_episodes wraps each record in its own episode."""
        rows = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        ds = InMemoryDataset(rows)
        ds.load()

        episodes = list(ds.iter_episodes())
        assert len(episodes) == 2
        assert len(episodes[0]) == 1
        assert len(episodes[1]) == 1
        assert episodes[0][0].problem == "Q1"
        assert episodes[1][0].problem == "Q2"


# ---------------------------------------------------------------------------
# Tests: Loading and iteration
# ---------------------------------------------------------------------------


class TestDatasetLoading:
    """Tests for dataset loading with various parameters."""

    @pytest.mark.spec("REQ-evals.dataset.load")
    def test_load_all_rows(self):
        rows = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        ds = InMemoryDataset(rows)
        ds.load()

        assert ds.size() == 3

    @pytest.mark.spec("REQ-evals.dataset.load")
    def test_load_with_max_samples(self):
        rows = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
        ds = InMemoryDataset(rows)
        ds.load(max_samples=3)

        assert ds.size() == 3

    @pytest.mark.spec("REQ-evals.dataset.load")
    def test_load_max_samples_exceeds_rows(self):
        rows = [{"question": "Q1", "answer": "A1"}]
        ds = InMemoryDataset(rows)
        ds.load(max_samples=100)

        assert ds.size() == 1

    @pytest.mark.spec("REQ-evals.dataset.load")
    def test_load_with_seed_shuffles(self):
        rows = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]
        ds1 = InMemoryDataset(rows)
        ds1.load(seed=42)
        ds2 = InMemoryDataset(rows)
        ds2.load(seed=42)

        records1 = list(ds1.iter_records())
        records2 = list(ds2.iter_records())

        # Same seed produces same order
        assert [r.problem for r in records1] == [r.problem for r in records2]

    @pytest.mark.spec("REQ-evals.dataset.load")
    def test_load_different_seeds_different_order(self):
        rows = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]
        ds1 = InMemoryDataset(rows)
        ds1.load(seed=42)
        ds2 = InMemoryDataset(rows)
        ds2.load(seed=99)

        records1 = [r.problem for r in ds1.iter_records()]
        records2 = [r.problem for r in ds2.iter_records()]

        # Different seeds should produce different order (probabilistically)
        assert records1 != records2


class TestDatasetIteration:
    """Tests for iter_records and record structure."""

    @pytest.mark.spec("REQ-evals.dataset.iteration")
    def test_iter_records_yields_eval_records(self):
        rows = [{"question": "What is 2+2?", "answer": "4", "category": "math"}]
        ds = InMemoryDataset(rows)
        ds.load()

        records = list(ds.iter_records())
        assert len(records) == 1
        assert isinstance(records[0], EvalRecord)

    @pytest.mark.spec("REQ-evals.dataset.iteration")
    def test_record_fields_populated(self):
        rows = [
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence",
                "category": "tech",
                "subject": "cs",
                "metadata": {"source": "test"},
            },
        ]
        ds = InMemoryDataset(rows)
        ds.load()

        record = list(ds.iter_records())[0]
        assert record.problem == "What is AI?"
        assert record.reference == "Artificial Intelligence"
        assert record.category == "tech"
        assert record.subject == "cs"
        assert record.metadata == {"source": "test"}

    @pytest.mark.spec("REQ-evals.dataset.iteration")
    def test_record_ids_are_unique(self):
        rows = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
        ds = InMemoryDataset(rows)
        ds.load()

        ids = [r.record_id for r in ds.iter_records()]
        assert len(ids) == len(set(ids))

    @pytest.mark.spec("REQ-evals.dataset.iteration")
    def test_iter_records_can_be_called_multiple_times(self):
        rows = [{"question": "Q1", "answer": "A1"}]
        ds = InMemoryDataset(rows)
        ds.load()

        list1 = list(ds.iter_records())
        list2 = list(ds.iter_records())
        assert len(list1) == len(list2) == 1

    @pytest.mark.spec("REQ-evals.dataset.iteration")
    def test_empty_dataset_iteration(self):
        ds = InMemoryDataset([])
        ds.load()

        records = list(ds.iter_records())
        assert records == []
        assert ds.size() == 0


# ---------------------------------------------------------------------------
# Tests: Episode iteration
# ---------------------------------------------------------------------------


class TestEpisodeIteration:
    """Tests for datasets with episode-based grouping."""

    @pytest.mark.spec("REQ-evals.dataset.episodes")
    def test_episode_dataset_groups_records(self):
        episodes = [
            [
                {"question": "E1-Q1", "answer": "A1"},
                {"question": "E1-Q2", "answer": "A2"},
            ],
            [
                {"question": "E2-Q1", "answer": "A3"},
            ],
        ]
        ds = EpisodeDataset(episodes)
        ds.load()

        eps = list(ds.iter_episodes())
        assert len(eps) == 2
        assert len(eps[0]) == 2
        assert len(eps[1]) == 1

    @pytest.mark.spec("REQ-evals.dataset.episodes")
    def test_episode_records_total(self):
        episodes = [
            [{"question": "Q1", "answer": "A1"}],
            [{"question": "Q2", "answer": "A2"}, {"question": "Q3", "answer": "A3"}],
        ]
        ds = EpisodeDataset(episodes)
        ds.load()

        assert ds.size() == 3
        all_records = list(ds.iter_records())
        assert len(all_records) == 3


# ---------------------------------------------------------------------------
# Tests: File-based loading (JSON)
# ---------------------------------------------------------------------------


class TestFileBasedLoading:
    """Tests for loading datasets from files using tmp_path."""

    @pytest.mark.spec("REQ-evals.dataset.file-load")
    def test_load_from_json_file(self, tmp_path):
        import json

        data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        json_file = tmp_path / "test_dataset.json"
        json_file.write_text(json.dumps(data))

        loaded = json.loads(json_file.read_text())
        ds = InMemoryDataset(loaded)
        ds.load()

        assert ds.size() == 2
        records = list(ds.iter_records())
        assert records[0].problem == "Q1"
        assert records[1].reference == "A2"

    @pytest.mark.spec("REQ-evals.dataset.file-load")
    def test_load_from_jsonl_file(self, tmp_path):
        import json

        rows = [
            {"question": "Q1", "answer": "A1", "category": "math"},
            {"question": "Q2", "answer": "A2", "category": "science"},
        ]
        jsonl_file = tmp_path / "test_dataset.jsonl"
        jsonl_file.write_text(
            "\n".join(json.dumps(row) for row in rows)
        )

        loaded = [
            json.loads(line)
            for line in jsonl_file.read_text().strip().split("\n")
        ]
        ds = InMemoryDataset(loaded)
        ds.load()

        assert ds.size() == 2
        records = list(ds.iter_records())
        assert records[0].category == "math"
        assert records[1].category == "science"

    @pytest.mark.spec("REQ-evals.dataset.file-load")
    def test_load_from_csv_file(self, tmp_path):
        import csv

        csv_file = tmp_path / "test_dataset.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerow({"question": "Q1", "answer": "A1"})
            writer.writerow({"question": "Q2", "answer": "A2"})

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            loaded = [dict(row) for row in reader]

        ds = InMemoryDataset(loaded)
        ds.load()

        assert ds.size() == 2


# ---------------------------------------------------------------------------
# Tests: EvalRecord dataclass
# ---------------------------------------------------------------------------


class TestEvalRecord:
    """Tests for EvalRecord data structure."""

    @pytest.mark.spec("REQ-evals.dataset.eval-record")
    def test_record_creation(self):
        record = EvalRecord(
            record_id="test-1",
            problem="What is 1+1?",
            reference="2",
            category="math",
        )
        assert record.record_id == "test-1"
        assert record.problem == "What is 1+1?"
        assert record.reference == "2"
        assert record.category == "math"
        assert record.subject == ""
        assert record.metadata == {}

    @pytest.mark.spec("REQ-evals.dataset.eval-record")
    def test_record_with_metadata(self):
        record = EvalRecord(
            record_id="test-2",
            problem="Q",
            reference="A",
            category="chat",
            metadata={"source": "test", "difficulty": "easy"},
        )
        assert record.metadata["source"] == "test"
        assert record.metadata["difficulty"] == "easy"

    @pytest.mark.spec("REQ-evals.dataset.eval-record")
    def test_record_with_subject(self):
        record = EvalRecord(
            record_id="test-3",
            problem="Q",
            reference="A",
            category="reasoning",
            subject="physics",
        )
        assert record.subject == "physics"
