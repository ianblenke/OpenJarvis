"""Tests for the daily_digest dataset."""

import pytest

from openjarvis.evals.datasets.daily_digest import DailyDigestDataset


@pytest.mark.spec("REQ-evals.datasets")
def test_dataset_loads():
    ds = DailyDigestDataset()
    ds.load(max_samples=5, seed=42)
    assert ds.size() == 5


@pytest.mark.spec("REQ-evals.datasets")
def test_dataset_full_size():
    ds = DailyDigestDataset()
    ds.load()
    assert ds.size() == 30


@pytest.mark.spec("REQ-evals.datasets")
def test_record_structure():
    ds = DailyDigestDataset()
    ds.load(max_samples=1, seed=42)
    record = next(ds.iter_records())
    assert record.record_id.startswith("daily-digest-")
    assert record.category == "agentic"
    assert record.metadata.get("role")
    assert record.metadata.get("company")
    assert record.metadata.get("must_mention")
    assert record.metadata.get("priority_order")
    assert isinstance(record.metadata["must_mention"], list)
    assert isinstance(record.metadata["priority_order"], list)


@pytest.mark.spec("REQ-evals.datasets")
def test_difficulty_tiers():
    ds = DailyDigestDataset()
    ds.load()
    subjects = {r.subject for r in ds.iter_records()}
    assert "easy" in subjects
    assert "medium" in subjects
    assert "hard" in subjects
