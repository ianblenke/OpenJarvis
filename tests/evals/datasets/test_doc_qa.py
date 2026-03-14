"""Tests for the doc_qa dataset."""

import pytest

from openjarvis.evals.datasets.doc_qa import DocQADataset


@pytest.mark.spec("REQ-evals.datasets")
def test_dataset_loads():
    ds = DocQADataset()
    ds.load(max_samples=5, seed=42)
    assert ds.size() == 5


@pytest.mark.spec("REQ-evals.datasets")
def test_dataset_full_size():
    ds = DocQADataset()
    ds.load()
    assert ds.size() == 30


@pytest.mark.spec("REQ-evals.datasets")
def test_record_structure():
    ds = DocQADataset()
    ds.load(max_samples=1, seed=42)
    record = next(ds.iter_records())
    assert record.record_id.startswith("doc-qa-")
    assert record.category == "agentic"
    assert record.metadata.get("question")
    assert record.metadata.get("documents")
    assert record.metadata.get("required_facts")
    assert isinstance(record.metadata["documents"], list)
    assert isinstance(record.metadata["required_facts"], list)


@pytest.mark.spec("REQ-evals.datasets")
def test_document_structure():
    ds = DocQADataset()
    ds.load(max_samples=1, seed=0)
    record = next(ds.iter_records())
    doc = record.metadata["documents"][0]
    assert "title" in doc
    assert "content" in doc


@pytest.mark.spec("REQ-evals.datasets")
def test_required_fact_structure():
    ds = DocQADataset()
    ds.load(max_samples=1, seed=0)
    record = next(ds.iter_records())
    fact = record.metadata["required_facts"][0]
    assert "fact" in fact
    assert "source_doc_index" in fact
    assert isinstance(fact["source_doc_index"], int)


@pytest.mark.spec("REQ-evals.datasets")
def test_difficulty_tiers():
    ds = DocQADataset()
    ds.load()
    subjects = {r.subject for r in ds.iter_records()}
    assert "easy" in subjects
    assert "medium" in subjects
    assert "hard" in subjects
