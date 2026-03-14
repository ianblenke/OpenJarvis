"""Coverage tests for FAISS backend — import error paths and edge cases.

Import-error tests work without faiss installed.
Functional tests are skipped when the library is not available.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Import error path (no faiss needed)
# ---------------------------------------------------------------------------


class TestFAISSImportError:
    @pytest.mark.spec("REQ-storage.faiss.import")
    def test_missing_faiss_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ImportError raised with helpful message when faiss is missing."""
        monkeypatch.delitem(
            sys.modules, "openjarvis.tools.storage.faiss_backend", raising=False,
        )
        monkeypatch.delitem(sys.modules, "faiss", raising=False)

        import builtins

        real_import = builtins.__import__

        def _block_faiss(name, *args, **kwargs):
            if name == "faiss":
                raise ImportError("mocked no faiss")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_faiss)

        with pytest.raises(ImportError, match="faiss"):
            importlib.import_module("openjarvis.tools.storage.faiss_backend")


# ---------------------------------------------------------------------------
# Functional tests (conditionally collected)
# ---------------------------------------------------------------------------

_has_faiss = importlib.util.find_spec("faiss") is not None


@pytest.mark.skipif(not _has_faiss, reason="faiss-cpu required")
class TestFAISSDeleteEdgeCases:
    def _make_backend(self):
        import numpy as np

        from openjarvis.core.registry import MemoryRegistry
        from openjarvis.tools.storage.embeddings import Embedder
        from openjarvis.tools.storage.faiss_backend import FAISSMemory

        class _FakeEmbedder(Embedder):
            def embed(self, texts: list[str]):
                results = []
                for text in texts:
                    rng = np.random.RandomState(hash(text) % 2**31)
                    vec = rng.randn(64).astype(np.float32)
                    results.append(vec)
                if results:
                    return np.array(results)
                return np.empty((0, 64), dtype=np.float32)

            def dim(self) -> int:
                return 64

        if not MemoryRegistry.contains("faiss"):
            MemoryRegistry.register_value("faiss", FAISSMemory)
        return FAISSMemory(embedder=_FakeEmbedder())

    @pytest.mark.spec("REQ-storage.faiss.delete")
    def test_delete_nonexistent_returns_false(self) -> None:
        backend = self._make_backend()
        assert backend.delete("nonexistent") is False

    @pytest.mark.spec("REQ-storage.faiss.delete")
    def test_double_delete_returns_false(self) -> None:
        backend = self._make_backend()
        doc_id = backend.store("content")
        assert backend.delete(doc_id) is True
        assert backend.delete(doc_id) is False

    @pytest.mark.spec("REQ-storage.faiss.retrieve")
    def test_retrieve_empty_query(self) -> None:
        backend = self._make_backend()
        backend.store("data")
        results = backend.retrieve("")
        assert results == []

    @pytest.mark.spec("REQ-storage.faiss.retrieve")
    def test_retrieve_empty_index(self) -> None:
        backend = self._make_backend()
        results = backend.retrieve("query")
        assert results == []

    @pytest.mark.spec("REQ-storage.faiss.clear")
    def test_clear_resets_everything(self) -> None:
        backend = self._make_backend()
        backend.store("a")
        backend.store("b")
        backend.clear()
        assert backend._index.ntotal == 0
        assert len(backend._documents) == 0

    @pytest.mark.spec("REQ-storage.faiss.metadata")
    def test_none_metadata_stored_as_empty(self) -> None:
        backend = self._make_backend()
        doc_id = backend.store("content", metadata=None)
        assert doc_id in backend._documents
        _, _, meta = backend._documents[doc_id]
        assert meta == {}
