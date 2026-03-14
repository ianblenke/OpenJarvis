"""Coverage tests for ColBERT backend — import error paths and edge cases.

The import-error tests work without colbert/torch installed.
Functional tests are skipped when the library is not available.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Import error path tests (no colbert/torch needed)
# ---------------------------------------------------------------------------


class TestColBERTImportErrors:
    @pytest.mark.spec("REQ-storage.colbert.import")
    def test_missing_torch_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ImportError raised with helpful message when torch is missing."""
        # Remove cached module so reimport actually fires
        monkeypatch.delitem(
            sys.modules, "openjarvis.tools.storage.colbert_backend", raising=False,
        )

        import builtins

        real_import = builtins.__import__

        def _block_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_torch)

        with pytest.raises(ImportError, match="PyTorch"):
            importlib.import_module("openjarvis.tools.storage.colbert_backend")


# ---------------------------------------------------------------------------
# Functional tests (conditionally collected)
# ---------------------------------------------------------------------------

_has_colbert = importlib.util.find_spec("colbert") is not None
_has_torch = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(
    not (_has_colbert and _has_torch),
    reason="colbert-ai and torch required",
)
class TestColBERTDeleteAndClear:
    @pytest.mark.spec("REQ-storage.colbert.delete")
    def test_delete_returns_false_for_nonexistent(self) -> None:
        from openjarvis.core.registry import MemoryRegistry
        from openjarvis.tools.storage.colbert_backend import ColBERTMemory

        if not MemoryRegistry.contains("colbert"):
            MemoryRegistry.register_value("colbert", ColBERTMemory)
        backend = ColBERTMemory()
        assert backend.delete("nope") is False

    @pytest.mark.spec("REQ-storage.colbert.delete")
    def test_delete_existing_doc(self) -> None:
        from openjarvis.core.registry import MemoryRegistry
        from openjarvis.tools.storage.colbert_backend import ColBERTMemory

        if not MemoryRegistry.contains("colbert"):
            MemoryRegistry.register_value("colbert", ColBERTMemory)
        backend = ColBERTMemory()
        doc_id = backend.store("content to delete")
        assert backend.count() == 1
        assert backend.delete(doc_id) is True
        assert backend.count() == 0

    @pytest.mark.spec("REQ-storage.colbert.clear")
    def test_clear_empties_everything(self) -> None:
        from openjarvis.core.registry import MemoryRegistry
        from openjarvis.tools.storage.colbert_backend import ColBERTMemory

        if not MemoryRegistry.contains("colbert"):
            MemoryRegistry.register_value("colbert", ColBERTMemory)
        backend = ColBERTMemory()
        backend.store("a")
        backend.store("b")
        assert backend.count() == 2
        backend.clear()
        assert backend.count() == 0


@pytest.mark.skipif(
    not (_has_colbert and _has_torch),
    reason="colbert-ai and torch required",
)
class TestColBERTRetrieveEdgeCases:
    @pytest.mark.spec("REQ-storage.colbert.retrieve")
    def test_retrieve_empty_query(self) -> None:
        from openjarvis.core.registry import MemoryRegistry
        from openjarvis.tools.storage.colbert_backend import ColBERTMemory

        if not MemoryRegistry.contains("colbert"):
            MemoryRegistry.register_value("colbert", ColBERTMemory)
        backend = ColBERTMemory()
        backend.store("some content")
        results = backend.retrieve("")
        assert results == []

    @pytest.mark.spec("REQ-storage.colbert.retrieve")
    def test_retrieve_no_documents(self) -> None:
        from openjarvis.core.registry import MemoryRegistry
        from openjarvis.tools.storage.colbert_backend import ColBERTMemory

        if not MemoryRegistry.contains("colbert"):
            MemoryRegistry.register_value("colbert", ColBERTMemory)
        backend = ColBERTMemory()
        results = backend.retrieve("anything")
        assert results == []
