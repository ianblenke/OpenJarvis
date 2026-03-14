"""Additional tests for openjarvis.tools.storage.embeddings.

The existing test_embeddings.py requires sentence-transformers (importorskip).
These tests cover the module without that dependency, using monkeypatch to
fake the library and test both success and error paths.
"""

from __future__ import annotations

import sys
import types

import pytest

from openjarvis.tools.storage.embeddings import Embedder

# ---------------------------------------------------------------------------
# ABC tests (no external dependency needed)
# ---------------------------------------------------------------------------


class TestEmbedderABC:
    @pytest.mark.spec("REQ-storage.embeddings.abc")
    def test_cannot_instantiate_abc(self) -> None:
        """Embedder ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]

    @pytest.mark.spec("REQ-storage.embeddings.abc")
    def test_concrete_subclass_works(self) -> None:
        """A concrete subclass implementing embed/dim can be instantiated."""

        class _TestEmbedder(Embedder):
            def embed(self, texts: list[str]):
                return [[0.0] * 4 for _ in texts]

            def dim(self) -> int:
                return 4

        e = _TestEmbedder()
        assert e.dim() == 4
        result = e.embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder with faked sentence-transformers
# ---------------------------------------------------------------------------


class _FakeSentenceTransformer:
    """Typed fake replacing SentenceTransformer model."""

    def __init__(self, model_name: str = "test") -> None:
        self._model_name = model_name
        self._dim = 128

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        return np.random.randn(len(texts), self._dim).astype("float32")


class TestSentenceTransformerEmbedderFaked:
    @pytest.mark.spec("REQ-storage.embeddings.sentence-transformer")
    def test_init_loads_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SentenceTransformerEmbedder calls SentenceTransformer() on init."""
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = (  # type: ignore[attr-defined]
            _FakeSentenceTransformer
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        from openjarvis.tools.storage.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name="test-model")
        assert embedder.dim() == 128

    @pytest.mark.spec("REQ-storage.embeddings.sentence-transformer")
    def test_embed_returns_correct_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = (  # type: ignore[attr-defined]
            _FakeSentenceTransformer
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        from openjarvis.tools.storage.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed(["hello", "world"])
        assert result.shape == (2, 128)

    @pytest.mark.spec("REQ-storage.embeddings.sentence-transformer")
    def test_embed_single_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = (  # type: ignore[attr-defined]
            _FakeSentenceTransformer
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        from openjarvis.tools.storage.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed(["single"])
        assert result.shape[0] == 1
        assert result.shape[1] == embedder.dim()

    @pytest.mark.spec("REQ-storage.embeddings.sentence-transformer")
    def test_import_error_raised_when_missing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ImportError with helpful message when sentence-transformers is absent."""
        import builtins

        real_import = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block)

        from openjarvis.tools.storage.embeddings import SentenceTransformerEmbedder

        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceTransformerEmbedder()
