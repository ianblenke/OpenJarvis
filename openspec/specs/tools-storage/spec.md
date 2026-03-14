# Tools Storage Module Spec

Memory/retrieval backends with chunking, embeddings, and context injection.

## MemoryBackend Protocol (`_stubs.py`)

### REQ-storage.protocol.store: Document storage
`MemoryBackend.store(content, *, source="", metadata=None) -> str` persists content, returns doc_id.

### REQ-storage.protocol.retrieve: Document retrieval
`retrieve(query, *, top_k=5, **kwargs) -> List[RetrievalResult]` searches and returns ranked results.

### REQ-storage.protocol.delete: Document deletion
`delete(doc_id) -> bool` returns True if document existed.

### REQ-storage.protocol.clear: Clear all
`clear()` removes all stored documents.

### REQ-storage.protocol.registration: Registry-based registration
All backends use `@MemoryRegistry.register("name")` decorator.

## RetrievalResult

### REQ-storage.result: Retrieval result
`RetrievalResult` with `content`, `score`, `source`, `metadata`.

## Backends

### REQ-storage.sqlite: SQLite FTS5
Persistent storage with FTS5 full-text search and BM25 ranking. Rust backend.

### REQ-storage.faiss: FAISS dense retrieval
In-memory dense vector retrieval using FAISS IndexFlatIP with L2-normalized embeddings. Default embedder: `all-MiniLM-L6-v2`.

### REQ-storage.faiss.import: FAISS import error handling
FAISS backend raises a helpful `ImportError` when the faiss library is not installed.

### REQ-storage.faiss.delete: FAISS document deletion
FAISS backend `delete(doc_id)` removes a stored document and returns True, or False if the document does not exist or was already deleted.

### REQ-storage.faiss.retrieve: FAISS retrieval edge cases
FAISS backend `retrieve()` returns an empty list for empty queries or when the index contains no documents.

### REQ-storage.faiss.clear: FAISS clear all documents
FAISS backend `clear()` resets the FAISS index and clears all stored documents.

### REQ-storage.faiss.metadata: FAISS metadata handling
FAISS backend stores `None` metadata as an empty dict.

### REQ-storage.bm25: BM25 term-frequency
In-memory BM25 scoring via Rust backend.

### REQ-storage.colbert: ColBERT late interaction
In-memory token-level embeddings with MaxSim scoring. Default checkpoint: `colbert-ir/colbertv2.0`.

### REQ-storage.colbert.import: ColBERT import error handling
ColBERT backend raises a helpful `ImportError` when PyTorch is not installed.

### REQ-storage.colbert.delete: ColBERT document deletion
ColBERT backend `delete(doc_id)` removes a stored document and returns True, or False if the document does not exist.

### REQ-storage.colbert.clear: ColBERT clear all documents
ColBERT backend `clear()` removes all stored documents and resets internal state.

### REQ-storage.colbert.retrieve: ColBERT retrieval edge cases
ColBERT backend `retrieve()` returns an empty list for empty queries or when no documents are stored.

### REQ-storage.hybrid: Hybrid RRF fusion
Combines sparse + dense retrievers with Reciprocal Rank Fusion (k=60, equal weighting, 3x over-fetch).

### REQ-storage.knowledge-graph: Knowledge graph
SQLite-backed entity-relation store with `add_entity()`, `add_relation()`, `neighbors()`, `query_pattern()`.

## Chunking

### REQ-storage.chunking: Document chunking
`chunk_text(text, *, source, config) -> List[Chunk]` splits by paragraph boundaries with configurable `chunk_size` (512 tokens), `chunk_overlap` (64 tokens), `min_chunk_size` (50 tokens).

## Embeddings

### REQ-storage.embeddings: Embedding abstraction
`Embedder` ABC with `embed(texts) -> ndarray` and `dim() -> int`. Default: `SentenceTransformerEmbedder`.

### REQ-storage.embeddings.abc: Embedder ABC enforcement
`Embedder` cannot be instantiated directly; concrete subclasses must implement `embed()` and `dim()`.

### REQ-storage.embeddings.sentence-transformer: SentenceTransformerEmbedder
`SentenceTransformerEmbedder` loads a sentence-transformers model, returns embeddings of correct shape, and raises a helpful `ImportError` when the library is missing.

## Context Injection

### REQ-storage.context: RAG context injection
`inject_context(query, messages, backend, *, config) -> List[Message]` retrieves relevant context, filters by min_score, truncates to max_context_tokens, prepends as system message.

## Document Ingestion

### REQ-storage.ingest: File/directory ingestion
`ingest_path(path, *, config) -> List[Chunk]` reads files (text, markdown, PDF), skips binary/sensitive/hidden files, recursively walks directories.

## Tests

- `tests/memory/test_*.py` - 11 memory backend test files
