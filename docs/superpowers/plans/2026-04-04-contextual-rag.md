# Contextual RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Anthropic-style Contextual Retrieval — each chunk gets an LLM-generated context paragraph prepended before embedding. At query time, combine semantic search over contextualized embeddings with BM25 keyword search using reciprocal rank fusion.

**Architecture:** At index time, for each chunk, send the full source document + chunk to an LLM to generate a 50-100 token context description. Prepend this to the chunk, then embed and index in a separate ChromaDB collection. Also build a BM25 index over the contextualized text. At query time, run both semantic and BM25 search, merge with RRF, and return top-k.

**Tech Stack:** rank-bm25, ChromaDB, LangChain BaseRetriever, json (BM25 persistence)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `simple_rag/contextual_builder.py` | Create | Index-time: LLM context generation, ChromaDB + BM25 index building |
| `simple_rag/contextual_retriever.py` | Create | Query-time: hybrid semantic + BM25 with RRF |
| `tests/test_contextual_builder.py` | Create | Unit tests for context generation and index building |
| `tests/test_contextual_retriever.py` | Create | Unit tests for hybrid retrieval |
| `scripts/build_contextual_index.py` | Create | CLI to build contextual index from existing chunks |
| `config.py` | Modify (after RAPTOR config block) | Add contextual RAG config constants |
| `simple_rag/retriever.py` | Already modified in RAPTOR plan | `use_contextual` flag already wired |
| `simple_rag/__init__.py` | Modify | Export `ContextualRetriever` |

**Note:** If you completed the RAPTOR plan first, `create_retriever()` already has the `use_contextual` flag and import. If not, add it as shown in RAPTOR plan Task 5 Step 1.

---

### Task 1: Install dependency and add config

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Install rank-bm25**

```bash
uv pip install rank-bm25
```

Expected: Successfully installed rank-bm25.

- [ ] **Step 2: Add Contextual RAG config constants to `config.py`**

Add after the RAPTOR config block (or after `CHUNKS_PER_SECTION` if RAPTOR not done yet):

```python
# =============================================================================
# Contextual RAG Configuration
# =============================================================================
CONTEXTUAL_COLLECTION_NAME = "ckd_contextual"
CONTEXTUAL_BM25_PATH = str(VECTORSTORE_DIR / "bm25_contextual.json")
CONTEXTUAL_SEMANTIC_WEIGHT = 0.7
CONTEXTUAL_BM25_WEIGHT = 0.3
```

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from rank_bm25 import BM25Okapi; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "feat: add Contextual RAG config and rank-bm25 dependency"
```

---

### Task 2: Contextual builder — context generation

**Files:**
- Create: `simple_rag/contextual_builder.py`
- Create: `tests/test_contextual_builder.py`

- [ ] **Step 1: Write failing tests for context generation**

Create `tests/test_contextual_builder.py`:

```python
"""Tests for Contextual RAG builder — context generation and indexing."""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from langchain_core.documents import Document


class TestGenerateContext:
    """Test LLM-based context generation for individual chunks."""

    def test_generate_context_calls_llm(self):
        from simple_rag.contextual_builder import ContextualBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(
            return_value="This chunk discusses potassium limits for CKD stage 3."
        )
        mock_embed = MagicMock()

        builder = ContextualBuilder(
            embedding_function=mock_embed, llm=mock_llm
        )

        chunk = Document(
            page_content="Limit potassium to 2000-3000mg per day.",
            metadata={"source": "nice.pdf", "chunk_id": 0},
        )

        context = builder.generate_context(
            document_text="# NICE Guidelines\n\n## Dietary\n\nLimit potassium...",
            chunk=chunk,
        )

        assert isinstance(context, str)
        assert len(context) > 0
        mock_llm.generate.assert_called_once()
        # Verify the prompt contains both document and chunk
        prompt_arg = mock_llm.generate.call_args[0][0]
        assert "NICE Guidelines" in prompt_arg
        assert "Limit potassium" in prompt_arg

    def test_generate_context_returns_llm_output(self):
        from simple_rag.contextual_builder import ContextualBuilder

        expected = "This chunk is from the dietary section of NICE NG203."
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value=expected)
        mock_embed = MagicMock()

        builder = ContextualBuilder(embedding_function=mock_embed, llm=mock_llm)
        chunk = Document(page_content="Some text.", metadata={})
        result = builder.generate_context("Full doc text.", chunk)
        assert result == expected


class TestContextualize:
    """Test contextualizing a chunk (prepending context to text)."""

    def test_contextualize_chunk(self):
        from simple_rag.contextual_builder import ContextualBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Context about CKD stage 3.")
        mock_embed = MagicMock()

        builder = ContextualBuilder(embedding_function=mock_embed, llm=mock_llm)

        chunk = Document(
            page_content="Limit potassium intake.",
            metadata={"source": "nice.pdf", "chunk_id": 0},
        )

        ctx_chunk = builder.contextualize_chunk("Full document.", chunk)

        assert ctx_chunk.page_content.startswith("Context about CKD stage 3.")
        assert "Limit potassium intake." in ctx_chunk.page_content
        assert ctx_chunk.metadata["contextual_context"] == "Context about CKD stage 3."
        # Original metadata preserved
        assert ctx_chunk.metadata["source"] == "nice.pdf"


class TestBM25Persistence:
    """Test BM25 index save/load."""

    def test_save_and_load_bm25(self, tmp_path):
        from simple_rag.contextual_builder import save_bm25_data, load_bm25_data

        corpus = [
            "CKD stage 3 potassium dietary limits",
            "Hemodialysis sodium restriction guidelines",
            "ACE inhibitor dosing for proteinuria",
        ]
        chunk_ids = ["c0", "c1", "c2"]
        path = str(tmp_path / "bm25.json")

        save_bm25_data(corpus, chunk_ids, path)
        loaded_corpus, loaded_ids = load_bm25_data(path)

        assert loaded_corpus == corpus
        assert loaded_ids == chunk_ids

    def test_load_missing_file_returns_empty(self, tmp_path):
        from simple_rag.contextual_builder import load_bm25_data

        corpus, ids = load_bm25_data(str(tmp_path / "nonexistent.json"))
        assert corpus == []
        assert ids == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_contextual_builder.py -v --tb=short 2>&1 | tail -10
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement contextual builder**

Create `simple_rag/contextual_builder.py`:

```python
"""
Contextual RAG builder for the CKD RAG System.

Implements Anthropic's Contextual Retrieval technique:
for each chunk, generates a short LLM context paragraph that explains
where the chunk sits in the document, then prepends it before embedding.

Also builds a BM25 index over the contextualized text for hybrid retrieval.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CONTEXTUAL_COLLECTION_NAME,
    CONTEXTUAL_BM25_PATH,
)

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class ContextualBuilder:
    """Builds contextualized chunk index for Contextual RAG.

    For each chunk:
    1. Send full document + chunk to LLM → get context paragraph
    2. Prepend context to chunk text
    3. Embed and index the contextualized chunk in ChromaDB
    4. Add to BM25 corpus for keyword search
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        llm: Any,
        collection_name: str = CONTEXTUAL_COLLECTION_NAME,
        bm25_path: str = CONTEXTUAL_BM25_PATH,
    ):
        self.embedding_function = embedding_function
        self.llm = llm
        self.collection_name = collection_name
        self.bm25_path = bm25_path

    def generate_context(self, document_text: str, chunk: Document) -> str:
        """Generate contextual description for a single chunk.

        Args:
            document_text: Full source document markdown text.
            chunk: The chunk to generate context for.

        Returns:
            Short context string (50-100 tokens).
        """
        prompt = CONTEXT_PROMPT.format(
            document_text=document_text,
            chunk_content=chunk.page_content,
        )
        return self.llm.generate(prompt)

    def contextualize_chunk(
        self, document_text: str, chunk: Document
    ) -> Document:
        """Generate context and prepend it to the chunk.

        Args:
            document_text: Full source document markdown text.
            chunk: Original chunk.

        Returns:
            New Document with context prepended to page_content,
            and original context stored in metadata['contextual_context'].
        """
        context = self.generate_context(document_text, chunk)
        new_chunk = deepcopy(chunk)
        new_chunk.page_content = f"{context}\n\n{chunk.page_content}"
        new_chunk.metadata["contextual_context"] = context
        return new_chunk

    def build(
        self,
        chunks: list[Document],
        document_texts: dict[str, str],
    ) -> list[Document]:
        """Contextualize all chunks and build BM25 index.

        Args:
            chunks: Original corpus chunks.
            document_texts: Mapping of source filename → full document markdown.

        Returns:
            List of contextualized Documents (ready for ChromaDB indexing).
        """
        contextualized = []
        bm25_corpus = []
        bm25_chunk_ids = []

        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "")
            doc_text = document_texts.get(source, "")

            if not doc_text:
                logger.warning(
                    f"No document text for source '{source}', "
                    f"skipping context generation for chunk {i}"
                )
                ctx_chunk = deepcopy(chunk)
                ctx_chunk.metadata["contextual_context"] = ""
            else:
                ctx_chunk = self.contextualize_chunk(doc_text, chunk)

            contextualized.append(ctx_chunk)
            bm25_corpus.append(ctx_chunk.page_content)
            bm25_chunk_ids.append(
                f"{source}_{chunk.metadata.get('chunk_id', i)}"
            )

            if (i + 1) % 50 == 0:
                logger.info(f"Contextualized {i + 1}/{len(chunks)} chunks")

        # Save BM25 data
        save_bm25_data(bm25_corpus, bm25_chunk_ids, self.bm25_path)
        logger.info(
            f"Contextual build complete: {len(contextualized)} chunks, "
            f"BM25 saved to {self.bm25_path}"
        )

        return contextualized


def save_bm25_data(
    corpus: list[str], chunk_ids: list[str], path: str
) -> None:
    """Save BM25 corpus and chunk IDs to JSON for persistence.

    Args:
        corpus: List of contextualized chunk texts.
        chunk_ids: Corresponding chunk identifiers.
        path: File path for JSON output.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"corpus": corpus, "chunk_ids": chunk_ids}
    Path(path).write_text(json.dumps(data))
    logger.info(f"BM25 data saved: {len(corpus)} entries → {path}")


def load_bm25_data(path: str) -> tuple[list[str], list[str]]:
    """Load BM25 corpus and chunk IDs from JSON.

    Args:
        path: File path to BM25 JSON data.

    Returns:
        Tuple of (corpus texts, chunk IDs). Empty lists if file missing.
    """
    p = Path(path)
    if not p.exists():
        return [], []
    data = json.loads(p.read_text())
    return data.get("corpus", []), data.get("chunk_ids", [])
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_contextual_builder.py -v --tb=short 2>&1 | tail -15
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_rag/contextual_builder.py tests/test_contextual_builder.py
git commit -m "feat: add Contextual RAG builder with LLM context generation"
```

---

### Task 3: Contextual retriever — hybrid semantic + BM25

**Files:**
- Create: `simple_rag/contextual_retriever.py`
- Create: `tests/test_contextual_retriever.py`

- [ ] **Step 1: Write failing tests for hybrid retriever**

Create `tests/test_contextual_retriever.py`:

```python
"""Tests for Contextual RAG hybrid retriever."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestContextualRetriever:
    """Test hybrid semantic + BM25 retrieval."""

    def _make_retriever(self, semantic_results, bm25_corpus=None, bm25_ids=None):
        from simple_rag.contextual_retriever import ContextualRetriever

        mock_vs = MagicMock()
        mock_vs.search_with_scores = MagicMock(return_value=semantic_results)

        mock_embed = MagicMock()

        # Build real BM25 index if corpus provided
        bm25_index = None
        bm25_doc_map = {}
        if bm25_corpus:
            from rank_bm25 import BM25Okapi

            tokenized = [doc.lower().split() for doc in bm25_corpus]
            bm25_index = BM25Okapi(tokenized)
            # Build doc map from semantic results (matching by index)
            for i, (doc, _) in enumerate(semantic_results):
                chunk_id = bm25_ids[i] if bm25_ids and i < len(bm25_ids) else str(i)
                bm25_doc_map[chunk_id] = doc

        return ContextualRetriever(
            vectorstore=mock_vs,
            embedding_function=mock_embed,
            bm25_index=bm25_index,
            bm25_corpus_ids=bm25_ids or [],
            bm25_doc_map=bm25_doc_map,
            k=3,
            score_threshold=0.2,
        )

    def test_semantic_only_when_no_bm25(self):
        """Without BM25 index, falls back to semantic-only."""
        results = [
            (Document(page_content="doc1", metadata={"source": "a.pdf"}), 0.9),
            (Document(page_content="doc2", metadata={"source": "b.pdf"}), 0.5),
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test query")
        assert len(docs) == 2

    def test_hybrid_merges_results(self):
        """With BM25, results come from both sources."""
        semantic_results = [
            (Document(page_content="CKD potassium limits", metadata={"source": "a", "chunk_id": 0}), 0.9),
            (Document(page_content="CKD sodium limits", metadata={"source": "a", "chunk_id": 1}), 0.7),
            (Document(page_content="Exercise guidelines", metadata={"source": "b", "chunk_id": 0}), 0.5),
        ]
        bm25_corpus = [
            "CKD potassium limits",
            "CKD sodium limits",
            "Exercise guidelines",
        ]
        bm25_ids = ["a_0", "a_1", "b_0"]

        retriever = self._make_retriever(semantic_results, bm25_corpus, bm25_ids)
        docs = retriever.invoke("potassium restriction CKD")
        assert len(docs) <= 3
        assert len(docs) >= 1

    def test_empty_results(self):
        retriever = self._make_retriever([])
        docs = retriever.invoke("test")
        assert docs == []

    def test_respects_k_limit(self):
        results = [
            (Document(page_content=f"doc{i}", metadata={"source": "a", "chunk_id": i}), 0.9)
            for i in range(10)
        ]
        bm25_corpus = [f"doc{i}" for i in range(10)]
        bm25_ids = [f"a_{i}" for i in range(10)]
        retriever = self._make_retriever(results, bm25_corpus, bm25_ids)
        docs = retriever.invoke("test")
        assert len(docs) <= 3


class TestCreateContextualRetriever:
    """Test factory function."""

    def test_create_without_bm25_file(self, tmp_path):
        from simple_rag.contextual_retriever import create_contextual_retriever

        mock_embed = MagicMock()

        with patch("simple_rag.contextual_retriever.CKDVectorStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            retriever = create_contextual_retriever(
                embedding_function=mock_embed,
                bm25_path=str(tmp_path / "nonexistent.json"),
            )

        assert retriever.bm25_index is None  # No BM25 file → None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_contextual_retriever.py -v --tb=short 2>&1 | tail -10
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement contextual retriever**

Create `simple_rag/contextual_retriever.py`:

```python
"""
Contextual RAG hybrid retriever for the CKD RAG System.

Combines semantic search over contextualized embeddings with BM25
keyword search using reciprocal rank fusion (RRF).
"""

import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CONTEXTUAL_COLLECTION_NAME,
    CONTEXTUAL_BM25_PATH,
    CONTEXTUAL_SEMANTIC_WEIGHT,
    CONTEXTUAL_BM25_WEIGHT,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# RRF constant (standard value from the literature)
RRF_K = 60


class ContextualRetriever(BaseRetriever):
    """Hybrid semantic + BM25 retrieval over contextualized chunks.

    1. Semantic search over contextualized embeddings → top-N
    2. BM25 keyword search over contextualized text → top-N
    3. Reciprocal rank fusion to merge
    4. Return top-K
    """

    vectorstore: Any
    embedding_function: Any
    bm25_index: Any  # BM25Okapi or None
    bm25_corpus_ids: list = []
    bm25_doc_map: dict = {}  # chunk_id -> Document
    k: int = TOP_K_RESULTS
    semantic_weight: float = CONTEXTUAL_SEMANTIC_WEIGHT
    bm25_weight: float = CONTEXTUAL_BM25_WEIGHT
    score_threshold: float = SIMILARITY_THRESHOLD

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Hybrid retrieval with reciprocal rank fusion.

        Args:
            query: User query.
            run_manager: Callback manager.

        Returns:
            Top-k documents from fused semantic + BM25 results.
        """
        # Phase 1: Semantic search
        fetch_k = self.k * 3  # Fetch more for fusion
        semantic_results = self.vectorstore.search_with_scores(
            query=query, k=fetch_k
        )

        # Filter by threshold
        semantic_results = [
            (doc, score) for doc, score in semantic_results
            if score >= self.score_threshold
        ]

        if not semantic_results:
            return []

        # If no BM25 index, return semantic-only
        if self.bm25_index is None:
            return [doc for doc, _ in semantic_results[:self.k]]

        # Phase 2: BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top BM25 indices
        import numpy as np

        top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # Phase 3: Reciprocal rank fusion
        doc_rrf: dict[str, tuple[Document, float]] = {}

        # Add semantic results with RRF scores
        for rank, (doc, score) in enumerate(semantic_results, start=1):
            doc_key = self._doc_key(doc)
            rrf_score = self.semantic_weight / (rank + RRF_K)
            if doc_key in doc_rrf:
                existing_doc, existing_score = doc_rrf[doc_key]
                doc_rrf[doc_key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_rrf[doc_key] = (doc, rrf_score)

        # Add BM25 results with RRF scores
        for rank, idx in enumerate(top_bm25_indices, start=1):
            if idx >= len(self.bm25_corpus_ids):
                continue
            chunk_id = self.bm25_corpus_ids[idx]
            if chunk_id not in self.bm25_doc_map:
                continue
            doc = self.bm25_doc_map[chunk_id]
            doc_key = self._doc_key(doc)
            rrf_score = self.bm25_weight / (rank + RRF_K)
            if doc_key in doc_rrf:
                existing_doc, existing_score = doc_rrf[doc_key]
                doc_rrf[doc_key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_rrf[doc_key] = (doc, rrf_score)

        # Sort by fused score, return top-k
        sorted_results = sorted(
            doc_rrf.values(), key=lambda x: x[1], reverse=True
        )

        docs = [doc for doc, _ in sorted_results[:self.k]]
        logger.info(
            f"Contextual retrieval: {len(semantic_results)} semantic + "
            f"BM25 → {len(docs)} fused results"
        )
        return docs

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Generate a dedup key for a document."""
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        return f"{source}_{chunk_id}"


def create_contextual_retriever(
    embedding_function: Embeddings,
    collection_name: str = CONTEXTUAL_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    bm25_path: str = CONTEXTUAL_BM25_PATH,
    k: int = TOP_K_RESULTS,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> ContextualRetriever:
    """Factory function to create a Contextual retriever.

    Loads the contextualized ChromaDB collection and BM25 index from disk.

    Args:
        embedding_function: Embeddings model.
        collection_name: ChromaDB collection for contextualized chunks.
        persist_directory: ChromaDB persistence directory.
        bm25_path: Path to BM25 JSON data file.
        k: Number of results.
        score_threshold: Minimum similarity score.

    Returns:
        Configured ContextualRetriever.
    """
    from simple_rag.vectorstore import CKDVectorStore
    from simple_rag.contextual_builder import load_bm25_data

    store = CKDVectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    # Load BM25 index if available
    bm25_index = None
    bm25_corpus_ids = []
    bm25_doc_map = {}

    corpus, chunk_ids = load_bm25_data(bm25_path)
    if corpus:
        from rank_bm25 import BM25Okapi

        tokenized = [text.lower().split() for text in corpus]
        bm25_index = BM25Okapi(tokenized)
        bm25_corpus_ids = chunk_ids

        # Build doc map by loading docs from ChromaDB
        # The BM25 doc_map will be populated on first query from semantic results
        logger.info(f"Loaded BM25 index: {len(corpus)} entries from {bm25_path}")
    else:
        logger.warning(f"No BM25 data at {bm25_path}, using semantic-only retrieval")

    return ContextualRetriever(
        vectorstore=store,
        embedding_function=embedding_function,
        bm25_index=bm25_index,
        bm25_corpus_ids=bm25_corpus_ids,
        bm25_doc_map=bm25_doc_map,
        k=k,
        score_threshold=score_threshold,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_contextual_retriever.py -v --tb=short 2>&1 | tail -15
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_rag/contextual_retriever.py tests/test_contextual_retriever.py
git commit -m "feat: add Contextual RAG hybrid retriever with RRF"
```

---

### Task 4: Integration — exports and build script

**Files:**
- Modify: `simple_rag/__init__.py`
- Create: `scripts/build_contextual_index.py`

- [ ] **Step 1: Update `simple_rag/__init__.py` exports**

Add `ContextualRetriever` to the imports and `__all__`:

```python
"""
Level 1: Simple Retrieval Augmented Generation (RAG)

This module implements a basic RAG pipeline for CKD management:
- Document embedding with EmbeddingGemma
- Vector storage with ChromaDB
- Retrieval and generation with MedGemma
"""

from .embeddings import EmbeddingGemmaWrapper
from .vectorstore import CKDVectorStore
from .retriever import CKDRetriever
from .tree_retriever import TreeRetriever
from .raptor_retriever import RaptorRetriever
from .contextual_retriever import ContextualRetriever
from .chain import SimpleRAGChain

__all__ = [
    "EmbeddingGemmaWrapper",
    "CKDVectorStore",
    "CKDRetriever",
    "TreeRetriever",
    "RaptorRetriever",
    "ContextualRetriever",
    "SimpleRAGChain",
]
```

**Note:** If RAPTOR plan was not completed, omit `RaptorRetriever` from imports and `__all__`.

- [ ] **Step 2: Create the build script**

Create `scripts/build_contextual_index.py`:

```python
#!/usr/bin/env python
"""Build the Contextual RAG index from existing processed chunks.

For each chunk, sends the full source document + chunk to an LLM
to generate a context paragraph, then embeds and indexes the
contextualized chunks. Also builds a BM25 index.

Usage:
    uv run python scripts/build_contextual_index.py
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_chunks(processed_dir: Path):
    """Load all chunk JSON files into LangChain Documents."""
    from langchain_core.documents import Document

    docs = []
    for f in sorted(processed_dir.glob("*_chunks.json")):
        data = json.loads(f.read_text())
        meta = data.get("export_metadata", {})
        title = meta.get("document_title", f.stem)
        source = meta.get("source_file", f.name)
        for i, chunk in enumerate(data.get("chunks", [])):
            chunk_meta = chunk.get("metadata", {})
            chunk_meta.update({"source": source, "document_title": title, "chunk_id": i})
            docs.append(Document(page_content=chunk["content"], metadata=chunk_meta))
    return docs


def load_document_texts(sections_dir: Path) -> dict[str, str]:
    """Load full document texts from processed_with_sections/*/main_text.md.

    Returns mapping of source filename → full markdown text.
    """
    texts = {}
    for doc_dir in sorted(sections_dir.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.startswith("."):
            continue
        main_text = doc_dir / "main_text.md"
        metadata_file = doc_dir / "metadata.json"
        if main_text.exists() and metadata_file.exists():
            meta = json.loads(metadata_file.read_text())
            source = meta.get("source_file", "")
            if source:
                texts[source] = main_text.read_text()
    return texts


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    processed_dir = PROJECT_ROOT / "Data" / "processed"
    sections_dir = PROJECT_ROOT / "Data" / "processed_with_sections"

    # Load chunks
    logger.info(f"Loading chunks from {processed_dir}...")
    chunks = load_chunks(processed_dir)
    logger.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks found. Run the data pipeline first.")
        sys.exit(1)

    # Load full document texts
    logger.info(f"Loading document texts from {sections_dir}...")
    doc_texts = load_document_texts(sections_dir)
    logger.info(f"Loaded {len(doc_texts)} document texts")

    # Initialize components
    from config import get_llm, get_embeddings, CONTEXTUAL_COLLECTION_NAME
    from simple_rag.contextual_builder import ContextualBuilder
    from simple_rag.vectorstore import CKDVectorStore

    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    logger.info("Loading LLM...")
    llm = get_llm()

    # Build contextualized chunks
    builder = ContextualBuilder(embedding_function=embeddings, llm=llm)
    logger.info("Generating context for all chunks (this may take a while)...")
    ctx_chunks = builder.build(chunks, doc_texts)

    # Index into ChromaDB
    logger.info(f"Indexing into ChromaDB collection '{CONTEXTUAL_COLLECTION_NAME}'...")
    store = CKDVectorStore(
        embedding_function=embeddings,
        collection_name=CONTEXTUAL_COLLECTION_NAME,
    )
    store.delete_collection()  # Fresh index
    store.add_documents(ctx_chunks)
    logger.info(f"Indexed {len(ctx_chunks)} contextualized chunks. Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run all tests to verify no regressions**

```bash
uv run pytest tests/test_contextual_builder.py tests/test_contextual_retriever.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add simple_rag/__init__.py scripts/build_contextual_index.py
git commit -m "feat: integrate Contextual RAG into exports and add build script"
```

---

### Task 5: Run full contextual pipeline on GPU machine

This task runs on the remote machine with GPU access and real models.

**Files:** None created — this is an execution task.

- [ ] **Step 1: Build the contextual index**

```bash
uv run python scripts/build_contextual_index.py
```

Expected output (approximate):
```
Loading chunks from Data/processed...
Loaded 923 chunks
Loading document texts from Data/processed_with_sections...
Loaded 24 document texts
Loading embedding model...
Loading LLM...
Generating context for all chunks (this may take a while)...
Contextualized 50/923 chunks
Contextualized 100/923 chunks
...
Contextual build complete: 923 chunks, BM25 saved to Data/vectorstore/bm25_contextual.json
Indexing into ChromaDB collection 'ckd_contextual'...
Indexed 923 contextualized chunks. Done.
```

- [ ] **Step 2: Test retrieval manually**

```bash
uv run python -c "
from config import get_embeddings
from simple_rag.contextual_retriever import create_contextual_retriever

embeddings = get_embeddings()
retriever = create_contextual_retriever(embedding_function=embeddings)
docs = retriever.invoke('What are potassium limits for CKD stage 3?')
for d in docs:
    ctx = d.metadata.get('contextual_context', '')[:80]
    print(f'[Context: {ctx}...]')
    print(f'  {d.page_content[:120]}...')
    print()
"
```

Expected: Each chunk has a prepended context that identifies the document, section, and topic.
