# RAPTOR Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RAPTOR retriever (recursive summarization tree with collapsed retrieval) as a new retrieval strategy alongside the existing CKDRetriever and TreeRetriever.

**Architecture:** Build a tree of summaries bottom-up by clustering leaf chunks with UMAP+GMM, summarizing each cluster via LLM, then recursing. At query time, flatten all layers into one ChromaDB collection and do standard top-k similarity search (collapsed retrieval). The builder runs once at index time; the retriever is zero-cost at query time.

**Tech Stack:** umap-learn, scikit-learn (GaussianMixture), ChromaDB, LangChain BaseRetriever, numpy

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `simple_rag/raptor_builder.py` | Create | Index-time: UMAP+GMM clustering, LLM summarization, recursive tree building |
| `simple_rag/raptor_retriever.py` | Create | Query-time: collapsed retrieval over RAPTOR ChromaDB collection |
| `tests/test_raptor_builder.py` | Create | Unit tests for clustering, summarization, tree construction |
| `tests/test_raptor_retriever.py` | Create | Unit tests for retrieval |
| `scripts/build_raptor_index.py` | Create | CLI to build RAPTOR index from existing chunks |
| `config.py` | Modify (line 64) | Add RAPTOR config constants |
| `simple_rag/retriever.py` | Modify (lines 228-264) | Add `use_raptor` flag to `create_retriever()` |
| `simple_rag/__init__.py` | Modify (lines 1-23) | Export `RaptorRetriever` |

---

### Task 1: Install dependency and add config

**Files:**
- Modify: `config.py:60-65`

- [ ] **Step 1: Install umap-learn**

```bash
uv pip install umap-learn
```

Expected: Successfully installed umap-learn and dependencies (numba, llvmlite, pynndescent).

- [ ] **Step 2: Add RAPTOR config constants to `config.py`**

Add after the `CHUNKS_PER_SECTION = 3` line (line 64) in `config.py`:

```python
# =============================================================================
# RAPTOR Configuration
# =============================================================================
RAPTOR_COLLECTION_NAME = "ckd_raptor"
RAPTOR_MAX_DEPTH = 3
RAPTOR_CLUSTER_DIM = 10          # UMAP target dimensions
RAPTOR_MIN_CLUSTER_PROB = 0.1    # GMM soft assignment threshold
```

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "import umap; from sklearn.mixture import GaussianMixture; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "feat: add RAPTOR config constants and umap-learn dependency"
```

---

### Task 2: RAPTOR builder — data structures and clustering

**Files:**
- Create: `simple_rag/raptor_builder.py`
- Create: `tests/test_raptor_builder.py`

- [ ] **Step 1: Write failing tests for data structures and clustering**

Create `tests/test_raptor_builder.py`:

```python
"""Tests for RAPTOR tree builder — data structures and clustering."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestRaptorNode:
    """Test RaptorNode dataclass."""

    def test_node_creation(self):
        from simple_rag.raptor_builder import RaptorNode

        node = RaptorNode(
            node_id="leaf_0",
            text="CKD stage 3 dietary guidance.",
            embedding=[0.1] * 10,
            layer=0,
            children=[],
            metadata={"source": "nice.pdf"},
        )
        assert node.node_id == "leaf_0"
        assert node.layer == 0
        assert len(node.embedding) == 10
        assert node.children == []

    def test_node_is_leaf(self):
        from simple_rag.raptor_builder import RaptorNode

        leaf = RaptorNode("l0", "text", [0.1], 0, [], {})
        summary = RaptorNode("s0", "text", [0.1], 1, ["l0"], {})
        assert leaf.layer == 0
        assert summary.layer == 1


class TestRaptorTree:
    """Test RaptorTree dataclass."""

    def test_tree_creation(self):
        from simple_rag.raptor_builder import RaptorNode, RaptorTree

        nodes = {
            "l0": RaptorNode("l0", "chunk 1", [0.1], 0, [], {}),
            "l1": RaptorNode("l1", "chunk 2", [0.2], 0, [], {}),
            "s0": RaptorNode("s0", "summary", [0.15], 1, ["l0", "l1"], {}),
        }
        tree = RaptorTree(nodes=nodes, depth=1)
        assert tree.depth == 1
        assert len(tree.nodes) == 3

    def test_all_documents(self):
        from simple_rag.raptor_builder import RaptorNode, RaptorTree

        nodes = {
            "l0": RaptorNode("l0", "chunk text", [0.1], 0, [], {"source": "a.pdf"}),
            "s0": RaptorNode("s0", "summary", [0.2], 1, ["l0"], {"layer": 1}),
        }
        tree = RaptorTree(nodes=nodes, depth=1)
        docs = tree.all_documents()
        assert len(docs) == 2
        assert docs[0].page_content == "chunk text"
        assert docs[0].metadata["raptor_layer"] == 0
        assert docs[1].metadata["raptor_layer"] == 1


class TestClustering:
    """Test UMAP + GMM clustering logic."""

    def test_cluster_embeddings_small_input(self):
        """With fewer than 3 embeddings, should return a single cluster."""
        from simple_rag.raptor_builder import cluster_embeddings

        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        labels = cluster_embeddings(embeddings, dim=2)
        # With only 2 items, everything goes in one cluster
        assert len(labels) == 2
        assert all(isinstance(lbl, list) for lbl in labels)

    def test_cluster_embeddings_soft_assignment(self):
        """Each item should get a list of cluster IDs (soft assignment)."""
        from simple_rag.raptor_builder import cluster_embeddings

        # 20 points in two clear clusters
        rng = np.random.RandomState(42)
        cluster_a = rng.randn(10, 50) + 5
        cluster_b = rng.randn(10, 50) - 5
        embeddings = np.vstack([cluster_a, cluster_b])

        labels = cluster_embeddings(embeddings, dim=5, min_prob=0.1)
        assert len(labels) == 20
        # Each label is a list of ints
        for lbl in labels:
            assert isinstance(lbl, list)
            assert len(lbl) >= 1

    def test_cluster_embeddings_returns_at_least_one_cluster(self):
        """Every item must belong to at least one cluster."""
        from simple_rag.raptor_builder import cluster_embeddings

        rng = np.random.RandomState(0)
        embeddings = rng.randn(15, 30)
        labels = cluster_embeddings(embeddings, dim=5, min_prob=0.01)
        for lbl in labels:
            assert len(lbl) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_raptor_builder.py -v --tb=short 2>&1 | tail -15
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` — `raptor_builder` doesn't exist yet.

- [ ] **Step 3: Implement data structures and clustering**

Create `simple_rag/raptor_builder.py`:

```python
"""
RAPTOR tree builder for the CKD RAG System.

Implements Recursive Abstractive Processing for Tree-Organized Retrieval
(Sarthi et al., ICLR 2024). Builds a tree of summaries bottom-up by:
1. Clustering leaf chunks with UMAP + GMM (soft assignments)
2. Summarizing each cluster with an LLM
3. Recursing until max depth or single cluster

The resulting tree is indexed into a separate ChromaDB collection
for collapsed retrieval (flat search over all layers).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAPTOR_COLLECTION_NAME,
    RAPTOR_MAX_DEPTH,
    RAPTOR_CLUSTER_DIM,
    RAPTOR_MIN_CLUSTER_PROB,
)

logger = logging.getLogger(__name__)


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree — either a leaf chunk or a summary."""

    node_id: str
    text: str
    embedding: list[float]
    layer: int  # 0 = leaf, 1+ = summary
    children: list[str]  # node IDs of children
    metadata: dict = field(default_factory=dict)


@dataclass
class RaptorTree:
    """Complete RAPTOR tree with all nodes across all layers."""

    nodes: dict[str, RaptorNode]  # node_id -> node
    depth: int

    def all_documents(self) -> list[Document]:
        """Convert all nodes to LangChain Documents for indexing."""
        docs = []
        for node in self.nodes.values():
            meta = {**node.metadata}
            meta["raptor_layer"] = node.layer
            meta["raptor_node_id"] = node.node_id
            meta["raptor_children"] = ",".join(node.children)
            docs.append(Document(page_content=node.text, metadata=meta))
        return docs


def cluster_embeddings(
    embeddings: np.ndarray,
    dim: int = RAPTOR_CLUSTER_DIM,
    min_prob: float = RAPTOR_MIN_CLUSTER_PROB,
) -> list[list[int]]:
    """Cluster embeddings using UMAP dimensionality reduction + GMM soft assignment.

    Args:
        embeddings: Array of shape (n_samples, n_features).
        dim: Target dimensionality for UMAP reduction.
        min_prob: Minimum probability threshold for soft cluster assignment.

    Returns:
        List of lists — for each embedding, the cluster IDs it belongs to.
    """
    n_samples = embeddings.shape[0]

    # Too few samples for meaningful clustering
    if n_samples < 3:
        return [[0]] * n_samples

    # UMAP dimensionality reduction
    reduced = _reduce_dimensions(embeddings, dim, n_samples)

    # GMM clustering with BIC-selected cluster count
    max_clusters = max(2, math.ceil(math.sqrt(n_samples)))
    labels = _gmm_soft_cluster(reduced, max_clusters, min_prob)

    return labels


def _reduce_dimensions(
    embeddings: np.ndarray, dim: int, n_samples: int
) -> np.ndarray:
    """Reduce embedding dimensions with UMAP."""
    import umap

    # Clamp dim and n_neighbors to sample count
    effective_dim = min(dim, embeddings.shape[1], n_samples - 1)
    n_neighbors = min(10, n_samples - 1)

    reducer = umap.UMAP(
        n_components=effective_dim,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _gmm_soft_cluster(
    data: np.ndarray, max_clusters: int, min_prob: float
) -> list[list[int]]:
    """Fit GMM, select best cluster count by BIC, return soft assignments."""
    from sklearn.mixture import GaussianMixture

    max_clusters = min(max_clusters, data.shape[0])

    # Test cluster counts, pick lowest BIC
    best_bic = float("inf")
    best_n = 1
    for n in range(1, max_clusters + 1):
        try:
            gmm = GaussianMixture(
                n_components=n, covariance_type="full", random_state=42
            )
            gmm.fit(data)
            bic = gmm.bic(data)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception:
            continue

    # Fit final model with best cluster count
    gmm = GaussianMixture(
        n_components=best_n, covariance_type="full", random_state=42
    )
    gmm.fit(data)
    probs = gmm.predict_proba(data)  # shape: (n_samples, best_n)

    # Soft assignment: each sample belongs to all clusters above threshold
    labels = []
    for row in probs:
        assigned = [i for i, p in enumerate(row) if p >= min_prob]
        if not assigned:
            assigned = [int(np.argmax(row))]
        labels.append(assigned)

    logger.info(f"GMM clustering: {best_n} clusters (BIC={best_bic:.1f})")
    return labels
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_raptor_builder.py -v --tb=short 2>&1 | tail -15
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_rag/raptor_builder.py tests/test_raptor_builder.py
git commit -m "feat: add RAPTOR data structures and UMAP+GMM clustering"
```

---

### Task 3: RAPTOR builder — summarization and tree construction

**Files:**
- Modify: `simple_rag/raptor_builder.py`
- Modify: `tests/test_raptor_builder.py`

- [ ] **Step 1: Write failing tests for summarization and tree building**

Append to `tests/test_raptor_builder.py`:

```python
class TestBuildLayer:
    """Test building one layer of summaries from nodes."""

    def test_build_layer_creates_summary_nodes(self):
        from simple_rag.raptor_builder import RaptorNode, RaptorBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Summary of cluster.")

        mock_embed = MagicMock()
        mock_embed.embed_documents = MagicMock(return_value=[[0.5] * 10])

        builder = RaptorBuilder(
            embedding_function=mock_embed,
            llm=mock_llm,
        )

        leaves = {
            "l0": RaptorNode("l0", "Potassium limits for CKD.", [0.1] * 10, 0, [], {}),
            "l1": RaptorNode("l1", "Sodium limits for CKD.", [0.2] * 10, 0, [], {}),
            "l2": RaptorNode("l2", "Protein intake guidance.", [0.3] * 10, 0, [], {}),
        }

        # Mock clustering to return one cluster with all 3 nodes
        with patch(
            "simple_rag.raptor_builder.cluster_embeddings",
            return_value=[[0], [0], [0]],
        ):
            new_nodes = builder._build_layer(leaves, layer=1)

        assert len(new_nodes) == 1
        node = list(new_nodes.values())[0]
        assert node.layer == 1
        assert len(node.children) == 3
        assert node.text == "Summary of cluster."
        mock_llm.generate.assert_called_once()

    def test_build_layer_soft_assignment_creates_overlapping_clusters(self):
        from simple_rag.raptor_builder import RaptorNode, RaptorBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Summary.")

        mock_embed = MagicMock()
        mock_embed.embed_documents = MagicMock(return_value=[[0.5] * 10, [0.6] * 10])

        builder = RaptorBuilder(embedding_function=mock_embed, llm=mock_llm)

        leaves = {
            "l0": RaptorNode("l0", "text A", [0.1] * 10, 0, [], {}),
            "l1": RaptorNode("l1", "text B", [0.2] * 10, 0, [], {}),
        }

        # l0 belongs to clusters 0 and 1, l1 belongs to cluster 0 only
        with patch(
            "simple_rag.raptor_builder.cluster_embeddings",
            return_value=[[0, 1], [0]],
        ):
            new_nodes = builder._build_layer(leaves, layer=1)

        # Should have 2 summary nodes (cluster 0 and cluster 1)
        assert len(new_nodes) == 2


class TestBuildTree:
    """Test full tree construction."""

    def test_build_returns_raptor_tree(self):
        from simple_rag.raptor_builder import RaptorBuilder, RaptorTree

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Summary text.")

        mock_embed = MagicMock()
        mock_embed.embed_documents = MagicMock(return_value=[[0.5] * 10])
        mock_embed.embed_query = MagicMock(return_value=[0.5] * 10)

        builder = RaptorBuilder(
            embedding_function=mock_embed, llm=mock_llm, max_depth=1
        )

        chunks = [
            Document(page_content=f"Chunk {i}", metadata={"source": "test.pdf", "chunk_id": i})
            for i in range(5)
        ]

        with patch(
            "simple_rag.raptor_builder.cluster_embeddings",
            return_value=[[0]] * 5,
        ):
            tree = builder.build(chunks)

        assert isinstance(tree, RaptorTree)
        assert tree.depth == 1
        # 5 leaves + 1 summary
        assert len(tree.nodes) == 6

    def test_build_stops_at_max_depth(self):
        from simple_rag.raptor_builder import RaptorBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Summary.")

        mock_embed = MagicMock()
        mock_embed.embed_documents = MagicMock(return_value=[[0.5] * 10])

        builder = RaptorBuilder(
            embedding_function=mock_embed, llm=mock_llm, max_depth=2
        )

        chunks = [
            Document(page_content=f"Chunk {i}", metadata={"chunk_id": i})
            for i in range(10)
        ]

        with patch(
            "simple_rag.raptor_builder.cluster_embeddings",
            return_value=[[0], [0], [1], [1], [0], [1], [0], [1], [0], [1]],
        ):
            tree = builder.build(chunks)

        assert tree.depth <= 2
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
uv run pytest tests/test_raptor_builder.py::TestBuildLayer -v --tb=short 2>&1 | tail -10
```

Expected: FAIL — `RaptorBuilder` class not found or `_build_layer` not defined.

- [ ] **Step 3: Implement RaptorBuilder class**

Add to the end of `simple_rag/raptor_builder.py`:

```python
SUMMARIZE_PROMPT = """You are summarizing sections of clinical guidelines about Chronic Kidney Disease.
Write a concise summary of the following text passages, preserving:
- Specific recommendations and their evidence grades
- Numerical thresholds (eGFR values, dosages, lab ranges)
- Which CKD stages the guidance applies to
- Source document names

Text:
{text}

Summary:"""


class RaptorBuilder:
    """Builds a RAPTOR tree from document chunks.

    Index-time pipeline:
    1. Create leaf nodes from chunks
    2. Cluster nodes with UMAP + GMM
    3. Summarize each cluster with LLM
    4. Embed summaries → new layer of nodes
    5. Recurse until max_depth or single cluster
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        llm: Any,
        max_depth: int = RAPTOR_MAX_DEPTH,
        cluster_dim: int = RAPTOR_CLUSTER_DIM,
        min_cluster_probability: float = RAPTOR_MIN_CLUSTER_PROB,
        collection_name: str = RAPTOR_COLLECTION_NAME,
    ):
        self.embedding_function = embedding_function
        self.llm = llm
        self.max_depth = max_depth
        self.cluster_dim = cluster_dim
        self.min_cluster_prob = min_cluster_probability
        self.collection_name = collection_name

    def build(self, chunks: list[Document]) -> RaptorTree:
        """Build the full RAPTOR tree from leaf chunks.

        Args:
            chunks: List of LangChain Documents (the original corpus chunks).

        Returns:
            RaptorTree containing all leaf and summary nodes.
        """
        # Create leaf nodes
        all_nodes: dict[str, RaptorNode] = {}
        embeddings = self.embedding_function.embed_documents(
            [c.page_content for c in chunks]
        )

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            node_id = f"leaf_{i}"
            all_nodes[node_id] = RaptorNode(
                node_id=node_id,
                text=chunk.page_content,
                embedding=emb,
                layer=0,
                children=[],
                metadata=dict(chunk.metadata),
            )

        # Build layers recursively
        current_layer_nodes = dict(all_nodes)
        depth = 0

        for d in range(1, self.max_depth + 1):
            if len(current_layer_nodes) <= 1:
                break

            logger.info(
                f"Building RAPTOR layer {d} from {len(current_layer_nodes)} nodes"
            )
            new_nodes = self._build_layer(current_layer_nodes, layer=d)

            if not new_nodes:
                break

            all_nodes.update(new_nodes)
            current_layer_nodes = new_nodes
            depth = d

        logger.info(
            f"RAPTOR tree complete: {len(all_nodes)} nodes, depth={depth}"
        )
        return RaptorTree(nodes=all_nodes, depth=depth)

    def _build_layer(
        self, nodes: dict[str, RaptorNode], layer: int
    ) -> dict[str, RaptorNode]:
        """Build one layer of summary nodes from the previous layer.

        Args:
            nodes: Nodes from the previous layer.
            layer: The layer number being built (1, 2, 3...).

        Returns:
            New summary nodes for this layer.
        """
        node_list = list(nodes.values())
        node_ids = list(nodes.keys())

        # Get embeddings as numpy array
        emb_matrix = np.array([n.embedding for n in node_list])

        # Cluster
        labels = cluster_embeddings(
            emb_matrix, dim=self.cluster_dim, min_prob=self.min_cluster_prob
        )

        # Group nodes by cluster
        clusters: dict[int, list[str]] = {}
        for i, cluster_ids in enumerate(labels):
            for cid in cluster_ids:
                clusters.setdefault(cid, []).append(node_ids[i])

        # Summarize each cluster
        new_nodes: dict[str, RaptorNode] = {}
        texts_to_embed = []
        node_ids_pending = []

        for cid, member_ids in sorted(clusters.items()):
            # Concatenate member texts
            member_texts = [nodes[mid].text for mid in member_ids]
            combined = "\n\n---\n\n".join(member_texts)

            # Summarize
            prompt = SUMMARIZE_PROMPT.format(text=combined)
            summary = self.llm.generate(prompt)

            node_id = f"summary_L{layer}_C{cid}"
            texts_to_embed.append(summary)
            node_ids_pending.append((node_id, summary, member_ids))

        # Batch embed all summaries
        if texts_to_embed:
            summary_embeddings = self.embedding_function.embed_documents(
                texts_to_embed
            )
        else:
            summary_embeddings = []

        for (node_id, summary, member_ids), emb in zip(
            node_ids_pending, summary_embeddings
        ):
            new_nodes[node_id] = RaptorNode(
                node_id=node_id,
                text=summary,
                embedding=emb,
                layer=layer,
                children=member_ids,
                metadata={"layer": layer, "cluster_size": len(member_ids)},
            )

        logger.info(f"Layer {layer}: {len(new_nodes)} summary nodes")
        return new_nodes
```

- [ ] **Step 4: Run all builder tests**

```bash
uv run pytest tests/test_raptor_builder.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_rag/raptor_builder.py tests/test_raptor_builder.py
git commit -m "feat: add RAPTOR tree builder with LLM summarization"
```

---

### Task 4: RAPTOR retriever

**Files:**
- Create: `simple_rag/raptor_retriever.py`
- Create: `tests/test_raptor_retriever.py`

- [ ] **Step 1: Write failing tests for RAPTOR retriever**

Create `tests/test_raptor_retriever.py`:

```python
"""Tests for RAPTOR collapsed retriever."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestRaptorRetriever:
    """Test collapsed retrieval over RAPTOR collection."""

    def _make_retriever(self, search_results):
        """Helper to create a RaptorRetriever with mocked vectorstore."""
        from simple_rag.raptor_retriever import RaptorRetriever

        mock_vs = MagicMock()
        mock_vs.search_with_scores = MagicMock(return_value=search_results)

        mock_embed = MagicMock()

        return RaptorRetriever(
            vectorstore=mock_vs,
            embedding_function=mock_embed,
            k=3,
            score_threshold=0.3,
        )

    def test_returns_documents_above_threshold(self):
        results = [
            (Document(page_content="doc1", metadata={"raptor_layer": 0}), 0.9),
            (Document(page_content="doc2", metadata={"raptor_layer": 1}), 0.5),
            (Document(page_content="doc3", metadata={"raptor_layer": 0}), 0.1),
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test query")
        assert len(docs) == 2  # doc3 below threshold
        assert docs[0].page_content == "doc1"

    def test_respects_k_limit(self):
        results = [
            (Document(page_content=f"doc{i}", metadata={}), 0.9 - i * 0.1)
            for i in range(5)
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test query")
        assert len(docs) <= 3  # k=3

    def test_empty_collection_returns_empty(self):
        retriever = self._make_retriever([])
        docs = retriever.invoke("test query")
        assert docs == []

    def test_metadata_preserved(self):
        results = [
            (Document(
                page_content="summary text",
                metadata={"raptor_layer": 2, "raptor_node_id": "summary_L2_C0"},
            ), 0.8),
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test")
        assert docs[0].metadata["raptor_layer"] == 2


class TestRaptorRetrieverFromCollection:
    """Test factory function that creates retriever from ChromaDB."""

    def test_create_raptor_retriever(self):
        from simple_rag.raptor_retriever import create_raptor_retriever

        mock_embed = MagicMock()
        # Patch Chroma and PersistentClient so no real DB is needed
        with patch("simple_rag.raptor_retriever.chromadb") as mock_chroma, \
             patch("simple_rag.raptor_retriever.Chroma") as mock_chroma_lc:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client

            retriever = create_raptor_retriever(embedding_function=mock_embed)

        assert retriever.k == 5  # default from config
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_raptor_retriever.py -v --tb=short 2>&1 | tail -10
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement RAPTOR retriever**

Create `simple_rag/raptor_retriever.py`:

```python
"""
RAPTOR collapsed retriever for the CKD RAG System.

Query-time component: performs flat top-k similarity search over all
RAPTOR tree layers (leaves + summaries) in a single ChromaDB collection.
"""

import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    RAPTOR_COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class RaptorRetriever(BaseRetriever):
    """Collapsed retrieval over all RAPTOR tree layers.

    Searches leaves and summaries in a single flat index.
    Higher-level summaries match broad thematic queries;
    leaf nodes match specific detail queries.
    """

    vectorstore: Any  # CKDVectorStore or Chroma-backed store
    embedding_function: Any
    k: int = TOP_K_RESULTS
    score_threshold: float = SIMILARITY_THRESHOLD

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Flat top-k similarity search over all RAPTOR layers.

        Args:
            query: User query.
            run_manager: Callback manager.

        Returns:
            Top-k documents above score threshold, from any tree layer.
        """
        results = self.vectorstore.search_with_scores(
            query=query,
            k=self.k,
        )

        docs = [
            doc for doc, score in results
            if score >= self.score_threshold
        ]

        logger.info(
            f"RAPTOR retrieval: {len(docs)}/{len(results)} docs "
            f"(threshold={self.score_threshold})"
        )
        return docs


def create_raptor_retriever(
    embedding_function: Embeddings,
    collection_name: str = RAPTOR_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    k: int = TOP_K_RESULTS,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> RaptorRetriever:
    """Factory function to create a RAPTOR retriever.

    Creates a CKDVectorStore-compatible wrapper around the RAPTOR
    ChromaDB collection.

    Args:
        embedding_function: Embeddings model.
        collection_name: ChromaDB collection name for RAPTOR nodes.
        persist_directory: ChromaDB persistence directory.
        k: Number of results to return.
        score_threshold: Minimum similarity score.

    Returns:
        Configured RaptorRetriever.
    """
    from simple_rag.vectorstore import CKDVectorStore

    store = CKDVectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return RaptorRetriever(
        vectorstore=store,
        embedding_function=embedding_function,
        k=k,
        score_threshold=score_threshold,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_raptor_retriever.py -v --tb=short 2>&1 | tail -15
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_rag/raptor_retriever.py tests/test_raptor_retriever.py
git commit -m "feat: add RAPTOR collapsed retriever"
```

---

### Task 5: Integration — factory, exports, and build script

**Files:**
- Modify: `simple_rag/retriever.py:228-264`
- Modify: `simple_rag/__init__.py`
- Create: `scripts/build_raptor_index.py`

- [ ] **Step 1: Add `use_raptor` to `create_retriever()` factory**

In `simple_rag/retriever.py`, replace the `create_retriever` function (lines 228-264) with:

```python
def create_retriever(
    vectorstore: Any,
    k: int = TOP_K_RESULTS,
    use_hybrid: bool = False,
    use_tree: bool = False,
    use_raptor: bool = False,
    use_contextual: bool = False,
    embedding_function: Any = None,
) -> BaseRetriever:
    """
    Factory function to create a retriever.

    Args:
        vectorstore: Vector store to retrieve from
        k: Number of results to return
        use_hybrid: Whether to use hybrid retrieval
        use_tree: Whether to use tree-based section routing
        use_raptor: Whether to use RAPTOR collapsed retrieval
        use_contextual: Whether to use contextual hybrid retrieval
        embedding_function: Required when use_tree=True or use_raptor=True

    Returns:
        Configured retriever instance
    """
    if use_raptor:
        from .raptor_retriever import create_raptor_retriever
        if embedding_function is None:
            raise ValueError("embedding_function is required for RAPTOR retrieval")
        return create_raptor_retriever(
            embedding_function=embedding_function,
            k=k,
        )

    if use_contextual:
        from .contextual_retriever import create_contextual_retriever
        if embedding_function is None:
            raise ValueError("embedding_function is required for contextual retrieval")
        return create_contextual_retriever(
            embedding_function=embedding_function,
            k=k,
        )

    if use_tree:
        from .tree_retriever import TreeRetriever
        if embedding_function is None:
            raise ValueError("embedding_function is required for tree-based retrieval")
        return TreeRetriever(
            vectorstore=vectorstore,
            embedding_function=embedding_function,
            k=k,
        )

    if use_hybrid:
        return HybridRetriever(vectorstore=vectorstore, k=k)

    return CKDRetriever(
        vectorstore=vectorstore,
        k=k,
    )
```

- [ ] **Step 2: Update `simple_rag/__init__.py` exports**

Replace the full content of `simple_rag/__init__.py` with:

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
from .chain import SimpleRAGChain

__all__ = [
    "EmbeddingGemmaWrapper",
    "CKDVectorStore",
    "CKDRetriever",
    "TreeRetriever",
    "RaptorRetriever",
    "SimpleRAGChain",
]
```

- [ ] **Step 3: Create the build script**

Create `scripts/build_raptor_index.py`:

```python
#!/usr/bin/env python
"""Build the RAPTOR index from existing processed chunks.

Usage:
    uv run python scripts/build_raptor_index.py
    uv run python scripts/build_raptor_index.py --max-depth 2
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Build RAPTOR index")
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # Load chunks
    processed_dir = PROJECT_ROOT / "Data" / "processed"
    logger.info(f"Loading chunks from {processed_dir}...")
    chunks = load_chunks(processed_dir)
    logger.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks found. Run the data pipeline first.")
        sys.exit(1)

    # Initialize components
    from config import get_llm, get_embeddings
    from simple_rag.raptor_builder import RaptorBuilder
    from simple_rag.vectorstore import CKDVectorStore
    from config import RAPTOR_COLLECTION_NAME

    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    logger.info("Loading LLM...")
    llm = get_llm()

    # Build RAPTOR tree
    builder = RaptorBuilder(
        embedding_function=embeddings,
        llm=llm,
        max_depth=args.max_depth,
    )

    logger.info("Building RAPTOR tree...")
    tree = builder.build(chunks)
    logger.info(f"Tree built: {len(tree.nodes)} nodes, depth={tree.depth}")

    # Index into ChromaDB
    logger.info(f"Indexing into ChromaDB collection '{RAPTOR_COLLECTION_NAME}'...")
    store = CKDVectorStore(
        embedding_function=embeddings,
        collection_name=RAPTOR_COLLECTION_NAME,
    )
    store.delete_collection()  # Fresh index
    docs = tree.all_documents()
    store.add_documents(docs)
    logger.info(f"Indexed {len(docs)} nodes. Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run existing tests to verify no regressions**

```bash
uv run pytest tests/test_raptor_builder.py tests/test_raptor_retriever.py -v --tb=short 2>&1 | tail -20
```

Expected: All tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add simple_rag/retriever.py simple_rag/__init__.py scripts/build_raptor_index.py
git commit -m "feat: integrate RAPTOR into retriever factory and add build script"
```

---

### Task 6: Run full RAPTOR pipeline on GPU machine

This task runs on the remote machine with GPU access and real models.

**Files:** None created — this is an execution task.

- [ ] **Step 1: Build the RAPTOR index**

```bash
uv run python scripts/build_raptor_index.py --max-depth 3
```

Expected output (approximate):
```
Loading chunks from Data/processed...
Loaded 923 chunks
Loading embedding model...
Loading LLM...
Building RAPTOR tree...
Building RAPTOR layer 1 from 923 nodes
GMM clustering: ~30 clusters (...)
Layer 1: ~30 summary nodes
Building RAPTOR layer 2 from ~30 nodes
Layer 2: ~5 summary nodes
Building RAPTOR layer 3 from ~5 nodes
Layer 3: 1 summary nodes
Tree built: ~960 nodes, depth=3
Indexing into ChromaDB collection 'ckd_raptor'...
Indexed ~960 nodes. Done.
```

- [ ] **Step 2: Test retrieval manually**

```bash
uv run python -c "
from config import get_embeddings
from simple_rag.raptor_retriever import create_raptor_retriever

embeddings = get_embeddings()
retriever = create_raptor_retriever(embedding_function=embeddings)
docs = retriever.invoke('What do guidelines say about potassium management?')
for d in docs:
    layer = d.metadata.get('raptor_layer', '?')
    print(f'[Layer {layer}] {d.page_content[:120]}...')
"
```

Expected: Mix of leaf chunks (layer 0) and summary nodes (layer 1+) in results.
