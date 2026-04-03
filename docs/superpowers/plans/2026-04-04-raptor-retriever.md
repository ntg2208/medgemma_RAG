# RAPTOR Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RAPTOR retriever (recursive summarization tree with collapsed retrieval) as a new retrieval strategy alongside the existing CKDRetriever and TreeRetriever.

**Architecture:** Build a tree of summaries bottom-up by clustering leaf chunks with UMAP+GMM, summarizing each cluster via LLM, then recursing. At query time, flatten all layers into one ChromaDB collection and do standard top-k similarity search (collapsed retrieval). The builder runs once at index time; the retriever is zero-cost at query time.

**Tech Stack:** umap-learn, scikit-learn (GaussianMixture), ChromaDB, LangChain BaseRetriever, numpy, pyvis

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `simple_rag/raptor_builder.py` | Create | Index-time: UMAP+GMM clustering, LLM summarization, recursive tree building |
| `simple_rag/raptor_retriever.py` | Create | Query-time: collapsed retrieval over RAPTOR ChromaDB collection |
| `simple_rag/raptor_viz.py` | Create | Pyvis interactive tree visualization |
| `tests/test_raptor_builder.py` | Create | Unit tests for clustering, summarization, tree construction |
| `tests/test_raptor_retriever.py` | Create | Unit tests for retrieval |
| `tests/test_raptor_viz.py` | Create | Unit tests for visualization |
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

---

### Task 7: Pyvis interactive tree visualization

**Files:**
- Create: `simple_rag/raptor_viz.py`
- Create: `tests/test_raptor_viz.py`
- Create: `scripts/visualize_raptor.py`

- [ ] **Step 1: Install pyvis**

```bash
uv pip install pyvis
```

Expected: Successfully installed pyvis.

- [ ] **Step 2: Write failing tests for visualization**

Create `tests/test_raptor_viz.py`:

```python
"""Tests for RAPTOR tree Pyvis visualization."""

import pytest
from pathlib import Path
from simple_rag.raptor_builder import RaptorNode, RaptorTree


def _make_sample_tree() -> RaptorTree:
    """Build a small tree for testing: 4 leaves, 2 L1 summaries, 1 L2 root."""
    nodes = {
        "leaf_0": RaptorNode(
            "leaf_0", "Potassium limits for CKD stage 3.", [0.1] * 10, 0, [],
            {"source": "nice.pdf", "section": "Dietary"},
        ),
        "leaf_1": RaptorNode(
            "leaf_1", "Sodium should be limited to 2000mg.", [0.2] * 10, 0, [],
            {"source": "nice.pdf", "section": "Dietary"},
        ),
        "leaf_2": RaptorNode(
            "leaf_2", "ACE inhibitors for proteinuria.", [0.3] * 10, 0, [],
            {"source": "kdigo.pdf", "section": "Medication"},
        ),
        "leaf_3": RaptorNode(
            "leaf_3", "eGFR monitoring every 3 months.", [0.4] * 10, 0, [],
            {"source": "kdigo.pdf", "section": "Monitoring"},
        ),
        "summary_L1_C0": RaptorNode(
            "summary_L1_C0", "Summary: dietary restrictions for CKD.",
            [0.15] * 10, 1, ["leaf_0", "leaf_1"],
            {"layer": 1, "cluster_size": 2},
        ),
        "summary_L1_C1": RaptorNode(
            "summary_L1_C1", "Summary: medication and monitoring.",
            [0.35] * 10, 1, ["leaf_2", "leaf_3"],
            {"layer": 1, "cluster_size": 2},
        ),
        "summary_L2_C0": RaptorNode(
            "summary_L2_C0", "Summary: CKD management overview.",
            [0.25] * 10, 2, ["summary_L1_C0", "summary_L1_C1"],
            {"layer": 2, "cluster_size": 2},
        ),
    }
    return RaptorTree(nodes=nodes, depth=2)


class TestVisualize:
    """Test Pyvis visualization output."""

    def test_generates_html_file(self, tmp_path):
        from simple_rag.raptor_viz import visualize_tree

        tree = _make_sample_tree()
        output = tmp_path / "raptor_tree.html"
        visualize_tree(tree, str(output))
        assert output.exists()
        html = output.read_text()
        assert "<html>" in html.lower() or "<!doctype" in html.lower()

    def test_html_contains_all_nodes(self, tmp_path):
        from simple_rag.raptor_viz import visualize_tree

        tree = _make_sample_tree()
        output = tmp_path / "tree.html"
        visualize_tree(tree, str(output))
        html = output.read_text()
        # All 7 node IDs should appear somewhere in the HTML
        for node_id in tree.nodes:
            assert node_id in html

    def test_html_contains_edges(self, tmp_path):
        from simple_rag.raptor_viz import visualize_tree

        tree = _make_sample_tree()
        output = tmp_path / "tree.html"
        visualize_tree(tree, str(output))
        html = output.read_text()
        # Pyvis encodes edges as JSON — check that parent→child links exist
        # summary_L2_C0 → summary_L1_C0 and summary_L1_C0 → leaf_0
        assert "summary_L2_C0" in html
        assert "leaf_0" in html

    def test_default_output_path(self, tmp_path, monkeypatch):
        from simple_rag.raptor_viz import visualize_tree

        monkeypatch.chdir(tmp_path)
        tree = _make_sample_tree()
        result_path = visualize_tree(tree)
        assert Path(result_path).exists()


class TestHighlightRetrieval:
    """Test highlighting retrieved nodes in the visualization."""

    def test_highlight_nodes(self, tmp_path):
        from simple_rag.raptor_viz import visualize_tree

        tree = _make_sample_tree()
        output = tmp_path / "highlighted.html"
        visualize_tree(
            tree,
            str(output),
            highlight_nodes=["leaf_0", "summary_L1_C0"],
        )
        html = output.read_text()
        assert output.exists()
        # Highlighted nodes should have a different color in the HTML
        # (we use red for highlighted, so check for the color code)
        assert "#e74c3c" in html or "red" in html.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_raptor_viz.py -v --tb=short 2>&1 | tail -10
```

Expected: FAIL — `raptor_viz` module not found.

- [ ] **Step 4: Implement Pyvis visualization**

Create `simple_rag/raptor_viz.py`:

```python
"""
Pyvis interactive visualization for RAPTOR trees.

Generates a standalone HTML file with an interactive network graph.
Nodes are colored by layer, sized by text length, and show
text previews on hover. Edges show parent→child relationships.

Usage:
    from simple_rag.raptor_viz import visualize_tree
    visualize_tree(tree, "raptor_tree.html")
    visualize_tree(tree, "debug.html", highlight_nodes=["leaf_0", "leaf_5"])
"""

import logging
from pathlib import Path
from typing import Optional

from pyvis.network import Network

logger = logging.getLogger(__name__)

# Color palette by layer (layer 0 = leaves, higher = summaries)
LAYER_COLORS = [
    "#3498db",  # Layer 0 (leaves): blue
    "#2ecc71",  # Layer 1: green
    "#f39c12",  # Layer 2: orange
    "#9b59b6",  # Layer 3: purple
    "#1abc9c",  # Layer 4+: teal
]
HIGHLIGHT_COLOR = "#e74c3c"  # Red for highlighted/retrieved nodes

# Node size range
MIN_SIZE = 10
MAX_SIZE = 40


def visualize_tree(
    tree,
    output_path: str = "raptor_tree.html",
    highlight_nodes: Optional[list[str]] = None,
    height: str = "900px",
    width: str = "100%",
) -> str:
    """Generate an interactive Pyvis visualization of a RAPTOR tree.

    Args:
        tree: RaptorTree instance with nodes and depth.
        output_path: Path for the output HTML file.
        highlight_nodes: Optional list of node IDs to highlight (e.g., retrieved nodes).
        height: Height of the visualization.
        width: Width of the visualization.

    Returns:
        Path to the generated HTML file.
    """
    highlight_set = set(highlight_nodes or [])

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
    )

    # Physics settings for hierarchical layout
    net.set_options("""
    {
        "layout": {
            "hierarchical": {
                "enabled": true,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 150,
                "nodeSpacing": 100
            }
        },
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 150
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    # Add nodes
    for node_id, node in tree.nodes.items():
        layer = node.layer
        color = HIGHLIGHT_COLOR if node_id in highlight_set else _layer_color(layer)

        # Size based on number of children (summaries bigger) + text length
        size = MIN_SIZE + min(len(node.text) // 50, MAX_SIZE - MIN_SIZE)
        if node.children:
            size = max(size, MIN_SIZE + len(node.children) * 5)

        # Label: short ID
        label = node_id

        # Title (hover tooltip): text preview + metadata
        text_preview = node.text[:200].replace("\n", " ")
        if len(node.text) > 200:
            text_preview += "..."

        source = node.metadata.get("source", "")
        section = node.metadata.get("section", "")
        cluster_size = node.metadata.get("cluster_size", "")

        title_parts = [f"<b>{node_id}</b> (Layer {layer})"]
        if source:
            title_parts.append(f"Source: {source}")
        if section:
            title_parts.append(f"Section: {section}")
        if cluster_size:
            title_parts.append(f"Cluster size: {cluster_size}")
        title_parts.append(f"<hr>{text_preview}")
        title = "<br>".join(title_parts)

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size,
            level=layer,
            shape="dot" if layer == 0 else "diamond",
            borderWidth=3 if node_id in highlight_set else 1,
        )

    # Add edges (parent → child)
    for node_id, node in tree.nodes.items():
        for child_id in node.children:
            if child_id in tree.nodes:
                net.add_edge(node_id, child_id, color="#cccccc", arrows="to")

    # Write HTML
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output))

    n_nodes = len(tree.nodes)
    n_edges = sum(len(n.children) for n in tree.nodes.values())
    n_highlighted = len(highlight_set & set(tree.nodes.keys()))
    logger.info(
        f"RAPTOR visualization: {n_nodes} nodes, {n_edges} edges, "
        f"{n_highlighted} highlighted → {output_path}"
    )

    return str(output)


def _layer_color(layer: int) -> str:
    """Get color for a given tree layer."""
    if layer < len(LAYER_COLORS):
        return LAYER_COLORS[layer]
    return LAYER_COLORS[-1]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_raptor_viz.py -v --tb=short 2>&1 | tail -15
```

Expected: All tests PASS.

- [ ] **Step 6: Create visualization CLI script**

Create `scripts/visualize_raptor.py`:

```python
#!/usr/bin/env python
"""Visualize a RAPTOR tree as an interactive HTML graph.

Loads the RAPTOR tree from ChromaDB and generates a Pyvis visualization.
Optionally highlights nodes that match a query.

Usage:
    uv run python scripts/visualize_raptor.py
    uv run python scripts/visualize_raptor.py --output raptor.html
    uv run python scripts/visualize_raptor.py --query "potassium limits CKD"
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Visualize RAPTOR tree")
    parser.add_argument(
        "--output", "-o",
        default="raptor_tree.html",
        help="Output HTML file path (default: raptor_tree.html)",
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Optional query — highlights retrieved nodes in the tree",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of results to highlight when using --query",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    from config import get_embeddings, RAPTOR_COLLECTION_NAME
    from simple_rag.vectorstore import CKDVectorStore
    from simple_rag.raptor_builder import RaptorNode, RaptorTree
    from simple_rag.raptor_viz import visualize_tree

    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    logger.info(f"Loading RAPTOR collection '{RAPTOR_COLLECTION_NAME}'...")
    store = CKDVectorStore(
        embedding_function=embeddings,
        collection_name=RAPTOR_COLLECTION_NAME,
    )

    stats = store.get_collection_stats()
    doc_count = stats["document_count"]
    if doc_count == 0:
        logger.error("RAPTOR collection is empty. Run build_raptor_index.py first.")
        sys.exit(1)

    logger.info(f"Collection has {doc_count} nodes")

    # Reconstruct RaptorTree from ChromaDB metadata
    collection = store._client.get_collection(RAPTOR_COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas"])

    nodes = {}
    for i, (doc_id, text, meta) in enumerate(
        zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    ):
        meta = meta or {}
        node_id = meta.get("raptor_node_id", doc_id)
        children_str = meta.get("raptor_children", "")
        children = [c for c in children_str.split(",") if c]
        layer = int(meta.get("raptor_layer", 0))

        nodes[node_id] = RaptorNode(
            node_id=node_id,
            text=text or "",
            embedding=[],  # Not needed for viz
            layer=layer,
            children=children,
            metadata={k: v for k, v in meta.items() if not k.startswith("raptor_")},
        )

    max_layer = max(n.layer for n in nodes.values()) if nodes else 0
    tree = RaptorTree(nodes=nodes, depth=max_layer)
    logger.info(f"Reconstructed tree: {len(nodes)} nodes, depth={max_layer}")

    # Optionally highlight query results
    highlight = []
    if args.query:
        logger.info(f"Running query: '{args.query}'")
        results = store.search_with_scores(query=args.query, k=args.k)
        for doc, score in results:
            nid = doc.metadata.get("raptor_node_id", "")
            if nid:
                highlight.append(nid)
                logger.info(f"  Hit: {nid} (score={score:.3f})")

    # Generate visualization
    output = visualize_tree(tree, args.output, highlight_nodes=highlight)
    logger.info(f"Visualization saved to {output}")
    print(f"\nOpen in browser: file://{Path(output).resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add simple_rag/raptor_viz.py tests/test_raptor_viz.py scripts/visualize_raptor.py
git commit -m "feat: add Pyvis interactive RAPTOR tree visualization"
```

---

### Task 8: CLI `--retriever` flag for all levels + show retrieved context

**Files:**
- Modify: `main.py:102-144` (init_components)
- Modify: `main.py:406-465` (argparse + main)
- Modify: `main.py:150-237` (chat_simple — show context)
- Modify: `main.py:242-328` (chat_agentic — pass selected retriever)
- Modify: `main.py:330-401` (chat_multi — pass selected retriever)

This task adds a `--retriever` flag to the CLI so the user can choose `flat`, `tree`, `raptor`, or `contextual` for any RAG level. It also adds a `--show-context` flag that prints retrieved chunks before the answer.

- [ ] **Step 1: Update argparse in `main()` (lines 406-428)**

Replace the `main()` function (from `def main():` to `args = parser.parse_args()`) with:

```python
def main():
    parser = argparse.ArgumentParser(
        description="CKD RAG Terminal Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  uv run python main.py simple                        # Level 1, default tree retriever
  uv run python main.py simple --retriever raptor      # Level 1 with RAPTOR
  uv run python main.py agentic --retriever contextual # Level 2 with Contextual RAG
  uv run python main.py multi --retriever flat         # Level 3 with flat retriever
  uv run python main.py simple --show-context          # Show retrieved chunks
  uv run python main.py simple -v                      # With debug logging
        """,
    )
    parser.add_argument(
        "level",
        choices=["simple", "agentic", "multi"],
        help="RAG level to use",
    )
    parser.add_argument(
        "--retriever", "-r",
        choices=["flat", "tree", "raptor", "contextual"],
        default="tree",
        help="Retriever strategy (default: tree)",
    )
    parser.add_argument(
        "--show-context", "-c",
        action="store_true",
        help="Show retrieved context chunks before the answer",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()
```

- [ ] **Step 2: Update `init_components()` to accept retriever choice (lines 102-144)**

Replace the `init_components()` function with:

```python
def init_components(retriever_type: str = "tree"):
    from config import get_llm, get_embeddings

    log_tool("embeddings", "loading embedding model...")
    embeddings = get_embeddings()

    log_tool("vectorstore", "loading ChromaDB...")
    rag_pkg = import_package("rag_pkg", PROJECT_ROOT / "simple_rag")
    vectorstore = rag_pkg.CKDVectorStore(embeddings)

    # If vectorstore is empty, load processed chunks
    stats = vectorstore.get_collection_stats()
    if stats["document_count"] == 0:
        processed_dir = PROJECT_ROOT / "Data" / "processed"
        log_tool("vectorstore", f"empty collection — loading chunks from {processed_dir}...")
        docs, n_files = load_processed_chunks(processed_dir)
        if docs:
            log_info(f"loaded {len(docs)} chunks from {n_files} files")
            vectorstore.add_documents(docs)
            log_info(f"indexed {len(docs)} chunks into ChromaDB")
        else:
            log_warn(f"no chunk files found in {processed_dir}")
    else:
        log_info(f"vectorstore has {stats['document_count']} documents")

    # Create the selected retriever
    log_tool("retriever", f"initializing {retriever_type} retriever...")
    from simple_rag.retriever import create_retriever

    retriever = create_retriever(
        vectorstore=vectorstore,
        embedding_function=embeddings,
        use_tree=(retriever_type == "tree"),
        use_raptor=(retriever_type == "raptor"),
        use_contextual=(retriever_type == "contextual"),
        use_hybrid=False,
    )

    log_tool("llm", "loading LLM...")
    llm = get_llm()

    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "retriever_type": retriever_type,
        "llm": llm,
        "rag_pkg": rag_pkg,
    }
```

- [ ] **Step 3: Update `chat_simple()` to show retrieved context (lines 150-237)**

Add a `show_context` parameter and context display. Replace the function signature and the section before streaming:

Find this block in `chat_simple` (the function signature and first few lines):

```python
def chat_simple(comps: dict):
    rag_pkg = comps["rag_pkg"]
    log_tool("simple_rag", "building RAG chain...")
    chain = rag_pkg.SimpleRAGChain(retriever=comps["retriever"], llm=comps["llm"])

    print(f"\n{C.BOLD}=== Simple RAG (Level 1) ==={C.RESET}")
    print(f"{C.GREY}Type 'quit' to exit.{C.RESET}\n")
```

Replace with:

```python
def chat_simple(comps: dict, show_context: bool = False):
    rag_pkg = comps["rag_pkg"]
    retriever_type = comps.get("retriever_type", "tree")
    log_tool("simple_rag", "building RAG chain...")
    chain = rag_pkg.SimpleRAGChain(retriever=comps["retriever"], llm=comps["llm"])

    print(f"\n{C.BOLD}=== Simple RAG (Level 1) [{retriever_type}] ==={C.RESET}")
    print(f"{C.GREY}Type 'quit' to exit.{C.RESET}\n")
```

Then find the line where the query is received and streaming begins. After `query = ...` and before the streaming block, add context display. Find:

```python
        query = raw
        log_info(f"query: {query[:80]}")
```

If that line doesn't exist, find the block right after the user input is captured (after `if not raw or raw.lower() in ...`) and before the streaming. Add context display by inserting after the query is set:

In the `chat_simple` while loop, after the user types their query and before the response is generated, add:

```python
        # Show retrieved context if requested
        if show_context:
            retrieved = comps["retriever"].invoke(query)
            if retrieved:
                print(f"\n{C.GREY}{'─' * 60}")
                print(f"{C.BOLD}Retrieved Context ({len(retrieved)} chunks):{C.RESET}")
                for i, doc in enumerate(retrieved, 1):
                    source = doc.metadata.get("source", "?")
                    section = doc.metadata.get("section", "")
                    layer = doc.metadata.get("raptor_layer", "")
                    ctx = doc.metadata.get("contextual_context", "")

                    # Header line
                    header_parts = [f"{C.CYAN}[{i}]{C.RESET} {C.YELLOW}{source}{C.RESET}"]
                    if section:
                        header_parts.append(f"§ {section}")
                    if layer != "":
                        header_parts.append(f"(layer {layer})")
                    print(f"  {'  '.join(header_parts)}")

                    # Contextual RAG context line
                    if ctx:
                        print(f"  {C.GREY}Context: {ctx[:100]}{'...' if len(ctx) > 100 else ''}{C.RESET}")

                    # Content preview
                    preview = doc.page_content[:150].replace("\n", " ")
                    if len(doc.page_content) > 150:
                        preview += "..."
                    print(f"  {C.GREY}{preview}{C.RESET}")
                    print()
                print(f"{C.GREY}{'─' * 60}{C.RESET}\n")
```

- [ ] **Step 4: Update `chat_agentic()` to use selected retriever (line 250)**

Find:

```python
        retriever=comps["flat_retriever"],
```

Replace with:

```python
        retriever=comps["retriever"],
```

And update the header to show retriever type:

Find:

```python
    print(f"\n{C.BOLD}=== Agentic RAG (Level 2) ==={C.RESET}")
```

Replace with:

```python
    retriever_type = comps.get("retriever_type", "flat")
    print(f"\n{C.BOLD}=== Agentic RAG (Level 2) [{retriever_type}] ==={C.RESET}")
```

- [ ] **Step 5: Update `chat_multi()` to use selected retriever (line 340)**

Find:

```python
        retriever=comps["flat_retriever"],
```

Replace with:

```python
        retriever=comps["retriever"],
```

And update the header:

Find:

```python
    print(f"\n{C.BOLD}=== Multi-Agent RAG (Level 3) ==={C.RESET}")
```

Replace with:

```python
    retriever_type = comps.get("retriever_type", "flat")
    print(f"\n{C.BOLD}=== Multi-Agent RAG (Level 3) [{retriever_type}] ==={C.RESET}")
```

- [ ] **Step 6: Wire args into main() (lines 440-460)**

Find the block that calls `init_components()` and the handler dispatch:

```python
    log_info("initializing components...")
    try:
        comps = init_components()
    except Exception as e:
        log_error(f"initialization failed: {e}")
        sys.exit(1)

    log_info("ready!\n")

    handlers = {
        "simple": chat_simple,
        "agentic": chat_agentic,
        "multi": chat_multi,
    }

    try:
        handlers[args.level](comps)
    except KeyboardInterrupt:
        pass
```

Replace with:

```python
    log_info(f"initializing components (retriever={args.retriever})...")
    try:
        comps = init_components(retriever_type=args.retriever)
    except Exception as e:
        log_error(f"initialization failed: {e}")
        sys.exit(1)

    log_info("ready!\n")

    if args.level == "simple":
        try:
            chat_simple(comps, show_context=args.show_context)
        except KeyboardInterrupt:
            pass
    elif args.level == "agentic":
        try:
            chat_agentic(comps)
        except KeyboardInterrupt:
            pass
    elif args.level == "multi":
        try:
            chat_multi(comps)
        except KeyboardInterrupt:
            pass
```

- [ ] **Step 7: Verify it parses correctly**

```bash
uv run python main.py --help
```

Expected:
```
usage: main.py [-h] [--retriever {flat,tree,raptor,contextual}]
               [--show-context] [-v]
               {simple,agentic,multi}

CKD RAG Terminal Chat

positional arguments:
  {simple,agentic,multi}
                        RAG level to use

options:
  -h, --help            show this help message and exit
  --retriever {flat,tree,raptor,contextual}, -r {flat,tree,raptor,contextual}
                        Retriever strategy (default: tree)
  --show-context, -c    Show retrieved context chunks before the answer
  -v, --verbose         Enable debug logging
```

- [ ] **Step 8: Commit**

```bash
git add main.py
git commit -m "feat: add --retriever and --show-context CLI flags for all RAG levels"
```
