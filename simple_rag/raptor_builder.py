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
