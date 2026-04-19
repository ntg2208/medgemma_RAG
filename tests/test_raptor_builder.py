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
        # First call: embed 5 leaf chunks -> 5 embeddings
        # Second call: embed 1 summary -> 1 embedding
        mock_embed.embed_documents = MagicMock(
            side_effect=[
                [[0.5] * 10] * 5,   # leaf embeddings for 5 chunks
                [[0.5] * 10],        # summary embedding for 1 cluster
            ]
        )
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
        # First call: 10 leaf embeddings
        # Subsequent calls: 2 summary embeddings per layer (2 clusters), then 1
        mock_embed.embed_documents = MagicMock(
            side_effect=[
                [[0.5] * 10] * 10,   # 10 leaf chunks
                [[0.5] * 10, [0.6] * 10],  # 2 summaries at layer 1
                [[0.5] * 10],              # 1 summary at layer 2 (collapsed)
            ]
        )

        builder = RaptorBuilder(
            embedding_function=mock_embed, llm=mock_llm, max_depth=2
        )

        chunks = [
            Document(page_content=f"Chunk {i}", metadata={"chunk_id": i})
            for i in range(10)
        ]

        # Layer 1: 10 nodes -> 2 clusters; Layer 2: 2 nodes -> 1 cluster
        cluster_side_effect = [
            [[0], [0], [1], [1], [0], [1], [0], [1], [0], [1]],  # layer 1: 10 nodes
            [[0], [0]],  # layer 2: 2 summary nodes collapse to 1
        ]
        with patch(
            "simple_rag.raptor_builder.cluster_embeddings",
            side_effect=cluster_side_effect,
        ):
            tree = builder.build(chunks)

        assert tree.depth <= 2
