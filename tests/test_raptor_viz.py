"""Tests for RAPTOR tree Pyvis visualization."""

import pytest
from pathlib import Path
from simple_rag.raptor_builder import RaptorNode, RaptorTree


def _make_sample_tree() -> RaptorTree:
    """Build a small tree for testing: 4 leaves, 2 L1 summaries, 1 L2 root."""
    nodes = {
        "leaf_0": RaptorNode("leaf_0", "Potassium limits for CKD stage 3.", [0.1] * 10, 0, [], {"source": "nice.pdf", "section": "Dietary"}),
        "leaf_1": RaptorNode("leaf_1", "Sodium should be limited to 2000mg.", [0.2] * 10, 0, [], {"source": "nice.pdf", "section": "Dietary"}),
        "leaf_2": RaptorNode("leaf_2", "ACE inhibitors for proteinuria.", [0.3] * 10, 0, [], {"source": "kdigo.pdf", "section": "Medication"}),
        "leaf_3": RaptorNode("leaf_3", "eGFR monitoring every 3 months.", [0.4] * 10, 0, [], {"source": "kdigo.pdf", "section": "Monitoring"}),
        "summary_L1_C0": RaptorNode("summary_L1_C0", "Summary: dietary restrictions for CKD.", [0.15] * 10, 1, ["leaf_0", "leaf_1"], {"layer": 1, "cluster_size": 2}),
        "summary_L1_C1": RaptorNode("summary_L1_C1", "Summary: medication and monitoring.", [0.35] * 10, 1, ["leaf_2", "leaf_3"], {"layer": 1, "cluster_size": 2}),
        "summary_L2_C0": RaptorNode("summary_L2_C0", "Summary: CKD management overview.", [0.25] * 10, 2, ["summary_L1_C0", "summary_L1_C1"], {"layer": 2, "cluster_size": 2}),
    }
    return RaptorTree(nodes=nodes, depth=2)


class TestVisualize:
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
        for node_id in tree.nodes:
            assert node_id in html

    def test_html_contains_edges(self, tmp_path):
        from simple_rag.raptor_viz import visualize_tree
        tree = _make_sample_tree()
        output = tmp_path / "tree.html"
        visualize_tree(tree, str(output))
        html = output.read_text()
        assert "summary_L2_C0" in html
        assert "leaf_0" in html

    def test_default_output_path(self, tmp_path, monkeypatch):
        from simple_rag.raptor_viz import visualize_tree
        monkeypatch.chdir(tmp_path)
        tree = _make_sample_tree()
        result_path = visualize_tree(tree)
        assert Path(result_path).exists()


class TestHighlightRetrieval:
    def test_highlight_nodes(self, tmp_path):
        from simple_rag.raptor_viz import visualize_tree
        tree = _make_sample_tree()
        output = tmp_path / "highlighted.html"
        visualize_tree(tree, str(output), highlight_nodes=["leaf_0", "summary_L1_C0"])
        html = output.read_text()
        assert output.exists()
        assert "#e74c3c" in html or "red" in html.lower()
