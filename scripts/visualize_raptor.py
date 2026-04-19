#!/usr/bin/env python
"""Visualize a RAPTOR tree as an interactive HTML graph.

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
    parser.add_argument("--output", "-o", default="raptor_tree.html", help="Output HTML file path")
    parser.add_argument("--query", "-q", default=None, help="Optional query — highlights retrieved nodes")
    parser.add_argument("--k", type=int, default=5, help="Number of results to highlight when using --query")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    from config import get_embeddings, RAPTOR_COLLECTION_NAME
    from simple_rag.vectorstore import CKDVectorStore
    from simple_rag.raptor_builder import RaptorNode, RaptorTree
    from simple_rag.raptor_viz import visualize_tree

    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    logger.info(f"Loading RAPTOR collection '{RAPTOR_COLLECTION_NAME}'...")
    store = CKDVectorStore(embedding_function=embeddings, collection_name=RAPTOR_COLLECTION_NAME)

    stats = store.get_collection_stats()
    doc_count = stats["document_count"]
    if doc_count == 0:
        logger.error("RAPTOR collection is empty. Run build_raptor_index.py first.")
        sys.exit(1)

    logger.info(f"Collection has {doc_count} nodes")

    collection = store._client.get_collection(RAPTOR_COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas"])

    nodes = {}
    for i, (doc_id, text, meta) in enumerate(zip(all_data["ids"], all_data["documents"], all_data["metadatas"])):
        meta = meta or {}
        node_id = meta.get("raptor_node_id", doc_id)
        children_str = meta.get("raptor_children", "")
        children = [c for c in children_str.split(",") if c]
        layer = int(meta.get("raptor_layer", 0))
        nodes[node_id] = RaptorNode(
            node_id=node_id, text=text or "", embedding=[], layer=layer,
            children=children, metadata={k: v for k, v in meta.items() if not k.startswith("raptor_")},
        )

    max_layer = max(n.layer for n in nodes.values()) if nodes else 0
    tree = RaptorTree(nodes=nodes, depth=max_layer)
    logger.info(f"Reconstructed tree: {len(nodes)} nodes, depth={max_layer}")

    highlight = []
    if args.query:
        logger.info(f"Running query: '{args.query}'")
        results = store.search_with_scores(query=args.query, k=args.k)
        for doc, score in results:
            nid = doc.metadata.get("raptor_node_id", "")
            if nid:
                highlight.append(nid)
                logger.info(f"  Hit: {nid} (score={score:.3f})")

    output = visualize_tree(tree, args.output, highlight_nodes=highlight)
    logger.info(f"Visualization saved to {output}")
    print(f"\nOpen in browser: file://{Path(output).resolve()}")


if __name__ == "__main__":
    main()
