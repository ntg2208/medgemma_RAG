"""
Pyvis interactive visualization for RAPTOR trees.

Generates a standalone HTML file with an interactive network graph.
Nodes are colored by layer, sized by text length, and show
text previews on hover. Edges show parent→child relationships.
"""

import logging
from pathlib import Path
from typing import Optional

from pyvis.network import Network

logger = logging.getLogger(__name__)

LAYER_COLORS = [
    "#3498db",  # Layer 0 (leaves): blue
    "#2ecc71",  # Layer 1: green
    "#f39c12",  # Layer 2: orange
    "#9b59b6",  # Layer 3: purple
    "#1abc9c",  # Layer 4+: teal
]
HIGHLIGHT_COLOR = "#e74c3c"  # Red for highlighted/retrieved nodes

MIN_SIZE = 10
MAX_SIZE = 40


def visualize_tree(
    tree,
    output_path: str = "raptor_tree.html",
    highlight_nodes: Optional[list[str]] = None,
    height: str = "900px",
    width: str = "100%",
) -> str:
    """Generate an interactive Pyvis visualization of a RAPTOR tree."""
    highlight_set = set(highlight_nodes or [])

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
    )

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

    for node_id, node in tree.nodes.items():
        layer = node.layer
        color = HIGHLIGHT_COLOR if node_id in highlight_set else _layer_color(layer)

        size = MIN_SIZE + min(len(node.text) // 50, MAX_SIZE - MIN_SIZE)
        if node.children:
            size = max(size, MIN_SIZE + len(node.children) * 5)

        label = node_id

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

    for node_id, node in tree.nodes.items():
        for child_id in node.children:
            if child_id in tree.nodes:
                net.add_edge(node_id, child_id, color="#cccccc", arrows="to")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output))

    n_nodes = len(tree.nodes)
    n_edges = sum(len(n.children) for n in tree.nodes.values())
    n_highlighted = len(highlight_set & set(tree.nodes.keys()))
    logger.info(
        f"RAPTOR visualization: {n_nodes} nodes, {n_edges} edges, "
        f"{n_highlighted} highlighted -> {output_path}"
    )

    return str(output)


def _layer_color(layer: int) -> str:
    if layer < len(LAYER_COLORS):
        return LAYER_COLORS[layer]
    return LAYER_COLORS[-1]
