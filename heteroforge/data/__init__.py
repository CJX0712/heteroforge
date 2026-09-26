"""数据层: 合成图生成、持久化、切分。

Example:
    >>> from heteroforge.data import generate_graph, split_nodes
    >>> from heteroforge.core.types import SplitSpec
    >>> out = generate_graph(n_nodes=60, num_classes=3, homophily=0.2, seed=42)
    >>> out.labels.shape
    (60,)
"""

from __future__ import annotations

from heteroforge.data.loader import load_graph_data, save_graph_data
from heteroforge.data.split import (
    EdgeSplit,
    split_edges,
    split_hash,
    split_nodes,
)
from heteroforge.data.synthetic import (
    SyntheticGraph,
    generate_graph,
    make_graph_data,
)

__all__ = [
    "SyntheticGraph",
    "generate_graph",
    "make_graph_data",
    "save_graph_data",
    "load_graph_data",
    "split_nodes",
    "split_edges",
    "split_hash",
    "EdgeSplit",
]
