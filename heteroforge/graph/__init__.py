"""度量层: 同配性度量与 feature-label signal 构造。

Example:
    >>> from heteroforge.graph import measure_homophily, build_features
    >>> import networkx as nx
    >>> g = nx.Graph(); g.add_edges_from([(0, 1), (2, 3)])
    >>> labels = __import__("numpy").array([0, 0, 1, 1])
    >>> rep = measure_homophily(g, labels)
    >>> rep.edge_homophily
    1.0
"""

from __future__ import annotations

from heteroforge.graph.features import (
    build_features,
    build_structural_features,
    signal_strength,
)
from heteroforge.graph.homophily import (
    measure_homophily,
    node_local_homophily,
)

__all__ = [
    "measure_homophily",
    "node_local_homophily",
    "build_features",
    "build_structural_features",
    "signal_strength",
]
