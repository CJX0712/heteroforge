"""通道 B: 单层 SGC 与 k-hop 局部前向。

Example:
    >>> from heteroforge.gnn import SGCBackend, k_hop_subgraph
    >>> import numpy as np, scipy.sparse as sp
    >>> adj = sp.csr_matrix(np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]))
    >>> nodes, sub, pos = k_hop_subgraph(adj, 0, 2)
    >>> nodes.tolist()
    [0, 1, 2]
"""

from __future__ import annotations

from heteroforge.gnn.channel_b import PropLRBackend, SGCBackend
from heteroforge.gnn.subgraph import k_hop_subgraph, symmetric_normalize

__all__ = ["SGCBackend", "PropLRBackend", "k_hop_subgraph", "symmetric_normalize"]
