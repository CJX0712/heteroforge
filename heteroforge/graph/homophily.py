"""同配性度量: edge / node / class-adjusted homophily 与 assortativity。

孤立节点策略固定: node_homophily_vec 置 NaN, 计入 isolated_count, 不参与均值。

Example:
    >>> import networkx as nx, numpy as np
    >>> from heteroforge.graph.homophily import measure_homophily
    >>> g = nx.Graph(); g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    >>> y = np.array([0, 1, 1])
    >>> rep = measure_homophily(g, y)
    >>> round(rep.edge_homophily, 6)
    0.333333
    >>> round(rep.node_homophily, 6)
    0.333333
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp

from heteroforge.core.types import HomophilyReport


def _as_graph(graph: nx.Graph | sp.csr_matrix) -> nx.Graph:
    """把 CSR 邻接或 networkx 图统一为 networkx.Graph。"""
    if isinstance(graph, nx.Graph):
        return graph
    return nx.from_scipy_sparse_array(sp.csr_matrix(graph))


def node_local_homophily(adjacency: sp.csr_matrix, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐节点同配比例; 度为 0 的节点为 NaN。

    Args:
        adjacency: (n,n) CSR 邻接。
        labels: (n,) int64 标签。

    Returns:
        (h_vec, degrees): 逐节点同配比例(float64, 孤立为 NaN) 与度数(int64)。
    """
    adjacency = sp.csr_matrix(adjacency)
    labels = np.asarray(labels, dtype=np.int64)
    n = adjacency.shape[0]
    degrees = np.asarray(adjacency.getnnz(axis=1), dtype=np.int64).reshape(-1)
    same = np.zeros(n, dtype=np.float64)
    indptr, indices = adjacency.indptr, adjacency.indices
    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            same[i] = float(np.count_nonzero(labels[indices[start:end]] == labels[i]))
    h_vec = np.full(n, np.nan, dtype=np.float64)
    positive = degrees > 0
    h_vec[positive] = same[positive] / degrees[positive].astype(np.float64)
    return h_vec, degrees


def measure_homophily(
    graph: nx.Graph | sp.csr_matrix,
    labels: np.ndarray,
) -> HomophilyReport:
    """计算全套同配性指标。

    Args:
        graph: networkx 图或 CSR 邻接。
        labels: (n,) int64 标签。

    Returns:
        HomophilyReport; edge_homophily 在 [0,1], 无边时为 nan。
    """
    nx_graph = _as_graph(graph)
    labels = np.asarray(labels, dtype=np.int64)
    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()
    adjacency = sp.csr_matrix(
        nx.to_scipy_sparse_array(nx_graph, nodelist=sorted(nx_graph.nodes()), format="csr")
    )
    h_vec, degrees = node_local_homophily(adjacency, labels)
    isolated = int(np.count_nonzero(degrees == 0))
    finite = h_vec[np.isfinite(h_vec)]
    node_h = float(finite.mean()) if finite.size else float("nan")
    if m == 0:
        edge_h = float("nan")
    else:
        same = sum(1 for u, v in nx_graph.edges() if labels[u] == labels[v])
        edge_h = float(same) / float(m)
    total_degree = float(degrees.sum())
    if total_degree > 0 and m > 0:
        shares = np.array(
            [float(degrees[labels == c].sum()) / total_degree for c in np.unique(labels)],
            dtype=np.float64,
        )
        baseline = float(np.sum(shares**2))
        h_adj = (edge_h - baseline) / (1.0 - baseline) if abs(1.0 - baseline) > 1e-12 else float("nan")
    else:
        h_adj = float("nan")
    degree_assort = _safe_assortativity(lambda: nx.degree_assortativity_coefficient(nx_graph))
    labelled = nx_graph.copy()
    nx.set_node_attributes(labelled, {int(i): int(v) for i, v in enumerate(labels)}, "label")
    attr_assort = _safe_assortativity(
        lambda: nx.attribute_assortativity_coefficient(labelled, "label")
    )
    return HomophilyReport(
        num_nodes=int(n),
        num_edges=int(m),
        num_classes=int(np.unique(labels[labels >= 0]).size),
        edge_homophily=float(edge_h),
        node_homophily=node_h,
        node_homophily_vec=h_vec,
        node_degrees=degrees,
        isolated_count=isolated,
        degree_assortativity=degree_assort,
        attribute_assortativity=attr_assort,
        class_adjusted_homophily=float(h_adj),
    )


def _safe_assortativity(func) -> float:
    """调用 networkx assortativity, 退化情形返回 nan 而不是抛异常。"""
    try:
        value = float(func())
    except Exception:  # noqa: BLE001 - 退化图(无边/单度数)时 networkx 会抛
        return float("nan")
    return value if np.isfinite(value) else float("nan")
