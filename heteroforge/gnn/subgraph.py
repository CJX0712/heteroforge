"""k-hop 诱导子图抽取: BFS 取点 -> 升序重编号 -> 诱导 CSR -> 局部对称归一化。

子图节点一律按升序编号, 中心节点位置由 center_pos 返回, 供 forward() 取行。

Example:
    >>> import numpy as np, scipy.sparse as sp
    >>> from heteroforge.gnn.subgraph import k_hop_subgraph, symmetric_normalize
    >>> adj = sp.csr_matrix(np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]))
    >>> nodes, sub, pos = k_hop_subgraph(adj, 2, 1)
    >>> nodes.tolist(), pos
    ([1, 2], 0)
    >>> sub.shape
    (2, 2)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from heteroforge.core.errors import HeteroForgeError


def symmetric_normalize(adjacency: sp.csr_matrix, add_self_loop: bool = True) -> sp.csr_matrix:
    """对称归一化 D^{-1/2} A D^{-1/2}; 孤立节点保持零行。

    Args:
        adjacency: (m, m) CSR 邻接。
        add_self_loop: 是否先加自环(SGC 的 self_embeddings=True 对应 True)。

    Returns:
        归一化后的 CSR。
    """
    adj = sp.csr_matrix(adjacency, dtype=np.float64)
    if add_self_loop:
        adj = adj + sp.eye(adj.shape[0], format="csr", dtype=np.float64)
    degrees = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    diag = sp.diags(inv_sqrt, format="csr")
    return sp.csr_matrix(diag @ adj @ diag)


def k_hop_subgraph(
    adjacency: sp.csr_matrix,
    center: int,
    k_hop: int = 2,
    raise_on_empty: bool = True,
) -> tuple[np.ndarray, sp.csr_matrix, int]:
    """抽取 center 的 k-hop 诱导子图。

    Args:
        adjacency: (n, n) CSR 邻接。
        center: 中心节点下标。
        k_hop: 跳数, >= 1。
        raise_on_empty: 邻域为空时是否抛 E404; False 时返回仅含中心的子图。

    Returns:
        (nodes, sub_adjacency, center_pos): 子图原节点下标(升序)、子图 CSR、中心位置。

    Raises:
        HeteroForgeError: E404, k-hop 邻域只有中心节点且 raise_on_empty=True。
    """
    adj = sp.csr_matrix(adjacency, dtype=np.float64)
    n = adj.shape[0]
    center = int(center)
    if not 0 <= center < n:
        raise HeteroForgeError("E404", "center node out of range", {"center": center})
    k_hop = max(1, int(k_hop))
    frontier = {center}
    visited = {center}
    for _ in range(k_hop):
        nxt: set[int] = set()
        for node in frontier:
            start, end = adj.indptr[node], adj.indptr[node + 1]
            nxt.update(int(v) for v in adj.indices[start:end])
        nxt -= visited
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    if len(visited) <= 1 and raise_on_empty:
        raise HeteroForgeError("E404", "k-hop subgraph contains only the center",
                               {"center": center, "k_hop": k_hop})
    nodes = np.asarray(sorted(visited), dtype=np.int64)
    sub = sp.csr_matrix(adj[nodes][:, nodes])
    center_pos = int(np.flatnonzero(nodes == center)[0])
    return nodes, sub, center_pos


def batch_k_hop_subgraphs(
    adjacency: sp.csr_matrix, centers: np.ndarray, k_hop: int = 2
) -> list[tuple[int, np.ndarray, sp.csr_matrix, int]]:
    """批量抽取; 邻域为空的中心返回仅含自身的 1x1 子图, 不抛异常。

    Returns:
        [(center, nodes, sub_adjacency, center_pos), ...] 与 centers 同序。
    """
    results: list[tuple[int, np.ndarray, sp.csr_matrix, int]] = []
    for center in np.asarray(centers, dtype=np.int64):
        try:
            nodes, sub, pos = k_hop_subgraph(adjacency, int(center), k_hop=k_hop, raise_on_empty=True)
        except HeteroForgeError:
            nodes = np.asarray([int(center)], dtype=np.int64)
            sub = sp.csr_matrix((1, 1), dtype=np.float64)
            pos = 0
        results.append((int(center), nodes, sub, pos))
    return results


def ego_size_profile(adjacency: sp.csr_matrix, k_hop: int = 2) -> np.ndarray:
    """返回每个节点 k-hop ego 图的节点数, 用于级联成本评估(架构 12.4)。

    用布尔矩阵幂累计可达集: reach = I | A | A^2 | ... | A^k, 逐行非零元数即 ego 规模。
    """
    adj = sp.csr_matrix(adjacency)
    n = adj.shape[0]
    edge = (adj > 0).astype(np.float64)
    reach = sp.eye(n, format="csr", dtype=np.float64)
    cur = sp.eye(n, format="csr", dtype=np.float64)
    for _ in range(max(1, int(k_hop))):
        cur = ((cur @ edge) > 0).astype(np.float64)
        reach = ((reach + cur) > 0).astype(np.float64)
    return np.asarray(reach.getnnz(axis=1), dtype=np.int64)
