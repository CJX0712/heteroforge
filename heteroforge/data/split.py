"""切分: 节点分层切分、链路生成树保护切分、split_hash。

节点切分只依赖 labels 与 seed; 链路切分保证训练图连通且三集互斥(E203/E205)。

Example:
    >>> import numpy as np
    >>> from heteroforge.core.types import SplitSpec
    >>> from heteroforge.data.split import split_nodes, split_hash
    >>> y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    >>> tr, va, te = split_nodes(y, SplitSpec(0.6, 0.2, 0.2), seed=1)
    >>> int(tr.sum() + va.sum() + te.sum())
    10
    >>> len(split_hash(tr, va, te))
    16
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp

from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import SplitSpec, canonical_edges
from heteroforge.core.utils import stable_hash


def split_nodes(
    labels: np.ndarray,
    spec: SplitSpec,
    seed: int = 0,
    known_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层切分节点; 交集恒为空。

    Args:
        labels: (n,) int64 标签, -1 表示未标注。
        spec: 切分规格。
        seed: 随机种子。
        known_mask: 可选, 仅在这些节点中切分; 其余节点三个 mask 均为 False。

    Returns:
        (train_mask, val_mask, test_mask), 均为 (n,) bool。
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.shape[0]
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    if known_mask is None:
        known_mask = labels >= 0
    known_mask = np.asarray(known_mask, dtype=bool)
    idx = np.flatnonzero(known_mask)
    if idx.size == 0:
        return train_mask, val_mask, test_mask
    rng = np.random.default_rng(int(seed))
    classes = np.unique(labels[idx])
    pools: list[np.ndarray] = []
    for cls in classes:
        members = idx[labels[idx] == cls]
        rng.shuffle(members)
        k = members.size
        n_train = int(round(k * spec.train_ratio))
        n_val = int(round(k * spec.val_ratio))
        n_train = min(max(n_train, 1 if k >= 3 else 0), k)
        n_val = min(max(n_val, 1 if k >= 3 else 0), k - n_train)
        if n_train + n_val > k:
            n_val = k - n_train
        n_test = k - n_train - n_val
        pools.append((members[:n_train], members[n_train:n_train + n_val],
                      members[n_train + n_val:n_train + n_val + n_test]))
    for tr, va, te in pools:
        train_mask[tr] = True
        val_mask[va] = True
        test_mask[te] = True
    if spec.stratify is False:
        merged = np.flatnonzero(train_mask | val_mask | test_mask)
        rng.shuffle(merged)
        k = merged.size
        n_train = int(round(k * spec.train_ratio))
        n_val = int(round(k * spec.val_ratio))
        train_mask[:] = False
        val_mask[:] = False
        test_mask[:] = False
        train_mask[merged[:n_train]] = True
        val_mask[merged[n_train:n_train + n_val]] = True
        test_mask[merged[n_train + n_val:]] = True
    return (np.ascontiguousarray(train_mask), np.ascontiguousarray(val_mask),
            np.ascontiguousarray(test_mask))


def split_hash(train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> str:
    """对三个 mask 求稳定哈希; 同 seed 同数据恒定。"""
    payload = [np.flatnonzero(np.asarray(m, dtype=bool)).tolist() for m in (train_mask, val_mask, test_mask)]
    return stable_hash(payload, 16)


@dataclass
class EdgeSplit:
    """链路切分结果: 三组边与统计。"""

    train_edges: np.ndarray
    val_edges: np.ndarray
    test_edges: np.ndarray
    protected_edges: np.ndarray
    num_edges_total: int = 0
    warnings: list[str] = field(default_factory=list)

    def sets(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
        """返回三组边的规范集合, 用于 E205 互斥校验。"""
        return (
            {(int(a), int(b)) for a, b in self.train_edges},
            {(int(a), int(b)) for a, b in self.val_edges},
            {(int(a), int(b)) for a, b in self.test_edges},
        )

    def assert_disjoint(self) -> None:
        """三组边必须两两不相交, 否则 E205。"""
        s_train, s_val, s_test = self.sets()
        if s_train & s_val or s_train & s_test or s_val & s_test:
            raise HeteroForgeError("E205", "edge splits overlap")


def split_edges(
    graph: nx.Graph | sp.csr_matrix,
    spec: SplitSpec,
    seed: int = 0,
    require_connected: bool | None = None,
) -> EdgeSplit:
    """生成树保护式链路切分: 先保护生成树, 再抽 val/test, 训练图必须连通。

    Args:
        graph: networkx 图或 CSR 邻接。
        spec: 切分规格。
        seed: 随机种子。
        require_connected: 覆盖 spec.require_connected_train_graph。

    Returns:
        EdgeSplit, 三组边均为 (k,2) int64 规范 (min,max) 元组。
    """
    need_connected = spec.require_connected_train_graph if require_connected is None else require_connected
    if isinstance(graph, sp.csr_matrix):
        adjacency = graph
        n_nodes = adjacency.shape[0]
        edges = canonical_edges(adjacency)
        nx_graph = nx.from_scipy_sparse_array(adjacency)
    else:
        nx_graph = graph
        n_nodes = graph.number_of_nodes()
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=sorted(graph.nodes()), format="csr")
        edges = canonical_edges(sp.csr_matrix(adjacency))
    if edges.shape[0] == 0:
        raise HeteroForgeError("E203", "graph has no edges to split")
    rng = np.random.default_rng(int(seed))
    protected = set()
    for component in nx.connected_components(nx_graph):
        subgraph = nx_graph.subgraph(component)
        if subgraph.number_of_nodes() > 1:
            for u, v in nx.minimum_spanning_tree(subgraph).edges():
                protected.add((int(min(u, v)), int(max(u, v))))
    candidates = np.asarray(
        [e for e in edges if (int(e[0]), int(e[1])) not in protected], dtype=np.int64
    ).reshape(-1, 2)
    if candidates.shape[0] == 0:
        raise HeteroForgeError("E203", "all edges are protected by spanning tree")
    order = rng.permutation(candidates.shape[0])
    shuffled = candidates[order]
    n_val = int(round(edges.shape[0] * spec.edge_val_ratio))
    n_test = int(round(edges.shape[0] * spec.edge_test_ratio))
    n_val = min(n_val, shuffled.shape[0])
    n_test = min(n_test, shuffled.shape[0] - n_val)
    val_edges = shuffled[:n_val]
    test_edges = shuffled[n_val:n_val + n_test]
    holdout = {(int(a), int(b)) for a, b in np.vstack([val_edges, test_edges]).reshape(-1, 2)}
    train_edges = np.asarray(
        [e for e in edges if (int(e[0]), int(e[1])) not in holdout], dtype=np.int64
    ).reshape(-1, 2)
    protected_arr = np.asarray(sorted(protected), dtype=np.int64).reshape(-1, 2)
    result = EdgeSplit(
        train_edges=np.ascontiguousarray(train_edges),
        val_edges=np.ascontiguousarray(val_edges),
        test_edges=np.ascontiguousarray(test_edges),
        protected_edges=np.ascontiguousarray(protected_arr),
        num_edges_total=int(edges.shape[0]),
    )
    result.assert_disjoint()
    if need_connected:
        train_graph = nx.Graph()
        train_graph.add_nodes_from(range(n_nodes))
        train_graph.add_edges_from((int(a), int(b)) for a, b in train_edges)
        if not nx.is_connected(train_graph):
            raise HeteroForgeError("E203", "train graph is not connected after edge split")
    return result


def train_adjacency_from_split(adjacency: sp.csr_matrix, split: EdgeSplit) -> sp.csr_matrix:
    """仅保留 train_edges 的 CSR, 供 embedding 使用(禁止看到 holdout 边)。"""
    n = adjacency.shape[0]
    coo = sp.coo_matrix(
        (
            np.ones(split.train_edges.shape[0], dtype=np.float64),
            (split.train_edges[:, 0], split.train_edges[:, 1]),
        ),
        shape=(n, n),
    )
    dense_free = sp.csr_matrix(coo)
    dense_free = dense_free + dense_free.T
    dense_free.setdiag(0.0)
    dense_free.eliminate_zeros()
    return sp.csr_matrix(dense_free)
