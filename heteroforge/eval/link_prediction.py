"""链路预测评测: degree-aware hard negatives + cosine 打分 + sklearn 对照(E605)。

Example:
    >>> import numpy as np
    >>> from heteroforge.eval.link_prediction import roc_auc_manual
    >>> y = np.array([1, 1, 0, 0]); s = np.array([0.9, 0.8, 0.2, 0.1])
    >>> round(roc_auc_manual(y, s), 6)
    1.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from heteroforge.core.errors import HeteroForgeError


def roc_auc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """手写 rank-based AUC, 供 sklearn 对照。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        raise HeteroForgeError("E601", "AUC undefined without both classes")
    order = np.argsort(y_score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def degree_aware_negatives(
    adjacency: sp.csr_matrix,
    sources: np.ndarray,
    negative_ratio: float = 1.0,
    seed: int = 0,
    pool_mult: int = 20,
) -> np.ndarray:
    """为每个源节点采样度数相近的未连边负样本(架构裁断 7)。

    Args:
        adjacency: 训练图 CSR(只允许看训练边)。
        sources: (k,) 源节点(正样本的 u 端)。
        negative_ratio: 每个正样本配的负样本数。
        seed: 随机种子。
        pool_mult: 候选池倍数。

    Returns:
        (k*n_neg, 2) int64 负边数组(规范 min,max)。
    """
    adjacency = sp.csr_matrix(adjacency)
    rng = np.random.default_rng(int(seed))
    n = adjacency.shape[0]
    degrees = np.asarray(adjacency.getnnz(axis=1), dtype=np.int64)
    n_neg = max(1, int(round(float(negative_ratio))))
    out: list[tuple[int, int]] = []
    for u in np.asarray(sources, dtype=np.int64):
        u = int(u)
        neighbors = adjacency.indices[adjacency.indptr[u]:adjacency.indptr[u + 1]]
        blocked = np.zeros(n, dtype=bool)
        blocked[u] = True
        blocked[neighbors] = True
        cand = np.flatnonzero(~blocked)
        if cand.size == 0:
            continue
        pool_size = min(cand.size, max(n_neg * pool_mult, n_neg))
        if cand.size > pool_size:
            pool_idx = rng.choice(cand.size, size=pool_size, replace=False)
            cand = cand[pool_idx]
        dist = np.abs(degrees[cand] - degrees[u])
        best = cand[np.argsort(dist, kind="stable")[:n_neg]]
        for v in best:
            a, b = (u, int(v)) if u < int(v) else (int(v), u)
            out.append((a, b))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def evaluate_link_prediction(
    representation: np.ndarray,
    train_adjacency: sp.csr_matrix,
    test_edges: np.ndarray,
    negative_ratio: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    """cosine 表示下的链路预测指标。

    Args:
        representation: (n, d) 节点表示(来自训练图, 禁止接触 holdout 边)。
        train_adjacency: 训练图 CSR(负采样只看它)。
        test_edges: (m, 2) 规范测试边。
        negative_ratio: 负采样比。
        seed: 随机种子。

    Returns:
        {"roc_auc": float, "average_precision": float}。

    Raises:
        HeteroForgeError: E601 测试边为空; E605 与 sklearn 对照差异超 1e-9。
    """
    test_edges = np.asarray(test_edges, dtype=np.int64).reshape(-1, 2)
    if test_edges.shape[0] == 0:
        raise HeteroForgeError("E601", "empty test edges for link prediction")
    repr_norm = np.asarray(representation, dtype=np.float64)
    norms = np.linalg.norm(repr_norm, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    repr_norm = repr_norm / norms

    def _score(pairs: np.ndarray) -> np.ndarray:
        u, v = pairs[:, 0], pairs[:, 1]
        return np.sum(repr_norm[u] * repr_norm[v], axis=1)

    neg_edges = degree_aware_negatives(
        train_adjacency, test_edges[:, 0], negative_ratio=negative_ratio, seed=int(seed)
    )
    y_true = np.concatenate([
        np.ones(test_edges.shape[0], dtype=np.int64),
        np.zeros(neg_edges.shape[0], dtype=np.int64),
    ])
    y_score = np.concatenate([_score(test_edges), _score(neg_edges)])
    auc = roc_auc_manual(y_true, y_score)
    from sklearn.metrics import average_precision_score as _sk_ap
    from sklearn.metrics import roc_auc_score as _sk_auc

    sk_auc = float(_sk_auc(y_true, y_score))
    if abs(sk_auc - auc) > 1e-9:
        raise HeteroForgeError(
            "E605", "manual AUC mismatches sklearn", {"manual": auc, "sklearn": sk_auc}
        )
    return {
        "roc_auc": sk_auc,
        "average_precision": float(_sk_ap(y_true, y_score)),
    }
