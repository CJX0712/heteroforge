"""HeteroForge 核心数据契约(dataclass)。

全部跨层传递的对象都定义在此, 便于静态审计与序列化。约定见架构文档第 4 节与共享知识 K1-K15。

Example:
    >>> import numpy as np, scipy.sparse as sp
    >>> adj = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    >>> gd = GraphData(
    ...     num_nodes=2, adjacency=adj,
    ...     features=np.zeros((2, 3)), labels=np.array([0, 1]),
    ...     train_mask=np.array([True, False]), val_mask=np.array([False, True]),
    ...     test_mask=np.array([False, False]), metadata={},
    ... )
    >>> gd.validate(); gd.num_nodes
    2
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from heteroforge.core.errors import HeteroForgeError

REASON_ACCEPTED = "ACCEPTED_BY_CHANNEL_A"
REASON_LOW_CONF = "LOW_CONFIDENCE_ROUTED"
REASON_LOW_HOMOPHILY = "LOW_LOCAL_HOMOPHILY_ROUTED"
REASON_BUDGET_CAP = "BUDGET_CAP_ROUTED"
REASON_FULL_DUAL = "FORCED_FULL_DUAL"
REASON_DEGREE_ZERO = "DEGREE_ZERO_ROUTED"

ROUTE_REASONS: tuple[str, ...] = (
    REASON_ACCEPTED,
    REASON_LOW_CONF,
    REASON_LOW_HOMOPHILY,
    REASON_BUDGET_CAP,
    REASON_FULL_DUAL,
    REASON_DEGREE_ZERO,
)

ROUTING_MODES: tuple[str, ...] = ("cascade", "full_dual")


def canonical_edges(adjacency: sp.csr_matrix) -> np.ndarray:
    """返回无向边的规范表示 (min,max) 排序后的 (m,2) int64 数组。

    Args:
        adjacency: CSR 邻接矩阵。

    Returns:
        shape (m, 2) 的 int64 数组, 行按字典序升序。
    """
    coo = sp.triu(adjacency, k=1, format="coo")
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    pairs = np.stack([lo, hi], axis=1)
    order = np.lexsort((hi, lo))
    return np.ascontiguousarray(pairs[order])


@dataclass
class GraphData:
    """图数据契约。规范节点 ID 恒等于 arange(n)。"""

    num_nodes: int
    adjacency: sp.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """强制全部不变量; 任一违反抛 E2xx。"""
        n = int(self.num_nodes)
        if not sp.isspmatrix_csr(self.adjacency):
            raise HeteroForgeError("E207", "adjacency must be scipy csr_matrix")
        self.adjacency = self.adjacency.tocsr().astype(np.float64)
        if self.adjacency.shape != (n, n):
            raise HeteroForgeError("E207", "adjacency shape mismatch", {"shape": self.adjacency.shape})
        diff = self.adjacency - self.adjacency.T
        if abs(diff).max() > 1e-12:
            raise HeteroForgeError("E207", "adjacency must be symmetric")
        if abs(self.adjacency.diagonal()).max() > 1e-12:
            raise HeteroForgeError("E207", "adjacency must not contain self loops")
        self.features = np.ascontiguousarray(np.asarray(self.features, dtype=np.float64))
        if self.features.shape[0] != n:
            raise HeteroForgeError("E208", "features row count mismatch")
        if not np.isfinite(self.features).all():
            raise HeteroForgeError("E208", "features contain NaN or Inf")
        self.labels = np.asarray(self.labels, dtype=np.int64)
        if self.labels.shape != (n,):
            raise HeteroForgeError("E206", "labels shape mismatch")
        if self.labels.min() < -1:
            raise HeteroForgeError("E206", "labels below -1 are invalid")
        for name in ("train_mask", "val_mask", "test_mask"):
            mask = np.asarray(getattr(self, name), dtype=bool)
            if mask.shape != (n,):
                raise HeteroForgeError("E206", f"{name} shape mismatch")
            setattr(self, name, np.ascontiguousarray(mask))
        if np.any(self.train_mask & self.val_mask):
            raise HeteroForgeError("E206", "train and val masks overlap")
        if np.any(self.train_mask & self.test_mask):
            raise HeteroForgeError("E206", "train and test masks overlap")
        if np.any(self.val_mask & self.test_mask):
            raise HeteroForgeError("E206", "val and test masks overlap")

    def num_classes(self) -> int:
        """根据 labels 推断类别数; 全为 -1 时返回 0。"""
        known = self.labels[self.labels >= 0]
        return int(known.max()) + 1 if known.size else 0

    def node_hash(self) -> str:
        """对 (n, 规范边集, features, labels) 求 sha256 前 16 位。"""
        hasher = hashlib.sha256()
        hasher.update(np.asarray([self.num_nodes], dtype=np.int64).tobytes())
        hasher.update(np.ascontiguousarray(canonical_edges(self.adjacency)).tobytes())
        hasher.update(np.ascontiguousarray(self.features).tobytes())
        hasher.update(np.ascontiguousarray(self.labels).tobytes())
        return hasher.hexdigest()[:16]

    def edge_set(self, undirected: bool = True) -> set[tuple[int, int]]:
        """返回边集。undirected=True 时规范为 (min,max)。"""
        pairs = canonical_edges(self.adjacency) if undirected else np.stack(
            np.nonzero(self.adjacency), axis=1
        )
        return {(int(a), int(b)) for a, b in pairs}

    def induced(self, node_index: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
        """抽取给定节点集合的诱导子图。

        Args:
            node_index: 原图节点下标, 允许重复与外部顺序。

        Returns:
            (sub_adjacency, sub_features): 子图 CSR 邻接与对应特征行。
        """
        idx = np.asarray(node_index, dtype=np.int64)
        sub_adj = self.adjacency[idx][:, idx]
        return sp.csr_matrix(sub_adj), np.ascontiguousarray(self.features[idx])

    def to_nx(self) -> "Any":
        """转换为 networkx.Graph(延迟 import, 避免内核层硬依赖)。"""
        import networkx as nx

        return nx.from_scipy_sparse_array(self.adjacency)


@dataclass(frozen=True)
class HomophilyReport:
    """同配性度量报告。孤立节点在 node_homophily_vec 中为 NaN 且不参与均值。"""

    num_nodes: int
    num_edges: int
    num_classes: int
    edge_homophily: float
    node_homophily: float
    node_homophily_vec: np.ndarray
    node_degrees: np.ndarray
    isolated_count: int
    degree_assortativity: float
    attribute_assortativity: float
    class_adjusted_homophily: float

    def within_target(self, target: float, tol: float = 0.03) -> bool:
        """实测 edge_homophily 与目标的误差是否在容差内。"""
        return abs(float(self.edge_homophily) - float(target)) <= float(tol)

    def normalized_gate_features(self) -> np.ndarray:
        """送入 router 的三元组, 全部压到 [0,1]。

        degree_assortativity 原值在 [-1,1], 此处按 (r+1)/2 映射(K6); 报告仍输出原值。

        Returns:
            shape (3,) 的 float64 数组: [edge_h, node_h, (assortativity+1)/2]。
        """
        assort = float(self.degree_assortativity)
        assort = 0.0 if not np.isfinite(assort) else assort
        return np.asarray(
            [float(self.edge_homophily), float(self.node_homophily), (assort + 1.0) / 2.0],
            dtype=np.float64,
        )


@dataclass
class ChannelOutput:
    """单通道输出。confidence 口径统一为越大越可信(K5)。"""

    channel: str
    backend: str
    backend_license: str
    fallback_used: bool
    fallback_from: str | None
    embedding: np.ndarray
    node_ids: np.ndarray
    proba: np.ndarray | None
    confidence: np.ndarray
    computed_mask: np.ndarray
    dim: int
    seed: int
    elapsed_sec: float
    peak_rss_mb: float
    params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def validate(self, num_nodes: int | None = None) -> None:
        """校验 shape、有限性与节点顺序; 违反抛 E3xx/E4xx。"""
        self.embedding = np.ascontiguousarray(np.asarray(self.embedding, dtype=np.float64))
        if self.embedding.ndim != 2:
            raise HeteroForgeError("E304", "embedding must be 2-D")
        if not np.isfinite(self.embedding).all():
            raise HeteroForgeError("E304", "embedding contains NaN or Inf")
        n = self.embedding.shape[0]
        if num_nodes is not None and n != int(num_nodes):
            raise HeteroForgeError("E304", "embedding row count mismatch")
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        if not np.array_equal(self.node_ids, np.arange(n)):
            raise HeteroForgeError("E209", "node_ids must equal arange(n)")
        self.computed_mask = np.ascontiguousarray(np.asarray(self.computed_mask, dtype=bool))
        self.confidence = np.ascontiguousarray(np.asarray(self.confidence, dtype=np.float64))
        if self.confidence.shape != (n,):
            raise HeteroForgeError("E304", "confidence shape mismatch")
        if self.proba is not None:
            self.proba = np.ascontiguousarray(np.asarray(self.proba, dtype=np.float64))
            if not np.isfinite(self.proba).all():
                raise HeteroForgeError("E405", "proba contains NaN or Inf")
            rowsum = self.proba.sum(axis=1)
            if not np.allclose(rowsum, 1.0, atol=1e-6):
                raise HeteroForgeError("E405", "proba rows must sum to 1")
        self.dim = int(self.embedding.shape[1])

    def fusion_matrix(self) -> np.ndarray:
        """返回用于融合的矩阵: 优先 proba, 否则 embedding。"""
        return self.proba if self.proba is not None else self.embedding


@dataclass
class RoutingDecision:
    """HAAR 路由决策。reason 取值限于 ROUTE_REASONS。"""

    mode: str
    alpha_base: float
    w_homophily: float
    threshold: float
    budget_ratio: float
    alpha_vec: np.ndarray
    routed_mask: np.ndarray
    routed_fraction: float
    reason: np.ndarray
    node_homophily: np.ndarray
    selected_by: str = "default"
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """校验模式、reason 枚举与 alpha 边界; 违反抛 E5xx。"""
        if self.mode not in ROUTING_MODES:
            raise HeteroForgeError("E502", "unknown routing mode", {"mode": self.mode})
        self.alpha_vec = np.ascontiguousarray(np.asarray(self.alpha_vec, dtype=np.float64))
        if self.alpha_vec.min() < -1e-12 or self.alpha_vec.max() > 1.0 + 1e-12:
            raise HeteroForgeError("E501", "alpha_vec out of [0,1]")
        self.reason = np.asarray(self.reason, dtype="<U32")
        bad = sorted({str(r) for r in self.reason} - set(ROUTE_REASONS))
        if bad:
            raise HeteroForgeError("E502", "unknown route reason", {"reasons": bad})
        self.routed_mask = np.ascontiguousarray(np.asarray(self.routed_mask, dtype=bool))
        n = self.alpha_vec.shape[0]
        if self.routed_mask.shape != (n,) or self.reason.shape != (n,):
            raise HeteroForgeError("E502", "routing arrays shape mismatch")

    def fuse(self, z_a: np.ndarray, z_b: np.ndarray, computed_mask: np.ndarray) -> np.ndarray:
        """按 alpha_vec 融合两个通道; 未计算的行一律拒绝读取(E505)。

        Args:
            z_a: (n, d) 通道 A 输出矩阵。
            z_b: (n, d) 通道 B 输出矩阵, 未计算行为 0。
            computed_mask: (n,) bool, 通道 B 真实计算过的行。

        Returns:
            (n, d) 融合矩阵; accepted 行直接取 z_a。
        """
        self.validate()
        z_a = np.asarray(z_a, dtype=np.float64)
        z_b = np.asarray(z_b, dtype=np.float64)
        if z_a.shape != z_b.shape:
            raise HeteroForgeError("E505", "fusion shape mismatch", {"a": z_a.shape, "b": z_b.shape})
        computed = np.ascontiguousarray(np.asarray(computed_mask, dtype=bool))
        missing = np.flatnonzero(self.routed_mask & ~computed)
        if missing.size:
            raise HeteroForgeError(
                "E505",
                "routed node has no computed channel B value",
                {"count": int(missing.size), "first": int(missing[0])},
            )
        alpha = self.alpha_vec.reshape(-1, 1)
        fused = alpha * z_a + (1.0 - alpha) * z_b
        accepted = ~self.routed_mask
        if accepted.any():
            fused[accepted] = z_a[accepted]
        return np.ascontiguousarray(fused)


@dataclass
class EvalResult:
    """评测结果。test_touch_count 必须恰为 1, 否则 E604。"""

    task: str
    system: str
    dataset_id: str
    homophily_bucket: float
    homophily_measured: float
    n_nodes: int
    n_samples: int
    seed: int
    metrics: dict[str, float | None]
    elapsed_sec: float
    peak_rss_mb: float
    routed_fraction: float
    is_primary: bool
    test_touch_count: int
    leakage_hash: str


@dataclass
class BenchmarkRow:
    """benchmark 网格的一行结果。generator_effective 如实记录降级后的生成器。"""

    generator: str
    generator_effective: str
    n_nodes: int
    n_edges: int
    homophily_target: float
    homophily_measured: float
    num_classes: int
    class_balance: str
    feature_signal: str
    task: str
    system: str
    seed: int
    macro_f1: float | None
    micro_f1: float | None
    accuracy: float | None
    roc_auc: float | None
    average_precision: float | None
    alpha_base: float
    routed_fraction: float
    budget_ratio: float
    mode: str
    backend_a: str
    backend_b: str
    elapsed_sec: float
    peak_rss_mb: float
    status: str
    warning_codes: str
    split_hash: str
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 安全字典, None 保持为 null(K8)。"""
        return json.loads(json.dumps(dict(vars(self)), default=str))


@dataclass(frozen=True)
class SplitSpec:
    """切分规格。比例和必须为 1, 由 config 层校验(E102)。"""

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    edge_train_ratio: float = 0.7
    edge_val_ratio: float = 0.15
    edge_test_ratio: float = 0.15
    stratify: bool = True
    require_connected_train_graph: bool = True
    version: str = "1.0"

    def ratios_sum(self) -> float:
        """节点三比例之和。"""
        return float(self.train_ratio + self.val_ratio + self.test_ratio)
