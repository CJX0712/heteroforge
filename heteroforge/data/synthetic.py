"""合成图生成: SBM 的 p_in/p_out 校准、LFR 的 mu 逼近、三次失败后降级 SBM。

P0-01: 生成后必须以实测 homophily 验收, 不能相信参数; 降级时 generator_effective 写 'sbm'。

Example:
    >>> from heteroforge.data.synthetic import generate_graph
    >>> out = generate_graph(n_nodes=120, num_classes=3, homophily=0.8, seed=42)
    >>> out.generator_effective
    'sbm'
    >>> out.labels.shape
    (120,)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.utils import derive_seed

BALANCE_MODES: tuple[str, ...] = ("balanced", "imbalanced_1_1_1_1_4")


@dataclass
class SyntheticGraph:
    """生成结果: 图、标签与降级审计字段。"""

    graph: nx.Graph
    labels: np.ndarray
    generator: str = "sbm"
    generator_effective: str = "sbm"
    homophily_target: float = 0.5
    homophily_measured: float = float("nan")
    num_classes: int = 2
    class_balance: str = "balanced"
    attempts: int = 1
    warnings: list[str] = field(default_factory=list)

    def to_csr(self) -> sp.csr_matrix:
        """转为 CSR 邻接(float64, 对称, 无自环)。"""
        adj = nx.to_scipy_sparse_array(self.graph, nodelist=sorted(self.graph.nodes()), format="csr")
        return sp.csr_matrix(adj.astype(np.float64))


def _class_sizes(n_nodes: int, num_classes: int, class_balance: str) -> list[int]:
    """按平衡模式切分各类节点数。"""
    if class_balance == "imbalanced_1_1_1_1_4" and num_classes >= 5:
        weights = [1.0] * (num_classes - 1) + [4.0]
    elif class_balance == "balanced":
        weights = [1.0] * num_classes
    else:
        weights = [1.0] * num_classes
    total = float(sum(weights))
    sizes = [max(1, int(round(n_nodes * w / total))) for w in weights]
    diff = int(n_nodes) - int(sum(sizes))
    idx = 0
    while diff != 0:
        if diff > 0:
            sizes[idx % num_classes] += 1
            diff -= 1
        else:
            if sizes[idx % num_classes] > 1:
                sizes[idx % num_classes] -= 1
                diff += 1
        idx += 1
    return sizes


def _pair_counts(sizes: list[int]) -> tuple[float, float]:
    """返回 (类内可能对计数, 类间可能对计数)。"""
    arr = np.asarray(sizes, dtype=np.float64)
    within = float(np.sum(arr * (arr - 1.0) / 2.0))
    across = float((arr.sum() ** 2 - np.sum(arr**2)) / 2.0)
    return within, across


def _initial_probabilities(
    sizes: list[int], homophily: float, avg_degree: float
) -> tuple[float, float, float]:
    """由目标 homophily 与平均度反解 p_in / p_out。"""
    within, across = _pair_counts(sizes)
    n_nodes = int(sum(sizes))
    m_target = n_nodes * float(avg_degree) / 2.0
    p_in = homophily * m_target / within if within > 0 else 0.0
    p_out = (1.0 - homophily) * m_target / across if across > 0 else 0.0
    scale = max(p_in, p_out, 1e-12)
    if scale > 1.0:
        p_in /= scale
        p_out /= scale
    return float(p_in), float(p_out), float(m_target)


def _edge_homophily(graph: nx.Graph, labels: np.ndarray) -> float:
    """实测 edge homophily; 无边时返回 nan。"""
    m = graph.number_of_edges()
    if m == 0:
        return float("nan")
    same = sum(1 for u, v in graph.edges() if labels[u] == labels[v])
    return float(same) / float(m)


def _build_sbm(sizes: list[int], p_in: float, p_out: float, seed: int) -> nx.Graph:
    """构造无向 SBM 图并规整节点编号。"""
    num_classes = len(sizes)
    probs = np.full((num_classes, num_classes), float(p_out), dtype=np.float64)
    np.fill_diagonal(probs, float(p_in))
    graph = nx.stochastic_block_model(sizes, probs, seed=int(seed), directed=False, selfloops=False)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")


def _attach_labels(graph: nx.Graph, labels: np.ndarray) -> nx.Graph:
    """把标签写入节点属性 label, 便于度量层与序列化。"""
    nx.set_node_attributes(graph, {int(i): int(v) for i, v in enumerate(labels)}, "label")
    return graph


def generate_sbm(
    n_nodes: int,
    num_classes: int,
    homophily: float,
    seed: int,
    class_balance: str = "balanced",
    avg_degree: float = 10.0,
    max_retry: int = 3,
    tolerance: float = 0.03,
) -> SyntheticGraph:
    """校准式 SBM 生成: 生成 -> 实测 -> 二分校正 p_in/p_out, 最多 max_retry 次。"""
    sizes = _class_sizes(n_nodes, num_classes, class_balance)
    labels = np.concatenate([np.full(s, c, dtype=np.int64) for c, s in enumerate(sizes)])
    p_in, p_out, m_target = _initial_probabilities(sizes, homophily, avg_degree)
    best: SyntheticGraph | None = None
    for attempt in range(1, max(1, int(max_retry)) + 1):
        graph = _build_sbm(sizes, p_in, p_out, seed + attempt)
        measured = _edge_homophily(graph, labels)
        result = SyntheticGraph(
            graph=_attach_labels(graph, labels),
            labels=labels.copy(),
            generator="sbm",
            generator_effective="sbm",
            homophily_target=float(homophily),
            homophily_measured=float(measured),
            num_classes=int(num_classes),
            class_balance=class_balance,
            attempts=attempt,
        )
        if np.isfinite(measured) and abs(measured - homophily) <= tolerance:
            return result
        if best is None or (
            np.isfinite(measured) and abs(measured - homophily) < abs(best.homophily_measured - homophily)
        ):
            best = result
        if not np.isfinite(measured):
            break
        # 校正: 同配边概率按 target/measured 缩放, 异配边按补空间缩放, 再回压到目标边数
        new_in = p_in * (homophily / measured) if measured > 1e-9 else p_in * 2.0
        new_out = p_out * ((1.0 - homophily) / (1.0 - measured)) if measured < 1.0 - 1e-9 else p_out * 2.0
        within, across = _pair_counts(sizes)
        m_new = within * new_in + across * new_out
        if m_new > 1e-9:
            new_in *= m_target / m_new
            new_out *= m_target / m_new
        scale = max(new_in, new_out, 1e-12)
        if scale > 1.0:
            new_in /= scale
            new_out /= scale
        p_in, p_out = float(min(new_in, 1.0)), float(min(new_out, 1.0))
    if best is None:
        raise HeteroForgeError("E201", "SBM generation failed", {"n_nodes": n_nodes})
    best.warnings.append("HOMOPHILY_TARGET_MISS")
    return best


def _lfr_attempt(n_nodes: int, mu: float, min_degree: int, max_degree: int, seed: int) -> nx.Graph:
    """单次 LFR 尝试; 失败由上层捕获。"""
    return nx.LFR_benchmark_graph(
        n=n_nodes,
        tau1=3.0,
        tau2=1.5,
        mu=float(mu),
        min_degree=int(min_degree),
        max_degree=int(max_degree),
        min_community=max(10, n_nodes // 20),
        max_community=max(20, n_nodes // 8),
        max_iters=100,
        seed=int(seed),
    )


def _communities_to_labels(graph: nx.Graph, n_nodes: int, num_classes: int) -> np.ndarray:
    """LFR 社区映射为 C 个类: 最大 C-1 个社区各占一类, 其余并入最后一类。

    社区取自 LFR 自带的 community 节点属性(每个节点一个 frozenset), 不做二次社区发现。
    """
    buckets: dict[frozenset, list[int]] = {}
    for node, data in graph.nodes(data=True):
        key = frozenset(data.get("community", frozenset([node])))
        buckets.setdefault(key, []).append(int(node))
    groups = sorted(buckets.values(), key=len, reverse=True)
    if len(groups) < num_classes:
        raise HeteroForgeError("E201", "LFR community count below num_classes",
                               {"communities": len(groups)})
    labels = np.full(n_nodes, num_classes - 1, dtype=np.int64)
    for cls in range(num_classes - 1):
        labels[np.asarray(groups[cls], dtype=np.int64)] = cls
    return labels


def generate_lfr(
    n_nodes: int,
    num_classes: int,
    homophily: float,
    seed: int,
    avg_degree: float = 10.0,
    max_retry: int = 3,
    tolerance: float = 0.03,
) -> SyntheticGraph:
    """LFR 生成: mu = 1 - homophily 起步, 失败或超差最多 3 次, 之后降级 SBM。"""
    mu = float(np.clip(1.0 - homophily, 0.01, 0.99))
    min_degree = max(3, int(avg_degree * 0.6))
    max_degree = max(min_degree + 2, int(avg_degree * 1.6))
    last_error = "unknown"
    for attempt in range(1, max(1, int(max_retry)) + 1):
        try:
            graph = _lfr_attempt(n_nodes, mu, min_degree, max_degree, seed + attempt)
            graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
            if graph.number_of_nodes() != n_nodes:
                raise HeteroForgeError("E201", "LFR node count mismatch")
            labels = _communities_to_labels(graph, n_nodes, num_classes)
            measured = _edge_homophily(graph, labels)
            if np.isfinite(measured) and abs(measured - homophily) <= tolerance:
                return SyntheticGraph(
                    graph=_attach_labels(graph, labels),
                    labels=labels,
                    generator="lfr",
                    generator_effective="lfr",
                    homophily_target=float(homophily),
                    homophily_measured=float(measured),
                    num_classes=int(num_classes),
                    attempts=attempt,
                )
            if np.isfinite(measured):
                mu = float(np.clip(mu - (homophily - measured) * 0.8, 0.01, 0.99))
                continue
            last_error = "empty graph"
        except Exception as exc:  # noqa: BLE001 - LFR 失败必须收敛到 SBM
            last_error = f"{type(exc).__name__}: {exc}"
            min_degree = max(2, min_degree - 1)
            max_degree = max(min_degree + 2, max_degree + 2)
    fallback = generate_sbm(
        n_nodes=n_nodes,
        num_classes=num_classes,
        homophily=homophily,
        seed=seed,
        avg_degree=avg_degree,
        max_retry=max_retry,
        tolerance=tolerance,
    )
    fallback.generator = "lfr"
    fallback.generator_effective = "sbm"
    fallback.warnings.append("LFR_TO_SBM")
    fallback.warnings.append(f"reason={last_error[:120]}")
    return fallback


def generate_graph(
    n_nodes: int,
    num_classes: int,
    homophily: float,
    seed: int,
    generator: str = "sbm",
    class_balance: str = "balanced",
    avg_degree: float = 10.0,
    max_retry: int = 3,
    tolerance: float = 0.03,
    limits: dict[str, Any] | None = None,
) -> SyntheticGraph:
    """统一生成入口; 先做规模守卫(E204), 再按 generator 分发。"""
    if limits is not None:
        if int(n_nodes) > int(limits.get("max_nodes", 5000)):
            raise HeteroForgeError("E204", "n_nodes exceeds limit", {"n_nodes": n_nodes})
        est_edges = int(n_nodes * avg_degree / 2)
        if est_edges > int(limits.get("max_edges", 200000)):
            raise HeteroForgeError("E204", "estimated edges exceed limit", {"edges": est_edges})
    if not 0.0 <= float(homophily) <= 1.0:
        raise HeteroForgeError("E102", "homophily must be in [0,1]", {"value": homophily})
    seed = int(derive_seed(seed, "data"))
    if generator == "lfr":
        return generate_lfr(
            n_nodes=n_nodes,
            num_classes=num_classes,
            homophily=homophily,
            seed=seed,
            avg_degree=avg_degree,
            max_retry=max_retry,
            tolerance=tolerance,
        )
    if generator != "sbm":
        raise HeteroForgeError("E102", "unknown generator", {"value": generator})
    return generate_sbm(
        n_nodes=n_nodes,
        num_classes=num_classes,
        homophily=homophily,
        seed=seed,
        class_balance=class_balance,
        avg_degree=avg_degree,
        max_retry=max_retry,
        tolerance=tolerance,
    )


def make_graph_data(
    n_nodes: int = 800,
    num_classes: int = 5,
    homophily: float = 0.5,
    seed: int = 42,
    generator: str = "sbm",
    class_balance: str = "balanced",
    feature_signal: str = "strong",
    feature_dim: int = 64,
    avg_degree: float = 10.0,
    split_spec: Any | None = None,
) -> Any:
    """便捷组装: 生成 -> 特征 -> 切分 -> GraphData(延迟 import 避免模块级跨层依赖)。"""
    from heteroforge.core.types import GraphData, SplitSpec
    from heteroforge.data.split import split_nodes
    from heteroforge.graph.features import build_features

    spec = split_spec if split_spec is not None else SplitSpec()
    out = generate_graph(
        n_nodes=n_nodes,
        num_classes=num_classes,
        homophily=homophily,
        seed=seed,
        generator=generator,
        class_balance=class_balance,
        avg_degree=avg_degree,
    )
    features = build_features(
        out.labels,
        num_classes=num_classes,
        signal=feature_signal,
        dim=feature_dim,
        seed=derive_seed(seed, "features"),
    )
    adjacency = out.to_csr()
    train_mask, val_mask, test_mask = split_nodes(
        out.labels, spec, seed=derive_seed(seed, "split_node")
    )
    graph = GraphData(
        num_nodes=int(n_nodes),
        adjacency=adjacency,
        features=features,
        labels=out.labels.astype(np.int64),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        metadata={
            "generator": generator,
            "generator_effective": out.generator_effective,
            "homophily_target": float(homophily),
            "homophily_measured": float(out.homophily_measured),
            "num_classes": int(num_classes),
            "class_balance": class_balance,
            "feature_signal": feature_signal,
            "seed": int(seed),
            "warnings": list(out.warnings),
        },
    )
    graph.validate()
    return graph
