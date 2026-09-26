"""feature-label signal 生成与结构统计特征。

strong/weak 两档控制 X 与 y 的互信息, 避免把结构收益与特征强度混淆(P0-02 附注)。

Example:
    >>> import numpy as np
    >>> from heteroforge.graph.features import build_features, signal_strength
    >>> y = np.repeat(np.arange(3), 20)
    >>> xs = build_features(y, 3, "strong", 16, seed=0)
    >>> xw = build_features(y, 3, "weak", 16, seed=0)
    >>> signal_strength(xs, y, seed=0) > signal_strength(xw, y, seed=0)
    True
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from heteroforge.core.errors import HeteroForgeError

SIGNAL_LEVELS: tuple[str, ...] = ("strong", "weak")

# 类中心间距与噪声标准差的比值, strong 明显大于 weak。
_SIGNAL_PARAMS: dict[str, tuple[float, float]] = {"strong": (2.5, 0.35), "weak": (0.45, 1.10)}


def build_features(
    labels: np.ndarray,
    num_classes: int,
    signal: str = "strong",
    dim: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """按类别中心加高斯噪声生成特征矩阵。

    Args:
        labels: (n,) int64 标签, -1 节点按类别 0 中心加噪。
        num_classes: 类别数。
        signal: 'strong' 或 'weak'。
        dim: 特征维度。
        seed: 随机种子。

    Returns:
        (n, dim) float64 C 连续矩阵, 无 NaN/Inf。
    """
    if signal not in _SIGNAL_PARAMS:
        raise HeteroForgeError("E102", "unknown feature signal", {"value": signal})
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.shape[0]
    separation, noise = _SIGNAL_PARAMS[signal]
    rng = np.random.default_rng(int(seed))
    centers = rng.normal(loc=0.0, scale=1.0, size=(int(num_classes), int(dim))) * separation
    safe = np.where(labels >= 0, labels, 0)
    if safe.max() >= int(num_classes):
        raise HeteroForgeError("E206", "label value exceeds num_classes")
    features = centers[safe] + rng.normal(loc=0.0, scale=noise, size=(n, int(dim)))
    features = np.ascontiguousarray(features.astype(np.float64))
    if not np.isfinite(features).all():
        raise HeteroForgeError("E208", "generated features contain NaN or Inf")
    return features


def signal_strength(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """用 LogisticRegression 的留出准确率度量 feature-label 可分离度。

    Args:
        features: (n, d) 特征。
        labels: (n,) 标签。
        seed: 随机种子。

    Returns:
        留出集准确率, 值越高表示 signal 越强; 样本不足时返回 nan。
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    known = labels >= 0
    x = features[known]
    y = labels[known]
    if x.shape[0] < 10 or np.unique(y).size < 2:
        return float("nan")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=int(seed), stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=200, random_state=int(seed), n_jobs=1)
    model.fit(scaler.transform(x_train), y_train)
    return float(model.score(scaler.transform(x_test), y_test))


def build_structural_features(adjacency: sp.csr_matrix, include_clustering: bool = False) -> np.ndarray:
    """结构统计特征: 度数、对数度数、度占比。

    Args:
        adjacency: (n,n) CSR 邻接。
        include_clustering: 是否追加局部聚类系数(n 大时较慢)。

    Returns:
        (n, d) float64 标准化后的结构特征。
    """
    adjacency = sp.csr_matrix(adjacency)
    degrees = np.asarray(adjacency.getnnz(axis=1), dtype=np.float64).reshape(-1, 1)
    log_degree = np.log1p(degrees)
    total = float(degrees.sum())
    share = degrees / total if total > 0 else np.zeros_like(degrees)
    columns = [degrees, log_degree, share]
    if include_clustering:
        import networkx as nx

        nx_graph = nx.from_scipy_sparse_array(adjacency)
        clustering = np.asarray(list(nx.clustering(nx_graph).values()), dtype=np.float64).reshape(-1, 1)
        columns.append(clustering)
    matrix = np.concatenate(columns, axis=1)
    matrix = np.ascontiguousarray(matrix.astype(np.float64))
    std = matrix.std(axis=0)
    std[std < 1e-12] = 1.0
    return np.ascontiguousarray((matrix - matrix.mean(axis=0)) / std)


def scale_features(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """只在给定 mask(如 train)上拟合 StandardScaler, 防泄漏。

    Args:
        features: (n, d) 特征。
        mask: 拟合用行; None 表示全量。

    Returns:
        标准化后的 (n, d) 矩阵。
    """
    features = np.asarray(features, dtype=np.float64)
    rows = features if mask is None else features[np.asarray(mask, dtype=bool)]
    if rows.shape[0] == 0:
        raise HeteroForgeError("E208", "cannot fit scaler on empty mask")
    scaler = StandardScaler().fit(rows)
    return np.ascontiguousarray(scaler.transform(features))
