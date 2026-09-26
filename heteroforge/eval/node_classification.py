"""节点分类评测: 统一 LR head + sklearn 交叉对照(E605)。

公平性口径: 每个系统(haar/channel_a/channel_b/feat_only)都在自己的表示上
用同一个 LogisticRegression head(train_mask 拟合, test_mask 预测一次,
test_touch_count 必须恰为 1, 违反抛 E604)。

Example:
    >>> from heteroforge.eval.node_classification import macro_f1_manual
    >>> import numpy as np
    >>> y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 1])
    >>> round(macro_f1_manual(y, p), 6)
    0.666667
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from heteroforge.core.config import EvalConfig
from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import EvalResult
from heteroforge.data.split import split_hash

_LAST_SKLEARN: dict[str, Any] = {}


def macro_f1_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """手写 Macro-F1(混淆矩阵实现), 供 sklearn 对照。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s: list[float] = []
    for cls in classes:
        tp = float(np.count_nonzero((y_pred == cls) & (y_true == cls)))
        fp = float(np.count_nonzero((y_pred == cls) & (y_true != cls)))
        fn = float(np.count_nonzero((y_pred != cls) & (y_true == cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _head(representation: np.ndarray) -> Any:
    """构造统一 LR head(n_jobs=1 保证可复现, K15)。"""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, n_jobs=1)


def head_predictions(
    representation: np.ndarray,
    graph: Any,
    fit_mask: np.ndarray,
    pred_mask: np.ndarray,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """在 representation 上拟合 LR head 并在 pred_mask 上预测。

    Args:
        representation: (n, d) 任意系统表示。
        graph: GraphData。
        fit_mask: 拟合用 mask(仅 train/val)。
        pred_mask: 预测用 mask。
        max_iter: 迭代上限。

    Returns:
        (y_true, y_pred): pred_mask 内的真实标签与预测标签。
    """
    representation = np.ascontiguousarray(np.asarray(representation, dtype=np.float64))
    idx_fit = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    idx_pred = np.flatnonzero(np.asarray(pred_mask, dtype=bool))
    if idx_fit.size == 0 or idx_pred.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    model = _head(representation)
    model.max_iter = int(max_iter)
    model.fit(representation[idx_fit], graph.labels[idx_fit])
    y_pred = np.asarray(model.predict(representation[idx_pred]), dtype=np.int64)
    return np.asarray(graph.labels[idx_pred], dtype=np.int64), y_pred


def evaluate_representation(
    representation: np.ndarray,
    graph: Any,
    system: str,
    dataset_id: str,
    homophily_bucket: float,
    homophily_measured: float,
    seed: int,
    routed_fraction: float = 0.0,
    elapsed_sec: float = 0.0,
    peak_rss_mb: float = 0.0,
    config: EvalConfig | None = None,
) -> EvalResult:
    """在 test_mask 上评估一个系统的节点分类表现。

    Args:
        representation: (n, d) 系统表示(fused proba/embedding/原始特征)。
        graph: GraphData。
        system: 系统名(haar/channel_a/channel_b/feat_only)。
        dataset_id: 数据集 id(含 seed 与规模)。
        homophily_bucket: 目标同配档位。
        homophily_measured: 实测 edge homophily。
        seed: 主种子。
        routed_fraction: 级联路由比例(haar 专有)。
        elapsed_sec: 上游耗时。
        peak_rss_mb: 峰值 RSS。
        config: 评测配置。

    Returns:
        EvalResult, test_touch_count 恒为 1。

    Raises:
        HeteroForgeError: E605 与 sklearn 对照差异超过 1e-9。
    """
    cfg = config if config is not None else EvalConfig()
    started = time.perf_counter()
    y_true, y_pred = head_predictions(
        representation, graph, fit_mask=graph.train_mask, pred_mask=graph.test_mask,
        max_iter=int(cfg.max_iter),
    )
    head_elapsed = time.perf_counter() - started
    if y_true.size == 0:
        raise HeteroForgeError("E601", "empty test mask for node classification")
    macro = macro_f1_manual(y_true, y_pred)
    from sklearn.metrics import accuracy_score as _sk_acc
    from sklearn.metrics import f1_score as _sk_f1

    sk_macro = float(_sk_f1(y_true, y_pred, average="macro", zero_division=0))
    if abs(sk_macro - macro) > 1e-9:
        raise HeteroForgeError(
            "E605", "manual macro-F1 mismatches sklearn",
            {"manual": macro, "sklearn": sk_macro},
        )
    _LAST_SKLEARN["macro_f1"] = sk_macro
    metrics: dict[str, float | None] = {
        "macro_f1": sk_macro,
        "micro_f1": float(_sk_f1(y_true, y_pred, average="micro", zero_division=0)),
        "accuracy": float(_sk_acc(y_true, y_pred)),
    }
    return EvalResult(
        task="node_classification",
        system=str(system),
        dataset_id=str(dataset_id),
        homophily_bucket=float(homophily_bucket),
        homophily_measured=float(homophily_measured),
        n_nodes=int(graph.num_nodes),
        n_samples=int(y_true.size),
        seed=int(seed),
        metrics=metrics,
        elapsed_sec=float(elapsed_sec + head_elapsed),
        peak_rss_mb=float(peak_rss_mb),
        routed_fraction=float(routed_fraction),
        is_primary=str(system) == "haar",
        test_touch_count=1,
        leakage_hash=split_hash(graph.train_mask, graph.val_mask, graph.test_mask),
    )
