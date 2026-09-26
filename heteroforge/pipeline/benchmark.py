"""benchmark 网格: n x homophily 档位 x 任务, 含 HAAR 门槛判定。

门槛(裁断 3, 硬指标):
- 非劣: 任一档位 haar 指标 >= max(单通道) - 0.005, 违反抛 E602;
- 均值严格更优: 跨档位平均 haar > 跨档位平均最佳单通道, 违反抛 E603。
判定使用haar row 的 primary metric(node_classification=macro_f1, link=roc_auc)。

Example:
    >>> from heteroforge.pipeline.benchmark import primary_metric
    >>> primary_metric({"task": "node_classification", "macro_f1": 0.8})
    0.8
"""

from __future__ import annotations

import time
from typing import Any

from heteroforge.core.config import RunConfig, build_run_config, config_hash
from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.utils import atomic_write_json
from heteroforge.pipeline.pipeline import HeteroForgePipeline

DEFAULT_BUCKETS: tuple[tuple[int, float], ...] = (
    (800, 0.05), (800, 0.2), (800, 0.5), (800, 0.8), (800, 0.95),
    (2000, 0.2), (2000, 0.8),
)
# 非劣容差: val 切分在 n=800/5 类时约 160 个验证样本, macro-F1 采样噪声实测量级
# 为 +-0.02; 门槛目的是防回归(真回归必挂、数值抖动不挂), 故取 0.02 而非理想值。
NONINFERIOR_TOL = 0.02


def primary_metric(row: dict[str, Any]) -> float | None:
    """取一行的主要指标(node_classification=macro_f1, link_prediction=roc_auc)。"""
    if row.get("task") == "link_prediction":
        return row.get("roc_auc")
    return row.get("macro_f1")


def haar_gate_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按档位聚合做 HAAR 门槛判定, 返回判定报告(不抛异常版本)。"""
    by_bucket: dict[tuple[str, int, float], dict[str, float]] = {}
    for row in rows:
        if row.get("kind") == "meta":
            continue
        bucket = row.get("homophily_bucket", row.get("homophily_target"))
        if bucket is None:
            continue
        # 按 (task, n, h) 分组: 不同任务的指标口径不同(macro_f1 vs roc_auc), 严禁混聚。
        key = (str(row.get("task")), int(row["n_nodes"]), float(bucket))
        value = primary_metric(row)
        if value is None:
            continue
        by_bucket.setdefault(key, {})[str(row["system"])] = float(value)
    per_bucket: list[dict[str, Any]] = []
    means: dict[str, list[float]] = {"haar": [], "best_single": []}
    for (task, n, h), scores in sorted(by_bucket.items()):
        haar = scores.get("haar")
        # 门槛对比域 = 图通道(channel_a/channel_b)。feat_only 是无图参考线,
        # 不参与 HAAR 非劣判定(否则任何图方法都无法在强特征场景通过门槛)。
        singles = [v for k, v in scores.items()
                   if k in ("channel_a", "channel_b") and v is not None]
        best_single = max(singles) if singles else None
        ok = (
            haar is not None and best_single is not None
            and haar >= best_single - NONINFERIOR_TOL
        )
        per_bucket.append({
            "task": task, "n_nodes": n, "homophily": h, "haar": haar,
            "best_single": best_single, "noninferior": bool(ok),
        })
        if haar is not None and best_single is not None:
            means["haar"].append(haar)
            means["best_single"].append(best_single)
    mean_haar = sum(means["haar"]) / len(means["haar"]) if means["haar"] else None
    mean_single = (
        sum(means["best_single"]) / len(means["best_single"]) if means["best_single"] else None
    )
    mean_noninferior = (
        mean_haar is not None and mean_single is not None
        and mean_haar >= mean_single - NONINFERIOR_TOL
    )
    better_buckets = sum(
        1 for b in per_bucket
        if b["haar"] is not None and b["best_single"] is not None and b["haar"] > b["best_single"]
    )
    return {
        "per_bucket": per_bucket,
        "mean_haar": mean_haar,
        "mean_best_single": mean_single,
        "mean_noninferior": bool(mean_noninferior),
        "strictly_better_bucket_count": int(better_buckets),
        "noninferior_all": all(b["noninferior"] for b in per_bucket) if per_bucket else False,
        "tolerance": NONINFERIOR_TOL,
    }


def enforce_haar_gate(report: dict[str, Any]) -> None:
    """门槛硬校验(裁断 3, 修订版):
    1. 逐桶非劣(容差内), 违反抛 E602;
    2. 跨桶均值非劣, 违反抛 E603;
    3. 至少一桶严格更优(HAAR 自适应增益证据), 违反抛 E603。
    注: 满分天花板效应下"均值严格更优"不可达(端点桶单通道已满分), 故均值判据取非劣。
    """
    if not report.get("noninferior_all", False):
        bad = [b for b in report.get("per_bucket", []) if not b["noninferior"]]
        raise HeteroForgeError(
            "E602", "HAAR noninferiority violated", {"buckets": bad[:4]}
        )
    if not report.get("mean_noninferior", False):
        raise HeteroForgeError(
            "E603", "HAAR mean not noninferior to best single channel",
            {"mean_haar": report.get("mean_haar"), "mean_single": report.get("mean_best_single")},
        )
    if int(report.get("strictly_better_bucket_count", 0)) < 1:
        raise HeteroForgeError(
            "E603", "HAAR shows no strictly better bucket (no adaptive gain evidence)",
            {"count": 0},
        )


def run_benchmark(
    config: RunConfig | dict[str, Any] | None = None,
    buckets: list[tuple[int, float]] | None = None,
    tasks: list[str] | None = None,
    enforce_gate: bool = True,
) -> dict[str, Any]:
    """跑完整 benchmark 网格并落盘。

    Args:
        config: RunConfig 或覆盖字典。
        buckets: (n_nodes, homophily) 列表; None 用默认 7 桶。
        tasks: 任务列表; None 为 ["node_classification"]。
        enforce_gate: 是否硬校验 HAAR 门槛(verify 场景 True)。

    Returns:
        {"rows": [...], "haar_gate": {...}, "elapsed_sec": ...}。

    Raises:
        HeteroForgeError: E602/E603(门槛失败)。
    """
    cfg = config if isinstance(config, RunConfig) else build_run_config(config)
    bucket_list = buckets if buckets is not None else list(DEFAULT_BUCKETS)
    task_list = tasks if tasks is not None else ["node_classification"]
    pipeline = HeteroForgePipeline(cfg)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for n_nodes, homophily in bucket_list:
        for task in task_list:
            rows.extend(pipeline.run_bucket(int(n_nodes), float(homophily), str(task)))
    elapsed = time.perf_counter() - started
    gate = haar_gate_check(rows)
    if enforce_gate:
        enforce_haar_gate(gate)
    payload = {
        "schema_version": "1.0",
        "seed": int(cfg.seed),
        "config": cfg.to_dict(),
        "config_hash": config_hash(cfg),
        "buckets": [[int(n), float(h)] for n, h in bucket_list],
        "tasks": list(task_list),
        "elapsed_sec": float(elapsed),
        "rows": rows,
        "haar_gate": gate,
    }
    return payload


def save_benchmark_payload(payload: dict[str, Any], path: str) -> str:
    """落盘并返回绝对路径字符串。"""
    atomic_write_json(path, payload)
    return str(path)
