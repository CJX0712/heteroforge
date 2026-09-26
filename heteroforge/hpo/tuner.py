"""per-bucket Optuna 路由校准: 每个同配性档位独立搜索 RoutingParams。

共享约定:
- 目标函数为 val_mask 上的 Macro-F1(HPO 禁止接触 test, 由 assert_no_test_access 静态拦截)。
- trial 内通道 B 使用真实 k-hop 局部前向(不走全图切片), 保证选出的参数
  在最终级联前向上同样最优。
- sampler 与 n_jobs 全部钉死(K15): TPESampler(seed=derive_seed(seed, "hpo")), n_jobs=1。

Example:
    >>> from heteroforge.hpo.tuner import HPOResult
    >>> from heteroforge.core.config import RoutingParams
    >>> r = HPOResult(params=RoutingParams(), study_summary={"n_trials": 0},
    ...               selected_by="default", warnings=["HPO_FALLBACK"])
    >>> r.selected_by
    'default'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from heteroforge.core.config import EmbedConfig, GNNConfig, RoutingParams, RunConfig
from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import ChannelOutput, GraphData, HomophilyReport
from heteroforge.core.utils import derive_seed
from heteroforge.eval.node_classification import head_predictions
from heteroforge.router.haar import HAARRouter, assert_no_test_access

DEFAULT_TRIALS = 30
DEFAULT_TIMEOUT_SEC = 300.0

# 单通道端点: HAAR 路由空间的两端恒等价于纯通道 A / 纯通道 B。
# 强制入队保证 HPO 结果在 val 上不低于最佳单通道(非劣性 by construction),
# TPE 自由搜索只负责在端点之间找更优的融合配置。
_ENDPOINT_TRIALS: list[dict[str, Any]] = [
    {"mode": "cascade", "alpha_base": 1.0, "w_homophily": 0.0,
     "threshold": 0.2, "budget_ratio": 1.0},
    {"mode": "full_dual", "alpha_base": 0.0, "w_homophily": 0.0,
     "threshold": 0.2, "budget_ratio": 1.0},
]


@dataclass
class HPOResult:
    """调参结果: 选中参数 + study 摘要 + 降级告警。"""

    params: RoutingParams
    study_summary: dict[str, Any]
    selected_by: str
    warnings: list[str] = field(default_factory=list)


def _suggest_params(trial: Any, base: RoutingParams) -> RoutingParams:
    """从 trial 采样一组 RoutingParams(mode 也作为超参由 val 选择)。"""
    mode = trial.suggest_categorical("mode", ["cascade", "full_dual"])
    alpha_base = trial.suggest_float("alpha_base", 0.0, 1.0)
    w_homophily = trial.suggest_float("w_homophily", 0.0, 4.0)
    threshold = trial.suggest_float("threshold", 0.2, 0.95)
    budget_ratio = trial.suggest_float("budget_ratio", 0.05, 1.0)
    return RoutingParams(
        mode=str(mode),
        alpha_base=float(alpha_base),
        w_homophily=float(w_homophily),
        threshold=float(threshold),
        budget_ratio=float(budget_ratio),
        k_hop=int(base.k_hop),
        selected_by="hpo",
    )


def tune_routing(
    graph: GraphData,
    report: HomophilyReport,
    out_a: ChannelOutput,
    gnn_backend: Any,
    seed: int,
    n_trials: int = DEFAULT_TRIALS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    base_params: RoutingParams | None = None,
    gnn_config: GNNConfig | None = None,
    max_iter: int = 1000,
) -> HPOResult:
    """对单个 homophily 档位做 Optuna 路由校准。

    Args:
        graph: GraphData(train/val 参与, test 永不进入)。
        report: 同配性报告。
        out_a: 通道 A 全图输出。
        gnn_backend: 已 fit 的 SGCBackend(权重固定, trial 只做前向)。
        seed: 主种子(内部按 tag "hpo" 派生)。
        n_trials: trial 数上限。
        timeout_sec: study 超时秒数, 超时保留已完成 trial。
        base_params: 参数搜索空间基准(k_hop 等沿用)。
        gnn_config: 通道 B 配置(k_hop 沿用)。
        max_iter: LR head 迭代上限。

    Returns:
        HPOResult; 无完成 trial 时降级为默认参数并携带 HPO_FALLBACK 告警。

    Raises:
        HeteroForgeError: E502 路由模式非法(由 RoutingDecision.validate 抛出)。
    """
    assert_no_test_access("train_mask", "val_mask")
    base = base_params if base_params is not None else RoutingParams()
    gcfg = gnn_config if gnn_config is not None else GNNConfig()
    hpo_seed = derive_seed(int(seed), "hpo")
    router = HAARRouter()

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, base)
        decision = router.route(graph, report, out_a, None, params)
        routed_idx = np.flatnonzero(decision.routed_mask)
        if routed_idx.size == 0:
            return 0.0
        out_b = gnn_backend.predict(graph, nodes=routed_idx, k_hop=int(params.k_hop))
        fused = decision.fuse(out_a.fusion_matrix(), out_b.fusion_matrix(), out_b.computed_mask)
        y_true, y_pred = head_predictions(
            fused, graph, fit_mask=graph.train_mask, pred_mask=graph.val_mask, max_iter=max_iter
        )
        if y_true.size == 0:
            return 0.0
        from heteroforge.core.errors import HeteroForgeError as _E
        from sklearn.metrics import f1_score as _sk_f1

        return float(_sk_f1(y_true, y_pred, average="macro", zero_division=0))

    warnings: list[str] = []
    study_summary: dict[str, Any] = {"n_trials": int(n_trials), "timeout_sec": float(timeout_sec)}
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=int(hpo_seed))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        if float(timeout_sec) > 0:
            for endpoint in _ENDPOINT_TRIALS:
                study.enqueue_trial(dict(endpoint))
            study.optimize(objective, n_trials=int(n_trials), timeout=float(timeout_sec), n_jobs=1)
        else:
            study_summary["skipped"] = "timeout_sec <= 0"
    except HeteroForgeError:
        raise
    except Exception as exc:  # noqa: BLE001 - optuna 链路异常统一降级, 不吞业务错误
        warnings.append("HPO_FALLBACK")
        study_summary["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return HPOResult(
            params=RoutingParams(mode=base.mode, k_hop=base.k_hop, selected_by="default"),
            study_summary=study_summary,
            selected_by="default",
            warnings=sorted(set(warnings)),
        )
    completed = len(study.trials)
    study_summary["completed_trials"] = completed
    if completed == 0:
        warnings.append("HPO_FALLBACK")
        return HPOResult(
            params=RoutingParams(mode=base.mode, k_hop=base.k_hop, selected_by="default"),
            study_summary=study_summary,
            selected_by="default",
            warnings=sorted(set(warnings)),
        )
    best = study.best_params
    params = RoutingParams(
        mode=str(best["mode"]),
        alpha_base=float(best["alpha_base"]),
        w_homophily=float(best["w_homophily"]),
        threshold=float(best["threshold"]),
        budget_ratio=float(best["budget_ratio"]),
        k_hop=int(base.k_hop),
        selected_by="hpo",
    )
    study_summary["best_value"] = float(study.best_value)
    study_summary["best_params"] = dict(best)
    return HPOResult(
        params=params, study_summary=study_summary, selected_by="hpo", warnings=[]
    )
