"""HAAR 路由器: 图级 alpha 门控 + 节点级预算级联。

设计语义(架构 4.4/12.4):
- alpha 融合口径: fused = alpha * z_a + (1 - alpha) * z_b, alpha 越大越信通道 A。
- 图级门控: alpha = clip(alpha_base + w_homophily * (0.5 - edge_homophily), 0, 1)。
  低同配图(edge_h -> 0)自动偏通道 A, 高同配图(edge_h -> 1)自动偏通道 B。
- 节点级级联(cascade): 被通道 A 接受的节点完全不计算通道 B(computed_mask=False),
  融合时读取未计算行抛 E505, 保证 routed_fraction 是真实算力节省。
- full_dual: 所有节点无条件双通道(reason=FORCED_FULL_DUAL)。

Example:
    >>> from heteroforge.router.haar import default_alpha_fn, assert_no_test_access
    >>> from heteroforge.core.config import RoutingParams
    >>> params = RoutingParams(alpha_base=0.5, w_homophily=1.0)
    >>> round(default_alpha_fn(0.2, params), 6)
    0.8
    >>> round(default_alpha_fn(0.8, params), 6)
    0.2
    >>> assert_no_test_access("train_mask", "val_mask") is None
    True
"""

from __future__ import annotations

import numpy as np

from heteroforge.core.config import RoutingParams
from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import (
    REASON_ACCEPTED,
    REASON_BUDGET_CAP,
    REASON_DEGREE_ZERO,
    REASON_FULL_DUAL,
    REASON_LOW_CONF,
    REASON_LOW_HOMOPHILY,
    ChannelOutput,
    GraphData,
    HomophilyReport,
    RoutingDecision,
)

_FORBIDDEN_FIELDS = ("test_mask", "test_labels", "test_indices", "test_mask_vec")


def assert_no_test_access(*used_fields: str) -> None:
    """静态审计路由器输入: 禁止任何 test 字段进入路由决策(E504)。

    Args:
        used_fields: 路由过程实际引用的字段名列表。

    Raises:
        HeteroForgeError: E504, 出现 test 相关字段。
    """
    for name in used_fields:
        if str(name).strip().lower() in _FORBIDDEN_FIELDS or "test" in str(name).lower():
            raise HeteroForgeError(
                "E504", "router must not access test data", {"field": str(name)}
            )


def default_alpha_fn(edge_homophily: float, params: RoutingParams) -> float:
    """图级 alpha 解析式: 低同配偏通道 A, 高同配偏通道 B。

    Args:
        edge_homophily: 图级 edge homophily, [0,1] 内, NaN 按中性 0.5 处理。
        params: 路由参数(alpha_base 与 w_homophily 参与)。

    Returns:
        [0,1] 内的 alpha, 乘在通道 A 输出上。
    """
    edge_h = float(edge_homophily)
    if not np.isfinite(edge_h):
        edge_h = 0.5
    alpha = float(params.alpha_base) + float(params.w_homophily) * (0.5 - edge_h)
    return float(np.clip(alpha, 0.0, 1.0))


class HAARRouter:
    """HAAR 双门控路由器。不拟合任何 val/test 标签(接口契约)。"""

    def __init__(self, alpha_fn=default_alpha_fn) -> None:
        self._alpha_fn = alpha_fn

    def route(
        self,
        graph: GraphData,
        report: HomophilyReport,
        out_a: ChannelOutput,
        out_b: ChannelOutput | None = None,
        params: RoutingParams | None = None,
    ) -> RoutingDecision:
        """生成逐节点 alpha_vec、routed_mask 与 reason。

        cascade 决策序列(预算 budget_max = ceil(budget_ratio * n)):
        1. 孤立节点强制路由(DEGREE_ZERO_ROUTED);
        2. 通道 A 低置信节点路由(LOW_CONFIDENCE_ROUTED); 超预算按 confidence 升序截断,
           被截断者回到通道 A(ACCEPTED_BY_CHANNEL_A);
        3. 预算有剩余时, 逐个纳入局部同配性最低的未路由节点(BUDGET_CAP_ROUTED);
        4. 若纳入的局部低同配节点本身满足 h_i 显著低于均值, 标 LOW_LOCAL_HOMOPHILY_ROUTED。

        Args:
            graph: GraphData(仅允许访问 adjacency/train_mask/labels 的 train 部分)。
            report: 同配性报告(提供 node_homophily_vec 快照)。
            out_a: 通道 A 输出(computed_mask 必须全 True)。
            out_b: 可选通道 B 输出(full_dual 模式或预算校核用; 可为 None)。
            params: 路由参数; None 时用默认 RoutingParams。

        Returns:
            RoutingDecision。

        Raises:
            HeteroForgeError: E501/E502/E503/E504/E505。
        """
        p = params if params is not None else RoutingParams()
        assert_no_test_access("adjacency", "train_mask", "val_mask", "node_homophily")
        graph.validate()
        n = int(graph.num_nodes)
        if out_a.computed_mask.shape != (n,) or not out_a.computed_mask.all():
            raise HeteroForgeError(
                "E505", "channel A must be computed for every node before routing"
            )
        alpha_base = self._alpha_fn(report.edge_homophily, p)
        alpha_vec = np.ones(n, dtype=np.float64)
        reason = np.full(n, REASON_ACCEPTED, dtype="<U32")
        warnings: list[str] = []

        if p.mode == "full_dual":
            routed_mask = np.ones(n, dtype=bool)
            reason[:] = REASON_FULL_DUAL
            alpha_vec[:] = alpha_base
            routed_fraction = 1.0
        elif p.mode == "cascade":
            routed_mask = np.zeros(n, dtype=bool)
            confidence = np.asarray(out_a.confidence, dtype=np.float64)
            h_vec = np.asarray(report.node_homophily_vec, dtype=np.float64)
            degrees = np.asarray(report.node_degrees, dtype=np.int64)
            budget_max = int(np.ceil(float(p.budget_ratio) * n))
            if budget_max < 1:
                raise HeteroForgeError(
                    "E503", "budget too small to route any node",
                    {"budget_ratio": float(p.budget_ratio), "n": n},
                )

            isolated = np.flatnonzero(degrees == 0)
            routed_mask[isolated] = True
            reason[isolated] = REASON_DEGREE_ZERO
            alpha_vec[isolated] = alpha_base

            low_conf = np.flatnonzero((confidence < float(p.threshold)) & ~routed_mask)
            if isolated.size + low_conf.size > budget_max:
                isolated_set = set(int(i) for i in isolated.tolist())
                order = np.argsort(np.where(np.isfinite(confidence), confidence, np.inf), kind="stable")
                cand = [int(i) for i in order
                        if int(i) in isolated_set or confidence[int(i)] < float(p.threshold)]
                keep = set(cand[:budget_max])
                for i in cand:
                    routed_mask[i] = i in keep
                    reason[i] = REASON_DEGREE_ZERO if i in isolated_set else REASON_LOW_CONF
                    alpha_vec[i] = alpha_base
                warnings.append("BUDGET_TRUNCATED")
            else:
                routed_mask[low_conf] = True
                reason[low_conf] = REASON_LOW_CONF
                alpha_vec[low_conf] = alpha_base

            remaining = budget_max - int(routed_mask.sum())
            if remaining > 0 and np.isfinite(h_vec).any():
                finite_h = h_vec[np.isfinite(h_vec)]
                h_mean = float(finite_h.mean())
                h_std = float(finite_h.std())
                unrouted = np.flatnonzero(~routed_mask & np.isfinite(h_vec))
                unrouted = unrouted[np.argsort(h_vec[unrouted], kind="stable")]
                for i in unrouted[:remaining]:
                    routed_mask[i] = True
                    alpha_vec[i] = alpha_base
                    reason[i] = (
                        REASON_LOW_HOMOPHILY
                        if h_vec[i] < h_mean - 0.5 * h_std
                        else REASON_BUDGET_CAP
                    )
            routed_fraction = float(routed_mask.sum()) / float(n)
        else:
            raise HeteroForgeError("E502", "unknown routing mode", {"mode": p.mode})

        decision = RoutingDecision(
            mode=str(p.mode),
            alpha_base=float(alpha_base),
            w_homophily=float(p.w_homophily),
            threshold=float(p.threshold),
            budget_ratio=float(p.budget_ratio),
            alpha_vec=alpha_vec,
            routed_mask=routed_mask,
            routed_fraction=float(routed_fraction),
            reason=reason,
            node_homophily=np.asarray(report.node_homophily_vec, dtype=np.float64),
            selected_by=str(p.selected_by),
            warnings=sorted(set(warnings)),
        )
        decision.validate()
        return decision

    def fuse(
        self, decision: RoutingDecision, out_a: ChannelOutput, out_b: ChannelOutput
    ) -> np.ndarray:
        """按 decision 融合双通道, 强校验 computed_mask(E505)。

        Args:
            decision: route() 产出的决策。
            out_a: 通道 A 输出。
            out_b: 通道 B 输出(未计算行必须为 0 且 computed_mask=False)。

        Returns:
            (n, d) 融合矩阵。
        """
        return decision.fuse(out_a.fusion_matrix(), out_b.fusion_matrix(), out_b.computed_mask)
