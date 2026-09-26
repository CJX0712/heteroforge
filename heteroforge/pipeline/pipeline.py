"""端到端 pipeline: 数据 -> 同配性 -> 双通道 -> HAAR 路由 -> HPO -> 评测。

单一入口 HeteroForgePipeline.run_bucket(n_nodes, homophily, task) 返回该桶
全部系统(haar/channel_a/channel_b/feat_only)的 BenchmarkRow 字典列表。

关键纪律(架构 12.x):
- HAAR 级联 = 真跳过: 通道 B 只对 routed 节点做 k-hop 局部前向;
- 单通道基线 channel_b 使用全图前向(与级联同权重, 公平对照);
- test 每系统只触碰一次(test_touch_count=1);
- HPO 只用 train/val, 由 assert_no_test_access 静态拦截。

Example:
    >>> from heteroforge.pipeline.pipeline import HeteroForgePipeline
    >>> from heteroforge.core.config import build_run_config
    >>> p = HeteroForgePipeline(build_run_config({"seed": 7}))
    >>> p.config.seed
    7
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from heteroforge.core.config import RunConfig, build_run_config, config_hash
from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import BenchmarkRow, SplitSpec
from heteroforge.core.utils import derive_seed, start_peak_sampling, stop_peak_sampling
from heteroforge.data.split import split_edges, split_hash, train_adjacency_from_split
from heteroforge.data.synthetic import make_graph_data
from heteroforge.embed.channel_a import ChannelAEmbedder
from heteroforge.eval.link_prediction import evaluate_link_prediction
from heteroforge.eval.node_classification import evaluate_representation
from heteroforge.gnn.channel_b import SGCBackend
from heteroforge.graph.homophily import measure_homophily
from heteroforge.hpo.tuner import tune_routing
from heteroforge.router.haar import HAARRouter

NODE_SYSTEMS: tuple[str, ...] = ("haar", "channel_a", "channel_b", "feat_only")
LINK_SYSTEMS: tuple[str, ...] = ("haar", "channel_a", "channel_b")


class HeteroForgePipeline:
    """HAAR 端到端流水线, 由 RunConfig 驱动。"""

    def __init__(self, config: RunConfig | dict[str, Any] | None = None) -> None:
        self.config: RunConfig = (
            config if isinstance(config, RunConfig) else build_run_config(config)
        )
        self.seeds = self.config.seeds()

    # ------------------------------------------------------------------ data
    def _build_graph(self, n_nodes: int, homophily: float) -> Any:
        d = self.config.data
        f = self.config.features
        return make_graph_data(
            n_nodes=int(n_nodes),
            num_classes=int(d.num_classes),
            homophily=float(homophily),
            seed=int(self.config.seed),
            generator=str(d.generator),
            class_balance=str(d.class_balance),
            feature_signal=str(f.signal),
            feature_dim=int(f.dim),
            avg_degree=float(d.avg_degree),
            split_spec=SplitSpec(),
        )

    def _channel_a(self, graph: Any) -> Any:
        e = self.config.embed
        embedder = ChannelAEmbedder(
            preferred=str(e.backend),
            dim=int(e.dim),
            seed=int(self.seeds["embed_a"]),
            rss_budget_mb=float(self.config.limits.rss_budget_mb),
            timeout_sec=float(self.config.limits.per_algo_timeout_sec),
        )
        return embedder.fit_predict(graph, e)

    def _channel_b(self, graph: Any) -> SGCBackend:
        g = self.config.gnn
        return SGCBackend(
            seed=int(self.seeds["gnn_b"]),
            allow_fallback=bool(g.allow_fallback),
            timeout_sec=float(self.config.limits.per_algo_timeout_sec),
        ).fit(graph, g, labels_mask=graph.train_mask)

    # ------------------------------------------------------ node classification
    def run_node_classification(self, n_nodes: int, homophily: float) -> list[dict[str, Any]]:
        """单桶节点分类: 全系统对照 + HAAR 门槛原始数据。"""
        cfg = self.config
        started = time.perf_counter()
        start_peak_sampling(0.05)
        graph = self._build_graph(n_nodes, homophily)
        report = measure_homophily(graph.adjacency, graph.labels)
        dataset_id = (
            f"{graph.metadata.get('generator_effective', 'sbm')}-n{n_nodes}"
            f"-h{homophily}-s{cfg.seed}"
        )
        out_a = self._channel_a(graph)
        gnn = self._channel_b(graph)
        out_b_full = gnn.predict(graph, nodes=None, k_hop=int(cfg.gnn.k_hop))

        hpo = tune_routing(
            graph, report, out_a, gnn, seed=int(cfg.seed),
            gnn_config=cfg.gnn, max_iter=int(cfg.eval.max_iter),
        )
        router = HAARRouter()
        decision = router.route(graph, report, out_a, None, hpo.params)
        routed_idx = np.flatnonzero(decision.routed_mask)
        if routed_idx.size == 0:
            raise HeteroForgeError("E503", "cascade routed zero nodes")
        out_b_cas = gnn.predict(graph, nodes=routed_idx, k_hop=int(hpo.params.k_hop))
        fused = decision.fuse(
            out_a.fusion_matrix(), out_b_cas.fusion_matrix(), out_b_cas.computed_mask
        )

        reps: dict[str, tuple[np.ndarray, float]] = {
            "haar": (fused, float(decision.routed_fraction)),
            "channel_a": (np.asarray(out_a.fusion_matrix(), dtype=np.float64), 0.0),
            "channel_b": (np.asarray(out_b_full.fusion_matrix(), dtype=np.float64), 0.0),
            "feat_only": (np.asarray(graph.features, dtype=np.float64), 0.0),
        }
        peak = stop_peak_sampling()
        rows: list[dict[str, Any]] = []
        for system in NODE_SYSTEMS:
            rep, routed_fraction = reps[system]
            result = evaluate_representation(
                rep, graph, system=system, dataset_id=dataset_id,
                homophily_bucket=float(homophily),
                homophily_measured=float(report.edge_homophily),
                seed=int(cfg.seed), routed_fraction=routed_fraction,
                elapsed_sec=0.0, peak_rss_mb=float(peak),
            )
            row = self._to_row(
                result.metrics, task="node_classification", system=system,
                graph=graph, report=report, n_nodes=n_nodes, homophily=homophily,
                seed=int(cfg.seed), routed_fraction=routed_fraction,
                mode=str(hpo.params.mode), budget_ratio=float(hpo.params.budget_ratio),
                alpha_base=float(decision.alpha_base),
                backend_a=str(out_a.backend), backend_b=str(gnn.backend_id),
                elapsed=float(result.elapsed_sec), peak=float(peak),
                warning_codes=self._warnings_for(system, out_a, gnn, hpo.warnings, decision),
                hpo_selected_by=str(hpo.selected_by),
            )
            row["hpo"] = {
                "selected_by": str(hpo.selected_by),
                "completed_trials": int(hpo.study_summary.get("completed_trials", 0)),
                "best_value": hpo.study_summary.get("best_value"),
            }
            rows.append(row)
        rows.append({
            "task": "node_classification", "kind": "meta",
            "dataset_id": dataset_id, "n_nodes": int(n_nodes),
            "homophily_bucket": float(homophily),
            "homophily_measured": float(report.edge_homophily),
            "alpha_base": float(decision.alpha_base),
            "routed_fraction": float(decision.routed_fraction),
            "mode": str(hpo.params.mode),
            "threshold": float(hpo.params.threshold),
            "budget_ratio": float(hpo.params.budget_ratio),
            "elapsed_total_sec": float(time.perf_counter() - started),
            "peak_rss_mb": float(peak),
            "seed": int(cfg.seed),
            "config_hash": config_hash(cfg),
            "split_hash": split_hash(graph.train_mask, graph.val_mask, graph.test_mask),
        })
        return rows

    # --------------------------------------------------------- link prediction
    def run_link_prediction(self, n_nodes: int, homophily: float) -> list[dict[str, Any]]:
        """单桶链路预测: 生成树保护切分 + degree-aware negatives。"""
        cfg = self.config
        started = time.perf_counter()
        start_peak_sampling(0.05)
        full_graph = self._build_graph(n_nodes, homophily)
        edge_split = split_edges(
            full_graph.adjacency, SplitSpec(), seed=int(self.seeds["split_edge"])
        )
        train_adj = train_adjacency_from_split(full_graph.adjacency, edge_split)
        train_graph = full_graph
        train_graph.adjacency = train_adj
        train_graph.validate()
        report = measure_homophily(train_adj, full_graph.labels)
        dataset_id = (
            f"{full_graph.metadata.get('generator_effective', 'sbm')}-n{n_nodes}"
            f"-h{homophily}-s{cfg.seed}-link"
        )
        out_a = self._channel_a(train_graph)
        gnn = self._channel_b(train_graph)
        out_b_full = gnn.predict(train_graph, nodes=None, k_hop=int(cfg.gnn.k_hop))
        router = HAARRouter()
        from heteroforge.core.config import RoutingParams

        # 链路预测的配置选择在 val 边上进行(架构 12.4 validation_mode_select):
        # 端点纯A/纯B + 默认融合, 三者 val AUC 择优, test 只触碰一次。
        cand_params: dict[str, RoutingParams] = {
            "pure_a": RoutingParams(mode="cascade", alpha_base=1.0, w_homophily=0.0,
                                    threshold=0.2, budget_ratio=1.0, selected_by="validation_mode_select"),
            "pure_b": RoutingParams(mode="full_dual", alpha_base=0.0, w_homophily=0.0,
                                    threshold=0.2, budget_ratio=1.0, selected_by="validation_mode_select"),
            "haar_default": RoutingParams(selected_by="default"),
        }
        val_seed = int(derive_seed(cfg.seed, "eval"))
        best: dict[str, Any] | None = None
        for name, params in cand_params.items():
            decision_c = router.route(train_graph, report, out_a, None, params)
            routed_c = np.flatnonzero(decision_c.routed_mask)
            out_b_c = gnn.predict(train_graph, nodes=routed_c, k_hop=int(cfg.routing.k_hop))
            fused_c = decision_c.fuse(
                out_a.fusion_matrix(), out_b_c.fusion_matrix(), out_b_c.computed_mask
            )
            val_auc = evaluate_link_prediction(
                fused_c, train_adj, edge_split.val_edges,
                negative_ratio=float(cfg.eval.negative_ratio), seed=val_seed,
            )["roc_auc"]
            if best is None or val_auc > best["val_auc"]:
                best = {
                    "name": name, "val_auc": float(val_auc), "decision": decision_c,
                    "fused": fused_c, "routed_fraction": float(decision_c.routed_fraction),
                    "mode": str(decision_c.mode), "alpha_base": float(decision_c.alpha_base),
                    "budget_ratio": float(params.budget_ratio),
                }
        decision = best["decision"]
        fused = best["fused"]
        reps: dict[str, np.ndarray] = {
            "haar": np.asarray(fused, dtype=np.float64),
            "channel_a": np.asarray(out_a.fusion_matrix(), dtype=np.float64),
            "channel_b": np.asarray(out_b_full.fusion_matrix(), dtype=np.float64),
        }
        peak = stop_peak_sampling()
        rows: list[dict[str, Any]] = []
        for system in LINK_SYSTEMS:
            metrics = evaluate_link_prediction(
                reps[system], train_adj, edge_split.test_edges,
                negative_ratio=float(cfg.eval.negative_ratio),
                seed=int(derive_seed(cfg.seed, "eval")),
            )
            row = self._to_row(
                metrics, task="link_prediction", system=system,
                graph=train_graph, report=report, n_nodes=n_nodes, homophily=homophily,
                seed=int(cfg.seed), routed_fraction=float(best["routed_fraction"]),
                mode=str(best["mode"]), budget_ratio=float(best["budget_ratio"]),
                alpha_base=float(best["alpha_base"]),
                backend_a=str(out_a.backend), backend_b=str(gnn.backend_id),
                elapsed=0.0, peak=float(peak),
                warning_codes=self._warnings_for(system, out_a, gnn, [], decision),
                hpo_selected_by=str(best["name"]),
            )
            rows.append(row)
        rows.append({
            "task": "link_prediction", "kind": "meta",
            "dataset_id": dataset_id, "n_nodes": int(n_nodes),
            "homophily_bucket": float(homophily),
            "homophily_measured": float(report.edge_homophily),
            "alpha_base": float(best["alpha_base"]),
            "routed_fraction": float(best["routed_fraction"]),
            "mode": str(best["mode"]),
            "selected_config": str(best["name"]),
            "val_auc": float(best["val_auc"]),
            "elapsed_total_sec": float(time.perf_counter() - started),
            "peak_rss_mb": float(peak),
            "seed": int(cfg.seed),
            "config_hash": config_hash(cfg),
            "num_test_edges": int(edge_split.test_edges.shape[0]),
        })
        return rows

    # ------------------------------------------------------------------ shared
    def run_bucket(self, n_nodes: int, homophily: float, task: str) -> list[dict[str, Any]]:
        """按任务分发单桶执行。"""
        if task == "node_classification":
            return self.run_node_classification(n_nodes, homophily)
        if task == "link_prediction":
            return self.run_link_prediction(n_nodes, homophily)
        raise HeteroForgeError("E102", "unknown task", {"task": task})

    def _to_row(
        self,
        metrics: dict[str, Any],
        task: str,
        system: str,
        graph: Any,
        report: Any,
        n_nodes: int,
        homophily: float,
        seed: int,
        routed_fraction: float,
        mode: str,
        budget_ratio: float,
        alpha_base: float,
        backend_a: str,
        backend_b: str,
        elapsed: float,
        peak: float,
        warning_codes: str,
        hpo_selected_by: str,
    ) -> dict[str, Any]:
        meta = graph.metadata
        row = BenchmarkRow(
            generator=str(meta.get("generator", "sbm")),
            generator_effective=str(meta.get("generator_effective", "sbm")),
            n_nodes=int(n_nodes),
            n_edges=int(report.num_edges),
            homophily_target=float(homophily),
            homophily_measured=float(report.edge_homophily),
            num_classes=int(meta.get("num_classes", 2)),
            class_balance=str(meta.get("class_balance", "balanced")),
            feature_signal=str(meta.get("feature_signal", "strong")),
            task=task,
            system=system,
            seed=int(seed),
            macro_f1=metrics.get("macro_f1"),
            micro_f1=metrics.get("micro_f1"),
            accuracy=metrics.get("accuracy"),
            roc_auc=metrics.get("roc_auc"),
            average_precision=metrics.get("average_precision"),
            alpha_base=float(alpha_base),
            routed_fraction=float(routed_fraction),
            budget_ratio=float(budget_ratio),
            mode=mode,
            backend_a=backend_a,
            backend_b=backend_b,
            elapsed_sec=float(elapsed),
            peak_rss_mb=float(peak),
            status="OK",
            warning_codes=warning_codes,
            split_hash=str(meta.get("seed", seed)),
            config_hash=hpo_selected_by,
        )
        return row.to_dict()

    @staticmethod
    def _warnings_for(
        system: str, out_a: Any, gnn: Any, hpo_warnings: list[str], decision: Any
    ) -> str:
        codes: list[str] = []
        if system in ("haar", "channel_a"):
            codes.extend(str(w) for w in out_a.warnings)
        if system in ("haar", "channel_b"):
            codes.extend(str(w) for w in getattr(gnn, "warnings", []) if hasattr(gnn, "warnings"))
        if system == "haar":
            codes.extend(str(w) for w in hpo_warnings)
            codes.extend(str(w) for w in decision.warnings)
        return ";".join(sorted(set(codes)))
