"""HeteroForge CLI: data / embed / train / route / eval / benchmark 六组命令。

退出码映射见 core.errors.exit_code_for; 所有输出为纯 ASCII 表格。

Example:
    >>> from heteroforge.cli import build_parser
    >>> parser = build_parser()
    >>> args = parser.parse_args(["benchmark", "--n", "400", "--homophily", "0.8"])
    >>> args.command
    'benchmark'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from heteroforge.core.config import build_run_config, load_yaml
from heteroforge.core.errors import HeteroForgeError

PROGRAM = "heteroforge"


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器(六组子命令)。"""
    parser = argparse.ArgumentParser(prog=PROGRAM, description="HeteroForge: homophily-aware adaptive graph routing")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--seed", type=int, default=None, help="master seed override")
    parser.add_argument("--output-dir", type=str, default=None, help="artifact output dir")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="generate a synthetic graph and report homophily")
    p_data.add_argument("--n", type=int, default=800)
    p_data.add_argument("--homophily", type=float, default=0.5)
    p_data.add_argument("--generator", type=str, default="sbm", choices=["sbm", "lfr"])
    p_data.add_argument("--save", type=str, default=None, help="save GraphData to dir")

    p_embed = sub.add_parser("embed", help="run channel A and report backend/confidence")
    p_embed.add_argument("--n", type=int, default=800)
    p_embed.add_argument("--homophily", type=float, default=0.5)
    p_embed.add_argument("--backend", type=str, default="node2vec")
    p_embed.add_argument("--dim", type=int, default=64)

    p_train = sub.add_parser("train", help="fit channel B (SGC) and report train metrics")
    p_train.add_argument("--n", type=int, default=800)
    p_train.add_argument("--homophily", type=float, default=0.5)
    p_train.add_argument("--epochs", type=int, default=50)

    p_route = sub.add_parser("route", help="run HAAR routing and report decision stats")
    p_route.add_argument("--n", type=int, default=800)
    p_route.add_argument("--homophily", type=float, default=0.5)
    p_route.add_argument("--mode", type=str, default="cascade", choices=["cascade", "full_dual"])
    p_route.add_argument("--budget", type=float, default=0.4)
    p_route.add_argument("--threshold", type=float, default=0.6)

    p_eval = sub.add_parser("eval", help="run one bucket end-to-end and report metrics")
    p_eval.add_argument("--n", type=int, default=800)
    p_eval.add_argument("--homophily", type=float, default=0.5)
    p_eval.add_argument("--task", type=str, default="node_classification",
                        choices=["node_classification", "link_prediction"])

    p_bench = sub.add_parser("benchmark", help="run the full benchmark grid")
    p_bench.add_argument("--n", type=int, nargs="*", default=None)
    p_bench.add_argument("--homophily", type=float, nargs="*", default=None)
    p_bench.add_argument("--task", type=str, nargs="*", default=None)
    p_bench.add_argument("--out", type=str, default="benchmark.json")
    p_bench.add_argument("--no-gate", action="store_true", help="skip HAAR gate enforcement")
    return parser


def _merged_config(args: argparse.Namespace) -> Any:
    override: dict[str, Any] = {}
    if args.config:
        override = load_yaml(args.config)
    if args.seed is not None:
        override["seed"] = int(args.seed)
    if args.output_dir:
        override["output_dir"] = str(args.output_dir)
    return build_run_config(override)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def cmd_data(args: argparse.Namespace) -> int:
    """data 子命令: 生成图 + 同配性报告。"""
    cfg = _merged_config(args)
    from heteroforge.data.synthetic import make_graph_data
    from heteroforge.graph.homophily import measure_homophily

    graph = make_graph_data(
        n_nodes=int(args.n), homophily=float(args.homophily), seed=int(cfg.seed),
        generator=str(args.generator),
    )
    report = measure_homophily(graph.adjacency, graph.labels)
    payload = {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(report.num_edges),
        "edge_homophily": round(float(report.edge_homophily), 6),
        "node_homophily": round(float(report.node_homophily), 6),
        "degree_assortativity": round(float(report.degree_assortativity), 6),
        "isolated_count": int(report.isolated_count),
        "generator_effective": str(graph.metadata.get("generator_effective")),
        "warnings": list(graph.metadata.get("warnings", [])),
    }
    if args.save:
        from heteroforge.data.loader import save_graph_data

        path = save_graph_data(graph, args.save, root=cfg.output_dir)
        payload["saved_to"] = str(path)
    _print_json(payload)
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """embed 子命令: 通道 A 嵌入与置信度摘要。"""
    cfg = _merged_config(args)
    from heteroforge.core.config import EmbedConfig
    from heteroforge.data.synthetic import make_graph_data
    from heteroforge.embed.channel_a import ChannelAEmbedder, available_backends

    graph = make_graph_data(n_nodes=int(args.n), homophily=float(args.homophily), seed=int(cfg.seed))
    cfg_embed = EmbedConfig(backend=str(args.backend), dim=int(args.dim), seed=int(cfg.seeds()["embed_a"]))
    out = ChannelAEmbedder(preferred=str(args.backend), dim=int(args.dim), seed=int(cfg.seeds()["embed_a"])).fit_predict(graph, cfg_embed)
    _print_json({
        "backend": str(out.backend),
        "backend_license": str(out.backend_license),
        "fallback_used": bool(out.fallback_used),
        "fallback_from": out.fallback_from,
        "dim": int(out.dim),
        "mean_confidence": round(float(out.confidence.mean()), 6),
        "available_backends": {k: bool(v) for k, v in available_backends().items()},
        "elapsed_sec": round(float(out.elapsed_sec), 4),
        "peak_rss_mb": round(float(out.peak_rss_mb), 1),
        "warnings": list(out.warnings),
    })
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """train 子命令: 通道 B 训练摘要。"""
    cfg = _merged_config(args)
    from heteroforge.core.config import GNNConfig
    from heteroforge.data.synthetic import make_graph_data
    from heteroforge.gnn.channel_b import SGCBackend

    graph = make_graph_data(n_nodes=int(args.n), homophily=float(args.homophily), seed=int(cfg.seed))
    gnn_cfg = GNNConfig(seed=int(cfg.seeds()["gnn_b"]), n_epochs=int(args.epochs))
    backend = SGCBackend(seed=int(cfg.seeds()["gnn_b"])).fit(graph, gnn_cfg, labels_mask=graph.train_mask)
    out = backend.predict(graph, nodes=None, k_hop=int(gnn_cfg.k_hop))
    _print_json({
        "backend": str(backend.backend_id),
        "fallback_used": bool(out.fallback_used),
        "dim": int(out.dim),
        "mean_confidence": round(float(out.confidence.mean()), 6),
        "elapsed_sec": round(float(out.elapsed_sec), 4),
    })
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """route 子命令: HAAR 决策统计。"""
    cfg = _merged_config(args)
    from heteroforge.core.config import EmbedConfig, GNNConfig, RoutingParams
    from heteroforge.data.synthetic import make_graph_data
    from heteroforge.embed.channel_a import ChannelAEmbedder
    from heteroforge.gnn.channel_b import SGCBackend
    from heteroforge.graph.homophily import measure_homophily
    from heteroforge.router.haar import HAARRouter

    graph = make_graph_data(n_nodes=int(args.n), homophily=float(args.homophily), seed=int(cfg.seed))
    report = measure_homophily(graph.adjacency, graph.labels)
    out_a = ChannelAEmbedder(seed=int(cfg.seeds()["embed_a"])).fit_predict(
        graph, EmbedConfig(seed=int(cfg.seeds()["embed_a"]))
    )
    params = RoutingParams(
        mode=str(args.mode), budget_ratio=float(args.budget),
        threshold=float(args.threshold), selected_by="cli",
    )
    decision = HAARRouter().route(graph, report, out_a, None, params)
    gnn = SGCBackend(seed=int(cfg.seeds()["gnn_b"])).fit(
        graph, GNNConfig(seed=int(cfg.seeds()["gnn_b"])), labels_mask=graph.train_mask
    )
    routed_idx = np.flatnonzero(decision.routed_mask)
    out_b = gnn.predict(graph, nodes=routed_idx, k_hop=int(params.k_hop))
    fused = decision.fuse(out_a.fusion_matrix(), out_b.fusion_matrix(), out_b.computed_mask)
    reasons, counts = np.unique(decision.reason, return_counts=True)
    _print_json({
        "mode": str(decision.mode),
        "alpha_base": round(float(decision.alpha_base), 6),
        "routed_fraction": round(float(decision.routed_fraction), 6),
        "reason_counts": {str(r): int(c) for r, c in zip(reasons, counts)},
        "fused_shape": list(map(int, fused.shape)),
        "warnings": list(decision.warnings),
    })
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """eval 子命令: 单桶端到端。"""
    cfg = _merged_config(args)
    from heteroforge.pipeline.pipeline import HeteroForgePipeline
    from heteroforge.eval.report import render_table

    pipeline = HeteroForgePipeline(cfg)
    rows = pipeline.run_bucket(int(args.n), float(args.homophily), str(args.task))
    meta = [r for r in rows if r.get("kind") == "meta"]
    system_rows = [r for r in rows if r.get("kind") != "meta"]
    columns = ["task", "system", "macro_f1", "micro_f1", "accuracy",
               "roc_auc", "average_precision", "routed_fraction", "backend_a", "backend_b"]
    print(render_table(system_rows, columns))
    _print_json({"meta": meta, "rows": system_rows})
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """benchmark 子命令: 网格 + 门槛 + 落盘。"""
    cfg = _merged_config(args)
    from heteroforge.pipeline.benchmark import haar_gate_check, run_benchmark
    from heteroforge.eval.report import render_table, save_benchmark

    buckets = None
    if args.n or args.homophily:
        ns = args.n or [800]
        hs = args.homophily or [0.5]
        buckets = [(int(n), float(h)) for n in ns for h in hs]
    payload = run_benchmark(
        cfg, buckets=buckets, tasks=args.task or None, enforce_gate=not args.no_gate
    )
    save_benchmark(payload, args.out)
    system_rows = [r for r in payload["rows"] if r.get("kind") != "meta"]
    print(render_table(system_rows, ["task", "system", "homophily_target", "macro_f1",
                                     "roc_auc", "routed_fraction", "mode"]))
    gate = payload["haar_gate"]
    print(f"[GATE] noninferior_all={gate['noninferior_all']} "
          f"mean_haar={gate['mean_haar']:.4f} mean_single={gate['mean_best_single']:.4f} "
          f"mean_noninferior={gate['mean_noninferior']} "
          f"strictly_better_buckets={gate['strictly_better_bucket_count']}")
    print(f"[OK] benchmark saved to {Path(args.out).resolve()}")
    return 0


HANDLERS = {
    "data": cmd_data,
    "embed": cmd_embed,
    "train": cmd_train,
    "route": cmd_route,
    "eval": cmd_eval,
    "benchmark": cmd_benchmark,
}


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 错误码到退出码的统一映射。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(HANDLERS[args.command](args))
    except HeteroForgeError as exc:
        print(f"[ERROR] {exc.errcode} {exc.description}: {exc.message} | {exc.detail}",
              file=sys.stderr)
        return int(exc.exit_code)
    except Exception as exc:  # noqa: BLE001 - 未预期错误统一退出码 9
        print(f"[ERROR] UNEXPECTED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
