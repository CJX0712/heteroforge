"""端到端演示: 三档同配性 x 节点分类全系统对照, 落盘 benchmark.json。

用法(仓库根):
    python examples/run_demo.py [--fast]

--fast 使用 400 节点小图, 全程约 1-2 分钟(默认 800 节点约 5-8 分钟)。

Example:
    >>> from heteroforge.pipeline.benchmark import DEFAULT_BUCKETS
    >>> len(DEFAULT_BUCKETS) >= 7
    True
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroforge.core.config import build_run_config  # noqa: E402
from heteroforge.eval.report import render_table, save_benchmark  # noqa: E402
from heteroforge.pipeline.benchmark import haar_gate_check, run_benchmark  # noqa: E402


def main() -> int:
    """跑演示网格并打印门槛结论。"""
    parser = argparse.ArgumentParser(description="HeteroForge end-to-end demo")
    parser.add_argument("--fast", action="store_true", help="use 400-node graphs")
    args = parser.parse_args()
    n = 400 if args.fast else 800
    buckets = [(n, 0.05), (n, 0.5), (n, 0.95)]
    cfg = build_run_config({
        "seed": 42,
        "output_dir": "runs",
        "embed": {"backend": "node2vec"},
        "gnn": {"n_epochs": 30 if args.fast else 50},
    })
    payload = run_benchmark(cfg, buckets=buckets, tasks=["node_classification"],
                            enforce_gate=False)
    out_path = Path(__file__).resolve().parents[1] / "benchmark.json"
    save_benchmark(payload, out_path)
    rows = [r for r in payload["rows"] if r.get("kind") != "meta"]
    print(render_table(rows, ["homophily_target", "system", "macro_f1", "micro_f1",
                              "accuracy", "routed_fraction", "mode", "backend_a", "backend_b"]))
    gate = haar_gate_check(payload["rows"])
    print()
    print(f"[GATE] noninferior_all={gate['noninferior_all']} "
          f"mean_haar={gate['mean_haar']:.4f} "
          f"mean_best_single={gate['mean_best_single']:.4f} "
          f"mean_noninferior={gate['mean_noninferior']} "
          f"strictly_better_buckets={gate['strictly_better_bucket_count']}")
    print(f"[OK] artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
