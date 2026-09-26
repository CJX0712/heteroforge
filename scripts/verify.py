"""一键自检 verify: 分阶段可独立失败, 报告落盘 verify_report.json。

阶段: chars -> imports -> pytest -> smoke -> gate; 任一阶段失败立即短路。

用法(仓库根):
    python scripts/verify.py [--fast]

--fast 用 300 节点小图冒烟(约 1 分钟); 默认 800 节点(约 5 分钟)。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PY = sys.executable
STAGES: list[str] = ["chars", "imports", "pytest", "smoke", "gate"]


def run_chars() -> dict:
    from scripts.scan_chars import scan

    findings = scan()
    return {"findings": len(findings), "detail": findings[:10], "ok": not findings}


def run_imports() -> dict:
    errors: list[str] = []
    modules = [
        "heteroforge", "heteroforge.cli", "heteroforge.core.config",
        "heteroforge.data.synthetic", "heteroforge.graph.homophily",
        "heteroforge.embed.channel_a", "heteroforge.gnn.channel_b",
        "heteroforge.router.haar", "heteroforge.hpo.tuner",
        "heteroforge.eval.node_classification", "heteroforge.eval.link_prediction",
        "heteroforge.pipeline.benchmark",
    ]
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - 逐模块隔离失败
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return {"modules": len(modules), "errors": errors, "ok": not errors}


def run_pytest() -> dict:
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-W", "ignore", "tests",
         "--basetemp=data/.pytest-tmp", "-p", "no:cacheprovider"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800,
        env={"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
             "PATH": __import__("os").environ.get("PATH", "")},
    )
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    passed = "passed" in (proc.stdout or "")
    return {"exit_code": proc.returncode, "tail": tail, "ok": proc.returncode == 0 and passed}


def run_smoke(fast: bool) -> dict:
    from heteroforge.core.config import build_run_config
    from heteroforge.pipeline.pipeline import HeteroForgePipeline

    n = 300 if fast else 800
    cfg = build_run_config({
        "seed": 42,
        "data": {"n_nodes": n},
        "embed": {"backend": "node2vec", "dim": 32},
        "gnn": {"n_epochs": 20 if fast else 50},
    })
    started = time.perf_counter()
    rows = HeteroForgePipeline(cfg).run_bucket(n, 0.5, "node_classification")
    systems = {r["system"]: r for r in rows if r.get("kind") != "meta"}
    return {
        "n_nodes": n,
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "macro_f1": {k: round(float(v["macro_f1"]), 4) for k, v in systems.items()},
        "routed_fraction": round(float(
            [r for r in rows if r.get("kind") == "meta"][0]["routed_fraction"]), 3),
        "ok": all(v is not None for v in
                  (s["macro_f1"] for s in systems.values())),
    }


def run_gate() -> dict:
    from heteroforge.pipeline.benchmark import DEFAULT_BUCKETS, haar_gate_check, run_benchmark

    buckets = [(800, 0.05), (800, 0.5), (800, 0.95)]
    payload = run_benchmark(
        {"seed": 42, "embed": {"backend": "node2vec"}, "gnn": {"n_epochs": 30}},
        buckets=buckets, tasks=["node_classification"], enforce_gate=False,
    )
    gate = haar_gate_check(payload["rows"])
    out = REPO_ROOT / "benchmark.json"
    from heteroforge.eval.report import save_benchmark

    save_benchmark(payload, out)
    gate["artifact"] = str(out)
    gate["ok"] = bool(
        gate["noninferior_all"] and gate["mean_noninferior"]
        and gate["strictly_better_bucket_count"] >= 1
    )
    return gate


def main(argv: list[str] | None = None) -> int:
    """分阶段执行并落盘报告。"""
    fast = bool(argv and "--fast" in argv)
    report: dict = {"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for stage in STAGES:
        started = time.perf_counter()
        try:
            if stage == "chars":
                result = run_chars()
            elif stage == "imports":
                result = run_imports()
            elif stage == "pytest":
                result = run_pytest()
            elif stage == "smoke":
                result = run_smoke(fast)
            else:
                result = run_gate()
        except Exception as exc:  # noqa: BLE001 - 阶段异常即失败
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed_sec"] = round(time.perf_counter() - started, 1)
        report["stages"][stage] = result
        status = "PASS" if result.get("ok") else "FAIL"
        print(f"[{status}] {stage} ({result['elapsed_sec']}s)")
        if not result.get("ok"):
            break
    report["all_pass"] = all(
        s.get("ok") for s in report["stages"].values()
    ) and len(report["stages"]) == len(STAGES)
    out = REPO_ROOT / "verify_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"[{'OK' if report['all_pass'] else 'FAIL'}] verify report -> {out}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
