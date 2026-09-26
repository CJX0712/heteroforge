"""pipeline/CLI/复现性/字符门禁 测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from heteroforge.cli import build_parser, main
from heteroforge.core.config import build_run_config
from heteroforge.pipeline.benchmark import haar_gate_check, run_benchmark
from heteroforge.pipeline.pipeline import HeteroForgePipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPipeline:
    def test_run_bucket_rows_complete(self):
        cfg = build_run_config({
            "seed": 5, "data": {"n_nodes": 200},
            "embed": {"backend": "svd", "dim": 16}, "gnn": {"n_epochs": 8},
        })
        pipeline = HeteroForgePipeline(cfg)
        rows = pipeline.run_bucket(200, 0.6, "node_classification")
        systems = {r["system"] for r in rows if r.get("kind") != "meta"}
        assert systems == {"haar", "channel_a", "channel_b", "feat_only"}
        for row in rows:
            if row.get("kind") != "meta":
                assert row["status"] == "OK"
                assert row["macro_f1"] is not None

    def test_link_prediction_bucket(self):
        cfg = build_run_config({
            "seed": 5, "embed": {"backend": "svd", "dim": 16}, "gnn": {"n_epochs": 8},
        })
        pipeline = HeteroForgePipeline(cfg)
        rows = pipeline.run_bucket(150, 0.6, "link_prediction")
        systems = {r["system"] for r in rows if r.get("kind") != "meta"}
        assert systems == {"haar", "channel_a", "channel_b"}
        for row in rows:
            if row.get("kind") != "meta":
                assert row["roc_auc"] is not None

    def test_reproducible_bitwise(self):
        cfg = build_run_config({
            "seed": 13, "data": {"n_nodes": 100},
            "embed": {"backend": "svd", "dim": 8}, "gnn": {"n_epochs": 5},
        })
        r1 = HeteroForgePipeline(build_run_config(cfg.to_dict())).run_bucket(100, 0.5, "node_classification")
        r2 = HeteroForgePipeline(build_run_config(cfg.to_dict())).run_bucket(100, 0.5, "node_classification")
        f1_1 = {r["system"]: r["macro_f1"] for r in r1 if r.get("kind") != "meta"}
        f1_2 = {r["system"]: r["macro_f1"] for r in r2 if r.get("kind") != "meta"}
        assert f1_1 == f1_2


class TestBenchmark:
    def test_small_grid_and_gate(self, tmp_path):
        payload = run_benchmark(
            {"seed": 3, "embed": {"backend": "svd", "dim": 16}, "gnn": {"n_epochs": 8}},
            buckets=[(120, 0.2), (120, 0.9)],
            tasks=["node_classification"],
            enforce_gate=False,
        )
        report = haar_gate_check(payload["rows"])
        assert report["per_bucket"]
        out = tmp_path / "benchmark.json"
        from heteroforge.eval.report import save_benchmark

        save_benchmark(payload, out)
        assert out.exists() and out.stat().st_size > 1000


class TestCLI:
    def test_parser(self):
        args = build_parser().parse_args(["benchmark", "--n", "400", "--homophily", "0.8"])
        assert args.command == "benchmark"

    def test_data_command_exit_zero(self, tmp_path):
        rc = main(["--seed", "9", "--output-dir", str(tmp_path), "data",
                   "--n", "60", "--homophily", "0.5", "--save", str(tmp_path / "g")])
        assert rc == 0

    def test_route_command_exit_zero(self):
        rc = main(["--seed", "9", "route", "--n", "60", "--homophily", "0.5"])
        assert rc == 0


class TestCharGate:
    """P0 字符门禁: 全仓源码禁 emoji / U+2500-U+257F / U+2192。"""

    FORBIDDEN = [(0x2500, 0x257F), (0x2190, 0x21FF), (0x1F300, 0x1FAFF), (0x2600, 0x27BF)]

    def _py_and_md_files(self):
        targets = []
        for pattern in ("heteroforge/**/*.py", "tests/**/*.py", "examples/*.py",
                        "scripts/*.py", "*.md", "docs/*.md"):
            targets.extend(p for p in REPO_ROOT.glob(pattern) if p.is_file())
        return targets

    def test_no_forbidden_chars(self):
        bad = []
        for path in self._py_and_md_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for ch in text:
                cp = ord(ch)
                if any(lo <= cp <= hi for lo, hi in self.FORBIDDEN):
                    bad.append((path.name, hex(cp)))
        assert not bad, f"forbidden chars: {bad[:10]}"

    def test_requirements_no_banned_packages(self):
        req = REPO_ROOT / "requirements.txt"
        if not req.exists():
            pytest.skip("requirements.txt not written yet")
        lines = [ln.strip() for ln in req.read_text(encoding="utf-8").lower().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        for banned in ("torch", "karateclub", "dgl", "gensim", "node2vec"):
            assert not any(ln.startswith(banned) for ln in lines), banned
