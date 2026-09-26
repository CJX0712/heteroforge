"""core 层测试: config 校验、错误码、utils 确定性。"""

from __future__ import annotations

import json

import pytest

from heteroforge.core.config import build_run_config, config_hash
from heteroforge.core.errors import HeteroForgeError, exit_code_for, is_warn_code, warn_message
from heteroforge.core.utils import atomic_write_json, derive_seed, peak_rss_mb, stable_hash


class TestConfig:
    def test_defaults(self):
        cfg = build_run_config()
        assert cfg.seed == 42 and cfg.data.n_nodes == 800

    def test_unknown_key_raises_e101(self):
        with pytest.raises(HeteroForgeError) as ei:
            build_run_config({"nope": 1})
        assert ei.value.errcode == "E101"

    def test_unknown_section_key_raises_e101(self):
        with pytest.raises(HeteroForgeError) as ei:
            build_run_config({"data": {"bad_key": 1}})
        assert ei.value.errcode == "E101"

    def test_out_of_range_raises_e102(self):
        with pytest.raises(HeteroForgeError) as ei:
            build_run_config({"routing": {"alpha_base": 1.5}})
        assert ei.value.errcode == "E102"
        with pytest.raises(HeteroForgeError):
            build_run_config({"routing": {"budget_ratio": 0.0}})

    def test_workers_must_be_one(self):
        with pytest.raises(HeteroForgeError) as ei:
            build_run_config({"embed": {"workers": 4}})
        assert ei.value.errcode == "E102"

    def test_schema_mismatch_e103(self):
        with pytest.raises(HeteroForgeError) as ei:
            build_run_config({"schema_version": "9.9"})
        assert ei.value.errcode == "E103"

    def test_config_hash_stable(self):
        a = build_run_config({"seed": 7})
        b = build_run_config({"seed": 7})
        assert config_hash(a) == config_hash(b)
        assert len(config_hash(a)) == 16

    def test_seeds_derived_distinct(self):
        cfg = build_run_config({"seed": 42})
        seeds = cfg.seeds()
        assert len(set(seeds.values())) == len(seeds)
        assert seeds == build_run_config({"seed": 42}).seeds()


class TestErrors:
    def test_exit_code_map(self):
        assert exit_code_for("E101") == 2
        assert exit_code_for("E204") == 3
        assert exit_code_for("E306") == 4
        assert exit_code_for("E405") == 5
        assert exit_code_for("E505") == 6
        assert exit_code_for("E602") == 7
        assert exit_code_for("E607") == 8
        assert exit_code_for("EXXX") == 9

    def test_warn_codes(self):
        assert is_warn_code("W-LFR_TO_SBM")
        assert "LFR" in warn_message("W-LFR_TO_SBM")

    def test_error_to_dict(self):
        err = HeteroForgeError("E204", "too large", {"n": 9})
        assert err.exit_code == 3
        assert json.loads(json.dumps(err.to_dict()))["errcode"] == "E204"


class TestUtils:
    def test_derive_seed_stable_and_tagged(self):
        assert derive_seed(42, "data") == derive_seed(42, "data")
        assert derive_seed(42, "data") != derive_seed(42, "gnn_b")

    def test_stable_hash_json_safe(self):
        assert stable_hash({"a": [1, 2]}) == stable_hash({"a": [1, 2]})

    def test_atomic_write_json(self, tmp_path):
        target = tmp_path / "out.json"
        atomic_write_json(target, {"x": 1.23456789})
        assert json.loads(target.read_text(encoding="utf-8"))["x"] == 1.23456789

    def test_peak_rss_positive_or_nan(self):
        value = peak_rss_mb()
        assert value > 0 or value != value
