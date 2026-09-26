"""RunConfig 配置树: YAML merge、越界校验、稳定 config_hash、按 tag 派生 seed。

配置优先级: CLI 覆盖 > YAML 覆盖 > 代码默认值。未知键一律 E101, 越界一律 E102。

Example:
    >>> from heteroforge.core.config import build_run_config, config_hash
    >>> cfg = build_run_config({"seed": 7, "data": {"homophily": 0.2}})
    >>> cfg.data.homophily
    0.2
    >>> cfg.seed
    7
    >>> len(config_hash(cfg))
    16
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.utils import SEED_TAGS, derive_seed, round_floats, stable_hash

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LimitsConfig:
    """资源上限, 见架构 11.2。"""

    per_algo_timeout_sec: float = 600.0
    per_run_timeout_sec: float = 900.0
    rss_budget_mb: float = 8192.0
    max_nodes: int = 5000
    max_edges: int = 200000
    dense_guard_nodes: int = 3000
    dense_guard_bytes: int = 2147483648


@dataclass(frozen=True)
class DataConfig:
    """合成图生成配置。"""

    generator: str = "sbm"
    n_nodes: int = 800
    num_classes: int = 5
    homophily: float = 0.5
    class_balance: str = "balanced"
    avg_degree: float = 10.0
    max_retry: int = 3
    tolerance: float = 0.03


@dataclass(frozen=True)
class FeatureConfig:
    """feature-label signal 配置。默认 weak: strong 下特征线性可分,
    图方法无增量空间, 评测不公平(实测 feat_only=1.0)。"""

    signal: str = "weak"
    dim: int = 64
    noise_scale: float = 1.0


@dataclass(frozen=True)
class EmbedConfig:
    """通道 A 配置。workers 恒为 1(K15)。"""

    backend: str = "node2vec"
    dim: int = 64
    seed: int = 0
    walk_length: int = 40
    num_walks: int = 5
    window: int = 5
    p: float = 1.0
    q: float = 1.0
    workers: int = 1
    head_max_iter: int = 1000


@dataclass(frozen=True)
class GNNConfig:
    """通道 B 配置。model 为 sgc 或 prop_lr。"""

    model: str = "sgc"
    seed: int = 0
    n_epochs: int = 50
    lr: float = 0.01
    k_hop: int = 2
    allow_fallback: bool = True


@dataclass(frozen=True)
class RoutingParams:
    """HAAR 路由参数。alpha_base/threshold 在 [0,1], budget_ratio 在 (0,1]。"""

    mode: str = "cascade"
    alpha_base: float = 0.5
    w_homophily: float = 1.0
    threshold: float = 0.6
    budget_ratio: float = 0.4
    k_hop: int = 2
    selected_by: str = "default"


@dataclass(frozen=True)
class EvalConfig:
    """评测配置。"""

    task: str = "node_classification"
    head: str = "logistic_regression"
    max_iter: int = 1000
    negative_ratio: float = 1.0
    degree_aware_negatives: bool = True


@dataclass(frozen=True)
class RunConfig:
    """顶层配置树, 构造后即冻结。"""

    seed: int = 42
    task: str = "node_classification"
    offline: bool = True
    output_dir: str = "runs"
    schema_version: str = SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    routing: RoutingParams = field(default_factory=RoutingParams)
    eval: EvalConfig = field(default_factory=EvalConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)

    def seeds(self) -> dict[str, int]:
        """按 K1 派生全部阶段子 seed。"""
        return {tag: derive_seed(self.seed, tag) for tag in SEED_TAGS}

    def to_dict(self) -> dict[str, Any]:
        """递归展开为普通字典(便于落盘与哈希)。"""
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = value if isinstance(value, (str, int, float, bool)) else dict(vars(value))
        return out


_SECTION_TYPES: dict[str, type] = {
    "data": DataConfig,
    "features": FeatureConfig,
    "embed": EmbedConfig,
    "gnn": GNNConfig,
    "routing": RoutingParams,
    "eval": EvalConfig,
    "limits": LimitsConfig,
}

_TOP_KEYS: set[str] = {"seed", "task", "offline", "output_dir", "schema_version", *_SECTION_TYPES.keys()}


def load_yaml(path: Path | str) -> dict[str, Any]:
    """读取 YAML; 解析失败或不可读抛 E104。"""
    target = Path(path)
    try:
        import yaml

        text = target.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except ImportError as exc:
        raise HeteroForgeError("E104", "PyYAML unavailable", {"path": str(target)}) from exc
    except OSError as exc:
        raise HeteroForgeError("E104", "config unreadable", {"path": str(target)}) from exc
    except Exception as exc:
        raise HeteroForgeError("E104", "yaml parse failed", {"path": str(target)}) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise HeteroForgeError("E104", "yaml root must be a mapping", {"path": str(target)})
    return data


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两棵配置字典, override 优先; 不修改入参。"""
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _check_unknown_keys(payload: dict[str, Any]) -> None:
    """未知键检测: 顶层与分段均拒绝未注册键(E101)。"""
    unknown_top = sorted(set(payload.keys()) - _TOP_KEYS)
    if unknown_top:
        raise HeteroForgeError("E101", "unknown config keys", {"keys": unknown_top})
    for section, payload_section in payload.items():
        if section not in _SECTION_TYPES or not isinstance(payload_section, dict):
            continue
        allowed = {f.name for f in fields(_SECTION_TYPES[section])}
        unknown = sorted(set(payload_section.keys()) - allowed)
        if unknown:
            raise HeteroForgeError("E101", f"unknown keys in section {section}", {"keys": unknown})


def _validate_bounds(cfg: RunConfig) -> None:
    """越界校验, 任一违反抛 E102。"""
    for name, value in (("alpha_base", cfg.routing.alpha_base), ("threshold", cfg.routing.threshold)):
        if not 0.0 <= float(value) <= 1.0:
            raise HeteroForgeError("E102", f"routing.{name} must be in [0,1]", {"value": value})
    if not 0.0 < float(cfg.routing.budget_ratio) <= 1.0:
        raise HeteroForgeError("E102", "routing.budget_ratio must be in (0,1]",
                               {"value": cfg.routing.budget_ratio})
    if not 0.0 <= float(cfg.routing.w_homophily) <= 4.0:
        raise HeteroForgeError("E102", "routing.w_homophily must be in [0,4]",
                               {"value": cfg.routing.w_homophily})
    if cfg.routing.mode not in ("cascade", "full_dual"):
        raise HeteroForgeError("E102", "routing.mode invalid", {"value": cfg.routing.mode})
    if not 0.0 <= float(cfg.data.homophily) <= 1.0:
        raise HeteroForgeError("E102", "data.homophily must be in [0,1]", {"value": cfg.data.homophily})
    if int(cfg.data.n_nodes) < 2:
        raise HeteroForgeError("E102", "data.n_nodes must be >= 2", {"value": cfg.data.n_nodes})
    if int(cfg.data.num_classes) < 2:
        raise HeteroForgeError("E102", "data.num_classes must be >= 2", {"value": cfg.data.num_classes})
    if int(cfg.embed.dim) < 1 or int(cfg.features.dim) < 1:
        raise HeteroForgeError("E102", "dim must be >= 1")
    if int(cfg.embed.workers) != 1:
        raise HeteroForgeError("E102", "embed.workers must be 1 for reproducibility (K15)",
                               {"value": cfg.embed.workers})


def build_run_config(override: dict[str, Any] | None = None) -> RunConfig:
    """按覆盖字典构造并校验 RunConfig。

    Args:
        override: 部分配置覆盖, 键需已注册。

    Returns:
        冻结后的 RunConfig。

    Raises:
        HeteroForgeError: E101 未知键 / E102 越界 / E103 schema 不符。
    """
    payload: dict[str, Any] = {} if override is None else copy.deepcopy(override)
    _check_unknown_keys(payload)
    version = str(payload.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise HeteroForgeError("E103", "schema_version mismatch",
                               {"expected": SCHEMA_VERSION, "got": version})
    cfg = RunConfig(
        seed=int(payload.get("seed", 42)),
        task=str(payload.get("task", "node_classification")),
        offline=bool(payload.get("offline", True)),
        output_dir=str(payload.get("output_dir", "runs")),
        schema_version=version,
        **{
            name: _SECTION_TYPES[name](**payload.get(name, {}))
            for name in _SECTION_TYPES
        },
    )
    _validate_bounds(cfg)
    return cfg


def config_hash(cfg: RunConfig) -> str:
    """稳定配置哈希: 先按 K7 舍入到 6 位再哈希。"""
    return stable_hash(round_floats(cfg.to_dict(), 6), 16)
