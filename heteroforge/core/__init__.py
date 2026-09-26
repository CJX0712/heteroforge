"""HeteroForge 内核层: 类型、错误码、配置、协议与跨切工具。

内核层依赖方向向下, 只允许 import stdlib 与 numpy/scipy, 禁止 import 任何上层包。

Example:
    >>> from heteroforge.core import HeteroForgeError, derive_seed
    >>> derive_seed(42, "data") == derive_seed(42, "data")
    True
    >>> derive_seed(42, "data") == derive_seed(42, "split_node")
    False
"""

from __future__ import annotations

from heteroforge.core.errors import (
    WARN_CODES,
    ErrorCode,
    HeteroForgeError,
    exit_code_for,
    is_warn_code,
    warn_message,
)
from heteroforge.core.types import (
    BenchmarkRow,
    ChannelOutput,
    EvalResult,
    GraphData,
    HomophilyReport,
    RoutingDecision,
    SplitSpec,
)
from heteroforge.core.utils import (
    SEED_TAGS,
    atomic_write_json,
    derive_seed,
    peak_rss_mb,
    rss_mb,
    stable_hash,
    start_peak_sampling,
    stop_peak_sampling,
)

__all__ = [
    "ErrorCode",
    "HeteroForgeError",
    "exit_code_for",
    "is_warn_code",
    "warn_message",
    "WARN_CODES",
    "GraphData",
    "HomophilyReport",
    "ChannelOutput",
    "RoutingDecision",
    "EvalResult",
    "BenchmarkRow",
    "SplitSpec",
    "SEED_TAGS",
    "derive_seed",
    "stable_hash",
    "atomic_write_json",
    "rss_mb",
    "peak_rss_mb",
    "start_peak_sampling",
    "stop_peak_sampling",
]
