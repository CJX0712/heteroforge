"""HeteroForge 统一错误码与异常。

错误码分段: E1xx 配置 / E2xx 数据 / E3xx 通道A / E4xx 通道B / E5xx 路由 / E6xx 评测。
退出码映射: E1xx->2, E2xx->3, E3xx->4, E4xx->5, E5xx->6, E6xx->7, E607->8, 未预期->9。
WARN 码不改变退出码(仍为 0), 但必须出现在 warnings.json 与汇总表中。

Example:
    >>> from heteroforge.core.errors import HeteroForgeError, exit_code_for
    >>> err = HeteroForgeError("E101", "unknown key", {"keys": ["foo"]})
    >>> err.errcode
    'E101'
    >>> exit_code_for("E101")
    2
    >>> exit_code_for("E607")
    8
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """错误码字符串枚举, 值即错误码本身。"""

    E101 = "E101"
    E102 = "E102"
    E103 = "E103"
    E104 = "E104"
    E105 = "E105"
    E106 = "E106"
    E201 = "E201"
    E202 = "E202"
    E203 = "E203"
    E204 = "E204"
    E205 = "E205"
    E206 = "E206"
    E207 = "E207"
    E208 = "E208"
    E209 = "E209"
    E301 = "E301"
    E302 = "E302"
    E303 = "E303"
    E304 = "E304"
    E305 = "E305"
    E306 = "E306"
    E401 = "E401"
    E402 = "E402"
    E403 = "E403"
    E404 = "E404"
    E405 = "E405"
    E406 = "E406"
    E501 = "E501"
    E502 = "E502"
    E503 = "E503"
    E504 = "E504"
    E505 = "E505"
    E601 = "E601"
    E602 = "E602"
    E603 = "E603"
    E604 = "E604"
    E605 = "E605"
    E606 = "E606"
    E607 = "E607"


ERROR_DESCRIPTIONS: dict[str, str] = {
    "E101": "CONFIG_KEY_UNKNOWN",
    "E102": "CONFIG_VALUE_OUT_OF_RANGE",
    "E103": "CONFIG_SCHEMA_MISMATCH",
    "E104": "CONFIG_UNREADABLE",
    "E105": "CLI_ARG_CONFLICT",
    "E106": "OUTPUT_PATH_TRAVERSAL",
    "E201": "GENERATION_FAILED",
    "E202": "HOMOPHILY_TARGET_MISS",
    "E203": "GRAPH_NOT_CONNECTED",
    "E204": "GRAPH_TOO_LARGE",
    "E205": "EDGE_SPLIT_LEAKAGE",
    "E206": "LABEL_SCHEMA_INVALID",
    "E207": "ADJACENCY_FORMAT_INVALID",
    "E208": "FEATURE_INVALID",
    "E209": "NODE_ORDER_MISMATCH",
    "E301": "OPTIONAL_BACKEND_UNAVAILABLE",
    "E302": "EMBED_TIMEOUT",
    "E303": "EMBED_MEMORY_GUARD",
    "E304": "EMBED_SHAPE_INVALID",
    "E305": "EMBED_NOT_REPRODUCIBLE",
    "E306": "ALL_EMBED_BACKENDS_FAILED",
    "E401": "GNN_FIT_FAILED",
    "E402": "GNN_LOSS_MISSING",
    "E403": "GNN_TIMEOUT",
    "E404": "GNN_SUBGRAPH_EMPTY",
    "E405": "GNN_PROBA_INVALID",
    "E406": "GNN_FALLBACK_USED",
    "E501": "ALPHA_OUT_OF_RANGE",
    "E502": "MODE_OR_REASON_UNKNOWN",
    "E503": "BUDGET_UNREACHABLE",
    "E504": "GATE_TEST_ACCESS",
    "E505": "FUSION_MASK_MISMATCH",
    "E601": "METRIC_TARGET_MISSING",
    "E602": "HAAR_NONINFERIORITY_VIOLATED",
    "E603": "HAAR_MEAN_NOT_BETTER",
    "E604": "TEST_EVALUATED_TWICE",
    "E605": "SKLEARN_MISMATCH",
    "E606": "ARTIFACT_WRITE_FAILED",
    "E607": "RESOURCE_LIMIT_EXCEEDED",
}

# WARN 码: 降级事件, 退出码仍为 0, 显示形如 W-OPTIONAL_BACKEND_UNAVAILABLE。
WARN_CODES: dict[str, str] = {
    "OPTIONAL_BACKEND_UNAVAILABLE": "可选后端不可用, 已按 ladder 降级",
    "EMBED_TIMEOUT_FALLBACK": "嵌入超时, 已降参数或切下一级后端",
    "MEMORY_GUARD": "预计内存超预算, 已切换更稀疏后端",
    "EMBED_NOT_REPRODUCIBLE": "同 seed 两次嵌入差异超过 1e-6",
    "LFR_TO_SBM": "LFR 三次失败, 已降级为同规模 SBM",
    "HOMOPHILY_TARGET_MISS": "实测 homophily 与目标误差超过 0.03",
    "GNN_FALLBACK_USED": "通道B 已降级为 prop_lr",
    "CASCADE_TO_FULL": "级联成本反增, 已切换 full_dual",
    "HPO_FALLBACK": "HPO 超时或无完成 trial, 使用显式默认参数",
    "HPO_NONREPRODUCIBLE": "同 seed 两次最优 trial 不一致",
    "PLATFORM_DRIFT": "跨平台同 seed 指标差异超过 1e-6",
}

# 降级可比性声明, 写入 warnings.json 的固定文案。
FALLBACK_COMPARABILITY_NOTE = (
    "result not directly comparable to original Node2Vec implementation"
)

_EXIT_BY_PREFIX: dict[str, int] = {"E1": 2, "E2": 3, "E3": 4, "E4": 5, "E5": 6, "E6": 7}
EXIT_UNKNOWN = 9


def exit_code_for(errcode: str) -> int:
    """错误码到进程退出码的映射。

    Args:
        errcode: 形如 "E101" 的错误码。

    Returns:
        退出码; E607 为 8, 未知码为 9。
    """
    if errcode == "E607":
        return 8
    prefix = errcode[:2] if len(errcode) >= 2 else ""
    return _EXIT_BY_PREFIX.get(prefix, EXIT_UNKNOWN)


def is_warn_code(code: str) -> bool:
    """判断是否为 WARN 码(去 "W-" 前缀后匹配)。"""
    return code.removeprefix("W-") in WARN_CODES


def warn_message(code: str, detail: str = "") -> str:
    """渲染一行 [WARN] 文本: 稳定 code + 原因。"""
    bare = code.removeprefix("W-")
    desc = WARN_CODES.get(bare, "unregistered warning")
    text = f"[WARN] W-{bare} {desc}"
    if detail:
        text = f"{text} | {detail}"
    return text


class HeteroForgeError(Exception):
    """HeteroForge 业务异常基类。

    Attributes:
        errcode: 错误码字符串, 如 "E501"。
        message: 人类可读说明, ASCII 优先, 可含中文。
        detail: 结构化上下文, 便于写 artifact 与排查。
    """

    def __init__(self, errcode: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(f"{errcode} {ERROR_DESCRIPTIONS.get(errcode, 'UNKNOWN')}: {message}")
        self.errcode = errcode
        self.message = message
        self.detail: dict[str, Any] = detail if detail is not None else {}

    @property
    def description(self) -> str:
        """错误码对应的稳定英文描述名。"""
        return ERROR_DESCRIPTIONS.get(self.errcode, "UNKNOWN")

    @property
    def exit_code(self) -> int:
        """该错误对应的进程退出码。"""
        return exit_code_for(self.errcode)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写盘的字典。"""
        return {
            "errcode": self.errcode,
            "description": self.description,
            "message": self.message,
            "detail": self.detail,
            "exit_code": self.exit_code,
        }

    def __repr__(self) -> str:
        return f"HeteroForgeError(errcode={self.errcode!r}, message={self.message!r})"
