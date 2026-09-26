"""HeteroForge 包入口: Homophily-Aware Adaptive Routing 双通道图学习系统。

职责: 版本号、logger 初始化、依赖污染守卫(架构文档 12.6)。

Example:
    >>> import heteroforge
    >>> heteroforge.__version__
    '0.1.0'
    >>> heteroforge.__author__
    '晨星'
"""

from __future__ import annotations

import logging
import sys

__version__ = "0.1.0"
__author__ = "晨星"
__license__ = "Apache-2.0"

# 硬约束: 运行时不得出现 torch / torch_geometric / dgl / karateclub。
BANNED_PREFIXES: tuple[str, ...] = ("torch", "torch_geometric", "dgl", "karateclub")


def _scan_banned_modules() -> list[str]:
    """扫描 sys.modules, 返回已导入的被禁模块名列表。

    Returns:
        命中的被禁模块名列表; 无命中时为空列表。
    """
    hits: list[str] = []
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in BANNED_PREFIXES):
            hits.append(name)
    return hits


def _build_logger() -> logging.Logger:
    """构造包级 logger, 只挂 NullHandler, 由 CLI 决定是否加 handler。"""
    log = logging.getLogger("heteroforge")
    log.setLevel(logging.INFO)
    if not any(isinstance(h, logging.NullHandler) for h in log.handlers):
        log.addHandler(logging.NullHandler())
    return log


logger = _build_logger()


def assert_clean_runtime() -> None:
    """断言运行时未被污染; 命中被禁依赖时抛 E105。

    Raises:
        HeteroForgeError: E105, 存在被禁模块。
    """
    from heteroforge.core.errors import HeteroForgeError

    hits = _scan_banned_modules()
    if hits:
        raise HeteroForgeError("E105", "banned module already imported", {"modules": hits})


__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "BANNED_PREFIXES",
    "logger",
    "assert_clean_runtime",
]
