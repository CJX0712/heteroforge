"""路由层: HAAR 双门控与级联融合。

Example:
    >>> from heteroforge.router import HAARRouter, default_alpha_fn
    >>> import numpy as np
    >>> from heteroforge.core.config import RoutingParams
    >>> params = RoutingParams(mode="full_dual")
    >>> isinstance(params.budget_ratio, float)
    True
"""

from __future__ import annotations

from heteroforge.router.haar import (
    HAARRouter,
    assert_no_test_access,
    default_alpha_fn,
)

__all__ = ["HAARRouter", "default_alpha_fn", "assert_no_test_access"]
