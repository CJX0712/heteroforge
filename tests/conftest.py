"""pytest 公共 fixture: 小图与种子(全部 function scope, 避免 GraphData 可变污染)。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroforge.data.synthetic import make_graph_data  # noqa: E402


@pytest.fixture()
def small_graph():
    """120 节点、5 类、homophily 0.6 的合成图。"""
    return make_graph_data(n_nodes=120, num_classes=5, homophily=0.6, seed=11)


@pytest.fixture()
def tiny_graph():
    """40 节点快图(通道/路由单测用)。"""
    return make_graph_data(n_nodes=40, num_classes=3, homophily=0.4, seed=3)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
