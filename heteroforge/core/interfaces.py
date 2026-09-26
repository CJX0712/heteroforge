"""跨层 Protocol 定义: 上层模块只依赖这些协议, 不互相 import 具体实现。

使用 runtime_checkable Protocol; 成员一律声明为 property/callable, 保证 isinstance 可用。

Example:
    >>> from heteroforge.core.interfaces import Embedder
    >>> class Dummy:
    ...     backend_id = "dummy"
    ...     license_id = "MIT"
    ...     def fit_predict(self, graph, config=None):
    ...         return None
    >>> isinstance(Dummy(), Embedder)
    True
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from heteroforge.core.config import EmbedConfig, EvalConfig, GNNConfig, RoutingParams
from heteroforge.core.types import ChannelOutput, EvalResult, GraphData, HomophilyReport, RoutingDecision


@runtime_checkable
class Embedder(Protocol):
    """通道 A 协议: 全图一次性浅层结构嵌入。禁止使用 val/test 标签。"""

    @property
    def backend_id(self) -> str:
        """后端算法 id, 写真实算法名。"""
        ...

    @property
    def license_id(self) -> str:
        """后端许可证 id。"""
        ...

    def fit_predict(self, graph: GraphData, config: EmbedConfig | None = None) -> ChannelOutput:
        """计算嵌入与置信度; 失败按 ladder 降级, 禁止抛业务异常。"""
        ...


@runtime_checkable
class GNNBackend(Protocol):
    """通道 B 协议: 全局 fit 一次, 局部 k-hop forward。"""

    @property
    def backend_id(self) -> str:
        """后端算法 id: sgc 或 prop_lr。"""
        ...

    def fit(self, graph: GraphData, config: GNNConfig, labels_mask: np.ndarray) -> "GNNBackend":
        """在 full train graph 上训练全局权重, 只允许 labels_mask 内标签参与。"""
        ...

    def predict(
        self,
        graph: GraphData,
        nodes: np.ndarray | None = None,
        k_hop: int = 2,
        z_a_fallback: np.ndarray | None = None,
    ) -> ChannelOutput:
        """nodes 为 None 表示全图前向; 否则只对 nodes 做 k-hop 子图前向。"""
        ...


@runtime_checkable
class Router(Protocol):
    """HAAR 路由协议: 不得拟合任何 val/test 标签。"""

    def route(
        self,
        graph: GraphData,
        report: HomophilyReport,
        out_a: ChannelOutput,
        out_b: ChannelOutput | None = None,
        params: RoutingParams | None = None,
    ) -> RoutingDecision:
        """生成逐节点 alpha_vec、routed_mask 与 reason。"""
        ...

    def fuse(self, decision: RoutingDecision, out_a: ChannelOutput, out_b: ChannelOutput) -> np.ndarray:
        """按 decision 融合双通道, 校验 computed_mask(E505)。"""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """评测协议: 指标与 sklearn 对照。"""

    @property
    def task(self) -> str:
        """任务名: node_classification 或 link_prediction。"""
        ...

    def evaluate(
        self,
        graph: GraphData,
        output: ChannelOutput | np.ndarray,
        config: EvalConfig,
    ) -> EvalResult:
        """在指定 mask 上评估并写 test_touch_count。"""
        ...

    def cross_check_sklearn(self, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
        """与 sklearn.metrics 对照, 差值 > 1e-12 抛 E605。"""
        ...


@runtime_checkable
class Generator(Protocol):
    """合成图生成器协议。"""

    @property
    def generator_id(self) -> str:
        """生成器 id: sbm 或 lfr。"""
        ...

    def generate(
        self,
        n_nodes: int,
        num_classes: int,
        homophily: float,
        seed: int,
        **kwargs: object,
    ) -> tuple[nx.Graph, np.ndarray]:
        """返回 (networkx 图, 标签数组)。"""
        ...

    def max_retry(self) -> int:
        """允许的重试次数; LFR 返回 3, SBM 返回 1。"""
        ...


@runtime_checkable
class SubgraphExtractor(Protocol):
    """k-hop 诱导子图抽取协议(通道 B 局部前向用)。"""

    def extract(
        self, adjacency: csr_matrix, center: int, k_hop: int
    ) -> tuple[np.ndarray, csr_matrix, int]:
        """返回 (子图节点下标, 子图 CSR 邻接, 中心在子图中的位置)。"""
        ...


__all__ = ["Embedder", "GNNBackend", "Router", "Evaluator", "Generator", "SubgraphExtractor"]
