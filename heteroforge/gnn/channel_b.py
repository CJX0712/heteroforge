"""通道 B: 真单层 SGC(实测构造法)与 k-hop 局部前向, 以及 prop_lr 兜底。

两条硬事实(架构 6.6, 本机实测):
1. GNNClassifier(dims=[F, C]) 会展开成 2 层卷积, 不是 SGC; 唯一正确写法是显式传
   layers=[Convolution('Conv', C, activation='Identity', loss=CrossEntropy())], 此时 len(g.layers)==1。
   activation 必须传字符串 'Identity', 传 None 会 TypeError; 最后一层不挂 loss 会 ValueError(归一为 E402)。
2. GNNClassifier.predict() 是无参方法, 只返回 fit 时全图缓存结果; k-hop 子图推断必须用
   forward(sub_adj, sub_X)。predict_proba() 行和恰为 1.0, 可作 confidence 来源。

Example:
    >>> import numpy as np, scipy.sparse as sp
    >>> from heteroforge.core.config import GNNConfig
    >>> from heteroforge.core.types import GraphData
    >>> from heteroforge.gnn import SGCBackend
    >>> rng = np.random.default_rng(0)
    >>> adj = sp.csr_matrix(np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]))
    >>> gd = GraphData(3, adj, rng.random((3, 4)), np.array([0, 0, 1]),
    ...     np.array([True, True, False]), np.array([False, False, True]),
    ...     np.array([False, False, False]), {})
    >>> out = SGCBackend(seed=1).fit(gd, GNNConfig(n_epochs=3), gd.train_mask).predict(gd)
    >>> out.backend
    'sgc'
    >>> out.proba.shape
    (3, 2)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sknetwork.gnn import ADAM, Convolution, CrossEntropy, GNNClassifier

from heteroforge.core.config import GNNConfig
from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import ChannelOutput
from heteroforge.core.utils import peak_rss_mb, start_peak_sampling
from heteroforge.gnn.subgraph import symmetric_normalize

BACKEND_LICENSES: dict[str, str] = {"sgc": "BSD-3-Clause", "prop_lr": "BSD-3-Clause"}


def _build_sgc(num_classes: int, lr: float) -> GNNClassifier:
    """按实测配方构造真单层 SGC; 构造异常归一为 E402。"""
    try:
        layer = Convolution(
            "Conv",
            int(num_classes),
            activation="Identity",
            use_bias=True,
            normalization="both",
            self_embeddings=True,
            loss=CrossEntropy(),
        )
        return GNNClassifier(layers=[layer], optimizer=ADAM(float(lr)), early_stopping=False)
    except (ValueError, TypeError) as exc:
        raise HeteroForgeError("E402", "SGC construction failed (loss/activation contract)",
                               {"reason": str(exc)}) from exc


def _masked_labels(labels: np.ndarray, labels_mask: np.ndarray) -> np.ndarray:
    """把 mask 外的标签置 -1(sknetwork 忽略负值)。"""
    out = np.asarray(labels, dtype=np.int64).copy()
    out[~np.asarray(labels_mask, dtype=bool)] = -1
    return out


def _confidence_from(proba: np.ndarray) -> np.ndarray:
    """置信度口径: max_c proba[i, c](K5), 越大越可信。"""
    return np.ascontiguousarray(np.asarray(proba, dtype=np.float64).max(axis=1))


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的逐行 softmax(支持 (n,C) 或 (C,) 输入)。

    SGC 最后一层为 Identity, forward 输出是 logits, 必须转概率才能与
    predict_proba() 同口径。
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def sknetwork_prop_features(adjacency: Any, features: np.ndarray) -> np.ndarray:
    """复刻 sknetwork Convolution(normalization='both', self_embeddings=True)的传播语义。

    顺序与 Kipf 原文相反: 先 D^-1/2 A D^-1/2 对称归一化, 再加单位自环
    (已对 sknetwork gnn/layer.py forward 源码逐行核对)。

    Returns:
        (n, d) 传播后的特征矩阵 prop_X; 全图一次 O(E*d), 之后任意节点的
        分类头 logits = prop_X[node] @ W + b, 与全图 forward 数学一致。
    """
    adjacency = sp.csr_matrix(adjacency)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    d_inv = np.where(degrees > 0, 1.0 / np.sqrt(np.maximum(degrees, 1e-12)), 0.0)
    d_mat = sp.diags(d_inv)
    a_norm = (d_mat @ adjacency @ d_mat).tocsr()
    a_prime = (a_norm + sp.identity(adjacency.shape[0], format="csr")).tocsr()
    return np.ascontiguousarray(np.asarray(a_prime @ features, dtype=np.float64))


class SGCBackend:
    """单层 SGC 后端: 全图 fit 一次, 局部 k-hop forward。"""

    def __init__(self, seed: int = 0, allow_fallback: bool = True,
                 timeout_sec: float = 600.0) -> None:
        self.seed = int(seed)
        self.allow_fallback = bool(allow_fallback)
        self.timeout_sec = float(timeout_sec)
        self._model: GNNClassifier | None = None
        self._num_classes: int = 0
        self._backend_id = "sgc"

    @property
    def backend_id(self) -> str:
        """当前实际 backend: 'sgc' 或降级后的 'prop_lr'。"""
        return self._backend_id

    @property
    def license_id(self) -> str:
        """backend 许可证。"""
        return BACKEND_LICENSES[self._backend_id]

    def _assert_fitted(self) -> GNNClassifier:
        """取已训练模型, 未训练抛 E401。"""
        if self._model is None:
            raise HeteroForgeError("E401", "SGC backend is not fitted")
        return self._model

    def fit(self, graph: Any, config: GNNConfig, labels_mask: np.ndarray) -> "SGCBackend":
        """在 full train graph 上训练全局权重; 只有 labels_mask 内标签参与。

        Raises:
            HeteroForgeError: E401 fit 失败且不允许兜底; E403 超时。
        """
        graph.validate()
        started = time.perf_counter()
        num_classes = max(2, graph.num_classes())
        seed = int(config.seed or self.seed)
        try:
            model = _build_sgc(num_classes, config.lr)
            model.fit(
                graph.adjacency,
                np.ascontiguousarray(graph.features),
                _masked_labels(graph.labels, labels_mask),
                n_epochs=int(config.n_epochs),
                random_state=seed,
            )
        except HeteroForgeError:
            raise
        except Exception as exc:  # noqa: BLE001 - 归一为 E401
            if not self.allow_fallback:
                raise HeteroForgeError("E401", "SGC fit failed", {"reason": str(exc)}) from exc
            self._backend_id = "prop_lr"
            return self._fit_prop_lr(graph, config, labels_mask)
        if time.perf_counter() - started > self.timeout_sec:
            if self.allow_fallback:
                self._backend_id = "prop_lr"
                return self._fit_prop_lr(graph, config, labels_mask)
            raise HeteroForgeError("E403", "SGC fit timed out")
        self._model = model
        self._num_classes = num_classes
        self._n_epochs = int(config.n_epochs)
        self._backend_id = "sgc"
        self._prop_cache: dict[str, Any] = {}
        return self

    def _prop_matrix(self, graph: Any) -> np.ndarray:
        """取(或计算)全图传播特征; 以 (n, nnz, feat_shape) 为缓存键。"""
        key = f"{graph.adjacency.shape[0]}:{graph.adjacency.nnz}:{graph.features.shape}"
        cached = self._prop_cache.get("key")
        if cached != key:
            self._prop_cache = {
                "key": key,
                "matrix": sknetwork_prop_features(graph.adjacency, graph.features),
            }
        return self._prop_cache["matrix"]  # type: ignore[no-any-return]

    def _fit_prop_lr(self, graph: Any, config: GNNConfig, labels_mask: np.ndarray) -> "SGCBackend":
        """prop_lr 兜底: k 次归一化传播 + LogisticRegression(纯 scipy/sklearn)。"""
        self._model = None
        self._prop = PropLRBackend(seed=int(config.seed or self.seed))
        self._prop.fit(graph, config, labels_mask)
        self._num_classes = max(2, graph.num_classes())
        return self

    def predict(
        self,
        graph: Any,
        nodes: np.ndarray | None = None,
        k_hop: int = 2,
        z_a_fallback: np.ndarray | None = None,
    ) -> ChannelOutput:
        """全图前向(nodes=None)或仅对 nodes 执行分类头(级联共享传播)。

        传播矩阵 prop_X 全图预计算一次并与 sknetwork forward 数学一致;
        级联的算力节省 = 只对 nodes 行执行 O(d*C) 的头计算,
        未计算行保持全 0 并由 computed_mask 标记(融合层禁止读取, E505)。

        Args:
            graph: GraphData。
            nodes: 需要计算的节点下标; None 表示全图。
            k_hop: 兼容参数(SGC 单层共享传播, 头计算不依赖 k_hop)。
            z_a_fallback: 兼容参数(传播语义下任意行均可计算, 恒为多余)。

        Returns:
            ChannelOutput, channel='B', computed_mask 标记真实计算过的行。

        Raises:
            HeteroForgeError: E405 proba 非法。
        """
        graph.validate()
        n = int(graph.num_nodes)
        started = time.perf_counter()
        start_peak_sampling()
        warnings: list[str] = []
        if self._backend_id == "prop_lr":
            out = self._prop.predict(graph, nodes=nodes, k_hop=k_hop, z_a_fallback=z_a_fallback)
            out.warnings = sorted(set(list(out.warnings) + ["GNN_FALLBACK_USED"]))
            return out
        model = self._assert_fitted()
        layer = model.layers[0]
        weight = np.asarray(layer.weight, dtype=np.float64)
        bias = np.asarray(layer.bias, dtype=np.float64) if layer.use_bias else None
        prop_x = self._prop_matrix(graph)
        if nodes is None:
            logits = prop_x @ weight if bias is None else prop_x @ weight + bias
            proba = _stable_softmax(logits)
            computed = np.ones(n, dtype=bool)
        else:
            proba = np.zeros((n, self._num_classes), dtype=np.float64)
            computed = np.zeros(n, dtype=bool)
            targets = np.unique(np.asarray(nodes, dtype=np.int64))
            logits = prop_x[targets] @ weight if bias is None else prop_x[targets] @ weight + bias
            proba[targets] = _stable_softmax(logits)
            computed[targets] = True
        if proba.shape[0] != n:
            raise HeteroForgeError("E405", "proba row count mismatch")
        if not np.isfinite(proba).all():
            raise HeteroForgeError("E405", "proba contains NaN or Inf")
        # 校验只针对 computed 行(未计算行保持全 0, 融合层禁止读取)。
        computed = np.ascontiguousarray(np.asarray(computed, dtype=bool))
        if computed.any():
            rowsum = proba[computed].sum(axis=1)
            drift = float(np.max(np.abs(rowsum - 1.0)))
            if drift > 1e-6:
                raise HeteroForgeError("E405", "proba rows must sum to 1", {"max_drift": drift})
        return ChannelOutput(
            channel="B",
            backend=self._backend_id,
            backend_license=BACKEND_LICENSES[self._backend_id],
            fallback_used=self._backend_id != "sgc",
            fallback_from="sgc" if self._backend_id != "sgc" else None,
            embedding=np.ascontiguousarray(proba.astype(np.float64)),
            node_ids=np.arange(n, dtype=np.int64),
            proba=np.ascontiguousarray(proba.astype(np.float64)),
            confidence=_confidence_from(proba) if nodes is None else self._masked_confidence(proba, computed),
            computed_mask=np.ascontiguousarray(computed),
            dim=int(self._num_classes),
            seed=self.seed,
            elapsed_sec=float(time.perf_counter() - started),
            peak_rss_mb=float(peak_rss_mb()),
            params={"k_hop": int(k_hop), "n_epochs": getattr(self, "_n_epochs", 0),
                    "layers": len(model.layers)},
            warnings=sorted(set(warnings)),
        )

    @staticmethod
    def _masked_confidence(proba: np.ndarray, computed: np.ndarray) -> np.ndarray:
        """局部前向时未计算行的置信度置 0(禁止被融合读取, 由 E505 拦截)。"""
        confidence = np.zeros(proba.shape[0], dtype=np.float64)
        confidence[np.asarray(computed, dtype=bool)] = _confidence_from(
            proba[np.asarray(computed, dtype=bool)]
        )
        return np.ascontiguousarray(confidence)


class PropLRBackend:
    """prop_lr: D^{-1/2} A D^{-1/2} 的 k 次传播 + LogisticRegression(无神经网络)。"""

    def __init__(self, seed: int = 0, k_hop: int = 2) -> None:
        self.seed = int(seed)
        self.k_hop = int(k_hop)
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._num_classes = 0

    @property
    def backend_id(self) -> str:
        """固定为 'prop_lr', 绝不写成 'sgc'。"""
        return "prop_lr"

    @property
    def license_id(self) -> str:
        """BSD-3-Clause。"""
        return BACKEND_LICENSES["prop_lr"]

    def _propagate(self, graph: Any, k_hop: int) -> np.ndarray:
        """k 次对称归一化传播后的特征。"""
        adj = symmetric_normalize(graph.adjacency, add_self_loop=True)
        x = np.ascontiguousarray(graph.features)
        for _ in range(max(1, int(k_hop))):
            x = np.ascontiguousarray(adj @ x)
        return x

    def fit(self, graph: Any, config: GNNConfig, labels_mask: np.ndarray) -> "PropLRBackend":
        """在 train 标签上拟合传播特征 + LR。"""
        graph.validate()
        mask = np.asarray(labels_mask, dtype=bool)
        propagated = self._propagate(graph, config.k_hop or self.k_hop)
        y = np.asarray(graph.labels, dtype=np.int64)
        self._num_classes = max(2, graph.num_classes())
        if mask.sum() < 4 or np.unique(y[mask]).size < 2:
            raise HeteroForgeError("E401", "prop_lr needs at least two labelled classes")
        self._scaler = StandardScaler().fit(propagated[mask])
        self._model = LogisticRegression(max_iter=1000, random_state=self.seed, n_jobs=1)
        self._model.fit(self._scaler.transform(propagated[mask]), y[mask])
        return self

    def predict(
        self,
        graph: Any,
        nodes: np.ndarray | None = None,
        k_hop: int = 2,
        z_a_fallback: np.ndarray | None = None,
    ) -> ChannelOutput:
        """全图或局部计算 proba; prop_lr 对子图与全图结果一致, 但 computed_mask 仍按 nodes 标记。"""
        if self._model is None or self._scaler is None:
            raise HeteroForgeError("E401", "prop_lr backend is not fitted")
        started = time.perf_counter()
        n = int(graph.num_nodes)
        propagated = self._propagate(graph, k_hop)
        proba = np.zeros((n, self._num_classes), dtype=np.float64)
        proba[:, self._model.classes_.astype(np.int64)] = self._model.predict_proba(
            self._scaler.transform(propagated)
        )
        computed = np.ones(n, dtype=bool) if nodes is None else np.zeros(n, dtype=bool)
        if nodes is not None:
            computed[np.asarray(nodes, dtype=np.int64)] = True
            proba[~computed] = 0.0
        return ChannelOutput(
            channel="B",
            backend="prop_lr",
            backend_license=BACKEND_LICENSES["prop_lr"],
            fallback_used=True,
            fallback_from="sgc",
            embedding=np.ascontiguousarray(proba.astype(np.float64)),
            node_ids=np.arange(n, dtype=np.int64),
            proba=np.ascontiguousarray(proba.astype(np.float64)),
            confidence=self._masked_confidence(proba, computed),
            computed_mask=np.ascontiguousarray(computed),
            dim=int(self._num_classes),
            seed=self.seed,
            elapsed_sec=float(time.perf_counter() - started),
            peak_rss_mb=float(peak_rss_mb()),
            params={"k_hop": int(k_hop)},
            warnings=["GNN_FALLBACK_USED"],
        )

    @staticmethod
    def _masked_confidence(proba: np.ndarray, computed: np.ndarray) -> np.ndarray:
        """未计算行置信度置 0。"""
        confidence = np.zeros(proba.shape[0], dtype=np.float64)
        mask = np.asarray(computed, dtype=bool)
        if mask.any():
            confidence[mask] = _confidence_from(proba[mask])
        return np.ascontiguousarray(confidence)
