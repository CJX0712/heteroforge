"""通道 A: node2vec -> spectral_rw -> svd 的降级阶梯, 输出 embedding 与 confidence。

可复现性要点(实测):
1. sknetwork.embedding.Spectral 内部的 LanczosEig 不接受 v0, ARPACK 随机初值导致同 seed
   两次结果差异可达 1.4(见 E305)。本模块改为 sknetwork.linalg.Laplacian + 显式 v0 的
   scipy eigsh, 数学等价且可复现。
2. sknetwork.embedding.SVD 的 solver 可注入, 用 _SeededLanczosSVD 提供确定性 v0。
3. gensim Word2Vec 与 node2vec 游走必须 workers=1, 并在调用前固定 random/np.random 种子。

Example:
    >>> import numpy as np, scipy.sparse as sp
    >>> from heteroforge.core.types import GraphData
    >>> from heteroforge.embed import ChannelAEmbedder
    >>> adj = sp.csr_matrix(np.array([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]]))
    >>> gd = GraphData(3, adj, np.zeros((3, 4)), np.array([0, 0, 1]),
    ...     np.array([True, True, False]), np.array([False, False, True]),
    ...     np.array([False, False, False]), {})
    >>> out = ChannelAEmbedder(preferred="svd", dim=4, seed=42).fit_predict(gd)
    >>> out.embedding.shape[1]
    4
"""

from __future__ import annotations

import random
import time
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sknetwork.embedding import SVD
from sknetwork.linalg import Laplacian, LanczosEig, LanczosSVD

from heteroforge.core.config import EmbedConfig
from heteroforge.core.errors import FALLBACK_COMPARABILITY_NOTE, HeteroForgeError
from heteroforge.core.types import ChannelOutput
from heteroforge.core.utils import estimate_bytes, peak_rss_mb, start_peak_sampling

LADDER: tuple[str, ...] = ("node2vec", "spectral_rw", "svd")
BACKEND_LICENSES: dict[str, str] = {
    "node2vec": "LGPL-2.1-only",
    "spectral_rw": "BSD-3-Clause",
    "svd": "BSD-3-Clause",
}


def _load_node2vec() -> Any:
    """延迟导入 node2vec; 缺失时抛 ImportError, 由 ladder 捕获并降级(E301)。"""
    from node2vec import Node2Vec

    return Node2Vec


def available_backends() -> dict[str, bool]:
    """探测三级 backend 的可用性; node2vec 缺失不影响退出码。"""
    status = {"node2vec": False, "spectral_rw": True, "svd": True}
    try:
        _load_node2vec()
        status["node2vec"] = True
    except ImportError:
        status["node2vec"] = False
    return status


def _start_vector(size: int, seed: int) -> np.ndarray:
    """构造确定性的 ARPACK 起始向量(单位范数)。"""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    vector = rng.standard_normal(max(1, int(size)))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.ones(max(1, int(size)))


class _SeededLanczosSVD(LanczosSVD):
    """为 sknetwork SVD 注入确定性起始向量的 solver。"""

    def __init__(self, seed: int = 0, n_iter: int | None = None, tol: float = 0.0) -> None:
        super().__init__(n_iter=n_iter, tol=tol)
        self._seed = int(seed)

    def fit(self, matrix: Any, n_components: int, init_vector: np.ndarray | None = None) -> Any:
        if init_vector is None:
            init_vector = _start_vector(min(matrix.shape), self._seed)
        return super().fit(matrix, n_components, init_vector=init_vector)


class _SeededLanczosEig(LanczosEig):
    """为谱分解注入确定性起始向量的 solver。"""

    def __init__(self, which: str = "SM", seed: int = 0, n_iter: int | None = None,
                 tol: float = 0.0) -> None:
        super().__init__(which=which, n_iter=n_iter, tol=tol)
        self._seed = int(seed)

    def fit(self, matrix: Any, n_components: int = 2) -> Any:
        v0 = _start_vector(matrix.shape[0], self._seed)
        self.eigenvalues_, self.eigenvectors_ = eigsh(
            matrix.astype(float), int(n_components), which=self.which,
            maxiter=self.n_iter, tol=self.tol, v0=v0,
        )
        return self


def _embed_spectral_rw(adjacency: sp.csr_matrix, dim: int, seed: int) -> np.ndarray:
    """随机游走归一化拉普拉斯谱嵌入, 与 Spectral(decomposition='rw') 等价但可复现。"""
    adjacency = sp.csr_matrix(adjacency)
    n = adjacency.shape[0]
    if n <= 3:
        return np.zeros((n, max(1, int(dim))), dtype=np.float64)
    k = max(1, min(int(dim), n - 2))
    k_fit = min(k + 1, n - 1)
    laplacian = Laplacian(adjacency, regularization=0.0, normalized_laplacian=True)
    solver = _SeededLanczosEig(which="SM", seed=seed)
    solver.fit(laplacian, k_fit)
    index = np.argsort(solver.eigenvalues_)[1:]
    vectors = solver.eigenvectors_[:, index]
    vectors = laplacian.norm_diag.dot(vectors)
    return np.ascontiguousarray(normalize(np.asarray(vectors, dtype=np.float64), p=2))


def _embed_svd(adjacency: sp.csr_matrix, dim: int, seed: int) -> np.ndarray:
    """截断 SVD 嵌入, solver 注入确定性起始向量。"""
    adjacency = sp.csr_matrix(adjacency)
    n = adjacency.shape[0]
    k = max(1, min(int(dim), n - 1))
    model = SVD(n_components=k, solver=_SeededLanczosSVD(seed=seed))
    return np.ascontiguousarray(np.asarray(model.fit_transform(adjacency), dtype=np.float64))


def _embed_node2vec(adjacency: sp.csr_matrix, config: EmbedConfig, seed: int) -> np.ndarray:
    """node2vec 游走(Grover 原实现) + gensim Word2Vec SGNS; workers=1 且固定 seed。

    注: 不调用 node2vec.Node2Vec.fit(), 因为 node2vec 0.3.2 向 gensim 传已移除的
    `size=` 参数(4.4.0 改名 vector_size), 触发 TypeError; 游走与训练解耦直调。
    """
    node2vec_cls = _load_node2vec()
    from gensim.models import Word2Vec

    adjacency = sp.csr_matrix(adjacency)
    n = adjacency.shape[0]
    graph = nx.from_scipy_sparse_array(adjacency)
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        walker = node2vec_cls(
            graph,
            dimensions=int(config.dim),
            walk_length=int(config.walk_length),
            num_walks=int(config.num_walks),
            p=float(config.p),
            q=float(config.q),
            workers=1,
            quiet=True,
        )
        walks = [list(map(str, walk)) for walk in walker.walks]
        model = Word2Vec(
            walks,
            vector_size=int(config.dim),
            window=int(config.window),
            min_count=1,
            sg=1,
            workers=1,
            seed=int(seed) % (2**31 - 1),
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    embedding = np.zeros((n, int(config.dim)), dtype=np.float64)
    for i in range(n):
        key = str(i)
        if key in model.wv.key_to_index:
            embedding[i] = np.asarray(model.wv[key], dtype=np.float64)
    return np.ascontiguousarray(embedding)


def _fit_head(embedding: np.ndarray, labels: np.ndarray, train_mask: np.ndarray,
              seed: int, max_iter: int) -> np.ndarray | None:
    """只用 train 标签拟合 LogisticRegression 头, 返回全图 proba。"""
    train = np.asarray(train_mask, dtype=bool)
    y = np.asarray(labels, dtype=np.int64)[train]
    if train.sum() < 4 or np.unique(y).size < 2:
        return None
    scaler = StandardScaler().fit(embedding[train])
    model = LogisticRegression(max_iter=int(max_iter), random_state=int(seed), n_jobs=1)
    model.fit(scaler.transform(embedding[train]), y)
    proba = model.predict_proba(scaler.transform(embedding))
    classes = int(model.classes_.max())
    if proba.shape[1] <= classes:
        return np.ascontiguousarray(proba.astype(np.float64))
    full = np.zeros((embedding.shape[0], classes + 1), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, col]
    return np.ascontiguousarray(full)


class ChannelAEmbedder:
    """通道 A 嵌入器: 单次探测 + ladder 降级 + 置信度头。"""

    def __init__(self, preferred: str = "node2vec", dim: int = 64, seed: int = 0,
                 rss_budget_mb: float = 8192.0, timeout_sec: float = 600.0) -> None:
        if preferred not in LADDER:
            raise HeteroForgeError("E102", "unknown channel A backend", {"value": preferred})
        self.preferred = preferred
        self.dim = int(dim)
        self.seed = int(seed)
        self.rss_budget_mb = float(rss_budget_mb)
        self.timeout_sec = float(timeout_sec)
        self._available = available_backends()

    @property
    def backend_id(self) -> str:
        """当前首选 backend id。"""
        return self.preferred

    @property
    def license_id(self) -> str:
        """当前首选 backend 的许可证。"""
        return BACKEND_LICENSES[self.preferred]

    def _run_backend(self, backend: str, graph: Any, config: EmbedConfig) -> np.ndarray:
        """执行单个 backend, 失败抛异常由上层降级。"""
        adjacency = graph.adjacency
        if backend == "node2vec":
            return _embed_node2vec(adjacency, config, config.seed or self.seed)
        if backend == "spectral_rw":
            return _embed_spectral_rw(adjacency, config.dim, config.seed or self.seed)
        return _embed_svd(adjacency, config.dim, config.seed or self.seed)

    def fit_predict(self, graph: Any, config: EmbedConfig | None = None) -> ChannelOutput:
        """计算全图嵌入与置信度; 任何失败都按 ladder 降级, 不抛业务异常。

        Args:
            graph: GraphData。
            config: 嵌入配置; None 时按实例默认值构造。

        Returns:
            ChannelOutput, channel='A'。

        Raises:
            HeteroForgeError: E306, ladder 全部失败。
        """
        cfg = config if config is not None else EmbedConfig(
            backend=self.preferred, dim=self.dim, seed=self.seed
        )
        graph.validate()
        n = int(graph.num_nodes)
        warnings: list[str] = []
        start_peak_sampling()
        started = time.perf_counter()
        est_mb = estimate_bytes(n, int(graph.adjacency.nnz / 2), cfg.dim,
                                max(2, graph.num_classes())) / (1024.0 * 1024.0)
        order = list(LADDER)
        start_at = order.index(self.preferred)
        if est_mb > self.rss_budget_mb and self.preferred == "node2vec":
            warnings.append("MEMORY_GUARD")
            start_at = max(start_at, 1)
        if not self._available.get(self.preferred, True):
            warnings.append("OPTIONAL_BACKEND_UNAVAILABLE")
        chosen: tuple[str, np.ndarray] | None = None
        for backend in order[start_at:]:
            if not self._available.get(backend, True):
                warnings.append("OPTIONAL_BACKEND_UNAVAILABLE")
                continue
            try:
                embedding = self._run_backend(backend, graph, cfg)
                if embedding.shape[0] != n or not np.isfinite(embedding).all():
                    raise HeteroForgeError("E304", "embedding shape or finiteness invalid")
                if time.perf_counter() - started > self.timeout_sec:
                    warnings.append("EMBED_TIMEOUT_FALLBACK")
                    continue
                chosen = (backend, embedding)
                break
            except Exception as exc:  # noqa: BLE001 - 任一 backend 失败都必须继续降级
                code = getattr(exc, "errcode", None)
                warnings.append(code if code else "OPTIONAL_BACKEND_UNAVAILABLE")
        if chosen is None:
            raise HeteroForgeError("E306", "all channel A backends failed",
                                   {"warnings": warnings, "num_nodes": n})
        backend, embedding = chosen
        fallback_used = backend != self.preferred
        proba = _fit_head(embedding, graph.labels, graph.train_mask,
                          cfg.seed or self.seed, cfg.head_max_iter)
        num_classes = max(2, graph.num_classes())
        if proba is not None:
            confidence = np.ascontiguousarray(proba.max(axis=1).astype(np.float64))
        else:
            confidence = np.full(n, 1.0 / num_classes, dtype=np.float64)
            warnings.append("OPTIONAL_BACKEND_UNAVAILABLE")
        if fallback_used:
            warnings.append(FALLBACK_COMPARABILITY_NOTE)
        elapsed = time.perf_counter() - started
        return ChannelOutput(
            channel="A",
            backend=backend,
            backend_license=BACKEND_LICENSES[backend],
            fallback_used=bool(fallback_used),
            fallback_from=self.preferred if fallback_used else None,
            embedding=np.ascontiguousarray(embedding.astype(np.float64)),
            node_ids=np.arange(n, dtype=np.int64),
            proba=proba,
            confidence=confidence,
            computed_mask=np.ones(n, dtype=bool),
            dim=int(embedding.shape[1]),
            seed=int(cfg.seed or self.seed),
            elapsed_sec=float(elapsed),
            peak_rss_mb=float(peak_rss_mb()),
            params={"dim": cfg.dim, "walk_length": cfg.walk_length, "num_walks": cfg.num_walks,
                    "window": cfg.window, "p": cfg.p, "q": cfg.q, "workers": cfg.workers},
            warnings=sorted(set(warnings)),
        )
