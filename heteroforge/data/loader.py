"""GraphData 持久化: npz + json, 原子写, schema 校验, 路径穿越拒绝(E106)。

Example:
    >>> import numpy as np, scipy.sparse as sp, tempfile
    >>> from heteroforge.core.types import GraphData
    >>> from heteroforge.data.loader import save_graph_data, load_graph_data
    >>> adj = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    >>> gd = GraphData(2, adj, np.zeros((2, 2)), np.array([0, 1]),
    ...     np.array([True, False]), np.array([False, True]), np.array([False, False]), {})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = save_graph_data(gd, tmp)
    ...     back = load_graph_data(tmp)
    ...     back.num_nodes
    2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from heteroforge.core.errors import HeteroForgeError
from heteroforge.core.types import GraphData
from heteroforge.core.utils import atomic_write_json, safe_under_root

SCHEMA_NAME = "heteroforge-graph"
SCHEMA_VERSION = "1.0"

ADJACENCY_FILE = "adjacency.npz"
META_FILE = "graph.meta.json"


def save_graph_data(graph: GraphData, output_dir: Path | str, root: Path | str | None = None) -> Path:
    """落盘图数据; 先写数据再写 meta, meta 写完才视为完整。

    Args:
        graph: 待保存图。
        output_dir: 目标目录。
        root: 允许写入的根, 给出时做路径穿越校验。

    Returns:
        输出目录路径。
    """
    graph.validate()
    target = Path(output_dir)
    if root is not None:
        target = safe_under_root(target, root)
    target.mkdir(parents=True, exist_ok=True)
    adj_path = target / ADJACENCY_FILE
    tmp_adj = target / f"_tmp_{ADJACENCY_FILE}"
    try:
        sp.save_npz(str(tmp_adj), graph.adjacency.tocsr().astype(np.float64))
        tmp_adj.replace(adj_path)
        np.savez_compressed(
            str(target / "_tmp_arrays.npz"),
            features=np.ascontiguousarray(graph.features),
            labels=np.ascontiguousarray(graph.labels),
            train_mask=np.ascontiguousarray(graph.train_mask),
            val_mask=np.ascontiguousarray(graph.val_mask),
            test_mask=np.ascontiguousarray(graph.test_mask),
        )
        (target / "_tmp_arrays.npz").replace(target / "arrays.npz")
    except OSError as exc:
        raise HeteroForgeError("E606", "artifact write failed", {"path": str(target)}) from exc
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "num_nodes": int(graph.num_nodes),
        "node_hash": graph.node_hash(),
        "metadata": graph.metadata,
    }
    atomic_write_json(target / META_FILE, payload)
    return target


def load_graph_data(input_dir: Path | str, root: Path | str | None = None) -> GraphData:
    """读取图数据并做 schema 校验。

    Args:
        input_dir: 数据目录。
        root: 允许读取的根, 给出时做路径穿越校验。

    Returns:
        校验通过的 GraphData。

    Raises:
        HeteroForgeError: E104 meta 缺失或 schema 不符; E207 邻接格式错。
    """
    source = Path(input_dir)
    if root is not None:
        source = safe_under_root(source, root)
    meta_path = source / META_FILE
    if not meta_path.exists():
        raise HeteroForgeError("E104", "graph meta missing", {"path": str(meta_path)})
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeteroForgeError("E104", "graph meta unreadable", {"path": str(meta_path)}) from exc
    if meta.get("schema") != SCHEMA_NAME or meta.get("schema_version") != SCHEMA_VERSION:
        raise HeteroForgeError("E104", "graph schema mismatch", {"meta": meta})
    try:
        adjacency = sp.load_npz(str(source / ADJACENCY_FILE)).tocsr()
        arrays = np.load(str(source / "arrays.npz"))
    except (OSError, ValueError, KeyError) as exc:
        raise HeteroForgeError("E104", "graph arrays unreadable", {"path": str(source)}) from exc
    if not sp.isspmatrix_csr(adjacency):
        raise HeteroForgeError("E207", "stored adjacency is not CSR")
    graph = GraphData(
        num_nodes=int(meta["num_nodes"]),
        adjacency=adjacency,
        features=np.asarray(arrays["features"], dtype=np.float64),
        labels=np.asarray(arrays["labels"], dtype=np.int64),
        train_mask=np.asarray(arrays["train_mask"], dtype=bool),
        val_mask=np.asarray(arrays["val_mask"], dtype=bool),
        test_mask=np.asarray(arrays["test_mask"], dtype=bool),
        metadata=dict(meta.get("metadata", {})),
    )
    graph.validate()
    if graph.node_hash() != meta.get("node_hash"):
        raise HeteroForgeError("E207", "node hash mismatch after load")
    return graph
