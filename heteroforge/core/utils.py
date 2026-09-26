"""跨切工具: seed 派生、稳定哈希、原子写、RSS 采样、计时。

不依赖 psutil(环境中没有), RSS 走 POSIX resource 与 Windows psapi 双路径。

Example:
    >>> from heteroforge.core.utils import derive_seed, stable_hash, rss_mb
    >>> s = derive_seed(42, "embed_a")
    >>> 0 <= s < 2**31 - 1
    True
    >>> stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})
    True
    >>> isinstance(rss_mb(), float)
    True
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from heteroforge.core.errors import HeteroForgeError

# K1: 固定 seed tag, 禁止跨阶段复用同一个 seed。
SEED_TAGS: tuple[str, ...] = (
    "data",
    "features",
    "split_node",
    "split_edge",
    "embed_a",
    "gnn_b",
    "router",
    "hpo",
    "head",
    "eval",
)

_SEED_MODULUS = 2**31 - 1


def derive_seed(master_seed: int, tag: str) -> int:
    """按 tag 派生子 seed(sha256, 禁止 Python hash)。

    Args:
        master_seed: 主种子。
        tag: 阶段标签, 取值限于 SEED_TAGS 或自定义字符串。

    Returns:
        [0, 2**31-2] 内的整数。
    """
    payload = f"{int(master_seed)}:{tag}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def stable_hash(obj: Any, length: int = 16) -> str:
    """对任意 JSON 可序列化对象求稳定哈希(键排序, 浮点保留 6 位)。

    Args:
        obj: 待哈希对象。
        length: 返回十六进制串长度。

    Returns:
        十六进制摘要串。
    """
    text = json.dumps(obj, sort_keys=True, default=_json_default, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _json_default(value: Any) -> Any:
    """json.dumps 的 default 回调: 处理 numpy 与 Path。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, np.ndarray):
        return [[round(float(v), 6) for v in row] for row in np.atleast_2d(value)]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """递归把浮点舍入到 ndigits 位, 用于 K7 的"舍入之后再哈希"。"""
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def atomic_write_json(path: Path | str, payload: Any, indent: int = 2) -> Path:
    """原子写 JSON: 写同目录 tmp -> fsync -> os.replace。

    Args:
        path: 目标路径。
        payload: 可 JSON 序列化对象。
        indent: 缩进。

    Returns:
        目标路径。

    Raises:
        HeteroForgeError: E606, 写入失败(保留 tmp 便于排查)。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="_tmp_", suffix=".json", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(target))
    except OSError as exc:
        raise HeteroForgeError("E606", "artifact write failed", {"path": str(target)}) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


def rss_mb() -> float:
    """读取当前进程 RSS(MB); 两条路径都失败时返回 nan 且不抛异常。"""
    try:
        if platform.system() == "Windows":
            return _rss_mb_windows()
        return _rss_mb_posix()
    except Exception:
        return float("nan")


def _rss_mb_posix() -> float:
    """POSIX: resource.getrusage; Linux 单位 KB, macOS 单位 bytes。"""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(peak) / (1024.0 * 1024.0)
    return float(peak) / 1024.0


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS 结构, WorkingSetSize 单位为 bytes。"""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _rss_mb_windows() -> float:
    """Windows: GetProcessMemoryInfo 读 WorkingSetSize(bytes)。"""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return float(counters.WorkingSetSize) / (1024.0 * 1024.0)


_PEAK_LOCK = threading.Lock()
_PEAK_STATE: dict[str, Any] = {"peak": float("nan"), "thread": None, "stop": None}


def _sample_loop(interval_sec: float) -> None:
    """采样线程主体: 每 interval_sec 更新一次峰值。"""
    while not _PEAK_STATE["stop"].is_set():
        value = rss_mb()
        with _PEAK_LOCK:
            current = _PEAK_STATE["peak"]
            if np.isnan(current) or (not np.isnan(value) and value > current):
                _PEAK_STATE["peak"] = value
            elif np.isnan(value) and np.isnan(current):
                _PEAK_STATE["peak"] = float("nan")
        _PEAK_STATE["stop"].wait(interval_sec)


def start_peak_sampling(interval_sec: float = 0.1) -> threading.Thread:
    """启动 100ms 采样的峰值 RSS 守护线程(重复调用幂等)。"""
    with _PEAK_LOCK:
        if _PEAK_STATE["thread"] is not None and _PEAK_STATE["thread"].is_alive():
            return _PEAK_STATE["thread"]
        _PEAK_STATE["stop"] = threading.Event()
        _PEAK_STATE["peak"] = rss_mb()
        thread = threading.Thread(
            target=_sample_loop, args=(interval_sec,), name="rss-peak-sampler", daemon=True
        )
        _PEAK_STATE["thread"] = thread
        thread.start()
        return thread


def stop_peak_sampling() -> float:
    """停止采样线程并返回已记录峰值(MB)。"""
    thread = _PEAK_STATE.get("thread")
    stop = _PEAK_STATE.get("stop")
    if stop is not None:
        stop.set()
    if thread is not None:
        thread.join(timeout=1.0)
    with _PEAK_LOCK:
        _PEAK_STATE["thread"] = None
        return float(_PEAK_STATE["peak"])


def peak_rss_mb() -> float:
    """返回采样到的峰值 RSS(MB); 未启动采样时回退到当前 RSS。"""
    with _PEAK_LOCK:
        peak = float(_PEAK_STATE["peak"])
    current = rss_mb()
    if np.isnan(peak):
        return current
    if np.isnan(current):
        return peak
    return max(peak, current)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """计时上下文: 进入记录 start, 退出写入 elapsed_sec(用 perf_counter, K14)。

    Yields:
        可变字典, 键 start / elapsed_sec / peak_rss_mb。
    """
    record: dict[str, float] = {"start": time.perf_counter(), "elapsed_sec": 0.0, "peak_rss_mb": 0.0}
    try:
        yield record
    finally:
        record["elapsed_sec"] = time.perf_counter() - record["start"]
        record["peak_rss_mb"] = peak_rss_mb()


def estimate_bytes(n_nodes: int, n_edges: int, dim: int, num_classes: int, dense_path: bool = False) -> float:
    """按架构 12.2 的公式估算内存字节数, 用于 E303 守卫。"""
    base = 8.0 * (2 * int(n_edges) + int(n_nodes))
    base += 8.0 * (int(n_nodes) * int(dim) * 2)
    base += 8.0 * (int(num_classes) * int(dim) + int(n_nodes) * int(num_classes))
    if dense_path:
        base += 16.0 * int(n_nodes) * int(n_nodes)
    return base


def safe_under_root(path: Path | str, root: Path | str) -> Path:
    """校验路径落在 root 内且不含 ".."; 违反抛 E106(不创建任何文件)。"""
    target = Path(path).resolve()
    base = Path(root).resolve()
    if ".." in Path(path).parts:
        raise HeteroForgeError("E106", "path traversal rejected", {"path": str(path)})
    if base not in target.parents and target != base:
        raise HeteroForgeError("E106", "path outside output root", {"path": str(path), "root": str(base)})
    return target
