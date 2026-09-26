"""报告渲染与落盘: benchmark 表格 + JSON artifact。

Example:
    >>> from heteroforge.eval.report import render_table
    >>> rows = [{"system": "haar", "macro_f1": 0.9, "routed_fraction": 0.4}]
    >>> text = render_table(rows, columns=["system", "macro_f1"])
    >>> "haar" in text and "macro_f1" in text
    True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from heteroforge.core.utils import atomic_write_json


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """渲染定宽对齐表格(CLI 输出, 纯 ASCII)。"""
    if not rows:
        return "(no rows)"
    widths = {c: len(str(c)) for c in columns}
    rendered: list[list[str]] = []
    for row in rows:
        cells = []
        for c in columns:
            value = row.get(c)
            if isinstance(value, float):
                text = f"{value:.4f}" if value == value else "nan"
            elif value is None:
                text = "null"
            else:
                text = str(value)
            widths[c] = max(widths[c], len(text))
            cells.append(text)
        rendered.append(cells)
    header = " | ".join(str(c).ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    lines = [header, sep]
    for cells in rendered:
        lines.append(" | ".join(cell.ljust(widths[c]) for cell, c in zip(cells, columns)))
    return "\n".join(lines)


def save_benchmark(payload: dict[str, Any], path: Path | str) -> Path:
    """benchmark 结果落盘(原子写, JSON 安全)。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(payload, default=str, ensure_ascii=False))
    return atomic_write_json(target, clean)
