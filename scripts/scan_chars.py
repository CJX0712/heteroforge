"""P0 字符门禁: 全仓源码与文档禁 emoji / 框线字符(U+2500-257F) / 箭头(U+2192)。

用法(仓库根):
    python scripts/scan_chars.py [paths ...]
默认扫描 heteroforge/ tests/ examples/ scripts/ *.md docs/*.md。
发现违规时打印 (文件, 行号, 码点) 并以退出码 1 结束。

Example:
    >>> from scripts.scan_chars import FORBIDDEN_RANGES, first_violation
    >>> first_violation("clean text -> ok") is None
    True
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (lo, hi, 说明) 全部为禁用码点区间。
FORBIDDEN_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x2500, 0x257F, "box drawing"),
    (0x2190, 0x21FF, "arrow"),
    (0x1F300, 0x1FAFF, "emoji"),
    (0x2600, 0x27BF, "misc symbol"),
)


def first_violation(text: str) -> tuple[int, str] | None:
    """返回第一个违规字符的 (码点, 说明); 无违规返回 None。"""
    for ch in text:
        cp = ord(ch)
        for lo, hi, label in FORBIDDEN_RANGES:
            if lo <= cp <= hi:
                return cp, label
    return None


def scan(paths: list[Path] | None = None) -> list[tuple[str, int, int, str]]:
    """扫描目标文件, 返回违规列表。"""
    if paths is None:
        targets: list[Path] = []
        for pattern in ("heteroforge/**/*.py", "tests/**/*.py", "examples/*.py",
                        "scripts/*.py", "*.md", "docs/*.md"):
            targets.extend(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    else:
        targets = paths
    findings: list[tuple[str, int, int, str]] = []
    for path in sorted(set(targets)):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            hit = first_violation(line)
            if hit is not None:
                cp, label = hit
                findings.append((str(path.relative_to(REPO_ROOT)), lineno, cp, label))
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 有违规退出 1, 干净退出 0。"""
    args = argv if argv is not None else sys.argv[1:]
    paths = [Path(a) if Path(a).is_absolute() else REPO_ROOT / a for a in args] or None
    findings = scan(paths)
    for file, lineno, cp, label in findings:
        print(f"[VIOLATION] {file}:{lineno} U+{cp:04X} ({label})")
    print(f"[SCAN] files scanned, findings = {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
