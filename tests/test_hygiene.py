"""Local CI-style guard for repo rules that are easy to break by accident."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EM_DASH = chr(0x2014)
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".ipynb"}
SCAN_ROOTS = ["src", "tests", "scripts", "configs", "notebooks", "docs", "README.md"]
TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def scanned_files() -> list[Path]:
    files = []
    for name in SCAN_ROOTS:
        root = ROOT / name
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in candidates:
            if path.is_file() and path.suffix in SUFFIXES and "reference" not in path.parts:
                files.append(path)
    return files


@pytest.mark.parametrize("path", scanned_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dash_and_no_token_shaped_strings(path):
    text = path.read_text(encoding="utf-8")
    assert EM_DASH not in text
    assert not [p.pattern for p in TOKEN_PATTERNS if p.search(text)]
