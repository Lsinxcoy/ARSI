"""Repo-anchored paths — SYNTHEX P0/S4.

All runtime artifacts resolve from ARSI project root (pyproject.toml),
never from process CWD. Callers must not invent parallel archive roots.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

# src/arsi/foundation/paths.py → parents[3] == project root
_PKG_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_CANDIDATE = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def project_root() -> Path:
    """ARSI project root: directory containing pyproject.toml."""
    env = __import__("os").environ.get("ARSI_PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "pyproject.toml").exists() or (p / "src" / "arsi").exists():
            return p
    for cand in (_PROJECT_CANDIDATE, Path.cwd(), *_PROJECT_CANDIDATE.parents):
        try:
            if (cand / "pyproject.toml").exists() and (cand / "src" / "arsi").exists():
                return cand.resolve()
        except Exception:
            continue
    return _PROJECT_CANDIDATE.resolve()


def archive_dir() -> Path:
    return project_root() / "archive"


def trace_pool_dir() -> Path:
    return archive_dir() / "trace_pool"


def eval_dir() -> Path:
    return archive_dir() / "eval"


def iwm_dir() -> Path:
    return archive_dir() / "iwm"


def tmp_dir() -> Path:
    return archive_dir() / "tmp"


def config_dir() -> Path:
    return project_root() / "config"


def ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json_once(path: Path, payload: dict, writer_id: str = "") -> Path:
    """Single write source for JSON artifacts (S3). Always absolute-anchored."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    ensure_dir(p.parent)
    import json
    data = dict(payload)
    data.setdefault("_arsi_write", {
        "writer_id": writer_id or "arsi.paths.write_json_once",
        "path": str(p),
        "root": str(project_root()),
    })
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def append_jsonl(path: Path, row: dict, writer_id: str = "") -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    ensure_dir(p.parent)
    import json
    rec = dict(row)
    rec.setdefault("_arsi_write", {"writer_id": writer_id or "arsi.paths.append_jsonl", "path": str(p)})
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return p


def identity_report() -> dict:
    """S2/S4 audit: where artifacts resolve."""
    root = project_root()
    return {
        "project_root": str(root),
        "archive": str(archive_dir()),
        "trace_pool": str(trace_pool_dir()),
        "eval": str(eval_dir()),
        "iwm": str(iwm_dir()),
        "config": str(config_dir()),
        "cwd_is_project": Path.cwd().resolve() == root,
        "pyproject_exists": (root / "pyproject.toml").exists(),
        "src_arsi_exists": (root / "src" / "arsi").exists(),
    }
