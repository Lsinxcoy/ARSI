r"""Live harness landscape — Digester+Planner over production traces.

Usage:
    set PYTHONPATH=E:/ARSI/src
    E:\ARSI\.venv\Scripts\python.exe scripts/harness_landscape.py
    ... --n 500 --json

Writes archive/eval/harness_landscape_latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.paths import archive_dir, eval_dir
from arsi.harness.audit import AuditLog
from arsi.harness.pipeline import load_label_counts, run_landscape, save_label_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="ARSI harness landscape (AEGIS digester/planner)")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from arsi.core import ARSI

    arsi = ARSI.from_config(str(PROJECT_ROOT / "config" / "arsi.yaml"))
    if getattr(arsi, "llm", None):
        arsi.llm._client = None

    traces = arsi.store.get_recent_traces(n=args.n)
    arch = archive_dir()
    audit = AuditLog(arch / "harness")
    label_path = arch / "harness" / "label_counts.json"
    report = run_landscape(
        traces,
        prior_labels=load_label_counts(label_path),
        audit=audit,
    )
    merged = report.pop("_merged_label_counts", None)
    if merged is not None:
        save_label_counts(label_path, merged)

    out = eval_dir() / "harness_landscape_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        land = report.get("landscape") or {}
        print("HARNESS LANDSCAPE")
        print(f"  traces={report['n_traces']} failures={report['n_failures']}")
        print(f"  clusters={len(report['clusters'])}")
        for c in report["clusters"][:8]:
            print(f"    {c['label']:20s} n={c['n']:3d} impl={c['implicated']} persist={c['persistent']}")
        print(f"  untried_edit_types={land.get('untried_edit_types')}")
        print(f"  untried_dims={land.get('untried_dims')}")
        for r in (land.get("recommendations") or [])[:5]:
            print(f"    rec {r['cluster']}: {r['prefer_edit_type']}/{r['prefer_dim']} ({r['rationale']})")
        print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
