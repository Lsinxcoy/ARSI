r"""One AEGIS evolve round: landscape → Change-Manifests → gates → store.

Usage:
    PYTHONPATH=E:/ARSI/src
    E:\ARSI\.venv\Scripts\python.exe scripts/harness_evolve.py
    ... --max-k 3 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.paths import archive_dir, eval_dir
from arsi.harness.audit import AuditLog
from arsi.harness.evolver import Evolver
from arsi.harness.manifest import ManifestStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="ARSI harness Evolver")
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    land_path = eval_dir() / "harness_landscape_latest.json"
    if not land_path.exists():
        print("no landscape — run scripts/harness_landscape.py first", file=sys.stderr)
        return 1
    landscape = json.loads(land_path.read_text(encoding="utf-8"))
    # accept either flat AdaptationLandscape or run_landscape report wrapper
    if "recommendations" not in landscape and isinstance(landscape.get("landscape"), dict):
        landscape = landscape["landscape"]

    llm = None
    if args.use_llm:
        try:
            from arsi.core import ARSI

            arsi = ARSI.from_config(str(PROJECT_ROOT / "config" / "arsi.yaml"))
            llm = arsi.llm
        except Exception as e:
            print(f"llm_unavailable:{e}", file=sys.stderr)

    arch = archive_dir()
    audit = AuditLog(arch / "harness")
    store = ManifestStore(arch / "harness")
    ev = Evolver(llm=llm)
    manifests = ev.propose_from_landscape(landscape, max_k=args.max_k)
    for m in manifests:
        store.append(m)
        audit.emit(
            "evolver",
            "propose",
            gate="critic+scope",
            ok=(m.status == "accepted"),
            manifest_id=m.manifest_id,
            status=m.status,
            edit_type=m.edit_type,
            dimension=m.dimension,
            reject_reason=m.reject_reason,
        )

    summary = {
        "n": len(manifests),
        "accepted": sum(1 for m in manifests if m.status == "accepted"),
        "rejected": sum(1 for m in manifests if m.status == "rejected"),
        "manifests": [m.to_dict() for m in manifests],
    }
    out = eval_dir() / "harness_evolve_latest.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print("HARNESS EVOLVE")
        print(f"  proposed={summary['n']} accepted={summary['accepted']} rejected={summary['rejected']}")
        for m in manifests:
            print(f"  [{m.status:8s}] {m.edit_type:9s} {m.dimension} {m.symbol} :: {m.summary[:70]}")
            if m.reject_reason:
                print(f"             reject: {m.reject_reason}")
        print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
