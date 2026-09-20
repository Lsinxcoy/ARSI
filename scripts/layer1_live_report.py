r"""Layer1 live accuracy report on production traces.

Usage:
    set PYTHONPATH=E:/ARSI/src
    python scripts/layer1_live_report.py
    python scripts/layer1_live_report.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.store import MnemosyneStore
from arsi.foundation.verified import verified_from_pytest
from arsi.world_model.siwm import SIWM, BehaviorPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "arsi.db"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--n", type=int, default=800)
    args = parser.parse_args()

    store = MnemosyneStore(args.db)
    siwm = SIWM(store)
    train = siwm.train_from_history()  # uses n=500 internally
    # also evaluate on wider window
    traces = store.get_recent_traces(n=args.n)
    wide_holdout = 0.0
    if len(traces) >= 10:
        split = int(len(traces) * 0.8)
        siwm.layer1.fit(traces[:split])
        wide_holdout = siwm.layer1.holdout_accuracy(traces[split:])

    report = {
        "db": args.db,
        "trace_count_available": len(traces),
        "train_result": train,
        "wide_holdout_accuracy": round(wide_holdout, 4),
        "rule_count": len(siwm.layer1.rules),
        "pair_rule_count": len(getattr(siwm.layer1, "_pair_rules", {}) or {}),
        "note": (
            "holdout is sequential next-action-category accuracy; "
            "mixed host traces (hermes tools) are harder than ARSI self-actions"
        ),
        "suite_verified": verified_from_pytest("arsi.layer1.suite", PROJECT_ROOT, pytest_args=["-q", "tests/test_iwm.py", "tests/test_runtime_fixes.py"]).to_dict(),
    }
    out = PROJECT_ROOT / "archive" / "eval" / "layer1_live_report.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(out)
    except Exception as e:
        report["report_error"] = str(e)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print("LAYER1 LIVE REPORT")
        print(f"  traces: {report['trace_count_available']}")
        print(f"  train:  {train}")
        print(f"  wide_holdout: {report['wide_holdout_accuracy']}")
        print(f"  rules: {report['rule_count']} pairs: {report['pair_rule_count']}")
        print(f"  report: {report.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
