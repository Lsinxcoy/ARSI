r"""Export c9 training bridge JSONL from production traces.

Usage:
    PYTHONPATH=E:/ARSI/src
    E:\ARSI\.venv\Scripts\python.exe scripts/export_training_bridge.py
    ... --n 500 --variant host:hermes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.paths import eval_dir
from arsi.harness.training_bridge import export_training_set

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="ARSI c9 training bridge export")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args()

    from arsi.core import ARSI

    arsi = ARSI.from_config(str(PROJECT_ROOT / "config" / "arsi.yaml"))
    if getattr(arsi, "llm", None):
        arsi.llm._client = None
    traces = arsi.store.get_recent_traces(n=args.n)
    out = eval_dir() / "c9_training_bridge.jsonl"
    man = export_training_set(
        traces,
        out,
        harness_variant=args.variant,
        include_excluded=args.include_excluded,
    )
    print("C9 TRAINING BRIDGE")
    print(f"  in={man['n_in']} exported={man['n_exported']} excluded={man['n_excluded']}")
    print(f"  variant={man['harness_variant']} grpo={man['grpo']} alignment={man['alignment']}")
    print(f"  out={man['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
