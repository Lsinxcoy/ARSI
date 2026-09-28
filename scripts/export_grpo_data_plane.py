r"""Export GRPO prerequisite data plane (groups + rewards + holdout split).

Usage:
    PYTHONPATH=E:/ARSI/src
    E:\ARSI\.venv\Scripts\python.exe scripts/export_grpo_data_plane.py
    ... --n 500 --variant host:hermes --private-ratio 0.2

Export only — never trains weights (API hosts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.foundation.paths import eval_dir
from arsi.harness.grpo_data import export_grpo_data_plane

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="ARSI GRPO pre-data-plane export")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--private-ratio", type=float, default=0.2)
    parser.add_argument("--from-pool", action="store_true", help="also build policy-matrix groups")
    parser.add_argument("--n-policies", type=int, default=5)
    parser.add_argument("--max-worlds", type=int, default=20)
    args = parser.parse_args()

    from arsi.core import ARSI

    arsi = ARSI.from_config(str(PROJECT_ROOT / "config" / "arsi.yaml"))
    if getattr(arsi, "llm", None):
        arsi.llm._client = None
    traces = arsi.store.get_recent_traces(n=args.n)
    out = eval_dir() / "grpo_data_plane"
    policies = None
    pool = None
    if args.from_pool:
        from arsi.harness.grpo_data import (
            _ChunkPool,
            default_policy_matrix,
            load_world_pool_snapshot,
            worlds_from_trace_chunks,
        )

        pool = getattr(arsi, "world_pool", None)
        if pool is None or getattr(pool, "size", 0) < 5:
            snap = load_world_pool_snapshot()
            if snap is not None and getattr(snap, "size", 0) > getattr(pool, "size", 0):
                pool = snap
        if pool is None or getattr(pool, "size", 0) < 3:
            chunks = worlds_from_trace_chunks(traces, n_worlds=max(4, args.max_worlds // 2))
            pool = _ChunkPool(chunks)
        policies = default_policy_matrix(args.n_policies)
    man = export_grpo_data_plane(
        traces,
        out,
        harness_variant=args.variant,
        private_ratio=args.private_ratio,
        world_pool=pool,
        policies=policies,
        max_worlds=args.max_worlds,
    )
    print("GRPO PRE-DATA-PLANE")
    print(f"  traces={man['n_traces']} rewards={man['n_exported_rewards']}")
    print(f"  groups={man['n_groups']} advantage_ready={man['n_advantage_ready_groups']} matrix_rows={man.get('n_policy_matrix_rows', 0)}")
    print(f"  split={man['split']} grpo_live={man['grpo_live']}")
    print(f"  out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
