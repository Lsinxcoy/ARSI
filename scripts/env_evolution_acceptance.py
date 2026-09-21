"""One-shot live acceptance for Environment Evolution P0.

Writes archive/eval/env_evolution_acceptance.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONPATH", r"E:\ARSI\src")
sys.path.insert(0, r"E:\ARSI\src")

from arsi.core import ARSI
from arsi.foundation.paths import archive_dir, write_json_once


def main() -> None:
    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    if arsi.llm:
        arsi.llm._client = None  # acceptance: heuristic paths only
    try:
        arsi.world_pool.configure_env_evolution(
            enabled=True,
            evolve_every_n=1,
            max_evolved_per_harvest=1,
            effort="high",
            min_seed_nodes=3,
            use_el_in_dream=True,
        )
        harv = arsi.harvest_term_tree()
        pool_before_eval = arsi.world_pool.size
        cycle = arsi.dream_rsi_cycle(skip_llm=True)
        stats = arsi.get_stats()
        env_h = stats.get("env_evolution") or {}
        pool_snap = {}
        snap_path = arsi._world_pool_path
        if Path(snap_path).exists():
            try:
                pool_snap = json.loads(Path(snap_path).read_text(encoding="utf-8"))
            except Exception:
                pool_snap = {}

        report = {
            "timestamp": datetime.now().isoformat(),
            "harvest": {k: harv.get(k) for k in (
                "harvested", "world_id", "pool_size", "node_count",
                "traces_kept", "env_difficulty", "lineage_id", "evolution",
            )},
            "pool_before_dream_eval": pool_before_eval,
            "pool_after": arsi.world_pool.size,
            "dream_rsi_ran": bool(cycle.get("ran")),
            "dream_reason": cycle.get("reason"),
            "dream_env_evolution": cycle.get("env_evolution"),
            "dream_deployed": cycle.get("deployed"),
            "dream_deployed_score": cycle.get("deployed_score"),
            "dream_pool_size": cycle.get("pool_size"),
            "live_capability_track": (cycle.get("eval_loop") or {}).get("live_best") if isinstance(cycle.get("eval_loop"), dict) else cycle.get("live_capability"),
            "env_evolution_health": {
                "config": env_h.get("config"),
                "difficulty": env_h.get("difficulty"),
                "el_advances": (env_h.get("el") or {}).get("advances"),
                "el_lineages": len((env_h.get("el") or {}).get("lineages") or {}),
                "evolved_worlds": env_h.get("evolved_worlds"),
                "evolution_accepted": env_h.get("evolution_accepted"),
                "evolution_rejected": env_h.get("evolution_rejected"),
            },
            "pool_snapshot_schema": pool_snap.get("schema"),
            "pool_el": {
                "tau": (pool_snap.get("el") or {}).get("tau"),
                "advances": (pool_snap.get("el") or {}).get("advances"),
                "n_lineages": len((pool_snap.get("el") or {}).get("lineages") or {}),
            },
            "pool_world_dt": [
                {
                    "world_id": w.get("world_id"),
                    "d_t": (w.get("env_difficulty") or {}).get("d_t"),
                    "generation": w.get("generation"),
                    "evolved": w.get("evolved"),
                    "direction": w.get("direction"),
                }
                for w in (pool_snap.get("worlds") or [])[-12:]
            ],
            "notes": [
                "live_capability track is separate from pool replay scores",
                "evolved worlds are sealed_eligible=False",
                "D_T rise is environment hardness, not live capability gain",
            ],
        }
        out = archive_dir() / "eval" / "env_evolution_acceptance.json"
        write_json_once(out, report, writer_id="scripts.env_evolution_acceptance")
        print(json.dumps({
            "out": str(out),
            "harvested": harv.get("harvested"),
            "pool": arsi.world_pool.size,
            "evolved": env_h.get("evolved_worlds"),
            "el_advances": (env_h.get("el") or {}).get("advances"),
            "d_t_mean": (env_h.get("difficulty") or {}).get("d_t_mean"),
            "d_t_spread": (env_h.get("difficulty") or {}).get("d_t_spread"),
            "dream_ran": cycle.get("ran"),
        }, ensure_ascii=False, indent=2))
    finally:
        try:
            arsi.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
