"""One-shot live acceptance: capability velocity field + RankMe (ODEWorld P0)."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"E:\ARSI\src")

from arsi.core import ARSI
from arsi.foundation.paths import archive_dir, write_json_once


def main() -> None:
    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    if arsi.llm:
        arsi.llm._client = None
    try:
        # seed an earlier wall-clock sample so velocity is non-zero
        arsi.capability_flow.observe(arsi.get_stats(), t_wall=time.time() - 45.0, note="t0_snap")
        harv = arsi.harvest_term_tree()
        cycle = arsi.dream_rsi_cycle(skip_llm=True)
        stats = arsi.get_stats()
        cf = stats.get("capability_flow") or {}
        ee = stats.get("env_evolution") or {}
        rk = (cf.get("last_rankme") or {}).get("z_window") or {}
        report = {
            "timestamp": datetime.now().isoformat(),
            "capability_flow_health": {
                "n_samples": cf.get("n_samples"),
                "n_v_updates": cf.get("n_v_updates"),
                "field_mse_ewma": cf.get("field_mse_ewma"),
                "last_dt": cf.get("last_dt"),
                "last_negative_organs": cf.get("last_negative_organs"),
                "rankme_z": rk,
                "last_z": cf.get("last_z"),
                "last_v_hat": cf.get("last_v_hat"),
                "last_v_gt": cf.get("last_v_gt"),
                "integrated_goal_sample": {
                    k: (cf.get("last_integrated_goal") or {}).get(k)
                    for k in ("self_trust", "live_ema", "d_t_mean")
                },
            },
            "pool_dyn_rankme": ee.get("pool_dyn_rankme") or {},
            "harvest_flow_note": (harv.get("capability_flow") or {}).get("note"),
            "dream_flow_note": (cycle.get("capability_flow") or {}).get("note"),
            "dream_env_rankme": (cycle.get("env_evolution") or {}).get("pool_dyn_rankme"),
            "harvest_d_t_dyn": harv.get("env_difficulty_dyn"),
            "tracks": ["live_capability", "pool_replay", "D_T", "dyn_velocity"],
            "notes": [
                "v_gt is wall-clock first-order, not tick index",
                "static context (beta/hosts) excluded from velocity fit",
                "RankMe collapse alarm is diagnostic, not a score",
            ],
        }
        out = archive_dir() / "eval" / "capability_flow_acceptance.json"
        write_json_once(out, report, writer_id="scripts.capability_flow_acceptance")
        print(json.dumps({
            "out": str(out),
            "n_samples": cf.get("n_samples"),
            "n_v_updates": cf.get("n_v_updates"),
            "mse": cf.get("field_mse_ewma"),
            "neg": cf.get("last_negative_organs"),
            "rankme": rk.get("effective_rank"),
            "collapse": rk.get("collapse"),
            "harvested": harv.get("harvested"),
            "dream_ran": cycle.get("ran"),
        }, ensure_ascii=False, indent=2))
    finally:
        arsi.close()


if __name__ == "__main__":
    main()
