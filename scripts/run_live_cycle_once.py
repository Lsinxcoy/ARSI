r"""Force one live ARSI cycle with current code (score-wall + S6/S7 + P0).

Writes: harvest + dream_rsi_cycle + eval + IWM health snapshot to archive.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"E:\ARSI\src")

from arsi.core import ARSI
from arsi.foundation.paths import archive_dir, identity_report


def main() -> int:
    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    print("paths", identity_report())
    print("world_min_verdict", arsi._world_min_verdict)
    print("score_mode", getattr(arsi.dream_rsi_params, "score_mode", None))

    # train layer1 + bind holdout into IWM
    train = arsi.siwm.train_from_history()
    print("layer1_train", train)
    if arsi.iwm is not None:
        arsi.iwm.observe_layer1_holdout(float(train.get("holdout_accuracy") or 0.0))

    # a few steps
    steps = []
    for _ in range(4):
        r = arsi.step()
        steps.append({
            "action": r.get("decision", {}).get("action"),
            "source": r.get("decision", {}).get("source"),
            "iwm_advice": r.get("iwm_advice"),
            "prediction": r.get("prediction"),
            "layer1": r.get("layer1_train"),
        })
        print("step", steps[-1]["action"], steps[-1]["source"], steps[-1].get("iwm_advice"))

    harvest = arsi.harvest_term_tree()
    print("harvest", {k: harvest.get(k) for k in ("harvested", "pool_size", "traces_kept", "quality_gate")})

    cycle = arsi.dream_rsi_cycle()
    slim_cycle = {
        k: cycle.get(k)
        for k in cycle
        if k not in ("feedback_excerpt", "best_fn")
    }
    print("dream_rsi", json.dumps({k: slim_cycle.get(k) for k in list(slim_cycle)[:20]}, ensure_ascii=False, default=str)[:1500])

    stats = arsi.get_stats()
    snapshot = {
        "tick": "oneshot_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "eta": stats.get("eta"),
        "trace_count": stats.get("trace_count"),
        "experience_count": stats.get("experience_count"),
        "world_pool_size": stats.get("world_pool_size"),
        "manifest_cycles": stats.get("manifest_cycles"),
        "beta": stats.get("beta"),
        "grid_plan": stats.get("grid_plan"),
        "iwm": stats.get("iwm"),
        "iwm_q_gate": stats.get("iwm_q_gate"),
        "layer1": stats.get("layer1"),
        "paths": stats.get("paths"),
        "vacuum": stats.get("vacuum"),
        "verified": stats.get("iwm_q_gate"),
        "steps": steps,
        "harvest": harvest,
        "cycle_keys": list(cycle.keys()) if isinstance(cycle, dict) else [],
        "cycle_deployed": cycle.get("deployed") if isinstance(cycle, dict) else None,
        "cycle_score": cycle.get("current_score") if isinstance(cycle, dict) else None,
        "eval_loop": cycle.get("eval_loop") if isinstance(cycle, dict) else None,
    }

    out = archive_dir() / "arsi_health.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")

    report = archive_dir() / "eval" / "oneshot_cycle_latest.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("WROTE", out)
    print("WROTE", report)
    arsi.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
