r"""Fast live acceptance snapshot — no LLM, harvest + stats + paired eval only."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"E:\ARSI\src")

from arsi.core import ARSI
from arsi.foundation.paths import archive_dir, identity_report, write_json_once
from arsi.meta.eval_loop import run_eval_loop
from arsi.meta.paired_ab import compare_paired


def main() -> int:
    arsi = ARSI.from_config(r"E:\ARSI\config\arsi.yaml")
    if arsi.llm:
        arsi.llm._client = None  # speed; heuristic paths only

    train = arsi.siwm.train_from_history()
    if arsi.iwm is not None:
        arsi.iwm.observe_layer1_holdout(float(train.get("holdout_accuracy") or 0.0))

    harvest = arsi.harvest_term_tree()
    # light eval
    eval_res = run_eval_loop(arsi, params=arsi.dream_rsi_params)
    pool_eval = arsi.world_pool.evaluate_policy_across_pool(
        __import__("arsi.governor.portfolio_policy", fromlist=["build_policy_fn"]).build_policy_fn(arsi.portfolio_policy),
        policy_name="dream_rsi",
    )
    fixed_fn = __import__("arsi.meta.eval_loop", fromlist=["fixed_exploration_fn"]).fixed_exploration_fn()
    fixed_eval = arsi.world_pool.evaluate_policy_across_pool(fixed_fn, policy_name="fixed")
    fixed_scores = [float(w.get("score") or 0) for w in fixed_eval.get("per_world") or []]
    dream_scores = [float(w.get("score") or 0) for w in pool_eval.get("per_world") or []]
    paired = compare_paired(fixed_scores, dream_scores, "fixed", "dream_rsi")

    stats = arsi.get_stats()
    # newest manifest
    mans = sorted(arsi.manifest_store.recent(5), key=lambda m: m.cycle_id)
    last_m = mans[-1].to_dict() if mans else {}

    snapshot = {
        "tick": "fast_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "paths": identity_report(),
        "world_min_verdict": arsi._world_min_verdict,
        "score_mode": getattr(arsi.dream_rsi_params, "score_mode", None),
        "layer1_train": train,
        "harvest": harvest,
        "eval": eval_res.to_dict(),
        "paired_ab": paired.to_dict(),
        "pool_eval": {
            "avg_score": pool_eval.get("avg_score"),
            "avg_quality": pool_eval.get("avg_quality"),
            "world_count": pool_eval.get("world_count"),
            "score_mode": pool_eval.get("score_mode"),
        },
        "fixed_eval": {
            "avg_score": fixed_eval.get("avg_score"),
            "avg_quality": fixed_eval.get("avg_quality"),
        },
        "iwm": stats.get("iwm"),
        "layer1": stats.get("layer1"),
        "vacuum": stats.get("vacuum"),
        "beta": stats.get("beta"),
        "world_pool_size": stats.get("world_pool_size"),
        "manifest_cycles": stats.get("manifest_cycles"),
        "last_manifest": {
            "cycle_id": last_m.get("cycle_id"),
            "best_score": last_m.get("best_score"),
            "deployed_policy": last_m.get("deployed_policy"),
            "notes": last_m.get("notes"),
            "quality_gate": last_m.get("quality_gate"),
        },
    }

    out_json = archive_dir() / "eval" / "fast_acceptance_latest.json"
    write_json_once(out_json, snapshot, writer_id="scripts.run_live_fast")
    with (archive_dir() / "arsi_health.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")

    # concise print
    print("LAYER1", train)
    print("HARVEST", {k: harvest.get(k) for k in ("harvested", "pool_size", "traces_kept", "quality_gate")})
    print("POOL", snapshot["pool_eval"])
    print("FIXED", snapshot["fixed_eval"])
    print("PAIRED", paired.reason, "promote", paired.promote, "mean_d", paired.mean_delta, "d", paired.cohen_d)
    print("EVAL", eval_res.recommendation, "delta", eval_res.delta_score)
    print("LAST_MANIFEST", snapshot["last_manifest"])
    print("IWM_LAYER1", (stats.get("iwm") or {}).get("iwm", {}).get("layer1_holdout"),
          (stats.get("iwm") or {}).get("iwm", {}).get("behavior_predictor_status"))
    print("WROTE", out_json)
    arsi.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
