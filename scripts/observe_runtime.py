r"""Runtime acceptance: pre/post score-wall comparison from health + manifests + eval.

Usage:
    set PYTHONPATH=E:/ARSI/src
    python scripts/observe_runtime.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(r"E:\ARSI")
ARCHIVE = ROOT / "archive"
OUT = Path(r"E:\Mimo 生成\docs\2026-09-20")


def load_health():
    rows = []
    p = ARCHIVE / "arsi_health.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_manifests():
    mans = []
    tp = ARCHIVE / "trace_pool"
    if not tp.exists():
        return mans
    for p in sorted(tp.glob("iter*/live_cycle_manifest.json")):
        try:
            mans.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return mans


def main():
    health = load_health()
    ok = [h for h in health if h.get("status") == "ok" and "trace_count" in h]
    mans = load_manifests()
    report = {"health_ok": len(ok), "manifests": len(mans)}

    if ok:
        etas = [float(h.get("eta") or 0) for h in ok]
        report["window"] = {"first": ok[0].get("timestamp"), "last": ok[-1].get("timestamp")}
        report["eta_last"] = round(etas[-1], 4)
        report["traces_last"] = ok[-1].get("trace_count")
        report["pool_last"] = ok[-1].get("world_pool_size")
        report["beta_last"] = ok[-1].get("beta")
        report["has_layer1"] = "layer1" in ok[-1]
        report["has_vacuum"] = "vacuum" in (ok[-1] or {})
        # score-wall markers in grid evidence
        lasts = ok[-5:]
        report["last_best_scores"] = [
            ((h.get("grid_plan") or {}).get("evidence") or {}).get("last_best")
            for h in lasts
        ]
        betas = [h.get("beta") for h in ok if h.get("beta") is not None]
        report["beta_hist_tail"] = Counter(round(float(b), 2) for b in betas[-30:]).most_common(6)
        iwm_rows = [h for h in ok if isinstance(h.get("iwm"), dict)]
        report["iwm_snapshots"] = len(iwm_rows)
        if iwm_rows:
            last = iwm_rows[-1]["iwm"]
            report["iwm_last"] = {
                "timestamp": iwm_rows[-1].get("timestamp"),
                "organ_trust": last.get("organ_trust"),
                "dream_loop": last.get("dream_loop"),
                "layer1_holdout": last.get("layer1_holdout"),
                "self_trust": last.get("self_trust"),
                "claim": (last.get("q_gate") or {}).get("claim"),
            }

    if mans:
        best = [float(m.get("best_score") or 0) for m in mans]
        report["best_last"] = best[-1]
        report["best_max"] = max(best)
        report["best_mean"] = round(mean(best), 4)
        report["deployed_last"] = mans[-1].get("deployed_policy")
        report["sweep_notes_last"] = [m.get("notes") for m in mans[-5:]]
        # post-fix detection: manifests after quality_anchored / frozen_plateau
        frozen = sum(1 for m in mans if str(m.get("deployed_policy", "")).startswith("frozen_plateau"))
        report["frozen_plateau_count"] = frozen
        report["gate_last"] = mans[-1].get("quality_gate")

    # eval compares
    eval_dir = ARCHIVE / "eval"
    cmps = []
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("compare_*.json")):
            try:
                cmps.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    report["compare_count"] = len(cmps)
    if cmps:
        last = cmps[-1]
        report["compare_last"] = {
            "ts": last.get("timestamp"),
            "world_count": last.get("world_count"),
            "fixed": last.get("fixed_avg_score"),
            "dream": last.get("dream_avg_score"),
            "delta": last.get("delta_score"),
            "rec": last.get("recommendation"),
            "paired_ab": (last.get("notes") or {}).get("paired_ab", {}).get("reason"),
            "effect_anchor": bool((last.get("notes") or {}).get("effect_anchor")),
        }

    # verdict
    wall = -3.673
    scores = [s for s in report.get("last_best_scores", []) if s is not None]
    if report.get("best_last") is not None:
        scores = scores + [report["best_last"]]
    left_wall = any(abs(float(s) - wall) > 0.05 for s in scores) if scores else None
    report["verdict"] = {
        "score_left_minus_3_67": left_wall,
        "frozen_beta_names": report.get("frozen_plateau_count", 0) > 0,
        "daemon_new_tick_has_layer1": report.get("has_layer1"),
        "note": "old health rows predate code deploy; look for timestamps after 2026-09-20 09:00",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "runtime-acceptance.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("WROTE", out)


if __name__ == "__main__":
    main()
