"""AEGIS pipeline glue — Digester → Planner → report (no LLM required)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from arsi.harness.audit import AuditLog
from arsi.harness.digester import Digester
from arsi.harness.planner import Planner


def load_label_counts(path: str | Path) -> Counter:
    p = Path(path)
    if not p.exists():
        return Counter()
    try:
        return Counter(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return Counter()


def save_label_counts(path: str | Path, counts: Counter) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(counts), ensure_ascii=False), encoding="utf-8")


def run_landscape(
    traces: Iterable[dict],
    *,
    prior_labels: Optional[Counter] = None,
    prior_manifests: Iterable[dict] = (),
    audit: Optional[AuditLog] = None,
) -> dict:
    """Digest failing traces and build adaptation landscape."""
    traces = [dict(t or {}) for t in (traces or [])]
    dig = Digester()
    prior = prior_labels if prior_labels is not None else Counter()
    clusters = dig.digest(traces, prior_labels=prior)
    # P0-2: feed accepted/proposed manifests so tried_edit_types is honest
    if not prior_manifests:
        try:
            from arsi.foundation.paths import archive_dir
            from arsi.harness.manifest import ManifestStore

            prior_manifests = ManifestStore(archive_dir() / "harness").list()
        except Exception:
            prior_manifests = []
    land = Planner().landscape(clusters, prior_manifests=prior_manifests)
    new_counts = Digester.label_counts(traces)
    # accumulate for persistence across ticks
    merged = Counter(prior)
    merged.update(new_counts)

    report = {
        "n_traces": len(traces),
        "n_failures": sum(1 for t in traces if dig.is_failure(t)),
        "clusters": [c.to_dict() for c in clusters],
        "landscape": land.to_dict(),
        "label_counts": dict(new_counts),
        "label_counts_cum": dict(merged),
        "note": "aegis_digester_planner_rulebased",
    }
    # RRSI Table-1 style tracks when score fields exist on traces
    try:
        from arsi.harness.rrsi import three_track_eval

        def _scores(key: str) -> list:
            out = []
            for t in traces:
                p = t.get("params") if isinstance(t.get("params"), dict) else {}
                v = p.get(key, t.get(key))
                if v is not None:
                    try:
                        out.append(float(v))
                    except Exception:
                        pass
            return out

        report["rrsi_tracks"] = three_track_eval(
            _scores("evolve_score") or _scores("score"),
            _scores("id_holdout_score"),
            _scores("ood_score"),
        ).to_dict()
    except Exception:
        report["rrsi_tracks"] = {}
    if audit is not None:
        audit.emit(
            "digester",
            "landscape",
            ok=bool(clusters or report["n_failures"] == 0),
            n_clusters=len(clusters),
            n_failures=report["n_failures"],
            untried_edit_types=land.untried_edit_types,
        )
    report["_merged_label_counts"] = merged
    return report
