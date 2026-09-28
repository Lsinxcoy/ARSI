"""Planner — adaptation landscape to fight under-exploration (AEGIS stage 2)."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from arsi.harness.digester import FailureCluster
from arsi.harness.taxonomy import DIMENSIONS, FROZEN_DIMS, MODULES

EDIT_TYPES = ("prompt", "processor", "tool", "config", "control")


@dataclass
class AdaptationLandscape:
    failing_clusters: list = field(default_factory=list)
    tried_edit_types: dict = field(default_factory=dict)
    untried_edit_types: list = field(default_factory=list)
    tried_dims: dict = field(default_factory=dict)
    untried_dims: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Planner:
    """Build landscape from digests + edit history; propose untried levers."""

    def landscape(
        self,
        clusters: Iterable[FailureCluster],
        prior_manifests: Iterable[dict] = (),
    ) -> AdaptationLandscape:
        clusters = list(clusters or [])
        manifests = [dict(m) for m in (prior_manifests or [])]
        tried_type = Counter(str(m.get("edit_type") or "") for m in manifests)
        tried_dim = Counter(str(m.get("dimension") or "") for m in manifests)

        untried_types = [t for t in EDIT_TYPES if tried_type.get(t, 0) == 0]
        untried_dims = [
            d for d, _name in DIMENSIONS.items() if d not in FROZEN_DIMS and tried_dim.get(d, 0) == 0
        ]
        # least-tried first when everything has been attempted (avoid prompt-loop)
        type_order = list(untried_types) + [
            t for t, _n in sorted(tried_type.items(), key=lambda kv: (kv[1], kv[0])) if t in EDIT_TYPES
        ] or list(EDIT_TYPES)
        dim_order = list(untried_dims) + [
            d for d, _n in sorted(tried_dim.items(), key=lambda kv: (kv[1], kv[0])) if d not in FROZEN_DIMS
        ] or ["c2"]

        # already-proposed (label, edit_type) pairs from history — dry-run dedup
        seen_pairs = {
            (str(m.get("summary", ""))[:40], str(m.get("edit_type") or ""))
            for m in manifests
        }
        seen_labels_types = {
            (str(m.get("evidence")[-1] if m.get("evidence") else ""), str(m.get("edit_type") or ""))
            for m in manifests
            if m.get("evidence")
        }

        recs: list[dict] = []
        for c in clusters:
            mods = list(c.implicated or [])
            dims = []
            for m in mods:
                sc = MODULES.get(m) or {}
                for d in sc.get("dims") or []:
                    if d not in FROZEN_DIMS and d not in dims:
                        dims.append(d)
            prefer_type = type_order[0] if type_order else "prompt"
            # skip type if this cluster label already has an accepted/proposed manifest
            for pt in type_order:
                key = (c.label, pt)
                if not any(k[0].endswith(c.cluster_id) or c.label in k[0] for k in seen_labels_types):
                    prefer_type = pt
                    break
                # rotate to next untried/least type
                if pt == prefer_type and len(type_order) > 1:
                    idx = type_order.index(pt)
                    prefer_type = type_order[(idx + 1) % len(type_order)]
            prefer_dim = dim_order[0] if dim_order else (dims[0] if dims else "c2")
            recs.append(
                {
                    "cluster": c.cluster_id,
                    "label": c.label,
                    "n": c.n,
                    "modules": mods,
                    "dims": dims,
                    "prefer_edit_type": prefer_type,
                    "prefer_dim": prefer_dim,
                    "persistent": c.persistent,
                    "hosts": list(getattr(c, "hosts", []) or []),
                    "fail_classes": dict(getattr(c, "fail_classes", {}) or {}),
                    "rationale": (
                        "untried_lever"
                        if untried_types or untried_dims
                        else "least_tried_lever_rotate"
                    ),
                }
            )

        return AdaptationLandscape(
            failing_clusters=[c.to_dict() for c in clusters],
            tried_edit_types=dict(tried_type),
            untried_edit_types=untried_types,
            tried_dims=dict(tried_dim),
            untried_dims=untried_dims,
            recommendations=recs,
            note="landscape_before_edit_generation",
        )
