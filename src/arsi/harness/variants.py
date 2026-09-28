"""Variant pool + ensemble routing (HarnessX §4.5).

seesaw → fork new variant instead of discarding a locally useful edit.
Iron laws / frozen dims are shared across all variants (not forked).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from arsi.harness.gates import GateResult


@dataclass
class HarnessVariant:
    variant_id: str = "default"
    label: str = "default"
    cluster_success: dict = field(default_factory=dict)  # cluster_id -> rate
    manifests: list = field(default_factory=list)
    active: bool = True
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


class VariantPool:
    def __init__(self, max_variants: int = 4):
        self.max_variants = max(1, int(max_variants))
        self.variants: dict[str, HarnessVariant] = {
            "default": HarnessVariant(variant_id="default", label="default")
        }

    def route(self, cluster_id: str) -> str:
        """Pick variant with highest estimated success on this cluster."""
        best_id, best_rate = "default", -1.0
        for vid, v in self.variants.items():
            if not v.active:
                continue
            rate = float(v.cluster_success.get(cluster_id, v.cluster_success.get("default", 0.5)) or 0.0)
            # unseen cluster → default bias
            if cluster_id not in v.cluster_success and vid != "default":
                rate -= 0.05
            if rate > best_rate:
                best_id, best_rate = vid, rate
        return best_id

    def observe(self, variant_id: str, cluster_id: str, success: bool, alpha: float = 0.3) -> None:
        v = self.variants.get(variant_id) or self.variants["default"]
        prev = float(v.cluster_success.get(cluster_id, 0.5) or 0.5)
        v.cluster_success[cluster_id] = round((1 - alpha) * prev + alpha * (1.0 if success else 0.0), 4)

    def ensure_hosts(self, host_ids: list[str]) -> list[str]:
        """Bind one harness variant per host agent (hermes / mimo / synthex…)."""
        created = []
        for hid in host_ids or []:
            hid = str(hid or "").strip()
            if not hid:
                continue
            vid = f"host:{hid}"
            if vid not in self.variants:
                if len([v for v in self.variants.values() if v.active]) >= self.max_variants:
                    self._retire_worst()
                self.variants[vid] = HarnessVariant(
                    variant_id=vid,
                    label=f"host:{hid}",
                    cluster_success={"default": 0.5},
                )
                created.append(vid)
        return created

    def route_for_host(self, host_id: str) -> str:
        """Prefer the host's own variant when bound; else cluster routing."""
        vid = f"host:{str(host_id or '').strip()}"
        if vid in self.variants and self.variants[vid].active:
            return vid
        return self.route(str(host_id or "default"))

    def observe_host(self, host_id: str, success: bool, cluster_id: str = "default") -> None:
        self.observe(self.route_for_host(host_id), cluster_id or "default", success)

    def fork_on_seesaw(
        self,
        seesaw: GateResult,
        parent_id: str = "default",
        label: str = "",
    ) -> HarnessVariant:
        """Create a child variant for the improving cluster side."""
        if seesaw.gate != "seesaw" or seesaw.ok:
            raise ValueError("fork_on_seesaw requires a failed seesaw gate")
        if len([v for v in self.variants.values() if v.active]) >= self.max_variants:
            self._retire_worst()
        vid = f"v-{uuid.uuid4().hex[:8]}"
        gains = [g.get("cluster") for g in (seesaw.detail.get("gains") or [])]
        child = HarnessVariant(
            variant_id=vid,
            label=label or f"fork:{parent_id}:{','.join(gains[:3])}",
            cluster_success={c: 0.55 for c in gains if c},
        )
        self.variants[vid] = child
        return child

    def _retire_worst(self) -> Optional[str]:
        candidates = [v for v in self.variants.values() if v.active and v.variant_id != "default"]
        if not candidates:
            return None
        def score(v: HarnessVariant) -> float:
            vals = list(v.cluster_success.values()) or [0.0]
            return sum(vals) / len(vals)
        worst = min(candidates, key=score)
        worst.active = False
        return worst.variant_id

    def report(self) -> dict:
        return {
            "max_variants": self.max_variants,
            "active": [v.variant_id for v in self.variants.values() if v.active],
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "note": "iron_laws_shared_across_variants",
        }
