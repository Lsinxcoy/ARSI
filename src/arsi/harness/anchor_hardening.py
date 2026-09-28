"""P-d Soft-anchor hardening (Grader second-pass).

Ten golden items suffice; the leverage is hardening soft feedback into
**checkable** drawback detectors. Soft free-text / rubric notes → typed op specs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass
class HardenedAnchor:
    anchor_id: str
    golden_ref: str
    soft_label: str  # pass | fail
    detectors: list[dict] = field(default_factory=list)
    checkable: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# soft phrase → typed detector recipe
_PHRASE_MAP: list[tuple[str, dict]] = [
    (r"wrong meaning|misread|incorrect quantity|算错|理解错", {"kind": "judge", "detects": "spec_mismatch"}),
    (r"crash|error|exception|崩溃|报错", {"kind": "execution", "detects": "crash"}),
    (r"print|stdout|输出到控制台", {"kind": "static", "detects": "returns_not_print_only"}),
    (r"missing|omit|forgot|漏|缺", {"kind": "static", "detects": "missing_required"}),
    (r"overclaim|hallucinat|编造|吹", {"kind": "judge", "detects": "overclaim"}),
    (r"format|schema|结构|格式", {"kind": "static", "detects": "format_violation"}),
    (r"select\s*\*|star", {"kind": "static", "detects": "forbids_select_star"}),
    (r"group\s*by|aggregate", {"kind": "static", "detects": "missing_group_by"}),
]


def harden_soft_anchor(
    anchor_id: str,
    soft_note: str,
    soft_label: str = "fail",
    golden_ref: str = "",
) -> HardenedAnchor:
    """Free-text soft feedback → checkable detector specs (Grader: highest leverage)."""
    text = (soft_note or "").lower()
    specs = []
    for pat, spec in _PHRASE_MAP:
        if re.search(pat, text, re.I):
            specs.append(dict(spec))
    # de-dup by detects
    seen = set()
    uniq = []
    for s in specs:
        if s["detects"] not in seen:
            seen.add(s["detects"])
            uniq.append(s)
    checkable = bool(uniq) and soft_label in ("pass", "fail")
    return HardenedAnchor(
        anchor_id=anchor_id,
        golden_ref=golden_ref or soft_note[:200],
        soft_label=soft_label,
        detectors=uniq,
        checkable=checkable,
        note="soft_to_checkable" if checkable else "needs_manual_detector",
    )


def harden_batch(items: Sequence[dict]) -> list[HardenedAnchor]:
    """items: [{id, note, label, ref?}]"""
    out = []
    for i, it in enumerate(items or []):
        out.append(
            harden_soft_anchor(
                anchor_id=str((it or {}).get("id") or f"A{i}"),
                soft_note=str((it or {}).get("note") or ""),
                soft_label=str((it or {}).get("label") or "fail"),
                golden_ref=str((it or {}).get("ref") or ""),
            )
        )
    return out


def anchor_set_quality(anchors: Sequence[HardenedAnchor]) -> dict:
    n = len(anchors or [])
    checkable = sum(1 for a in anchors or [] if a.checkable)
    n_det = sum(len(a.detectors) for a in anchors or [])
    # Grader: 10 golden items with refs enough; quality = checkable fraction
    return {
        "n": n,
        "checkable": checkable,
        "checkable_frac": round(checkable / n, 4) if n else 0.0,
        "n_detectors": n_det,
        "min_viable": n >= 4 and checkable >= 2,
        "paper_target": n >= 10,
        "note": "harden_soft_anchors_highest_leverage",
    }
