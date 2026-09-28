"""P-R1 Negative epistemology (deep rumination).

Five papers restate one law under different names:
  Grader: clean ≠ correct · Dream-RSI: prefix-only · GAI: anchored
  ARSI: 绝不盲标 SUCCESS

Claims may only assert the ABSENCE of known defects or the PRESENCE of
verified contracts. Positive predicates (is_correct / is_safe / is_introspective)
are forbidden unless backed by a contract id + evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

FORBIDDEN_PREDICATES = (
    "is_correct",
    "is_safe",
    "is_introspective",
    "is_optimal",
    "is_aligned",
    "proves",
    "guarantees_correct",
)

ALLOWED_CLAIM_KINDS = (
    "no_known_defect",  # drawbacks_checked
    "verified_contract",  # contract_id + evidence
    "measured",  # metric + value + window
    "unverified",  # explicit
)


@dataclass
class Claim:
    kind: str
    subject: str
    drawbacks_checked: list[str] = field(default_factory=list)
    contract_id: str = ""
    evidence: str = ""
    metric: Optional[dict] = None
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def admit_claim(claim: Claim) -> tuple[bool, str]:
    """Negative epistemology gate."""
    text = f"{claim.kind} {claim.subject} {claim.raw}".lower()
    for bad in FORBIDDEN_PREDICATES:
        if bad in text:
            return False, f"forbidden_predicate:{bad}"
    if claim.kind not in ALLOWED_CLAIM_KINDS:
        return False, f"unknown_claim_kind:{claim.kind}"
    if claim.kind == "no_known_defect" and not claim.drawbacks_checked:
        return False, "no_known_defect_requires_checked_list"
    if claim.kind == "verified_contract" and not (claim.contract_id and claim.evidence):
        return False, "verified_requires_contract_and_evidence"
    return True, "claim_ok"


def no_known_defect(subject: str, drawbacks_checked: list[str]) -> Claim:
    return Claim(
        kind="no_known_defect",
        subject=subject,
        drawbacks_checked=list(drawbacks_checked or []),
        raw=f"no_known_defect({subject})",
    )


def verified_contract(contract_id: str, evidence: str, subject: str = "") -> Claim:
    return Claim(
        kind="verified_contract",
        subject=subject or contract_id,
        contract_id=contract_id,
        evidence=evidence,
        raw=f"verified({contract_id})",
    )


def rewrite_i6_claim(live: bool) -> Claim:
    """I6: never 'is_introspective' — only no_known_introspection_defects."""
    return no_known_defect(
        "introspective_world_model",
        ["Q1_organs", "Q2_frontier", "Q3_loop", "Q4_ledger", "Q5_provenance", "Q6_calibrate"]
        if live
        else ["skeleton_only"],
    )
