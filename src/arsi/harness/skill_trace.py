"""Recuris R1 — structured trace Γ + failure → memory component localization.

Γ = (w_t, E_t, a_t, o_t, w~_{t+1}, c_t, w_{t+1})
Attribution is not always a patch (paper: some failures are harness's fault).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

COMPONENT_EXPERIENTIAL = "experiential"  # skill library
COMPONENT_WORKING = "working"  # WM spec
COMPONENT_INVOCATION = "invocation"  # rho
COMPONENT_CHECKER = "checker"  # C
COMPONENT_HARNESS = "harness"  # not memory's to repair


@dataclass
class TraceStep:
    w_t: dict
    skills: list = field(default_factory=list)
    action: str = ""
    observation: str = ""
    w_proposed: dict = field(default_factory=dict)
    checker: dict = field(default_factory=dict)
    w_next: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalizationResult:
    component: str
    confidence: float
    reason: str
    evidence: list = field(default_factory=list)
    patchable: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def build_gamma(
    task_id: str,
    steps: Sequence[TraceStep],
    outcome: int | bool = 0,
) -> dict:
    return {
        "task_id": task_id,
        "steps": [s.to_dict() for s in (steps or [])],
        "outcome": int(bool(outcome)),
        "note": "recuris_structured_gamma",
    }


def localize_failure(gamma: dict) -> LocalizationResult:
    """Attribute failed run to a memory component from structured evidence."""
    steps = list((gamma or {}).get("steps") or [])
    if not steps:
        return LocalizationResult(COMPONENT_HARNESS, 0.2, "no_structured_steps", patchable=False)

    # explicit fail_class on steps (report schema) wins
    for s in steps:
        fc = str((s.get("checker") or {}).get("fail_class") or s.get("fail_class") or "")
        reason = str((s.get("checker") or {}).get("reason") or "")
        if "fail_class=" in reason:
            fc = reason.split("fail_class=", 1)[1].split()[0]
        if fc:
            comp = component_for_fail_class(fc)
            return LocalizationResult(
                comp,
                0.75,
                f"fail_class={fc}",
                evidence=[reason or fc],
                patchable=comp != COMPONENT_HARNESS,
            )

    # checker bounces → C or W
    bounces = []
    no_skill = 0
    wrong_skill_hint = 0
    env_fail = 0
    for s in steps:
        ck = s.get("checker") or {}
        if ck.get("accepted") is False:
            bounces.append(ck.get("reason") or "reject")
        if not s.get("skills"):
            no_skill += 1
        if s.get("skills") and "stale" in str(s.get("observation") or "").lower():
            wrong_skill_hint += 1
        if "fail" in str(s.get("observation") or "").lower() or "fail" in str(ck.get("reason") or "").lower():
            env_fail += 1

    n = len(steps)
    if any("no_env_evidence" in b or "env_signal" in b for b in bounces):
        return LocalizationResult(
            COMPONENT_CHECKER,
            0.7,
            "checker_bounced_unverified_done",
            evidence=bounces[:5],
            patchable=True,
        )
    if wrong_skill_hint >= max(1, n // 3):
        return LocalizationResult(
            COMPONENT_INVOCATION,
            0.65,
            "skills_stale_or_misaligned",
            evidence=[f"stale_hint={wrong_skill_hint}"],
            patchable=True,
        )
    if no_skill >= max(1, n // 2) and env_fail:
        return LocalizationResult(
            COMPONENT_EXPERIENTIAL,
            0.6,
            "missing_skill_on_failing_path",
            evidence=[f"no_skill={no_skill}"],
            patchable=True,
        )
    if env_fail:
        return LocalizationResult(
            COMPONENT_EXPERIENTIAL,
            0.55,
            "env_failures_without_checker_bounce",
            evidence=[f"env_fail={env_fail}"],
            patchable=True,
        )
    # paper case: some clusters are harness — do not patch memory
    return LocalizationResult(
        COMPONENT_HARNESS,
        0.4,
        "not_memory_fault_leave_alone",
        patchable=False,
    )


def component_for_fail_class(fail_class: str) -> str:
    fc = str(fail_class or "").lower()
    if "gate" in fc or "checker" in fc:
        return COMPONENT_CHECKER
    if "tool_loop" in fc or "invok" in fc:
        return COMPONENT_INVOCATION
    if "compile" in fc or "skill" in fc or "tool_error" in fc:
        return COMPONENT_EXPERIENTIAL
    if "context" in fc or "memory" in fc:
        return COMPONENT_WORKING
    return COMPONENT_EXPERIENTIAL
