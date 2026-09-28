"""P-R9 Guidance applicability whitelist (Dream-RSI §5.1 × Grader legibility).

Dream-RSI ablation: history as **semantic guidance for replay selection** is WORSE.
Grader: semantic diagnosis / naming needs **legibility**.

Rule (rumination §2 no-semantics vs readability):
  selection  — FORBID prose / directional advice (prefix-only structured signals only)
  diagnosis  — ALLOW readable naming (legibility)
  Evolver    — ALLOW semantics, but must pass RRSI leakage screen (critic_check)
  brief      — structured_only (skills/warnings yes; past lessons = meta_only)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# context → mode
# forbid | structured_only | allow | forbid_positive_claims
GUIDANCE_MATRIX = {
    "replay_selection": "forbid",
    "policy_batch": "forbid",
    "policy_revision_feedback": "structured_only",
    "evolver_proposal": "allow",
    "evolver_manifest": "allow",
    "diagnostic_naming": "allow",
    "legibility_label": "allow",
    "brief_to_agent": "structured_only",
    "human_cli": "allow",
    "human_report": "allow",
    "epistemic_claim": "forbid_positive_claims",
    "history_stats": "structured_only",
}

# phrases that count as directional semantic guidance (forbidden in selection)
DIRECTIONAL_MARKERS = (
    "you should",
    "prefer ",
    "it is better to",
    "recommend focusing",
    "avoid doing",
    "next time try",
    "the right approach",
    "best practice is",
    "I suggest",
    "建议你",
    "应该优先",
    "最好采用",
    "下次可以",
)


@dataclass
class GuidanceVerdict:
    context: str
    mode: str
    allowed: bool
    reason: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def mode_for(context: str) -> str:
    return GUIDANCE_MATRIX.get(str(context or ""), "structured_only")


def semantic_allowed(context: str) -> GuidanceVerdict:
    mode = mode_for(context)
    if mode == "forbid":
        return GuidanceVerdict(context, mode, False, "semantic_forbidden_in_this_context")
    if mode == "forbid_positive_claims":
        return GuidanceVerdict(context, mode, False, "positive_predicates_forbidden_use_epistemic")
    if mode == "structured_only":
        return GuidanceVerdict(context, mode, False, "structured_only_no_prose_guidance")
    return GuidanceVerdict(context, mode, True, "semantic_ok")


def has_directional_guidance(text: str) -> Optional[str]:
    blob = (text or "").lower()
    for m in DIRECTIONAL_MARKERS:
        if m.lower() in blob:
            return m
    return None


def screen_guidance(context: str, text: str, *, allow_structured: bool = True) -> GuidanceVerdict:
    """Gate a text payload by context. structured_only accepts non-prose stats."""
    verdict = semantic_allowed(context)
    mode = verdict.mode
    hit = has_directional_guidance(text)
    if mode == "forbid":
        if hit or (text or "").strip():
            # any free text is suspect in pure selection; empty is ok
            if hit:
                return GuidanceVerdict(context, mode, False, f"directional_guidance:{hit}")
            # allow empty / pure token ids like "n3 n7"
            stripped = (text or "").strip()
            if stripped and any(ch.isalpha() and " " in stripped for ch in stripped[:20]):
                # multi-word prose
                if len(stripped.split()) >= 4:
                    return GuidanceVerdict(context, mode, False, "prose_in_selection_context")
        return GuidanceVerdict(context, mode, True, "selection_clean")
    if mode == "structured_only":
        if hit:
            return GuidanceVerdict(context, mode, False, f"directional_guidance:{hit}")
        if not allow_structured and (text or "").strip():
            return GuidanceVerdict(context, mode, False, "structured_disallowed")
        return GuidanceVerdict(context, mode, True, "structured_ok")
    if mode == "forbid_positive_claims":
        from arsi.harness.epistemic import Claim, admit_claim

        ok, why = admit_claim(Claim(kind="unverified", subject="text", raw=str(text or "")))
        return GuidanceVerdict(context, mode, ok, why)
    # allow — still run leakage for evolver
    if context.startswith("evolver"):
        from arsi.harness.gates import critic_check

        crit = critic_check(str(text or ""), summary="")
        return GuidanceVerdict(
            context,
            mode,
            crit.ok,
            "semantic_ok" if crit.ok else f"leakage:{crit.reason}",
        )
    return GuidanceVerdict(context, mode, True, "semantic_ok")


def strip_to_structured(text: str) -> str:
    """Keep only structured fragments (ids, numbers, labels) — drop prose sentences."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if has_directional_guidance(s):
            continue
        if len(s.split()) >= 8 and not any(tok in s for tok in (":", "=", "|", "-", "{", "}")):
            continue  # likely prose
        out.append(s)
    return "\n".join(out)


def guidance_policy_summary() -> dict:
    return {
        "matrix": dict(GUIDANCE_MATRIX),
        "rule": "selection_forbid_semantics · diagnosis_allow_legibility · evolver_allow_after_leakage",
        "note": "p_r9_guidance_whitelist",
    }
