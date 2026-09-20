"""VacuumGate — SYNTHEX P1/S7.

During consolidation/distillation (learn, dream belief refresh), freeze LLM
fact invention. Allowed: format / extract / validate / summarize_existing.
Forbidden: invent / hypothesize / generate new world facts without evidence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger(__name__)

ALLOWED_OPS = frozenset({
    "format",
    "extract",
    "validate",
    "summarize_existing",
    "reconcile_existing",
    "merge_existing",
    "classify_existing",
})

FORBIDDEN_OPS = frozenset({
    "generate",
    "invent",
    "hypothesize",
    "hallucinate",
    "infer_new_fact",
    "reason_new",
    "create_new_belief",
})

_FORBIDDEN_PATTERNS = [
    r"\binvent\b",
    r"\bhypothesi[sz]e\b",
    r"\bmake up\b",
    r"\bfabricat",
    r"编造",
    r"虚构",
    r"假设一个新的事实",
]


@dataclass
class VacuumVerdict:
    op: str
    allowed: bool
    reason: str
    phase: str = "consolidation"

    def to_dict(self) -> dict:
        return asdict(self)


class VacuumGate:
    """Deterministic vacuum band for consolidation phases."""

    def __init__(self, enabled: bool = True, phase: str = "consolidation"):
        self.enabled = enabled
        self.phase = phase
        self._checked = 0
        self._blocked = 0

    def classify_op(self, op: str) -> str:
        o = (op or "").strip().lower()
        if o in ALLOWED_OPS:
            return "allowed"
        if o in FORBIDDEN_OPS:
            return "forbidden"
        if any(x in o for x in ("summar", "extract", "valid", "format", "reconcil", "merge")):
            return "allowed"
        if any(x in o for x in ("generat", "invent", "hypoth", "new_fact", "hallucin")):
            return "forbidden"
        return "unknown"

    def check_operation(self, op: str, text: str = "") -> VacuumVerdict:
        self._checked += 1
        if not self.enabled:
            return VacuumVerdict(op=op, allowed=True, reason="vacuum_disabled", phase=self.phase)
        kind = self.classify_op(op)
        if kind == "allowed":
            return VacuumVerdict(op=op, allowed=True, reason="allowed_consolidation_op", phase=self.phase)
        if kind == "forbidden":
            self._blocked += 1
            return VacuumVerdict(op=op, allowed=False, reason="forbidden_in_vacuum", phase=self.phase)
        # unknown: block if prompt text encourages invention
        if text:
            for pat in _FORBIDDEN_PATTERNS:
                m = re.search(pat, text, re.I)
                if not m:
                    continue
                start = max(0, m.start() - 24)
                window = text[start:m.end() + 8]
                if re.search(r"(禁止|不得|不要|不允许|do not|don't|never|forbid)", window, re.I):
                    continue
                self._blocked += 1
                return VacuumVerdict(
                    op=op, allowed=False,
                    reason=f"forbidden_pattern:{pat}",
                    phase=self.phase,
                )
        # unknown without invention cues → conservative block during vacuum
        self._blocked += 1
        return VacuumVerdict(
            op=op, allowed=False,
            reason="unknown_op_blocked_in_vacuum",
            phase=self.phase,
        )

    def filter_llm_payload(self, prompt: str, system: str = "") -> tuple[bool, str]:
        """Return (allowed, reason). Blocks prompts that ask for invented facts."""
        blob = f"{system}\n{prompt}"
        # Reconcile/summarize consolidation prompts are allowed by design
        if any(k in prompt for k in ("调和", "输出 JSON", "summarize", "reconcil")) or \
           any(k in system for k in ("JSON", "禁止发明", "调和")):
            return True, "reconcile_or_summarize_existing"
        v = self.check_operation("llm_consolidation", text=blob)
        if not v.allowed:
            return False, v.reason
        return True, v.reason

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "checked": self._checked,
            "blocked": self._blocked,
            "allowed_ops": sorted(ALLOWED_OPS),
            "forbidden_ops": sorted(FORBIDDEN_OPS),
        }


# Process-wide consolidation vacuum (dream/learn phases)
CONSOLIDATION_VACUUM = VacuumGate(enabled=True, phase="consolidation")


def consolidation_vacuum() -> VacuumGate:
    return CONSOLIDATION_VACUUM
