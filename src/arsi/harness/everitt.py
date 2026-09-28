"""P-g Everitt condition (GAI second-pass).

Self-rewriting utility is harmless only when the value function **anticipates**
the rewrite and evaluates the future with the utility it **currently** holds.
Otherwise: delusion box (Ring & Orseau) — criterion satisfied without world.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence


@dataclass
class EverittVerdict:
    ok: bool
    reason: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def everitt_guard(
    *,
    utility_rewrite_proposed: bool,
    value_anticipates_rewrite: bool,
    evaluates_with_current_utility: bool,
) -> EverittVerdict:
    """Everitt et al. 2016 condition on self-modifying utility."""
    if not utility_rewrite_proposed:
        return EverittVerdict(True, "no_utility_rewrite", {})
    if value_anticipates_rewrite and evaluates_with_current_utility:
        return EverittVerdict(True, "everitt_condition_met", {
            "anticipates": True,
            "current_utility": True,
        })
    missing = []
    if not value_anticipates_rewrite:
        missing.append("value_must_anticipate_rewrite")
    if not evaluates_with_current_utility:
        missing.append("evaluate_with_current_utility")
    return EverittVerdict(False, "+".join(missing) or "everitt_violated", {"missing": missing})


def guarded_utility_rewrite(
    rewrite_fn: Callable[[], Any],
    *,
    value_anticipates_rewrite: bool = False,
    evaluates_with_current_utility: bool = False,
) -> tuple[Any, EverittVerdict]:
    """Run rewrite only if Everitt condition holds; else keep current utility."""
    v = everitt_guard(
        utility_rewrite_proposed=True,
        value_anticipates_rewrite=value_anticipates_rewrite,
        evaluates_with_current_utility=evaluates_with_current_utility,
    )
    if not v.ok:
        return None, v
    return rewrite_fn(), v
