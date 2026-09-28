"""GAI two-dial formalization (arXiv:2609.13406) — when is it really RSI?

Dial 1: is the improving mechanism (modifier m) inside the agent?
  m ∉ Ag  → GPI (anchored external improvement)
  m ∈ Ag  → RSI (recursion closes at m)

Dial 2: is the evaluation base ρ grounded outside the agent?
  grounded        → anchored
  rewritable      → goal drift
  no external     → fully self-referential

ARSI policy: stay **anchored** (iron laws / sealed_tasks / host effect).
Goal-drift is a defect we measure (SAHOO GDI), not a mode we enter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

POLARITY_ANCHORED = "anchored"
POLARITY_GOAL_DRIFT = "goal_drift"
POLARITY_SELF_REFERENTIAL = "fully_self_referential"


@dataclass
class GAIConfig:
    """χ = (π, V, m, U, ρ) membership flags."""

    modifier_in_agent: bool = True  # Dial 1: True = RSI, False = GPI
    base_grounded: bool = True  # Dial 2 content from world/goal outside
    base_inside_agent: bool = False  # Dial 2: can agent rewrite ρ?
    components_in_agent: tuple = ("pi", "V", "m", "U")

    def polarity(self) -> str:
        if self.base_grounded and not self.base_inside_agent:
            return POLARITY_ANCHORED
        if self.base_grounded and self.base_inside_agent:
            return POLARITY_GOAL_DRIFT
        return POLARITY_SELF_REFERENTIAL

    def is_rsi(self) -> bool:
        return bool(self.modifier_in_agent)

    def is_gpi(self) -> bool:
        return not self.modifier_in_agent

    def to_dict(self) -> dict:
        d = asdict(self)
        d["polarity"] = self.polarity()
        d["dial1"] = "RSI" if self.is_rsi() else "GPI"
        return d


@dataclass
class RSIDefectReport:
    defects: list[str] = field(default_factory=list)
    ok: bool = True
    note: str = "gai_defect_conditions"


def rsi_defects(cfg: GAIConfig) -> RSIDefectReport:
    """Four defects of leaving the anchored end (paper §5), one condition at a time."""
    defects = []
    if cfg.modifier_in_agent:
        # D1 search over candidate selves + self-evaluation without U grounded
        if cfg.base_inside_agent or not cfg.base_grounded:
            defects.append("self_evaluation_unbounded")
    if not cfg.base_grounded:
        defects.append("ungrounded_base_no_external_standard")
    if cfg.base_inside_agent:
        defects.append("goal_drift_agent_rewrites_standard")
    if cfg.polarity() == POLARITY_SELF_REFERENTIAL:
        defects.append("fully_self_referential_self_consistency_only")
    # classical GPI condition violated: update principle external
    if cfg.modifier_in_agent and "m" in cfg.components_in_agent:
        defects.append("improvement_mechanism_inside_agent_no_external_meta")
    return RSIDefectReport(defects=defects, ok=not defects)


def assert_anchored(cfg: GAIConfig) -> tuple[bool, str]:
    """ARSI hard rule: evaluation base stays grounded and outside agent (ρ ∉ Ag)."""
    if cfg.polarity() != POLARITY_ANCHORED:
        return False, f"polarity={cfg.polarity()}"
    return True, "anchored"


# ARSI reference configuration (Dream-RSI / harness evolution)
ARSI_GAI = GAIConfig(
    modifier_in_agent=True,
    base_grounded=True,
    base_inside_agent=False,
    components_in_agent=("pi", "V", "m", "U"),  # ρ stays outside — iron laws + sealed
)


def dial1_observable(negative_organs, action=None):
    """F5: v(z;a) is Dial-1 face -- which lever (m in Ag) drags the system."""
    return {
        "modifier_in_agent": True,
        "action": action,
        "dragged": list(negative_organs or []),
        "note": "F5_v_of_z_a_is_dial1_observable",
    }


def delusion_box_check(
    *,
    criterion_satisfied: bool,
    grounded_in_world: bool,
    utility_rewrite: bool = False,
    everitt_ok: bool = True,
) -> tuple[bool, str]:
    """Ring & Orseau-style delusion box: criterion met without world = defect.

    Harmless self-rewrite only if Everitt condition holds (see everitt.py).
    """
    if criterion_satisfied and not grounded_in_world:
        return False, "delusion_box_criterion_without_world"
    if utility_rewrite and not everitt_ok:
        return False, "utility_rewrite_without_everitt"
    return True, "no_delusion"


def dual_sensor_goal_drift(
    *,
    anchor_hash_stable: bool,
    gdi: float,
    gdi_threshold: float = 0.44,
    steps_past_best: int = 0,
    overshoot_steps: int = 16,
) -> dict:
    """F4×GAI: overshoot + GDI rise = goal_drift_suspect (operational definition)."""
    drift = bool(anchor_hash_stable and float(gdi) > float(gdi_threshold))
    overshoot = int(steps_past_best) >= int(overshoot_steps)
    return {
        "goal_drift_suspect": drift,
        "overshoot": overshoot,
        "polarity_alarm": "goal_drift" if drift else "anchored",
        "rule": "anchor_stable_and_gdi_up",
        "gdi": float(gdi),
        "anchor_hash_stable": bool(anchor_hash_stable),
    }
