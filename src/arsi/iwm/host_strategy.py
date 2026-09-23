"""Host empowerment strategy from IWM (Route A).

Turns IWM governor advice / organ trust / frontier into **structured
host-facing strategy facts** — not free-form lessons (Dream-RSI §5.1).

Rules:
- structured facts only (focus, skill priors, warnings, confidence)
- grounded in IWM organs + frontier + calibrator
- dual-gate aware: high memory_trust → trust memory/learn path;
  unreliable memory → re-ingest first
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from arsi.iwm.organ_self import (
    ORGAN_BEHAVIOR_PREDICTOR,
    ORGAN_DREAM,
    ORGAN_DYNAMICS,
    ORGAN_MEMORY,
    ORGAN_PORTFOLIO,
    STATUS_OK,
    STATUS_UNRELIABLE,
    STATUS_UNKNOWN,
)


@dataclass
class HostStrategy:
    agent_id: str = ""
    focus: str = "learn"
    confidence: float = 0.5
    skill_priorities: list[str] = field(default_factory=list)
    operational_warnings: list[str] = field(default_factory=list)
    control_flags: dict = field(default_factory=dict)
    organ_snapshot: dict = field(default_factory=dict)
    frontier: dict = field(default_factory=dict)
    # P1 ODEWorld: first-order dyn flow (evidence-only)
    negative_organs: list[str] = field(default_factory=list)
    pre_failure_organs: list[str] = field(default_factory=list)
    z_pre: dict = field(default_factory=dict)
    dyn_velocity: dict = field(default_factory=dict)
    z_subgoal: dict = field(default_factory=dict)
    receipt_policy: str = "require_grounded_receipt"
    note: str = "structured IWM strategy facts only"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def as_structured_block(self) -> list[str]:
        """Host-facing lines for brief (structured, not guidance prose)."""
        lines = [
            "## IWM Strategy (structured facts)",
            f"- focus: {self.focus}",
            f"- confidence: {self.confidence:.2f}",
            f"- receipt_policy: {self.receipt_policy}",
        ]
        if self.skill_priorities:
            lines.append("- skill_priorities:")
            for s in self.skill_priorities:
                lines.append(f"    - {s}")
        flags = {k: self.control_flags[k] for k in sorted(self.control_flags) if self.control_flags[k]}
        if flags:
            lines.append(f"- control_flags: {flags}")
        if self.operational_warnings:
            lines.append("- operational_warnings:")
            for w in self.operational_warnings:
                lines.append(f"    - {w}")
        ot = self.organ_snapshot.get("organ_trust") or self.organ_snapshot.get("organs") or {}
        if ot:
            lines.append(f"- organ_trust: {ot}")
        if self.negative_organs or self.pre_failure_organs:
            lines.append(f"- negative_velocity_organs: {self.negative_organs}")
            if self.pre_failure_organs:
                lines.append(f"- pre_failure_organs: {self.pre_failure_organs}")
        if self.z_pre:
            compact_pre = {
                k: self.z_pre.get(k)
                for k in ("self_trust", "memory_trust", "live_ema", "eta")
                if k in self.z_pre
            }
            if compact_pre:
                lines.append(f"- z_pre: {compact_pre}")
        if self.z_subgoal:
            compact = {
                k: self.z_subgoal.get(k)
                for k in ("self_trust", "memory_trust", "live_ema", "d_t_mean", "eta")
                if k in self.z_subgoal
            }
            if compact:
                lines.append(f"- z_subgoal: {compact}")
        if self.frontier.get("explore_bias"):
            lines.append(f"- frontier_explore: {self.frontier.get('explore_bias')[:6]}")
        if self.frontier.get("exploit_bias"):
            lines.append(f"- frontier_exploit: {self.frontier.get('exploit_bias')[:6]}")
        return lines

    def meta_only_recommendations(self) -> list[str]:
        """For full/debug policy only — still IWM-grounded, not free-form lessons."""
        recs = []
        if self.control_flags.get("forbid_default_dream"):
            recs.append("Do not rely on dream to fix self-model; complete host task with evidence.")
        if self.control_flags.get("downweight_memory_ops"):
            recs.append("Memory organ weak: re-ingest/observe outcomes before trusting distilled advice.")
        if self.control_flags.get("trust_memory_for_learn"):
            recs.append("Memory organ trusted: prefer learn/distill paths on host traces.")
        if self.focus == "calibrate":
            recs.append("Focus on calibration: declare confidence vs outcomes explicitly.")
        if self.focus == "reingest":
            recs.append("Focus on re-ingest: ensure new evidence flows into ARSI before claiming completion.")
        if self.focus == "repair_organs" and (self.pre_failure_organs or self.negative_organs):
            organs = list(self.pre_failure_organs or self.negative_organs)
            recs.append(
                "Pre-failure / negative-velocity organs: "
                + ",".join(organs)
                + " — repair with fresh measured outcomes (z_pre is unverified reconstruction)."
            )
        return recs


def _skills_from_strategy(focus: str, organ_trust: dict, agent_id: str, negative_organs: Optional[list[str]] = None) -> list[str]:
    """Skill priors for empowerment content — map IWM state to known ARSI skills."""
    skills = []
    if focus in ("learn", "reingest"):
        skills.extend(["arsi-knowledge", "arsi-environment"])
    if focus == "calibrate":
        skills.append("arsi-calibration")
    if focus == "repair_organs":
        skills.append("arsi-metacognition")
    neg = set(negative_organs or [])
    if "memory_trust" in neg:
        skills.extend(["arsi-environment", "arsi-knowledge"])
    if "behavior_predictor_trust" in neg:
        skills.append("arsi-calibration")
    if organ_trust.get(ORGAN_BEHAVIOR_PREDICTOR, 0) < 0.5:
        skills.append("arsi-metacognition")
    if agent_id.startswith("hermes") or agent_id.startswith("synthex"):
        skills.append("arsi-environment")
    if agent_id.startswith("mimo"):
        skills.append("arsi-calibration")
    # unique preserve order
    out = []
    for s in skills:
        if s not in out:
            out.append(s)
    return out[:5]


def build_host_strategy(arsi=None, agent_id: str = "host") -> HostStrategy:
    """Derive host empowerment strategy from IWM + frontier."""
    advice: dict[str, Any] = {}
    organ_snap: dict = {}
    frontier: dict = {}
    if arsi is not None and getattr(arsi, "iwm", None) is not None:
        try:
            state = arsi.siwm.get_state() if getattr(arsi, "siwm", None) else None
            advice = arsi.iwm.governor_advice(state)
            organ_snap = {
                "organ_trust": advice.get("organ_trust") or {},
                "memory_status": advice.get("memory_status"),
                "layer1_holdout": advice.get("organ_trust", {}).get(ORGAN_BEHAVIOR_PREDICTOR),
            }
            frontier = {
                "explore_bias": advice.get("explore_bias") or [],
                "exploit_bias": advice.get("exploit_bias") or [],
            }
        except Exception:
            advice = {}

    memory_trust = float(advice.get("memory_trust") or 0.0)
    memory_status = advice.get("memory_status") or STATUS_UNKNOWN
    organ_trust = advice.get("organ_trust") or {}

    # Capability flow (ODEWorld P1): first-order dyn evidence only
    flow_guide: dict = {}
    negative_organs: list[str] = []
    pre_fail: list[str] = []
    z_pre: dict = {}
    dyn_velocity: dict = {}
    z_subgoal: dict = {}
    if arsi is not None and getattr(arsi, "capability_flow", None) is not None:
        try:
            flow_guide = arsi.capability_flow.flow_guidance()
            negative_organs = list(flow_guide.get("negative_organs") or [])
            pre_fail = list(flow_guide.get("pre_failure_organs") or [])
            z_pre = dict(flow_guide.get("z_pre") or {})
            dyn_velocity = dict(flow_guide.get("v_hat") or {})
            z_subgoal = dict(flow_guide.get("z_subgoal") or {})
        except Exception:
            flow_guide = {}

    # A-line focus decision (IWM first; dyn-flow can override only with evidence)
    if advice.get("downweight_memory_ops") or memory_status == STATUS_UNRELIABLE:
        focus = "reingest"
    elif advice.get("trust_memory_for_learn") or memory_trust >= 0.5:
        focus = "learn"
    elif organ_trust.get(ORGAN_BEHAVIOR_PREDICTOR, 0) < 0.35:
        focus = "calibrate"
    elif frontier.get("explore_bias"):
        focus = "explore_frontier"
    else:
        focus = "execute_with_evidence"

    # P1: negative-velocity / pre-failure organs can set focus when they name a weakness
    flow_focus = str(flow_guide.get("focus") or "")
    focus_src = list(pre_fail) + [x for x in negative_organs if x not in pre_fail]
    if focus_src and flow_focus in ("reingest", "calibrate", "repair_organs", "explore_frontier"):
        if "memory_trust" in focus_src and flow_focus == "reingest":
            # memory velocity/pre-failure → evidence path first (even over calibrate)
            focus = "reingest"
        elif "behavior_predictor_trust" in focus_src or "self_trust" in focus_src:
            focus = "calibrate"
        elif focus in ("learn", "execute_with_evidence") and flow_focus in ("repair_organs", "explore_frontier"):
            focus = flow_focus

    # Operational warnings from IWM (structured, not lessons)
    warnings = []
    if advice.get("forbid_default_dream"):
        warnings.append(f"dream_forbidden:{advice.get('forbid_dream_reason') or 'loop_trial'}")
    unreliable = advice.get("unreliable_organs") or []
    if unreliable:
        warnings.append(f"unreliable_organs:{','.join(unreliable)}")
    if advice.get("downweight_pre_enactment"):
        warnings.append("dynamics_untrusted_skip_preenactment")
    if advice.get("degrade_to_baseline"):
        warnings.append("iwm_degrade_to_baseline")
    if memory_status == STATUS_UNRELIABLE:
        warnings.append("memory_organ_unreliable")
    if negative_organs:
        warnings.append(f"negative_velocity:{','.join(negative_organs)}")
    if pre_fail:
        warnings.append(f"pre_failure_degraded:{','.join(pre_fail)}")
        warnings.append("z_pre_is_model_reconstruction_unverified")

    # Confidence: IWM self_trust + memory trust + layer1 quality
    self_trust = float(advice.get("self_trust") or 0.5)
    layer1 = float((organ_snap.get("organ_trust") or {}).get(ORGAN_BEHAVIOR_PREDICTOR, 0) or 0)
    conf = 0.35 * self_trust + 0.40 * memory_trust + 0.25 * layer1
    if advice.get("degrade_to_baseline"):
        conf *= 0.7
    conf = max(0.05, min(0.95, conf))

    skills = _skills_from_strategy(focus, organ_trust, agent_id, negative_organs=focus_src)

    control_flags = {
        "prefer_learn": bool(advice.get("prefer_learn")),
        "trust_memory_for_learn": bool(advice.get("trust_memory_for_learn")),
        "downweight_memory_ops": bool(advice.get("downweight_memory_ops")),
        "prefer_remember_ingest": bool(advice.get("prefer_remember_ingest")),
        "forbid_default_dream": bool(advice.get("forbid_default_dream")),
        "downweight_pre_enactment": bool(advice.get("downweight_pre_enactment")),
        "degrade_to_baseline": bool(advice.get("degrade_to_baseline")),
        "use_z_subgoal": bool(z_subgoal),
        "negative_velocity": bool(negative_organs),
        "pre_failure_reverse": bool(pre_fail),
    }

    return HostStrategy(
        agent_id=agent_id,
        focus=focus,
        confidence=round(conf, 4),
        skill_priorities=skills,
        operational_warnings=warnings,
        control_flags=control_flags,
        organ_snapshot=organ_snap,
        frontier=frontier,
        negative_organs=negative_organs,
        pre_failure_organs=pre_fail,
        z_pre=z_pre,
        dyn_velocity=dyn_velocity,
        z_subgoal=z_subgoal,
        receipt_policy="require_grounded_receipt",
        note="structured IWM strategy facts; dyn-flow + reverse z_pre only from v_hat evidence",
    )


def strategy_for_skill_content(strategy: HostStrategy) -> str:
    """Ground skill markdown in IWM strategy (A-line empowerment content)."""
    flags = strategy.control_flags
    lines = [
        "",
        "## IWM Strategy Binding",
        f"- focus: {strategy.focus}",
        f"- confidence: {strategy.confidence:.2f}",
        f"- memory_trust: {(strategy.organ_snapshot.get('organ_trust') or {}).get(ORGAN_MEMORY, 0)}",
    ]
    if flags.get("trust_memory_for_learn"):
        lines.append("- policy: memory trusted → prefer learn/distill evidence")
    if flags.get("downweight_memory_ops"):
        lines.append("- policy: memory weak → require re-ingest before claims")
    if flags.get("forbid_default_dream"):
        lines.append("- policy: dream forbidden by IWM loop evidence")
    if strategy.operational_warnings:
        lines.append(f"- warnings: {strategy.operational_warnings}")
    if strategy.negative_organs:
        lines.append(f"- negative_velocity_organs: {strategy.negative_organs}")
    if strategy.pre_failure_organs:
        lines.append(f"- pre_failure_organs: {strategy.pre_failure_organs}")
    if strategy.z_pre:
        lines.append(f"- z_pre_keys: {sorted(list(strategy.z_pre))[:8]} (model_reconstruction)")
    if strategy.z_subgoal:
        lines.append(f"- z_subgoal_keys: {sorted(list(strategy.z_subgoal))[:8]}")
    lines.append(f"- receipt_policy: {strategy.receipt_policy}")
    return "\n".join(lines)
