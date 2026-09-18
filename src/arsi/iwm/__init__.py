"""IWM — Introspective World Model facade.

Real introspection requires:
  Q1 organ reliability → control hooks
  Q2 knowledge frontier → explore/exploit bias
  Q3 loop efficacy → evidence-based self-metric updates (no η hand-twisting)
  Q4 transition ledger → self-change audit
  Q5 decision provenance → replay why a was chosen
  Q6 calibrate → degrade to baseline when introspector is untrustworthy
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from arsi.iwm.calibrate import IntrospectorCalibrator
from arsi.iwm.frontier import KnowledgeFrontier
from arsi.iwm.loop_efficacy import (
    VERDICT_HURT,
    VERDICT_NEUTRAL,
    LoopEfficacy,
    LoopTrial,
)
from arsi.iwm.organ_self import (
    ORGAN_BEHAVIOR_PREDICTOR,
    ORGAN_DREAM,
    ORGAN_DYNAMICS,
    ORGAN_MEMORY,
    ORGAN_PORTFOLIO,
    STATUS_UNKNOWN,
    OrganSelfModel,
)
from arsi.iwm.provenance import DecisionProvenance
from arsi.iwm.transition_ledger import TransitionLedger, TransitionRecord

__all__ = [
    "IWM",
    "OrganSelfModel",
    "KnowledgeFrontier",
    "LoopEfficacy",
    "LoopTrial",
    "TransitionLedger",
    "TransitionRecord",
    "DecisionProvenance",
    "IntrospectorCalibrator",
    "ORGAN_BEHAVIOR_PREDICTOR",
    "ORGAN_DYNAMICS",
    "ORGAN_DREAM",
    "ORGAN_PORTFOLIO",
    "ORGAN_MEMORY",
]


class IWM:
    """Facade wiring all introspective organs."""

    def __init__(self, archive_dir: Optional[str | Path] = None):
        ledger_path = None
        self.archive_dir = Path(archive_dir) if archive_dir else None
        if self.archive_dir:
            ledger_path = self.archive_dir / "transition_ledger.jsonl"
        self.organ = OrganSelfModel()
        self.ledger = TransitionLedger(path=ledger_path)
        self.loops = LoopEfficacy()
        self.frontier = KnowledgeFrontier()
        self.provenance = DecisionProvenance()
        self.calibrator = IntrospectorCalibrator()
        self._open_trials: dict[str, str] = {}  # intervention -> trial_id
        self._hooks_applied_count = 0
        self._advice_count = 0

    # ── observation ingest ───────────────────────────────────────

    def observe_behavior(
        self,
        action: str,
        outcome: str = "",
        effect: float = 0.0,
        predicted: Optional[str] = None,
        predicted_correct: Optional[bool] = None,
        dimension: str = "",
    ) -> None:
        self.frontier.observe(action, outcome=outcome, effect=effect, dimension=dimension)
        if predicted is not None or predicted_correct is not None:
            correct = predicted_correct
            if correct is None:
                from arsi.world_model.siwm import BehaviorPredictor
                correct = BehaviorPredictor.categorize_action(predicted or "") == (
                    BehaviorPredictor.categorize_action(action)
                )
            self.organ.record(ORGAN_BEHAVIOR_PREDICTOR, bool(correct))
            self.ledger.record_prediction(
                organ=ORGAN_BEHAVIOR_PREDICTOR,
                action=action,
                predicted=predicted,
                actual=action,
                correct=bool(correct),
                evidence_refs=["layer1_predict"],
            )

    def observe_dynamics(
        self,
        action: str,
        predicted_state: Any,
        actual_state: Any,
        error: Optional[float] = None,
    ) -> None:
        if isinstance(predicted_state, dict) and isinstance(actual_state, dict):
            keys = set(predicted_state) | set(actual_state)
            errs = []
            for k in keys:
                pv = predicted_state.get(k)
                av = actual_state.get(k)
                if isinstance(pv, (int, float)) and isinstance(av, (int, float)):
                    errs.append(abs(float(pv) - float(av)))
            if errs and error is None:
                error = sum(errs) / len(errs)
        if error is None:
            error = 0.0 if predicted_state == actual_state else 1.0
        correct = float(error) < 0.35
        self.organ.record(ORGAN_DYNAMICS, correct, control_hint=f"dynamics_err={error:.4f}")
        self.ledger.record_prediction(
            organ=ORGAN_DYNAMICS,
            action=action,
            predicted=predicted_state,
            actual=actual_state,
            error=float(error),
            evidence_refs=["siwm_layer2"],
        )

    def observe_memory(self, helped: bool, note: str = "") -> None:
        self.organ.record(ORGAN_MEMORY, helped, control_hint=note)

    def observe_eval_loop(self, compare_result: Any) -> None:
        """Portfolio organ evidence from Phase D7 fixed-vs-dream compare."""
        d = compare_result.to_dict() if hasattr(compare_result, "to_dict") else dict(compare_result or {})
        delta = float(d.get("delta_score", 0.0) or 0.0)
        live_reg = bool(d.get("live_regression", False))
        helped = (not live_reg) and delta >= 0
        self.organ.record(
            ORGAN_PORTFOLIO,
            helped,
            control_hint=d.get("recommendation", ""),
        )
        self.ledger.record_prediction(
            organ=ORGAN_PORTFOLIO,
            action="dream_rsi_eval",
            predicted=1.0 if helped else 0.0,
            actual=delta,
            error=abs(0.0 if helped else delta) if live_reg else max(0.0, -delta),
            correct=helped,
            evidence_refs=["eval_loop"],
            meta={"delta_score": delta, "live_regression": live_reg},
        )

    # ── loop trials (Q3) ─────────────────────────────────────────

    def begin_loop_trial(
        self,
        intervention: str,
        before_error: float,
        eta_before: Optional[float] = None,
        evidence_refs: Optional[list[str]] = None,
        **notes,
    ) -> LoopTrial:
        trial = self.loops.begin(
            intervention=intervention,
            before=before_error,
            eta_before=eta_before,
            evidence_refs=evidence_refs,
            notes=notes,
        )
        self._open_trials[intervention] = trial.trial_id
        return trial

    def complete_loop_trial(
        self,
        intervention: str,
        after_error: float,
        eta_after: Optional[float] = None,
        evidence_refs: Optional[list[str]] = None,
        **notes,
    ) -> LoopTrial:
        trial_id = self._open_trials.pop(intervention, None)
        if trial_id is None:
            return self.loops.record_verdict(
                intervention=intervention,
                before=float(after_error),
                after=float(after_error),
                eta_after=eta_after,
                evidence_refs=evidence_refs,
                **notes,
            )
        trial = self.loops.complete(
            trial_id,
            after=after_error,
            eta_after=eta_after,
            evidence_refs=evidence_refs,
            notes=notes,
        )
        # Dream organ reliability from loop verdict
        if intervention == "dream":
            self.organ.record(
                ORGAN_DREAM,
                trial.verdict == "helped",
                control_hint=trial.verdict,
            )
            self.ledger.record_prediction(
                organ=ORGAN_DREAM,
                action="dream",
                predicted="helped",
                actual=trial.verdict,
                correct=trial.verdict == "helped",
                evidence_refs=trial.evidence_refs or ["loop_trial"],
                meta={"before": trial.before, "after": trial.after, "delta": trial.delta},
            )
        return trial

    def evidence_based_eta(
        self,
        current_eta: float,
        measured_error: float,
        intervention: str = "dream",
    ) -> tuple[float, str]:
        return self.loops.evidence_based_value(
            current=float(current_eta),
            measured=float(measured_error),
            intervention=intervention,
        )

    # ── control advice (behavior hooks) ──────────────────────────

    def governor_advice(self, state=None) -> dict:
        self._advice_count += 1
        hooks = self.organ.control_hooks()
        dream_summary = self.loops.intervention_summary("dream")
        degrade = self.calibrator.should_degrade_to_baseline()
        explore = self.frontier.explore_bias()
        exploit = self.frontier.exploit_bias()
        self_trust = self.calibrator.self_trust()

        # Behavioral rules
        forbid_dream = hooks.get("forbid_default_dream", False) or (
            not dream_summary.get("allowed_default", True)
            and dream_summary.get("count", 0) >= 2
        )
        downweight_dyn = hooks.get("downweight_pre_enactment", False)
        dyn_status = self.organ.status(ORGAN_DYNAMICS)
        if dyn_status == STATUS_UNKNOWN:
            downweight_dyn = True

        advice = {
            "self_trust": self_trust,
            "degrade_to_baseline": degrade,
            "downweight_pre_enactment": downweight_dyn,
            "forbid_default_dream": forbid_dream,
            "forbid_dream_reason": (
                dream_summary.get("last_verdict", "")
                if forbid_dream
                else ""
            ),
            "prefer_learn": hooks.get("prefer_learn_over_evolve", False),
            "downweight_portfolio": hooks.get("downweight_portfolio", False),
            "explore_bias": explore,
            "exploit_bias": exploit,
            "unreliable_organs": hooks.get("unreliable_organs", []),
            "organ_trust": {
                name: self.organ.trust_weight(name)
                for name in (
                    ORGAN_BEHAVIOR_PREDICTOR,
                    ORGAN_DYNAMICS,
                    ORGAN_DREAM,
                    ORGAN_PORTFOLIO,
                    ORGAN_MEMORY,
                )
            },
            "dream_loop": dream_summary,
        }
        return advice

    def apply_outcome(self, decision_id: str, used_iwm: bool, success: bool, note: str = "") -> None:
        self.calibrator.record(decision_id, used_iwm=used_iwm, success=success, note=note)
        self.provenance.mark_outcome(decision_id, outcome="success" if success else "failure")

    def record_decision(self, **kwargs):
        advice = kwargs.pop("iwm_snapshot", None)
        if advice is None:
            advice = self.governor_advice()
        hooks = kwargs.pop("hooks_applied", None)
        if hooks is None:
            hooks = []
            for k in (
                "downweight_pre_enactment",
                "forbid_default_dream",
                "prefer_learn",
                "degrade_to_baseline",
            ):
                if advice.get(k):
                    hooks.append(k)
        if hooks:
            self._hooks_applied_count += 1
        return self.provenance.record(iwm_snapshot=advice, hooks_applied=hooks, **kwargs)

    # ── first-person / health ────────────────────────────────────

    def first_person_report(self, world_state=None) -> dict:
        """Ω_ε with real introspective content — not four vanity numbers."""
        eta = getattr(world_state, "eta", None)
        report = {
            "eta": eta,
            "organs": self.organ.report(),
            "frontier": self.frontier.report(),
            "loop_efficacy": self.loops.report(),
            "ledger": self.ledger.stats(),
            "provenance": self.provenance.stats(),
            "calibrator": self.calibrator.report(),
        }
        if world_state is not None:
            phi = getattr(world_state, "phi", None)
            psi = getattr(world_state, "psi", None)
            report["generation"] = getattr(phi, "generation", None)
            report["belief_count"] = len(getattr(psi, "beliefs", []) or [])
            report["storage"] = getattr(phi, "storage_stats", {}) if phi else {}
        return report

    def health(self) -> dict:
        advice = self.governor_advice()
        return {
            "iwm": {
                "self_trust": advice["self_trust"],
                "degrade_to_baseline": advice["degrade_to_baseline"],
                "unreliable_organs": advice["unreliable_organs"],
                "organ_trust": advice["organ_trust"],
                "dream_loop": advice["dream_loop"],
                "explore_bias": advice["explore_bias"][:8],
                "frontier_weak": [
                    c["key"] for c in self.frontier.frontier().get("weak", [])
                ],
                "ledger_size": self.ledger.size,
                "loop_trials": self.loops.report()["trial_count"],
                "hooks_applied": self._hooks_applied_count,
                "advice_count": self._advice_count,
            }
        }

    def calibrate(self) -> dict:
        return self.calibrator.report()

    def q_scores(self) -> dict:
        """Acceptance scores Q1–Q6 for probes."""
        dream = self.loops.intervention_summary("dream")
        fr = self.frontier.frontier()
        provenance_ok = self.provenance.count > 0
        return {
            "Q1_organs": {
                "pass": any(
                    self.organ.status(o) != STATUS_UNKNOWN
                    for o in (ORGAN_BEHAVIOR_PREDICTOR, ORGAN_DYNAMICS, ORGAN_DREAM)
                ),
                "hooks_present": bool(self.organ.control_hooks()),
                "unreliable": self.organ.unreliable_organs(),
            },
            "Q2_frontier": {
                "pass": bool(fr.get("known") or fr.get("weak") or fr.get("underexplored")),
                "explore_bias": self.frontier.explore_bias(),
            },
            "Q3_loop": {
                "pass": dream.get("count", 0) > 0 and "allowed_default" in dream,
                "dream_summary": dream,
                "may_adjust_eta": self.loops.may_adjust_metric("dream"),
            },
            "Q4_ledger": {
                "pass": self.ledger.size > 0,
                "size": self.ledger.size,
            },
            "Q5_provenance": {
                "pass": provenance_ok,
                "count": self.provenance.count,
            },
            "Q6_calibrate": {
                "pass": True,  # always answerable; trust value is the content
                **self.calibrator.report(),
            },
        }

    def q_gate(self) -> tuple[bool, dict]:
        """Q1–Q3 any two fail → not an introspective world model."""
        qs = self.q_scores()
        fails = []
        for key in ("Q1_organs", "Q2_frontier", "Q3_loop"):
            if not qs[key].get("pass", False):
                fails.append(key)
        # For v1 claim we need evidence on Q1-Q3; empty system fails by design
        ok = len(fails) < 2
        return ok, {"failed": fails, "scores": qs, "claim": "introspective_skeleton" if not ok else "introspective_v1_candidate"}
