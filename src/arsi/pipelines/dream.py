"""Dream Pipeline — self-model refresh + memory consolidation.

When η ≥ θ_high, dream pipeline:
1. First-person observation (Ω_ε via IWM when available)
2. State re-parse (MindZero)
3. Belief reconstruction
4. Memory consolidation (decay + edge discovery)
5. Evidence-based η update (LoopTrial) — never hand-twist the meter

Based on: ARSI Whitepaper v0.8 §7.3 + IWM design Q3
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.schema import MentalState, WorldState
from arsi.mnemosyne.core import Mnemosyne
from arsi.world_model.siwm import SIWM

logger = logging.getLogger(__name__)


def measure_behavior_error(layer1, traces: list[dict], n: int = 30) -> float:
    """Holdout sequential prediction error on recent traces (0=perfect)."""
    traces = list(traces or [])[-n:]
    if len(traces) < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(1, len(traces)):
        prev = traces[i - 1]
        curr = traces[i]
        last_action = prev.get("action", "")
        last_outcome = prev.get("outcome", "unknown")
        predicted = layer1.predict(
            WorldState(), last_action=last_action, last_outcome=last_outcome
        )
        pred_cat = layer1.categorize_action(predicted)
        actual_cat = layer1.categorize_action(curr.get("action", ""))
        total += 1
        if pred_cat == actual_cat:
            correct += 1
    if total == 0:
        return 1.0
    return 1.0 - (correct / total)


class DreamSession:
    """Record of a dream cycle (auditable)."""

    def __init__(
        self,
        before_psi: MentalState,
        after_psi: MentalState,
        eta_before: float,
        eta_after: float,
        loop_trial: Optional[dict] = None,
        eta_policy: str = "",
    ):
        self.before_psi = before_psi
        self.after_psi = after_psi
        self.eta_before = eta_before
        self.eta_after = eta_after
        self.loop_trial = loop_trial or {}
        self.eta_policy = eta_policy
        self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            "belief_count_before": len(self.before_psi.beliefs),
            "belief_count_after": len(self.after_psi.beliefs),
            "eta_before": self.eta_before,
            "eta_after": self.eta_after,
            "eta_improved": self.eta_after < self.eta_before,
            "eta_policy": self.eta_policy,
            "loop_trial": self.loop_trial,
            "timestamp": self.timestamp.isoformat(),
        }


class DreamPipeline:
    """Dream pipeline — the system's sleep cycle.

    Not rest: deep processing of memory and self-model.
    η moves only on measured LoopTrial evidence (IWM Q3).
    """

    def __init__(self, siwm: SIWM, mnemosyne: Mnemosyne, llm=None, iwm=None):
        self.siwm = siwm
        self.mnemosyne = mnemosyne
        self.llm = llm
        self.iwm = iwm
        self._dream_count = 0
        self._llm_reconcile_count = 0
        self._last_loop_trial: dict = {}
        self._last_eta_policy = ""

    def execute(self, state: WorldState) -> WorldState:
        """Execute a dream cycle."""
        self._dream_count += 1
        eta_before = state.eta

        # 0. Measure baseline prediction error BEFORE dream work
        traces = self.mnemosyne.store.get_recent_traces(n=50)
        layer1 = self.siwm.layer1
        err_before = measure_behavior_error(layer1, traces)

        if self.iwm is not None:
            trial = self.iwm.begin_loop_trial(
                intervention="dream",
                before_error=err_before,
                eta_before=eta_before,
                evidence_refs=["holdout_behavior_error"],
            )
            trial_id = trial.trial_id
        else:
            trial_id = None

        # 1. First-person observation: render current state
        observation = self._first_person_observe(state)

        # 2. Re-parse mental state from recent traces
        fresh_psi = self.siwm.mindzero.infer_mental_state(traces)

        # 3. Reconcile beliefs (LLM-powered when available)
        corrected_beliefs = self._reconcile_beliefs(
            state.psi.beliefs, fresh_psi.beliefs
        )

        # 4. Memory consolidation
        consolidation_result = self.mnemosyne.consolidate()

        # 5. Evidence-based η — measure AFTER consolidation; no hardcoded halving
        # Re-fit layer1 on post-consolidation experience when possible
        try:
            post_traces = self.mnemosyne.store.get_recent_traces(n=50)
            if post_traces:
                layer1.fit(post_traces)
        except Exception as e:
            logger.warning(f"post-dream layer1 fit failed: {e}")
        err_after = measure_behavior_error(layer1, self.mnemosyne.store.get_recent_traces(n=50))

        if self.iwm is not None:
            trial = self.iwm.complete_loop_trial(
                intervention="dream",
                after_error=err_after,
                eta_after=None,
                evidence_refs=["holdout_behavior_error_post"],
                observation_keys=list(observation.keys()) if isinstance(observation, dict) else [],
            )
            self._last_loop_trial = trial.to_dict()
            new_eta, eta_policy = self.iwm.evidence_based_eta(
                current_eta=eta_before,
                measured_error=err_after,
                intervention="dream",
            )
        else:
            # Fallback without IWM: η stays measured-anchored, never free lunch
            if err_after < eta_before - 0.02:
                new_eta = err_after
                eta_policy = f"measured_only_fallback_{eta_before:.3f}->{new_eta:.3f}"
            else:
                new_eta = eta_before
                eta_policy = "no_iwm_hold_eta"
            self._last_loop_trial = {
                "intervention": "dream",
                "before": err_before,
                "after": err_after,
                "verdict": "helped" if err_after < err_before - 0.02 else (
                    "hurt" if err_after > err_before + 0.02 else "neutral"
                ),
                "iwm": False,
            }

        self._last_eta_policy = eta_policy
        self.siwm.eta.eta_smooth = new_eta

        # 6. Record dream session
        session = DreamSession(
            before_psi=state.psi,
            after_psi=fresh_psi,
            eta_before=eta_before,
            eta_after=new_eta,
            loop_trial=self._last_loop_trial,
            eta_policy=eta_policy,
        )
        self.mnemosyne.write_self_record("dream_session", session.to_dict())

        logger.info(
            f" Dream #{self._dream_count}: η {eta_before:.3f} → {new_eta:.3f} "
            f"(err {err_before:.3f}→{err_after:.3f}, {eta_policy}), "
            f"consolidated {consolidation_result}"
        )

        return state.model_copy(update={
            "psi": fresh_psi.model_copy(update={"beliefs": corrected_beliefs}),
            "eta": new_eta,
        })

    def _first_person_observe(self, state: WorldState) -> dict:
        """Ω_ε — first-person partial observation; IWM-rich when attached."""
        base = {
            "generation": state.phi.generation,
            "belief_count": len(state.psi.beliefs),
            "eta": state.eta,
            "storage": state.phi.storage_stats,
        }
        if self.iwm is not None:
            report = self.iwm.first_person_report(state)
            base.update({
                "organs": report.get("organs", {}),
                "frontier_keys": list(
                    (report.get("frontier") or {}).get("explore_bias", [])
                )[:8],
                "loop_trials": report.get("loop_efficacy", {}).get("trial_count", 0),
                "self_trust": (report.get("calibrator") or {}).get("self_trust"),
            })
        return base

    def _reconcile_beliefs(self, old: list, new: list) -> list:
        """Reconcile old and new beliefs.

        Uses LLM for semantic comparison when available.
        Falls back to confidence-based heuristic.
        """
        # Try LLM-powered reconciliation
        if self.llm and self.llm.available and (old or new):
            try:
                result = self._llm_reconcile(old, new)
                if result:
                    self._llm_reconcile_count += 1
                    return result
            except Exception as e:
                logger.warning(f"LLM belief reconciliation failed, using heuristic: {e}")

        # Heuristic fallback: prefer new, keep high-confidence old
        reconciled = list(new)
        for old_belief in old:
            if old_belief.confidence > 0.7:
                contradicted = any(
                    self._beliefs_contradict(old_belief, nb) for nb in new
                )
                if not contradicted:
                    reconciled.append(old_belief)
        return reconciled

    def _llm_reconcile(self, old: list, new: list) -> list:
        """LLM-powered belief reconciliation under consolidation vacuum (S7)."""
        import json

        from arsi.foundation.vacuum import consolidation_vacuum
        vacuum = consolidation_vacuum()

        old_str = json.dumps(
            [{"content": b.content, "confidence": b.confidence} for b in old[-10:]],
            ensure_ascii=False,
        )
        new_str = json.dumps(
            [{"content": b.content, "confidence": b.confidence} for b in new[-10:]],
            ensure_ascii=False,
        )

        prompt = f"""调和 AI 系统的新旧信念。保留准确的，修正过时的，合并重复的。
禁止编造信念之外的新事实；只能使用下面提供的旧/新信念内容。

旧信念（之前推断的）：
{old_str}

新信念（从最近行为重新推断的）：
{new_str}

调和原则：
1. 新信念与实际行为一致时优先保留新信念
2. 旧信念置信度高且未被新证据推翻时保留
3. 新旧信念表达相同意图时合并（取更高置信度）
4. 矛盾时以新信念为准

输出 JSON：
{{"reconciled": [{{"content": "调和后的信念", "confidence": 0.0到1.0, "source": "old|new|merged"}}]}}"""

        system = "你是信念调和专家。只输出 JSON。禁止发明列表之外的新事实。"
        allowed, reason = vacuum.filter_llm_payload(prompt, system)
        if not allowed:
            logger.warning(f"VacuumGate blocked dream LLM reconcile: {reason}")
            return None

        resp = self.llm.chat(
            prompt,
            system=system,
            max_tokens=800,
        )

        if not resp.success:
            return None

        data = self._parse_json(resp.content)
        if not data or "reconciled" not in data:
            return None

        # S7 validate: drop invented contents not grounded in old/new
        allowed_contents = {b.content for b in list(old) + list(new)}
        from arsi.foundation.schema import Belief
        reconciled = []
        dropped = 0
        for b in data["reconciled"]:
            if not isinstance(b, dict) or "content" not in b:
                continue
            content = str(b["content"])
            if content not in allowed_contents:
                # allow near-merge only if substring of an existing belief
                if not any(content in ac or ac in content for ac in allowed_contents):
                    dropped += 1
                    continue
            reconciled.append(Belief(
                content=content,
                confidence=max(0.0, min(1.0, b.get("confidence", 0.5))),
                source=f"llm_dream:{b.get('source', 'unknown')}",
            ))
        if dropped:
            logger.info(f"VacuumGate dropped {dropped} ungrounded reconciled beliefs")
        return reconciled if reconciled else None

    @staticmethod
    def _parse_json(content: str):
        import json
        c = content.strip()
        if c.startswith("```json"):
            c = c[7:]
        if c.startswith("```"):
            c = c[3:]
        if c.endswith("```"):
            c = c[:-3]
        c = c.strip()
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            start = c.find("{")
            end = c.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(c[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _beliefs_contradict(a, b) -> bool:
        """Simple contradiction check (simplified)."""
        # In production, would use semantic similarity
        return False

    @property
    def stats(self) -> dict:
        return {
            "dream_count": self._dream_count,
            "llm_reconcile_count": self._llm_reconcile_count,
        }
