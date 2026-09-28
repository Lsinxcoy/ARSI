"""Introspector self-check (Q6) — is the introspector itself trustworthy?

Low self_trust → IWM control advice is ignored; system degrades to baseline.

C1-1 (CTM arXiv:2505.05522 t2): certainty must align with correctness.
High claimed confidence with wrong outcomes lowers self_trust via ECE.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional


@dataclass
class ControlOutcome:
    decision_id: str
    used_iwm: bool
    success: bool
    note: str = ""
    timestamp: str = ""
    confidence: float = 0.5


class IntrospectorCalibrator:
    def __init__(
        self,
        min_samples: int = 5,
        degrade_below: float = 0.35,
        require_baseline: bool = False,
        persist_path: Optional[str] = None,
    ):
        self.min_samples = min_samples
        self.degrade_below = degrade_below
        self.require_baseline = require_baseline
        self.persist_path = persist_path
        self._outcomes: Deque[ControlOutcome] = deque(maxlen=100)
        self._iwm_hits = 0
        self._iwm_total = 0
        self._baseline_hits = 0
        self._baseline_total = 0
        # C1-1: (confidence, correct) pairs for reliability / ECE
        self._conf_pairs: Deque[tuple[float, bool]] = deque(maxlen=200)
        self._load()

    @staticmethod
    def is_meaningful_success(success: bool, note: str = "") -> bool:
        """Stricter than exec_ok: empty/no-op actions are not success."""
        n = (note or "").lower()
        if "no new traces" in n or "waiting" in n or "unknown action" in n:
            return False
        if "skipped" in n or "insufficient" in n or "no_distill" in n or "learn_no_distill" in n:
            return False
        return bool(success)

    def record(
        self,
        decision_id: str,
        used_iwm: bool,
        success: bool,
        note: str = "",
        confidence: Optional[float] = None,
    ) -> None:
        success = self.is_meaningful_success(success, note)
        ts = datetime.now().isoformat()
        conf = 0.5 if confidence is None else max(0.0, min(1.0, float(confidence)))
        self._outcomes.append(ControlOutcome(decision_id, used_iwm, success, note, ts, conf))
        # Only explicit confidence claims enter reliability (t2) — never invent 0.5
        if confidence is not None:
            self._conf_pairs.append((max(0.0, min(1.0, float(confidence))), success))
        if used_iwm:
            self._iwm_total += 1
            if success:
                self._iwm_hits += 1
        else:
            self._baseline_total += 1
            if success:
                self._baseline_hits += 1
        self._save()

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            import json
            from pathlib import Path

            p = Path(self.persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        "iwm_hits": self._iwm_hits,
                        "iwm_total": self._iwm_total,
                        "baseline_hits": self._baseline_hits,
                        "baseline_total": self._baseline_total,
                        "conf_pairs": [[c, bool(ok)] for c, ok in self._conf_pairs],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load(self) -> None:
        if not self.persist_path:
            return
        try:
            import json
            from pathlib import Path

            p = Path(self.persist_path)
            if not p.exists():
                return
            d = json.loads(p.read_text(encoding="utf-8"))
            self._iwm_hits = int(d.get("iwm_hits") or 0)
            self._iwm_total = int(d.get("iwm_total") or 0)
            self._baseline_hits = int(d.get("baseline_hits") or 0)
            self._baseline_total = int(d.get("baseline_total") or 0)
            for c, ok in d.get("conf_pairs") or []:
                self._conf_pairs.append((float(c), bool(ok)))
        except Exception:
            pass

    @property
    def iwm_success_rate(self) -> float:
        if self._iwm_total == 0:
            return 0.0
        return self._iwm_hits / self._iwm_total

    @property
    def baseline_success_rate(self) -> float:
        if self._baseline_total == 0:
            return 0.0
        return self._baseline_hits / self._baseline_total

    def reliability(self, n_bins: int = 5) -> dict:
        """CTM-style certainty vs accuracy bins + ECE (lower is better)."""
        pairs = list(self._conf_pairs)
        if not pairs:
            return {"ece": 0.0, "n": 0, "bins": [], "note": "empty_confidence_pairs"}
        bins = []
        ece = 0.0
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            members = [
                (c, ok) for c, ok in pairs if (lo <= c < hi) or (b == n_bins - 1 and c >= lo and c <= 1.0)
            ]
            if not members:
                continue
            conf_mean = sum(c for c, _ in members) / len(members)
            acc = sum(1 for _, ok in members if ok) / len(members)
            gap = abs(conf_mean - acc)
            ece += (len(members) / len(pairs)) * gap
            bins.append(
                {
                    "lo": round(lo, 3),
                    "hi": round(hi, 3),
                    "n": len(members),
                    "conf_mean": round(conf_mean, 4),
                    "accuracy": round(acc, 4),
                    "gap": round(gap, 4),
                }
            )
        overconf = any(b["conf_mean"] - b["accuracy"] > 0.15 for b in bins if b["n"] >= 3)
        return {
            "ece": round(ece, 4),
            "n": len(pairs),
            "bins": bins,
            "overconfident": overconf,
            "note": "certainty_correctness_alignment",
        }

    def self_trust(self) -> float:
        """Trust in IWM-based control. Cap without baseline; penalize overconfidence."""
        if self._iwm_total < self.min_samples:
            return 0.5
        rate = self.iwm_success_rate
        if self._baseline_total < self.min_samples:
            # No control arm → cannot claim full trust (Q6 honesty)
            rate = min(rate, 0.75)
        # C1-1: bad calibration (esp. overconfidence) must not keep high trust
        rel = self.reliability()
        if rel.get("n", 0) >= self.min_samples:
            ece = float(rel.get("ece") or 0.0)
            if rel.get("overconfident"):
                rate *= 0.75
            rate *= max(0.5, 1.0 - min(0.5, ece))
        return max(0.0, min(1.0, rate))

    def should_degrade_to_baseline(self) -> bool:
        if self._iwm_total < self.min_samples:
            return False
        if self.iwm_success_rate < self.degrade_below:
            return True
        if (
            self._baseline_total >= self.min_samples
            and self.iwm_success_rate + 0.1 < self.baseline_success_rate
        ):
            return True
        rel = self.reliability()
        # Severe overconfidence with enough samples → degrade (t2 discipline)
        if rel.get("n", 0) >= self.min_samples and rel.get("overconfident") and rel.get("ece", 0) > 0.25:
            return True
        # Perfect rate with zero baseline evidence is not trustworthy enough to dominate
        if self.require_baseline and self._baseline_total < self.min_samples and self.iwm_success_rate >= 0.99:
            return False  # not degrade, but trust is capped above
        return False

    def report(self) -> dict:
        return {
            "iwm_total": self._iwm_total,
            "iwm_success_rate": round(self.iwm_success_rate, 4),
            "baseline_total": self._baseline_total,
            "baseline_success_rate": round(self.baseline_success_rate, 4),
            "self_trust": round(self.self_trust(), 4),
            "degrade_to_baseline": self.should_degrade_to_baseline(),
            "min_samples": self.min_samples,
            "degrade_below": self.degrade_below,
            "trust_capped_without_baseline": self._baseline_total < self.min_samples,
            "reliability": self.reliability(),
        }
