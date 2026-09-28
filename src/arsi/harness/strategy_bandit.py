"""P-b Strategy-arm bandit for Evolver (AIDE² AIDE_85 second-pass).

UCB1 over edit_type arms + 30% softmax exploration; pick strategy first,
then generate within that arm. Diversity lever = arms, not individual nodes.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

DEFAULT_ARMS = (
    "conservative",
    "aggressive_rewrite",
    "ensemble",
    "tuned_specialist",
    "robust_simple",
)
EDIT_TYPE_ARMS = ("prompt", "processor", "tool", "config", "control")
SOFTMAX_EPS = 0.30  # AIDE85: 30% steps sample softmax over arm bests


@dataclass
class ArmStats:
    n: int = 0
    reward_sum: float = 0.0
    best: float = -1e9

    @property
    def mean(self) -> float:
        return self.reward_sum / self.n if self.n else 0.0


class StrategyBandit:
    """UCB1 over harness edit-type arms."""

    def __init__(self, arms: Sequence[str] = DEFAULT_ARMS, c: float = 1.4, seed: int = 7):
        self.arms = list(arms)
        self.c = float(c)
        self.stats: dict[str, ArmStats] = {a: ArmStats() for a in self.arms}
        self.total = 0
        self.rng = random.Random(seed)

    def _ucb(self, a: str) -> float:
        s = self.stats[a]
        if s.n == 0:
            return float("inf")
        bonus = self.c * math.sqrt(math.log(max(2, self.total)) / s.n)
        return s.mean + bonus

    def select(self) -> str:
        # 30% softmax over arm bests (AIDE85)
        if self.rng.random() < SOFTMAX_EPS:
            bests = [max(0.0, self.stats[a].best) for a in self.arms]
            mx = max(bests) if bests else 1.0
            exps = [math.exp(b - mx) for b in bests]
            z = sum(exps) or 1.0
            r = self.rng.random() * z
            acc = 0.0
            for a, e in zip(self.arms, exps):
                acc += e
                if r <= acc:
                    return a
            return self.arms[-1]
        return max(self.arms, key=self._ucb)

    def observe(self, arm: str, reward: float) -> None:
        s = self.stats.setdefault(arm, ArmStats())
        s.n += 1
        s.reward_sum += float(reward)
        s.best = max(s.best, float(reward))
        self.total += 1

    def report(self) -> dict:
        return {
            "total": self.total,
            "softmax_eps": SOFTMAX_EPS,
            "arms": {a: asdict(self.stats[a]) | {"mean": self.stats[a].mean, "ucb": self._ucb(a)} for a in self.arms},
            "note": "aide85_strategy_arm_bandit",
        }
