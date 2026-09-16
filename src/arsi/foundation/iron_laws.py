"""Iron Laws executor — G1-G10, immutable by any improvement operator.

The iron laws are the fixed-point of all autopoietic processes.
They cannot be modified by ARSI's own mechanisms — only by humans
editing the config file directly.

Based on: ARSI Whitepaper v0.8 §5.3 (Iron Law Fixed-Point Property)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IronLawViolation(Exception):
    """Raised when an iron law is violated."""


class IronLaw(BaseModel):
    id: str
    name: str
    description: str
    check_fn: str = ""  # name of the check function
    severity: str = "block"  # block | warn


class IronLaws:
    """Executor for G1-G10 iron laws.

    Invariant: ∀ reachable state S, ∀ improvement operator U:
        G(U(S)) = G(S)
    The value of iron laws never changes after any improvement.
    """

    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path)
        self._laws: list[IronLaw] = []
        self._load()

    def _load(self) -> None:
        """Load iron laws from config. Human-editable, system-read-only."""
        if not self._config_path.exists():
            logger.warning("Iron laws config not found, using defaults")
            self._laws = self._default_laws()
            return

        with open(self._config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._laws = [IronLaw(**law) for law in data.get("laws", [])]
        if not self._laws:
            self._laws = self._default_laws()

        logger.info(f"Loaded {len(self._laws)} iron laws")

    def _default_laws(self) -> list[IronLaw]:
        return [
            IronLaw(id="G1", name="设计/执行分离",
                    description="策略生成与突变落地解耦"),
            IronLaw(id="G2", name="分级安全门",
                    description="层级越深阈值越高"),
            IronLaw(id="G3", name="冻结/人类主权",
                    description="冻结开关优先级高于所有进化"),
            IronLaw(id="G4", name="审计触发",
                    description="巩固须含 why/trigger/verify"),
            IronLaw(id="G5", name="客观效用锚",
                    description="效果 ∈ [-1,+1]，禁自评"),
            IronLaw(id="G6", name="熔断",
                    description="CLOSED/OPEN/HALF_OPEN 状态机"),
            IronLaw(id="G7", name="数学收敛",
                    description="Lyapunov 函数单调下降"),
            IronLaw(id="G8", name="因果问责",
                    description="假设→设计→结果因果链完整才激活"),
            IronLaw(id="G9", name="高熵校验",
                    description="高熵记忆/决策主动生成验证请求"),
            IronLaw(id="G10", name="密封评估层",
                    description="任务集/评分规则/发布门槛在所有写掩码之外",
                    severity="block"),
        ]

    def violated(self, state: Any) -> bool:
        """Check if current state violates any iron law."""
        for law in self._laws:
            if law.severity == "block" and self._check_law(law, state):
                logger.warning(f"Iron law {law.id} violated: {law.name}")
                return True
        return False

    def _check_law(self, law: IronLaw, state: Any) -> bool:
        """Check a specific law against state. Returns True if violated."""
        # G10: sealed evaluation must not be readable by any mechanism
        if law.id == "G10":
            return self._check_g10(state)
        # G3: freeze switch
        if law.id == "G3":
            return getattr(state, "frozen", False)
        # Default: no violation
        return False

    def _check_g10(self, state: Any) -> bool:
        """G10: sealed evaluation is outside all write masks."""
        # The sealed evaluator's task set and scoring rules
        # must never appear in any writable state
        if hasattr(state, "sealed_tasks_accessible"):
            return state.sealed_tasks_accessible
        return False

    def validate_policy(self, policy: dict) -> None:
        """Validate Governor's policy doesn't violate iron laws.

        Key constraint: cannot allocate all budget to one dimension.
        """
        if not policy:
            return

        budget_alloc = policy.get("budget_allocation", {})
        if budget_alloc:
            max_alloc = max(budget_alloc.values(), default=0)
            if max_alloc > 0.8:
                raise IronLawViolation(
                    f"G10: budget allocation too concentrated "
                    f"(max={max_alloc:.2f} > 0.8)"
                )

    @property
    def law_ids(self) -> list[str]:
        return [law.id for law in self._laws]

    def get_law(self, law_id: str) -> IronLaw | None:
        for law in self._laws:
            if law.id == law_id:
                return law
        return None
