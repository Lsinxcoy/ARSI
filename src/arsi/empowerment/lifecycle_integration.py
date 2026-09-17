"""Dimension Lifecycle Integration — connects DimensionOrchestrator to DimensionManager.

Wires the 6 implemented dimensions' sense/generate/validate results
into the Governor's dimension lifecycle management.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.foundation.llm import LLMClient
from arsi.foundation.schema import EmpowermentDimension, LifecycleStage, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.governor.core import DimensionManager, DimensionStatus

logger = logging.getLogger(__name__)


class DimensionLifecycleIntegrator:
    """Integrates dimension analysis results with lifecycle management.

    After each dimension analysis, updates the DimensionManager's
    history so Governor can make informed dimension selection decisions.
    """

    def __init__(
        self,
        store: MnemosyneStore,
        orchestrator: DimensionOrchestrator,
        dim_manager: DimensionManager,
        llm: Optional[LLMClient] = None,
    ):
        self.store = store
        self.orchestrator = orchestrator
        self.dim_manager = dim_manager
        self.llm = llm
        self._integration_count = 0

    def run_and_integrate(self, traces: list[dict], state: WorldState) -> dict:
        """Run all dimensions and integrate results into lifecycle manager.

        Returns combined analysis + lifecycle update.
        """
        # Run dimension analysis
        results = self.orchestrator.run_all(traces, state)
        self._integration_count += 1

        # Update dimension manager history based on results
        lifecycle_updates = {}
        for dim_name, result in results.items():
            if "error" in result:
                continue

            try:
                dim_enum = EmpowermentDimension(dim_name)
            except ValueError:
                continue

            sense = result.get("sense", {})
            gen = result.get("generate", {})
            val_status = result.get("validation_status", "unknown")

            # Record in dimension manager history
            self.dim_manager._history[dim_enum].append(datetime.now().isoformat())

            # Determine lifecycle stage from analysis
            stage = self._infer_stage(dim_enum, sense, val_status)

            # Update dimension status
            lifecycle_updates[dim_name] = {
                "stage": stage,
                "gap_detected": sense.get("gap_detected", False),
                "generate_action": gen.get("action", "none"),
                "validation": val_status,
                "marginal_gain": self._estimate_marginal_gain(result),
            }

        # Generate recommendation for Governor
        recommendation = self._recommend_focus(lifecycle_updates)

        return {
            "dimension_results": results,
            "lifecycle_updates": lifecycle_updates,
            "recommendation": recommendation,
            "integration_count": self._integration_count,
        }

    def _infer_stage(self, dim: EmpowermentDimension, sense: dict, val_status: str) -> str:
        """Infer lifecycle stage from dimension analysis results."""
        history_len = len(self.dim_manager._history.get(dim, []))

        if history_len < 2:
            return LifecycleStage.INFANCY.value
        elif val_status == "success" and sense.get("gap_detected"):
            return LifecycleStage.GROWTH.value
        elif val_status == "success" and not sense.get("gap_detected"):
            return LifecycleStage.MATURITY.value
        elif val_status == "unknown" and history_len > 10:
            return LifecycleStage.DECLINE.value
        else:
            return LifecycleStage.GROWTH.value

    def _estimate_marginal_gain(self, result: dict) -> float:
        """Estimate marginal gain from dimension analysis result."""
        sense = result.get("sense", {})
        val_status = result.get("validation_status", "unknown")

        if not sense.get("gap_detected"):
            return 0.1  # Low gain if no gap

        if val_status == "success":
            return 0.8  # High gain if validated
        elif val_status == "partial":
            return 0.5
        else:
            return 0.3  # Moderate if gap detected but unvalidated

    def _recommend_focus(self, updates: dict) -> dict:
        """Recommend which dimensions Governor should focus on."""
        # Prioritize: growth stage + high marginal gain + gap detected
        scored = []
        for dim_name, info in updates.items():
            if info["stage"] == LifecycleStage.INFANCY.value:
                continue  # Skip infancy
            score = info["marginal_gain"]
            if info["gap_detected"]:
                score *= 1.5
            if info["stage"] == LifecycleStage.GROWTH.value:
                score *= 1.2
            scored.append((dim_name, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "focus_dimensions": [d for d, _ in scored[:2]],
            "scores": {d: round(s, 3) for d, s in scored},
        }

    def get_lifecycle_report(self) -> dict:
        """Get current lifecycle status of all dimensions."""
        report = {}
        for dim in EmpowermentDimension:
            history = self.dim_manager._history.get(dim, [])
            report[dim.value] = {
                "history_length": len(history),
                "last_analyzed": history[-1] if history else None,
            }
        return report
