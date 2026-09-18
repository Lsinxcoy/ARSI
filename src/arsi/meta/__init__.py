"""ARSI meta-layer: live manifests, grid planning, beta sweep, eval loop.

Phase D (Dream-RSI unmined mechanisms) — arXiv:2609.14858 Appendix B.2 adapted
to agent-consulting domain:
  - GridPlan.W = parallel focus count (dimensions/operators), not OS threads
  - GridPlan.R = refinement steps per focus
"""
from arsi.meta.live_manifest import LiveCycleManifest, ManifestStore
from arsi.meta.grid_plan import GridPlan, GridPlanningContext, plan_grid
from arsi.meta.beta_sweep import BetaSweepResult, sweep_beta
from arsi.meta.dream_rsi_params import DreamRSIParams, load_dream_rsi_params
from arsi.meta.eval_loop import CompareResult, detect_live_regression, run_eval_loop

__all__ = [
    "LiveCycleManifest",
    "ManifestStore",
    "GridPlan",
    "GridPlanningContext",
    "plan_grid",
    "BetaSweepResult",
    "sweep_beta",
    "DreamRSIParams",
    "load_dream_rsi_params",
    "CompareResult",
    "detect_live_regression",
    "run_eval_loop",
]
