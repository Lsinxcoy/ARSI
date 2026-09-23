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
from arsi.meta.env_difficulty import (
    DifficultyComponents,
    ReferenceCorpus,
    compute_env_difficulty,
    compute_env_difficulty_dyn,
    pool_difficulty_stats,
    weakness_vs_difficulty,
)
from arsi.meta.el_scheduler import ELScheduler, LineageState
from arsi.meta.difficulty_flow_gate import (
    apply_dual_gate_to_selection,
    d_t_is_rising,
    difficulty_flow_gate,
    extract_d_t_history,
    flow_progress_score,
    same_generation_pairs,
)

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
    "DifficultyComponents",
    "ReferenceCorpus",
    "compute_env_difficulty",
    "compute_env_difficulty_dyn",
    "pool_difficulty_stats",
    "weakness_vs_difficulty",
    "ELScheduler",
    "LineageState",
    "apply_dual_gate_to_selection",
    "d_t_is_rising",
    "difficulty_flow_gate",
    "extract_d_t_history",
    "flow_progress_score",
    "same_generation_pairs",
]
