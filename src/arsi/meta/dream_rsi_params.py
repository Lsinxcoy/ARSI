"""Dream-RSI parameter loader (Phase D).

Official hyperparams unpublished → ARSI defaults in config/dream_rsi_params.yaml.
Backfill path: edit that YAML only when official code is released.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[3] / "config" / "dream_rsi_params.yaml"


@dataclass
class DreamRSIParams:
    beta1_cost_penalty: float = 0.1
    beta2_parallel_bonus: float = 0.05
    parallel_lambda: float = 0.05
    M_revisions_per_cycle: int = 2
    K2_replay_max_rounds: int = 20
    dream_every_n_steps: int = 8
    dream_every_n_ticks: int = 8
    beta_default_uncertain: float = 0.6
    beta_sweep_grid: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    plateau_raise_step: float = 0.15
    wasteful_lower_step: float = 0.15
    bootstrap_branch_count: int = 2
    bootstrap_refine_count: int = 3
    hard_max_branch_count: int = 4
    hard_max_refine_count: int = 8
    cost_unit_per_focus_step: float = 1.0
    default_cost_budget: float = 8.0
    world_min_verdict: str = "PASS"
    max_warn_ratio: float = 0.25
    max_admitted: int = 120
    score_mode: str = "quality_anchored"
    weak_quality_cost_scale: float = 0.2
    min_quality_signal: float = 0.08
    freeze_beta_on_degenerate: bool = True
    rollback_window: int = 3
    rollback_delta_eps: float = 0.02
    brief_policy: str = "structured"
    official_code_status: str = "not_released"
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "DreamRSIParams":
        data = data or {}
        obj = cls()
        rp = data.get("replay_objective") or {}
        obj.beta1_cost_penalty = float(rp.get("beta1_cost_penalty", obj.beta1_cost_penalty))
        obj.beta2_parallel_bonus = float(rp.get("beta2_parallel_bonus", obj.beta2_parallel_bonus))
        po = data.get("pareto") or {}
        obj.parallel_lambda = float(po.get("parallel_lambda", obj.parallel_lambda))
        loop = data.get("loop") or {}
        obj.M_revisions_per_cycle = int(loop.get("M_revisions_per_cycle", obj.M_revisions_per_cycle))
        obj.K2_replay_max_rounds = int(loop.get("K2_replay_max_rounds", obj.K2_replay_max_rounds))
        obj.dream_every_n_steps = int(loop.get("dream_every_n_steps", obj.dream_every_n_steps))
        obj.dream_every_n_ticks = int(loop.get("dream_every_n_ticks", obj.dream_every_n_ticks))
        beta = data.get("beta") or {}
        obj.beta_default_uncertain = float(beta.get("default_uncertain", obj.beta_default_uncertain))
        grid = beta.get("sweep_grid")
        if grid:
            obj.beta_sweep_grid = [float(x) for x in grid]
        obj.plateau_raise_step = float(beta.get("plateau_raise_step", obj.plateau_raise_step))
        obj.wasteful_lower_step = float(beta.get("wasteful_lower_step", obj.wasteful_lower_step))
        gp = data.get("grid_plan") or {}
        obj.bootstrap_branch_count = int(gp.get("bootstrap_branch_count", obj.bootstrap_branch_count))
        obj.bootstrap_refine_count = int(gp.get("bootstrap_refine_count", obj.bootstrap_refine_count))
        obj.hard_max_branch_count = int(gp.get("hard_max_branch_count", obj.hard_max_branch_count))
        obj.hard_max_refine_count = int(gp.get("hard_max_refine_count", obj.hard_max_refine_count))
        obj.cost_unit_per_focus_step = float(gp.get("cost_unit_per_focus_step", obj.cost_unit_per_focus_step))
        obj.default_cost_budget = float(gp.get("default_cost_budget", obj.default_cost_budget))
        qg = data.get("quality_gate") or {}
        obj.world_min_verdict = str(qg.get("world_min_verdict", obj.world_min_verdict))
        obj.max_warn_ratio = float(qg.get("max_warn_ratio", obj.max_warn_ratio))
        obj.max_admitted = int(qg.get("max_admitted", obj.max_admitted))
        obj.score_mode = str(qg.get("score_mode", obj.score_mode))
        obj.weak_quality_cost_scale = float(qg.get("weak_quality_cost_scale", obj.weak_quality_cost_scale))
        obj.min_quality_signal = float(qg.get("min_quality_signal", obj.min_quality_signal))
        obj.freeze_beta_on_degenerate = bool(qg.get("freeze_beta_on_degenerate", True))
        el = data.get("eval_loop") or {}
        obj.rollback_window = int(el.get("rollback_window", obj.rollback_window))
        obj.rollback_delta_eps = float(el.get("rollback_delta_eps", obj.rollback_delta_eps))
        br = data.get("brief") or {}
        obj.brief_policy = str(br.get("policy", obj.brief_policy))
        paper = data.get("paper") or {}
        obj.official_code_status = str(paper.get("official_code_status", obj.official_code_status))
        obj.raw = data
        return obj

    def to_dict(self) -> dict:
        return {
            "beta1_cost_penalty": self.beta1_cost_penalty,
            "beta2_parallel_bonus": self.beta2_parallel_bonus,
            "parallel_lambda": self.parallel_lambda,
            "M_revisions_per_cycle": self.M_revisions_per_cycle,
            "K2_replay_max_rounds": self.K2_replay_max_rounds,
            "dream_every_n_steps": self.dream_every_n_steps,
            "beta_default_uncertain": self.beta_default_uncertain,
            "beta_sweep_grid": self.beta_sweep_grid,
            "grid_bootstrap": [self.bootstrap_branch_count, self.bootstrap_refine_count],
            "grid_hard_max": [self.hard_max_branch_count, self.hard_max_refine_count],
            "world_min_verdict": self.world_min_verdict,
            "max_warn_ratio": self.max_warn_ratio,
            "score_mode": self.score_mode,
            "freeze_beta_on_degenerate": self.freeze_beta_on_degenerate,
            "rollback_window": self.rollback_window,
            "rollback_delta_eps": self.rollback_delta_eps,
            "brief_policy": self.brief_policy,
            "official_code_status": self.official_code_status,
        }


def load_dream_rsi_params(path: str | Path | None = None) -> DreamRSIParams:
    p = Path(path) if path else DEFAULT_PARAMS_PATH
    if not p.exists():
        logger.warning(f"dream_rsi_params.yaml not found at {p}, using built-in defaults")
        return DreamRSIParams()
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return DreamRSIParams.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load dream_rsi params ({e}), using defaults")
        return DreamRSIParams()
