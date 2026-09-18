"""Evaluation closed loop — Phase D7.

Pool-monotone (Dream-RSI selection guarantee) ≠ live-monotone.
This module:
  1. Compares exploration_mode=fixed vs dream_rsi on replay + recent live scores
  2. Detects live regression window
  3. Auto-rolls back deployed policy/beta when dream underperforms
  4. Writes compare JSON under archive/eval/
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn
from arsi.meta.dream_rsi_params import DreamRSIParams

logger = logging.getLogger(__name__)


@dataclass
class CompareResult:
    timestamp: str = ""
    world_count: int = 0
    fixed_avg_score: float = 0.0
    fixed_avg_quality: float = 0.0
    dream_avg_score: float = 0.0
    dream_avg_quality: float = 0.0
    delta_score: float = 0.0
    live_scores: list[float] = field(default_factory=list)
    live_regression: bool = False
    rollback_triggered: bool = False
    rollback_to: str = ""
    recommendation: str = ""
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def dream_wins_pool(self) -> bool:
        return self.dream_avg_score >= self.fixed_avg_score - 1e-9

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_exploration_fn(max_workers: int = 2):
    """Controlled baseline: deterministic first-k eligible (no portfolio adaptivity)."""

    def _fn(observed, legal, max_parallelism=3):
        return list(legal)[: max(1, min(max_workers, max_parallelism or max_workers))]

    return _fn


def detect_live_regression(
    live_scores: list[float],
    window: int = 3,
    eps: float = 0.02,
) -> tuple[bool, str]:
    """True if recent live best scores regress vs prior window."""
    if len(live_scores) < window + 1:
        return False, "insufficient_live_history"
    recent = live_scores[-window:]
    prior = live_scores[-(2 * window):-window]
    rec = sum(recent) / len(recent)
    pri = sum(prior) / len(prior)
    if rec + eps < pri:
        return True, f"live_avg {rec:.4f} < prior_avg {pri:.4f} - eps={eps}"
    # also: strictly non-increasing last `window` steps with drop
    if all(live_scores[-i] <= live_scores[-i - 1] + 1e-9 for i in range(1, window)):
        drop = live_scores[-window] - live_scores[-1]
        if drop >= eps:
            return True, f"monotone_drop_{drop:.4f}_over_{window}_cycles"
    return False, "ok"


def run_eval_loop(
    arsi,
    params: Optional[DreamRSIParams] = None,
    report_dir: Optional[str | Path] = None,
) -> CompareResult:
    """Compare fixed vs dream_rsi; rollback on live regression."""
    params = params or getattr(arsi, "dream_rsi_params", None) or DreamRSIParams()
    result = CompareResult()
    pool = arsi.world_pool

    if pool.size == 0:
        result.recommendation = "empty_pool"
        return result

    result.world_count = pool.size
    fixed_eval = pool.evaluate_policy_across_pool(
        fixed_exploration_fn(), policy_name="fixed"
    )
    dream_eval = pool.evaluate_policy_across_pool(
        build_policy_fn(arsi.portfolio_policy), policy_name="dream_rsi"
    )
    result.fixed_avg_score = float(fixed_eval.get("avg_score", 0.0))
    result.fixed_avg_quality = float(fixed_eval.get("avg_quality", 0.0))
    result.dream_avg_score = float(dream_eval.get("avg_score", 0.0))
    result.dream_avg_quality = float(dream_eval.get("avg_quality", 0.0))
    result.delta_score = round(result.dream_avg_score - result.fixed_avg_score, 4)

    live = list(getattr(arsi, "_term_best_scores", []) or [])
    # merge manifest live scores
    try:
        for row in arsi.manifest_store.beta_history(n=20):
            if "best_score" in row:
                live.append(float(row["best_score"]))
    except Exception:
        pass
    result.live_scores = [float(x) for x in live]
    regressed, reg_reason = detect_live_regression(
        result.live_scores,
        window=params.rollback_window,
        eps=params.rollback_delta_eps,
    )
    result.live_regression = regressed
    result.notes["regression_reason"] = reg_reason
    result.notes["pool_dream_wins"] = result.dream_wins_pool
    result.notes["pool_delta"] = result.delta_score
    result.notes["official_code_status"] = params.official_code_status
    result.notes["params"] = params.to_dict()

    # D7 gate: pool monotone does NOT guarantee live monotone
    if regressed:
        prev_beta = float(result.notes.get("prev_beta", getattr(arsi.portfolio_policy, "beta", 0.6)))
        fallback_beta = params.beta_default_uncertain
        # rollback beta one step toward conservative
        new_beta = max(0.0, min(1.0, prev_beta - params.wasteful_lower_step))
        if abs(new_beta - prev_beta) < 1e-6:
            new_beta = fallback_beta
        arsi.portfolio_policy = PortfolioPolicy(
            beta=new_beta,
            max_workers=max(1, getattr(arsi.portfolio_policy, "max_workers", 2)),
            name=f"rollback_beta_{new_beta:.2f}",
        )
        result.rollback_triggered = True
        result.rollback_to = arsi.portfolio_policy.name
        result.recommendation = f"ROLLBACK_LIVE_REGRESSION:{reg_reason}"
        logger.warning(f"D7 auto-rollback: {reg_reason} → beta={new_beta}")
    elif not result.dream_wins_pool:
        result.recommendation = "POOL_FIXED_WINS_MONITOR_LIVE"
    else:
        result.recommendation = "DREAM_OK"

    # Persist report
    rdir = Path(report_dir) if report_dir else (
        Path(getattr(arsi, "project_root", Path.cwd())) / "archive" / "eval"
    )
    # prefer E:\ARSI\archive\eval when running from project
    default_eval = Path(__file__).resolve().parents[3] / "archive" / "eval"
    if report_dir is None:
        rdir = default_eval
    try:
        rdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = rdir / f"compare_{ts}.json"
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        result.notes["report_path"] = str(path)
        # also append line to rolling compare log
        log_path = rdir / "compare_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write eval report: {e}")
        result.notes["report_error"] = str(e)

    return result
