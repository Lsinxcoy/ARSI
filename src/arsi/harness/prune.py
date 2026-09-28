"""RRSI L1-style structural pruning targets (arXiv:2609.24972 Eq.14).

B_t = {ℓ ∈ T_t : g_t(ℓ) ≤ 0} — exercised but no strictly positive recent gain.
"""
from __future__ import annotations

from arsi.harness.credit import CreditLedger


def prune_targets(ledger: CreditLedger) -> list[str]:
    return ledger.prune_targets()


def exploration_directive(
    ledger: CreditLedger,
    all_components: set[str],
    score_now: float,
    score_w_ago: float,
    delta: float,
    m_draft: int = 1,
) -> dict:
    """E_t = (σ_t, U_t, m_draft) — stall reserves slots for unexercised components."""
    stalled = (float(score_now) - float(score_w_ago)) <= float(delta)
    unex = sorted(set(all_components) - ledger.exercised())
    return {
        "stalled": stalled,
        "unexercised": unex,
        "m_draft": int(m_draft) if stalled else 0,
        "prune_targets": prune_targets(ledger),
    }
