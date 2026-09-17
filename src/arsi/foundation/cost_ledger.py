"""Cost Accounting — four non-fungible ledgers.

Tracks: token consumption, wall-clock time, compute, verifier queries.
Based on: ARSI Whitepaper §9.3 + MetaRSI budget B = (β_tok, β_clk, β_gpu, β_qry)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CostLedger:
    """Four non-fungible cost ledgers."""
    # Token consumption (LLM calls)
    tokens_input: int = 0
    tokens_output: int = 0
    llm_calls: int = 0

    # Wall-clock time
    start_time: float = field(default_factory=time.time)
    active_seconds: float = 0.0
    _last_active: float = field(default_factory=time.time)

    # Compute (placeholder for GPU)
    compute_units: float = 0.0

    # Verifier queries (sealed evaluation)
    verifier_queries: int = 0

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record an LLM call."""
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.llm_calls += 1
        self._touch()

    def record_verifier_query(self, count: int = 1) -> None:
        """Record sealed evaluation queries."""
        self.verifier_queries += count
        self._touch()

    def record_compute(self, units: float) -> None:
        """Record compute usage."""
        self.compute_units += units
        self._touch()

    def _touch(self) -> None:
        """Update active time tracking."""
        now = time.time()
        self.active_seconds += now - self._last_active
        self._last_active = now

    def snapshot(self) -> dict:
        """Get current cost snapshot."""
        self._touch()
        total_tokens = self.tokens_input + self.tokens_output
        return {
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": total_tokens,
                "llm_calls": self.llm_calls,
            },
            "time": {
                "active_seconds": round(self.active_seconds, 2),
                "active_minutes": round(self.active_seconds / 60, 2),
            },
            "compute": {
                "units": self.compute_units,
            },
            "verifier": {
                "queries": self.verifier_queries,
            },
        }

    def cost_benefit(self, gain: float) -> dict:
        """Compute cost-benefit ratio."""
        snap = self.snapshot()
        total_tokens = snap["tokens"]["total"]
        total_seconds = snap["time"]["active_seconds"]

        return {
            "gain": round(gain, 4),
            "cost": {
                "tokens": total_tokens,
                "seconds": round(total_seconds, 2),
                "verifier_queries": snap["verifier"]["queries"],
            },
            "efficiency": {
                "gain_per_1k_tokens": round(gain / max(total_tokens / 1000, 0.001), 4),
                "gain_per_minute": round(gain / max(total_seconds / 60, 0.001), 4),
                "gain_per_query": round(gain / max(snap["verifier"]["queries"], 1), 4),
            },
        }

    def reset(self) -> None:
        """Reset all ledgers (start of new term)."""
        self.tokens_input = 0
        self.tokens_output = 0
        self.llm_calls = 0
        self.active_seconds = 0.0
        self._last_active = time.time()
        self.compute_units = 0.0
        self.verifier_queries = 0
