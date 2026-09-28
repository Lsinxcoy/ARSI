"""P-e Detectability investment matrix (Grader second-pass).

Mechanical detectors buy agreement; semantic ones buy legibility.
Invest here before semantic LLM judges.
"""
from __future__ import annotations

# kind → (agreement_yield, legibility_yield, cost)  qualitative ranks 1–3
SPECTRUM = {
    "static_syntax": {"agreement": 3, "legibility": 2, "cost": 1, "examples": ["parse", "format", "select_star"]},
    "execution_probe": {"agreement": 3, "legibility": 3, "cost": 2, "examples": ["crash", "returns_not_print"]},
    "mechanical_rule": {"agreement": 2, "legibility": 3, "cost": 1, "examples": ["missing_group_by", "missing_required"]},
    "semantic_judge": {"agreement": 1, "legibility": 2, "cost": 3, "examples": ["spec_mismatch", "overclaim"]},
}


def investment_order() -> list[str]:
    """Build mechanical first; semantic last (detectability spectrum)."""
    return sorted(
        SPECTRUM.keys(),
        key=lambda k: (-(SPECTRUM[k]["agreement"] + SPECTRUM[k]["legibility"]) / SPECTRUM[k]["cost"], k),
    )


def recommend_next(existing: set[str]) -> dict:
    """Given already-built detector kinds, recommend the next investment."""
    order = investment_order()
    for k in order:
        if k not in (existing or set()):
            return {"next": k, "why": SPECTRUM[k], "order": order}
    return {"next": None, "why": "spectrum_covered", "order": order}
