"""ARSI architecture pillars after RSI second-pass (2026-09-23).

L0 Constitution (GAI) · L1 Contracted Computation (SEVerA FGGM)
L2 Regularized MetaRSI (RRSI × AIDE²) · L3 Introspective World + vitals (SAHOO)
L4 Dream-RSI · L5 Evaluative Epistemology (Grader)
Autopoiesis = charter across layers, not a peer column.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

PILLARS: dict[str, dict] = {
    "L0_constitution": {
        "name": "Anchored Constitution",
        "source": "GAI 2609.13406 × Autopoiesis (P-R6)",
        "question": "What are we allowed to be?",
        "core": [
            "dial1_modifier_in_agent=RSI",
            "dial2_base_grounded",
            "iron_laws",
            "never_goal_drift",
            # P-R6 organization/structure dichotomy (Maturana)
            "organization=rho_iron_laws_sealed_goals",
            "structure=omega_harness_host_policies",
            "stage3_may_change_structure_never_organization",
        ],
        "modules": ["harness.gai", "foundation.iron_laws", "harness.contracts", "harness.epistemic"],
    },
    "L1_contract": {
        "name": "Contracted Computation",
        "source": "SEVerA 2603.25111",
        "question": "What is always true at every call?",
        "core": ["FGGM Φ/Ψ", "check_or_verified_fallback", "forall_theta_soundness"],
        "modules": ["harness.contracts"],
    },
    "L2_metarsi": {
        "name": "Regularized MetaRSI",
        "source": "RRSI 2609.24972 × AIDE² 2609.26457",
        "question": "How do we change without fooling ourselves?",
        "core": ["bilevel pub≠priv", "fixed_budget", "annealed_L0", "noise_floor", "cost_rule", "hidden_grade"],
        "modules": [
            "harness.edit_budget",
            "harness.noise_floor",
            "harness.credit",
            "harness.accept",
            "harness.prune",
            "harness.holdout",
            "harness.hack_kpi",
        ],
    },
    "L3_world_self": {
        "name": "Introspective World + Vitals",
        "source": "SIWM/ODEWorld/CTM + SAHOO 2603.06333",
        "question": "How do world and self flow, and when do we drift?",
        "core": ["eta", "capability_flow", "sync", "GDI", "CAR", "regression_risk"],
        "modules": ["iwm", "world_model.capability_flow", "harness.sahoo", "foundation.sync_repr"],
    },
    "L4_dream_rsi": {
        "name": "Dream-RSI Policy Meta-layer",
        "source": "Dream-RSI 2609.14858",
        "question": "How are exploration policies dreamed?",
        "core": ["world_pool", "replay", "beta", "V_star_ge_V0"],
        "modules": ["world_model.world_pool", "meta.eval_loop", "governor.portfolio_policy"],
    },
    "L5_epistemology": {
        "name": "Evaluative Epistemology",
        "source": "Grader 2607.12790",
        "question": "What counts as evidence?",
        "core": ["drawback_not_goodness", "validity_neq_sufficiency", "harden_soft_anchors", "detectability_spectrum"],
        "modules": ["harness.drawback", "foundation.verified", "harness.holdout"],
    },
}

CHARTER = {
    "name": "Autopoiesis",
    "role": "charter_across_layers",
    "question": "Why is the system alive?",
    "core": [
        "boundary",
        "self_production",
        "governor",
        # P-R6 / P-R8
        "organization_invariant_structure_open",
        "metabolism=traces_effects_cost",
        "membrane=iron_laws",
    ],
    "modules_note": "harness.metabolism · harness.epistemic · harness.multiscale",
}


@dataclass
class PillarAudit:
    missing_modules: list[str] = field(default_factory=list)
    ok: bool = True
    pillars: dict = field(default_factory=dict)


def audit_pillars() -> PillarAudit:
    """Check declared modules are importable — pillars must map to real code."""
    missing = []
    mapped = {}
    for pid, meta in PILLARS.items():
        mods = []
        for m in meta.get("modules") or []:
            try:
                __import__(m if m.startswith("arsi") else f"arsi.{m}" if "." not in m else f"arsi.{m}")
                mods.append((m, True))
            except Exception:
                # allow dotted under arsi
                try:
                    __import__(f"arsi.{m}")
                    mods.append((m, True))
                except Exception:
                    mods.append((m, False))
                    missing.append(m)
        mapped[pid] = mods
    return PillarAudit(missing_modules=missing, ok=not missing, pillars=mapped)


def pillar_summary() -> dict:
    return {"charter": CHARTER, "pillars": PILLARS, "count": len(PILLARS)}
