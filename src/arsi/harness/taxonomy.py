"""Nine-dimensional harness taxonomy (HarnessX §3.3) × five modules (ModularRSI §3.2).

c7 control/safety = Iron Laws — **frozen**, never an evolution target.
"""
from __future__ import annotations

from typing import Optional, Sequence

# HarnessX nine dimensions
DIMENSIONS: dict[str, str] = {
    "c1": "model_selection",
    "c2": "context_assembly",
    "c3": "memory_management",
    "c4": "tool_ecosystem",
    "c5": "execution_environment",
    "c6": "evaluation_and_reward",
    "c7": "control_and_safety",
    "c8": "observability",
    "c9": "training_bridge",
}

# Never evolve these (ARSI: iron laws / freeze / circuit-break)
FROZEN_DIMS = frozenset({"c7"})


def aegis_fidelity_check(
    *,
    frozen_dims: Optional[Sequence[str]] = None,
    has_digester: bool = True,
    has_planner: bool = True,
    has_evolver: bool = True,
    has_gates: bool = True,
    variant_isolation: bool = True,
) -> dict:
    """HarnessX AEGIS invariants: full pipeline + c7 frozen + variant isolation."""
    fr = set(frozen_dims if frozen_dims is not None else FROZEN_DIMS)
    ok = (
        "c7" in fr
        and has_digester
        and has_planner
        and has_evolver
        and has_gates
        and variant_isolation
    )
    return {
        "ok": ok,
        "frozen_dims": sorted(fr),
        "pipeline": {
            "digester": has_digester,
            "planner": has_planner,
            "evolver": has_evolver,
            "gates": has_gates,
        },
        "variant_isolation": variant_isolation,
        "note": "harnessx_aegis_fidelity",
    }


def contrastive_batches(
    outcomes: Sequence[tuple[str, bool]],
) -> dict:
    """ModularRSI contrastive sampling over (task_id, success) rollouts.

    Positive: all success · Contrastive: mixed · Negative: all fail.
    """
    by_task: dict[str, list[bool]] = {}
    for tid, ok in outcomes or []:
        by_task.setdefault(str(tid), []).append(bool(ok))
    pos, con, neg = [], [], []
    for tid, rows in by_task.items():
        if all(rows):
            pos.append(tid)
        elif any(rows):
            con.append(tid)
        else:
            neg.append(tid)
    return {
        "positive": sorted(pos),
        "contrastive": sorted(con),
        "negative": sorted(neg),
        "n_tasks": len(by_task),
        "note": "modular_rsi_contrastive_sampling",
    }


def module_scope_ok(module: str, file_path: str) -> bool:
    """Modification must stay inside the module's declared file scope."""
    meta = MODULES.get(str(module) or "")
    if not meta:
        return False
    files = meta.get("files") or []
    fp = str(file_path or "")
    return any(fp.startswith(f) or f in fp for f in files)

# ModularRSI five behavioral modules → default dimension ownership
MODULES: dict[str, dict] = {
    "agent_loop": {
        "dims": ["c2", "c5", "c7"],
        "primary_dim": "c2",
        "files": ["src/arsi/core.py", "src/arsi/governor/core.py", "scripts/arsi_daemon.py"],
    },
    "tool_use": {
        "dims": ["c4"],
        "primary_dim": "c4",
        "files": ["src/arsi/empowerment/applier.py", "src/arsi/adapters/"],
    },
    "observation_management": {
        "dims": ["c8", "c3"],
        "primary_dim": "c8",
        "files": ["src/arsi/mnemosyne/", "src/arsi/foundation/observation_pack.py"],
    },
    "context_management": {
        "dims": ["c3", "c2"],
        "primary_dim": "c3",
        "files": ["src/arsi/mnemosyne/", "src/arsi/adapters/bidirectional_interface.py"],
    },
    "task_completion_detection": {
        "dims": ["c6"],
        "primary_dim": "c6",
        "files": ["src/arsi/sealed_eval/", "src/arsi/foundation/verified.py"],
    },
    "infrastructure": {
        "dims": ["c5", "c1", "c7", "c9"],
        "primary_dim": "c5",
        "files": ["src/arsi/foundation/llm.py", "src/arsi/foundation/iron_laws.py"],
        "frozen": True,
    },
}

# Symbol path hints → dimension (for scope fence / dimension_of_symbol)
_SYMBOL_HINTS: list[tuple[str, str]] = [
    ("iron_laws", "c7"),
    ("IronLaws", "c7"),
    ("freeze", "c7"),
    ("llm", "c1"),
    ("LLMClient", "c1"),
    ("prompt", "c2"),
    ("brief", "c2"),
    ("memory", "c3"),
    ("Mnemosyne", "c3"),
    ("empowerment", "c4"),
    ("skill", "c4"),
    ("adapter", "c4"),
    ("sandbox", "c5"),
    ("daemon", "c5"),
    ("code_verifier", "c6"),
    ("sealed", "c6"),
    ("verified", "c6"),
    ("quality_gate", "c6"),
    ("observ", "c8"),
    ("health", "c8"),
    ("sync", "c8"),
    ("iwm", "c8"),
    ("train", "c9"),
    ("grpo", "c9"),
]


def dimension_of_symbol(symbol: str) -> str:
    """Best-effort map of a code symbol/path to a taxonomy dimension."""
    s = str(symbol or "")
    for hint, dim in _SYMBOL_HINTS:
        if hint in s:
            return dim
    return "c2"


def module_scope(module: str) -> dict:
    return dict(MODULES.get(module) or {"dims": [], "primary_dim": "", "files": [], "unknown": True})


def is_edit_allowed(dim: str, module: str = "", symbol: str = "") -> tuple[bool, str]:
    """Scope fence: frozen dims and frozen modules reject edits."""
    d = dim or dimension_of_symbol(symbol)
    if d in FROZEN_DIMS:
        return False, f"frozen_dimension:{d}"
    mod = MODULES.get(module or "")
    if mod and mod.get("frozen"):
        return False, f"frozen_module:{module}"
    if module and mod and d not in (mod.get("dims") or []) and d not in FROZEN_DIMS:
        # soft: still allow if dim is not frozen but outside module — warn
        return True, f"dim_outside_module:{d}"
    return True, "ok"
