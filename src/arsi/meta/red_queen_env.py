"""P-R5 Red Queen mutual pressure → EnvEvolution (Co-evo × multi-agent).

Hosts (hermes / mimo-desktop / synthex) are selection-pressure sources, not
just data sources. Minimal rule from rumination:

  success_rate↑  →  peer/self task difficulty D_T target↑  →  WorldEvolver effort↑

Red Queen gauge already exists in coevolution.red_queen_pressure; this module
wires the gauge into env-evolution commands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from arsi.harness.coevolution import red_queen_pressure

# default hosts in ARSI multi-agent
DEFAULT_HOSTS = ("hermes", "mimo-desktop", "synthex-mothernest")


@dataclass
class DifficultyCommand:
    host: str
    success_rate: float = 0.0
    success_trend: float = 0.0
    target_d_t: float = 0.0
    effort: str = "high"  # low | high | max
    pressure: float = 0.0
    reason: str = ""
    note: str = "red_queen_success_up_difficulty_up"

    def to_dict(self) -> dict:
        return asdict(self)


def success_rate(successes: Sequence[bool | int | float], window: int = 20) -> float:
    xs = list(successes or [])[-max(1, int(window)) :]
    if not xs:
        return 0.0
    hits = sum(1 for x in xs if float(x) >= 0.5 or x is True)
    return hits / len(xs)


def success_trend(successes: Sequence[bool | int | float], window: int = 5) -> float:
    """Recent mean − prior mean over binary/float success stream."""
    xs = [1.0 if (x is True or float(x) >= 0.5) else 0.0 for x in (successes or [])]
    if len(xs) < window + 1:
        return 0.0
    rec = sum(xs[-window:]) / window
    prior = sum(xs[-(2 * window) : -window]) / window
    return rec - prior


def effort_from_pressure(pressure: float) -> str:
    if pressure >= 0.25:
        return "max"
    if pressure >= 0.10:
        return "high"
    return "low"


def difficulty_command_for_host(
    host: str,
    successes: Sequence[bool | int | float],
    base_d_t: float = 1.0,
    peer_success_mean: Optional[float] = None,
) -> DifficultyCommand:
    """Map host success stream → D_T target + evolver effort.

    pressure = max(0, success_rate − peer_mean) + max(0, success_trend)
    target_d_t rises with pressure (Red Queen).
    """
    sr = success_rate(successes)
    tr = success_trend(successes)
    peer = float(peer_success_mean) if peer_success_mean is not None else sr
    pressure = max(0.0, sr - peer) + max(0.0, tr)
    target = float(base_d_t) * (1.0 + 2.0 * pressure)
    return DifficultyCommand(
        host=str(host),
        success_rate=round(sr, 4),
        success_trend=round(tr, 4),
        target_d_t=round(target, 4),
        effort=effort_from_pressure(pressure),
        pressure=round(pressure, 4),
        reason="success_up_raise_difficulty" if pressure > 0.05 else "pressure_low_hold",
    )


def mutual_pressure_plan(
    host_success: dict[str, Sequence],
    base_d_t: float = 1.0,
) -> dict:
    """All hosts: compute commands; peer mean excludes self (true mutual pressure)."""
    rates = {h: success_rate(s) for h, s in (host_success or {}).items()}
    commands = []
    for host, stream in (host_success or {}).items():
        peers = [v for h, v in rates.items() if h != host]
        peer_mean = (sum(peers) / len(peers)) if peers else rates.get(host, 0.0)
        commands.append(
            difficulty_command_for_host(host, stream, base_d_t=base_d_t, peer_success_mean=peer_mean)
        )
    return {
        "commands": [c.to_dict() for c in commands],
        "host_success_rates": {h: round(v, 4) for h, v in rates.items()},
        "max_effort": max((c.effort for c in commands), key=lambda e: {"low": 0, "high": 1, "max": 2}.get(e, 0), default="low"),
        "note": "p_r5_red_queen_env_pressure",
    }


def red_queen_coupling_report(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    a_task_diff: Sequence[float],
    b_task_diff: Sequence[float],
) -> dict:
    """Expose coevolution.red_queen_pressure for env-evolution health."""
    return red_queen_pressure(a_scores, b_scores, a_task_diff, b_task_diff)


def apply_to_evolver_plan(
    plan: dict,
    *,
    default_direction: str = "skill",
) -> list[dict]:
    """Turn difficulty commands into WorldEvolver job specs (no side effects)."""
    jobs = []
    for c in (plan or {}).get("commands") or []:
        effort = c.get("effort") or "high"
        direction = default_direction
        if effort == "max":
            direction = "length"  # harder horizon when Red Queen is hot
        elif effort == "low":
            direction = "scenario"
        jobs.append(
            {
                "host": c.get("host"),
                "direction": direction,
                "effort": effort,
                "target_d_t": c.get("target_d_t"),
                "pressure": c.get("pressure"),
                "reason": c.get("reason"),
            }
        )
    return jobs


def host_success_streams_from_pool(world_pool, window: int = 20) -> dict[str, list]:
    """Build per-host success streams from recent host outcomes / pool lineage."""
    streams: dict[str, list] = {h: [] for h in DEFAULT_HOSTS}
    try:
        store = getattr(world_pool, "_arsi_store", None)
        # fall back: multi-agent health is not here — use evolution lineage pass proxies
        for w in list(getattr(world_pool, "worlds", []) or [])[-window:]:
            meta = {}
            for row in getattr(world_pool, "_manifest", []) or []:
                if row.get("world_id") == getattr(w, "world_id", None):
                    meta = row
                    break
            hosts = []
            full = getattr(w, "_full", {}) or {}
            for node in list(full.values())[:50]:
                h = str((node or {}).get("agent_id") or "")
                if h and h not in hosts:
                    hosts.append(h)
            ok = any(
                "success" in str((node or {}).get("outcome") or "").lower()
                for node in list(full.values())[:20]
            )
            for h in hosts:
                key = h if h in streams else None
                if key is None:
                    for known in DEFAULT_HOSTS:
                        if known.startswith(h) or h.startswith(known.split("-")[0]):
                            key = known
                            break
                if key:
                    streams[key].append(1.0 if ok else 0.0)
    except Exception:
        pass
    return streams


def red_queen_effort_for_pool(world_pool) -> dict:
    """P2-2: map pool/host success → evolver effort (success↑ ⇒ harder)."""
    streams = host_success_streams_from_pool(world_pool)
    # drop empty hosts so peer_mean is meaningful
    filled = {h: s for h, s in streams.items() if s}
    if not filled:
        filled = {h: [0.5] for h in DEFAULT_HOSTS}
    plan = mutual_pressure_plan(filled, base_d_t=1.0)
    jobs = apply_to_evolver_plan(plan)
    efforts = [j.get("effort") or "low" for j in jobs]
    rank = {"low": 0, "high": 1, "max": 2}
    top = max(efforts, key=lambda e: rank.get(e, 0), default="high")
    return {
        "effort": top,
        "plan": plan,
        "jobs": jobs,
        "streams_len": {h: len(s) for h, s in filled.items()},
        "note": "p2_2_red_queen_env_pressure",
    }
