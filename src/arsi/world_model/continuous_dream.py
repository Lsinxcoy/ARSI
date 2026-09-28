"""P-R2 Continuous dream synthesizer (Dream-RSI × ODEWorld × CTM).

Layering (rumination §2 exact-discrete vs continuous-time):
  tree nodes  — Dream-RSI exact discrete replay (scores only at nodes)
  between — ODEWorld v(z;a) integration interpolates the path
  co-move     — CTM S(a) marks which organs move together along the path

Counterfactual: "what if we were half a step off" via local v step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from arsi.world_model.capability_flow import Z_KEYS, canon_action

VelocityFn = Callable[[dict, Optional[str]], dict]


@dataclass
class DreamStep:
    kind: str  # exact_node | interpolated | counterfactual
    z: dict
    action: str = "unknown"
    score: Optional[float] = None  # only at exact nodes
    cost: float = 0.0
    node_id: str = ""
    parent_id: str = ""
    sync_pairs: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContinuousDream:
    steps: list[DreamStep] = field(default_factory=list)
    n_exact: int = 0
    n_interp: int = 0
    n_counterfactual: int = 0
    note: str = "tree_exact_plus_v_interp_plus_S"

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "n_exact": self.n_exact,
            "n_interp": self.n_interp,
            "n_counterfactual": self.n_counterfactual,
            "note": self.note,
        }

    def exact_scores(self) -> list[float]:
        return [float(s.score) for s in self.steps if s.kind == "exact_node" and s.score is not None]


def _z_from_node(node) -> dict:
    """Normalize DiscoveryNode / Observation / dict to z-like feature map."""
    if node is None:
        return {}
    if isinstance(node, dict):
        src = node
    else:
        src = getattr(node, "metadata", None) or {}
        if not isinstance(src, dict):
            src = {}
        base = {
            "score": getattr(node, "score", 0.0) or 0.0,
            "cost": getattr(node, "cost", 0.0) or 0.0,
            "depth": getattr(node, "depth", 0) or 0,
        }
        base.update(src)
        src = base
    out = {}
    for k in Z_KEYS:
        if k in src:
            out[k] = float(src.get(k) or 0.0)
    if not out:
        # fall back: scalar proxies so interpolation still has a state
        out = {
            "live_last": float(src.get("score") or src.get("effect") or 0.0),
            "live_ema": float(src.get("score") or src.get("effect") or 0.0),
            "pool_last": float(src.get("cost") or 0.0),
            "pool_ema": float(src.get("cost") or 0.0),
            "eta": float(src.get("depth") or 0.0) / 10.0,
        }
    return out


def _lerp_z(z_a: dict, z_b: dict, t: float) -> dict:
    keys = set(z_a) | set(z_b)
    return {
        k: (1.0 - t) * float(z_a.get(k, 0.0) or 0.0) + t * float(z_b.get(k, 0.0) or 0.0)
        for k in sorted(keys)
    }


def co_move_pairs(z_delta: dict, min_abs: float = 1e-6) -> list[list]:
    """CTM-style S(a): channels that moved together (same sign, non-trivial)."""
    ups = [k for k, v in z_delta.items() if float(v) > min_abs]
    downs = [k for k, v in z_delta.items() if float(v) < -min_abs]
    pairs = []
    for i, a in enumerate(ups):
        for b in ups[i + 1 :]:
            pairs.append([a, b, "co_up"])
    for i, a in enumerate(downs):
        for b in downs[i + 1 :]:
            pairs.append([a, b, "co_down"])
    return pairs


def default_v_fn(z: dict, action: Optional[str] = None) -> dict:
    """Fallback velocity: mild drift toward live_ema (no trained field required)."""
    z = z or {}
    out = {}
    for k in z:
        target = float(z.get("live_ema", z.get(k, 0.0)) or 0.0)
        out[k] = 0.1 * (target - float(z.get(k, 0.0) or 0.0))
    return out


def interpolate_segment(
    z_a: dict,
    z_b: dict,
    action: str = "unknown",
    steps: int = 3,
    v_fn: Optional[VelocityFn] = None,
) -> list[DreamStep]:
    """Between two exact nodes: linear blend refined by one v(z;a) nudge."""
    vf = v_fn or default_v_fn
    act = canon_action(action)
    out: list[DreamStep] = []
    n = max(1, int(steps))
    prev = dict(z_a)
    for i in range(1, n + 1):
        t = i / (n + 1)
        z = _lerp_z(z_a, z_b, t)
        try:
            v = vf(z, act) or {}
            for k, dv in v.items():
                z[k] = float(z.get(k, 0.0) or 0.0) + 0.25 * float(dv or 0.0) / n
        except Exception:
            pass
        delta = {k: float(z.get(k, 0.0) or 0.0) - float(prev.get(k, 0.0) or 0.0) for k in z}
        out.append(
            DreamStep(
                kind="interpolated",
                z=z,
                action=act,
                score=None,
                sync_pairs=co_move_pairs(delta),
                note=f"v_interp_t={t:.3f}",
            )
        )
        prev = z
    return out


def counterfactual_half_step(
    z: dict,
    action: str = "unknown",
    v_fn: Optional[VelocityFn] = None,
    scale: float = 0.5,
) -> DreamStep:
    """P-R2 counterfactual: local half-step along v(z;a) from an exact node."""
    vf = v_fn or default_v_fn
    act = canon_action(action)
    try:
        v = vf(dict(z or {}), act) or {}
    except Exception:
        v = {}
    z_cf = {k: float((z or {}).get(k, 0.0) or 0.0) + scale * float(v.get(k, 0.0) or 0.0) for k in (z or v)}
    delta = {k: float(z_cf.get(k, 0.0) or 0.0) - float((z or {}).get(k, 0.0) or 0.0) for k in z_cf}
    return DreamStep(
        kind="counterfactual",
        z=z_cf,
        action=act,
        score=None,
        sync_pairs=co_move_pairs(delta),
        note="half_step_along_v",
    )


def synthesize_continuous_dream(
    nodes: Sequence,
    actions: Optional[Sequence[str]] = None,
    steps_between: int = 3,
    v_fn: Optional[VelocityFn] = None,
    with_counterfactual: bool = False,
) -> ContinuousDream:
    """Compose exact tree nodes + interpolated paths + optional half-step CFs.

    nodes: ordered parent→child DiscoveryNode / Observation / dict (with score).
    """
    dream = ContinuousDream()
    if not nodes:
        return dream
    zs = [_z_from_node(n) for n in nodes]
    acts = list(actions or [])
    while len(acts) < len(nodes):
        raw = nodes[len(acts)]
        a = ""
        if isinstance(raw, dict):
            a = str(raw.get("action") or "")
        else:
            a = str(getattr(raw, "action", "") or "")
        acts.append(a or "unknown")

    for i, node in enumerate(nodes):
        nid = getattr(node, "id", None) or (node.get("id", "") if isinstance(node, dict) else "")
        parent = getattr(node, "parent_id", None) or (node.get("parent_id", "") if isinstance(node, dict) else "")
        score = getattr(node, "score", None)
        if score is None and isinstance(node, dict):
            score = node.get("score")
        cost = getattr(node, "cost", 0.0) or (node.get("cost", 0.0) if isinstance(node, dict) else 0.0) or 0.0
        step = DreamStep(
            kind="exact_node",
            z=zs[i],
            action=canon_action(acts[i]),
            score=float(score) if score is not None else None,
            cost=float(cost or 0.0),
            node_id=str(nid or f"n{i}"),
            parent_id=str(parent or ""),
            sync_pairs=[],
            note="exact_tree_node",
        )
        if i > 0:
            delta = {k: float(zs[i].get(k, 0.0) or 0.0) - float(zs[i - 1].get(k, 0.0) or 0.0) for k in set(zs[i]) | set(zs[i - 1])}
            step.sync_pairs = co_move_pairs(delta)
        dream.steps.append(step)
        dream.n_exact += 1

        if with_counterfactual:
            dream.steps.append(counterfactual_half_step(zs[i], acts[i], v_fn=v_fn))
            dream.n_counterfactual += 1

        if i + 1 < len(nodes):
            seg = interpolate_segment(zs[i], zs[i + 1], acts[i], steps=steps_between, v_fn=v_fn)
            dream.steps.extend(seg)
            dream.n_interp += len(seg)
    return dream


def synthesize_from_discovery_tree(tree, steps_between: int = 3, v_fn: Optional[VelocityFn] = None) -> ContinuousDream:
    """Walk DiscoveryTree parent→children order (DFS from root)."""
    if tree is None or not getattr(tree, "nodes", None):
        return ContinuousDream(note="empty_tree")
    nodes = tree.nodes
    root_id = getattr(tree, "root_id", "") or "root"
    order = []
    seen = set()

    def _dfs(nid: str) -> None:
        if not nid or nid in seen or nid not in nodes:
            return
        seen.add(nid)
        order.append(nodes[nid])
        for c in getattr(nodes[nid], "children", []) or []:
            _dfs(c)

    _dfs(root_id if root_id in nodes else next(iter(nodes)))
    return synthesize_continuous_dream(order, steps_between=steps_between, v_fn=v_fn)
