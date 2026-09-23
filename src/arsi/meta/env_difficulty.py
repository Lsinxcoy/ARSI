"""Environment difficulty D_T — off-policy, model-agnostic (arXiv:2609.04128).

Policy-independent difficulty from multi-turn structure:
  (L, scenario novelty, skill rarity) under a reference distribution T.

ARSI worlds = discovery/replay trees from host traces.
Agent weakness Δ_θ is separate: excess difficulty vs current policy scores.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Optional


@dataclass
class DifficultyComponents:
    L: float = 0.0
    scenario_novelty: float = 0.0
    skill_rarity: float = 0.0
    d_t: float = 0.0
    w_l: float = 0.45
    w_sigma: float = 0.30
    w_kappa: float = 0.25
    n_nodes: int = 0
    n_agents: int = 0
    n_actions: int = 0
    unique_pairs: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceCorpus:
    """Reference distribution T grounded in ARSI host action vocabulary."""

    action_counts: Counter = None
    pair_counts: Counter = None
    agent_counts: Counter = None
    total_actions: int = 0

    def __post_init__(self):
        self.action_counts = self.action_counts or Counter()
        self.pair_counts = self.pair_counts or Counter()
        self.agent_counts = self.agent_counts or Counter()
        self.total_actions = int(self.total_actions or sum(self.action_counts.values()))

    @classmethod
    def from_trace_batches(cls, batches: Iterable[list[dict]]) -> "ReferenceCorpus":
        ac: Counter = Counter()
        pc: Counter = Counter()
        ag: Counter = Counter()
        for traces in batches:
            for t in traces or []:
                action = str(t.get("action", "unknown"))
                agent = str(t.get("agent_id", t.get("params", {}).get("source", "unknown")))
                ac[action] += 1
                pc[(agent, action)] += 1
                ag[agent] += 1
        return cls(action_counts=ac, pair_counts=pc, agent_counts=ag, total_actions=sum(ac.values()))

    @classmethod
    def uniform(cls, actions: Optional[Iterable[str]] = None) -> "ReferenceCorpus":
        acts = list(actions or ["learn", "remember", "dream", "maintain", "evolve", "empower"])
        ac = Counter({a: 1 for a in acts})
        pc = Counter({("unknown", a): 1 for a in acts})
        ag = Counter({"unknown": 1})
        return cls(action_counts=ac, pair_counts=pc, agent_counts=ag, total_actions=len(acts))

    def p_action(self, action: str) -> float:
        n = self.action_counts.get(action, 0)
        return (n / self.total_actions) if self.total_actions else 1e-6

    def p_pair(self, agent: str, action: str) -> float:
        n = self.pair_counts.get((agent, action), 0)
        tot = sum(self.pair_counts.values()) or 1
        return (n / tot) if tot else 1e-6


def _nodes_from_world(world) -> list[dict]:
    """Extract node dicts from ReplayWorld / DiscoveryTree / raw dict."""
    if world is None:
        return []
    if isinstance(world, dict):
        nodes = world.get("nodes") or world
        if isinstance(nodes, dict):
            return [v for k, v in nodes.items() if k != "root"]
        return []
    nodes = getattr(world, "_full", None)
    if isinstance(nodes, dict):
        return [v for k, v in nodes.items() if k != "root"]
    tree_nodes = getattr(world, "nodes", None)
    if isinstance(tree_nodes, dict):
        out = []
        for nid, node in tree_nodes.items():
            if nid == "root":
                continue
            if hasattr(node, "to_dict"):
                out.append(node.to_dict())
            elif isinstance(node, dict):
                out.append(node)
        return out
    return []


def compute_env_difficulty(
    world,
    reference: Optional[ReferenceCorpus] = None,
    w_l: float = 0.45,
    w_sigma: float = 0.30,
    w_kappa: float = 0.25,
) -> DifficultyComponents:
    """Off-policy D_T for one ARSI world."""
    ref = reference or ReferenceCorpus.uniform()
    nodes = _nodes_from_world(world)
    if not nodes:
        return DifficultyComponents(note="empty_world", w_l=w_l, w_sigma=w_sigma, w_kappa=w_kappa)

    depths = [int(n.get("depth") or 0) for n in nodes]
    L = float(max(depths) if depths else len(nodes))
    # length difficulty: log scale so huge trees don't dominate forever
    L_term = math.log1p(max(0.0, L))

    pairs = []
    actions = []
    agents = []
    sigma_terms = []
    kappa_terms = []
    for n in nodes:
        action = str(n.get("action", "unknown"))
        agent = str(n.get("agent_id", "unknown"))
        actions.append(action)
        agents.append(agent)
        pairs.append((agent, action))
        p_sig = ref.p_pair(agent, action)
        p_kap = ref.p_action(action)
        sigma_terms.append(-math.log(max(p_sig, 1e-9)))
        kappa_terms.append(-math.log(max(p_kap, 1e-9)))

    scenario_novelty = sum(sigma_terms) / len(sigma_terms) if sigma_terms else 0.0
    skill_rarity = sum(kappa_terms) / len(kappa_terms) if kappa_terms else 0.0

    # normalize components to comparable ~[0, 5] scales then weighted sum
    # L_term already log; scenario/skill already -log p
    d_t = w_l * L_term + w_sigma * scenario_novelty + w_kappa * skill_rarity

    return DifficultyComponents(
        L=L,
        scenario_novelty=round(scenario_novelty, 4),
        skill_rarity=round(skill_rarity, 4),
        d_t=round(d_t, 4),
        w_l=w_l,
        w_sigma=w_sigma,
        w_kappa=w_kappa,
        n_nodes=len(nodes),
        n_agents=len(set(agents)),
        n_actions=len(set(actions)),
        unique_pairs=len(set(pairs)),
        note="off_policy_env_difficulty",
    )


def compute_env_difficulty_dyn(
    world,
    reference: Optional[ReferenceCorpus] = None,
    w_l: float = 0.50,
    w_sigma: float = 0.25,
    w_kappa: float = 0.25,
) -> DifficultyComponents:
    """D_T on dyn features only (ODEWorld decoupling): no host/agent static identity.

    scenario novelty = (action, fail_class) rarity — skill structure, not agent name noise.
    """
    ref = reference or ReferenceCorpus.uniform()
    nodes = _nodes_from_world(world)
    if not nodes:
        return DifficultyComponents(note="empty_world_dyn", w_l=w_l, w_sigma=w_sigma, w_kappa=w_kappa)

    depths = [int(n.get("depth") or 0) for n in nodes]
    L = float(max(depths) if depths else len(nodes))
    L_term = math.log1p(max(0.0, L))

    actions = []
    fails = []
    pairs = []
    sigma_terms = []
    kappa_terms = []
    for n in nodes:
        action = str(n.get("action", "unknown"))
        meta = n.get("metadata") or n.get("params") or {}
        if not isinstance(meta, dict):
            meta = {}
        fail = str(
            n.get("fail_class")
            or meta.get("fail_class")
            or ("ok" if "success" in str(n.get("outcome", "")).lower() else "unknown")
        )
        actions.append(action)
        fails.append(fail)
        pairs.append((action, fail))
        p_sig = ref.p_pair(fail, action)  # pair keyed by fail×action (dyn), not agent
        p_kap = ref.p_action(action)
        sigma_terms.append(-math.log(max(p_sig, 1e-9)))
        kappa_terms.append(-math.log(max(p_kap, 1e-9)))

    scenario_novelty = sum(sigma_terms) / len(sigma_terms) if sigma_terms else 0.0
    skill_rarity = sum(kappa_terms) / len(kappa_terms) if kappa_terms else 0.0
    d_t = w_l * L_term + w_sigma * scenario_novelty + w_kappa * skill_rarity

    return DifficultyComponents(
        L=L,
        scenario_novelty=round(scenario_novelty, 4),
        skill_rarity=round(skill_rarity, 4),
        d_t=round(d_t, 4),
        w_l=w_l,
        w_sigma=w_sigma,
        w_kappa=w_kappa,
        n_nodes=len(nodes),
        n_agents=len(set(n.get("agent_id", "unknown") for n in nodes)),
        n_actions=len(set(actions)),
        unique_pairs=len(set(pairs)),
        note="off_policy_env_difficulty_dyn_only",
    )


def weakness_vs_difficulty(
    d_t: float,
    policy_score: float,
    reference_score: Optional[float] = None,
) -> dict:
    """Δ_θ-like: how much harder this world is for the current policy vs baseline.

    Lower replay score on higher D_T → larger weakness.
    """
    ref = 0.0 if reference_score is None else float(reference_score)
    gap = float(ref) - float(policy_score)
    return {
        "d_t": float(d_t),
        "policy_score": float(policy_score),
        "reference_score": ref,
        "excess_difficulty": round(max(0.0, gap + 0.15 * float(d_t)), 4),
        "weakness": round(max(0.0, gap), 4),
        "timestamp": datetime.now().isoformat(),
    }


def pool_difficulty_stats(worlds: list, reference: Optional[ReferenceCorpus] = None) -> dict:
    if not worlds:
        return {"world_count": 0, "d_t_mean": 0.0, "d_t_max": 0.0, "homogeneous": True}
    dts = []
    for w in worlds:
        try:
            dts.append(compute_env_difficulty(w, reference=reference).d_t)
        except Exception:
            dts.append(0.0)
    mean = sum(dts) / len(dts)
    dmax = max(dts)
    dmin = min(dts)
    return {
        "world_count": len(worlds),
        "d_t_mean": round(mean, 4),
        "d_t_max": round(dmax, 4),
        "d_t_min": round(dmin, 4),
        "d_t_spread": round(dmax - dmin, 4),
        "homogeneous": (dmax - dmin) < 0.35,
        "components_sample": compute_env_difficulty(worlds[-1], reference=reference).to_dict(),
    }
