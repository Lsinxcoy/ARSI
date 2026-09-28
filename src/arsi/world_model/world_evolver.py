"""WorldEvolver — off-policy environment evolution (arXiv:2609.04128).

Three directions on ARSI discovery worlds:
  - scenario: swap agent/host / context labels (novelty)
  - skill:   swap actions / fail_class / operator tags (rarity)
  - length:  insert extra (agent, action) steps (horizon)

Verifiers (adapted):
  - Oracle: evolved world retains at least one success-scored node
  - Invalid-test: empty/no-op world must fail (never accepted)
  - Quality: min nodes, has outcomes, difficulty > seed (high/max)

Evolution effort: low | high | max
"""
from __future__ import annotations

import copy
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from arsi.meta.env_difficulty import ReferenceCorpus, compute_env_difficulty
from arsi.world_model.discovery_tree import DiscoveryNode, DiscoveryTree
from arsi.world_model.replay_world import ReplayWorld

DIRECTIONS = ("scenario", "skill", "length")
EFFORTS = ("low", "high", "max")

# host/scenario vocabulary for evolution (ARSI multi-host)
SCENARIO_POOL = [
    "hermes", "mimo-desktop", "synthex-mothernest",
    "mimo", "synthex", "host-unknown", "edge-cli", "batch-runner",
]
SKILL_POOL = [
    "learn", "remember", "dream", "maintain", "evolve", "empower",
    "calibrate", "ingest", "verify", "repair", "deploy_skill", "analyze_trace",
]
FAIL_CLASSES = ["ok", "timeout", "tool_error", "invalid_input", "unknown"]


@dataclass
class VerifierResult:
    oracle_ok: bool = False
    invalid_rejected: bool = False
    quality_ok: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.oracle_ok and self.invalid_rejected and self.quality_ok

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


@dataclass
class EvolutionResult:
    seed_id: str = ""
    child_id: str = ""
    lineage_id: str = ""
    generation: int = 1
    direction: str = ""
    effort: str = "high"
    accepted: bool = False
    verifier: VerifierResult = field(default_factory=VerifierResult)
    seed_difficulty: dict = field(default_factory=dict)
    child_difficulty: dict = field(default_factory=dict)
    node_count: int = 0
    note: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verifier"] = self.verifier.to_dict() if hasattr(self.verifier, "to_dict") else self.verifier
        return d


def _nodes_to_traces(world) -> list[dict]:
    """Normalize ReplayWorld/DiscoveryTree/dict nodes to trace-like dicts."""
    traces = []
    nodes = []
    if world is None:
        return traces
    if isinstance(world, dict):
        raw = world.get("nodes") or world
        if isinstance(raw, dict):
            nodes = [v for k, v in raw.items() if k != "root"]
    else:
        full = getattr(world, "_full", None)
        if isinstance(full, dict):
            nodes = [v for k, v in full.items() if k != "root"]
        else:
            tn = getattr(world, "nodes", None)
            if isinstance(tn, dict):
                for nid, node in tn.items():
                    if nid == "root":
                        continue
                    traces.append(node.to_dict() if hasattr(node, "to_dict") else dict(node))
                return traces
    for n in nodes:
        if isinstance(n, dict):
            traces.append({
                "action": n.get("action", "unknown"),
                "agent_id": n.get("agent_id", "unknown"),
                "outcome": n.get("outcome", "unknown"),
                "effect": float(n.get("score") or 0.0),
                "params": n.get("metadata") or n.get("params") or {
                    "fail_class": n.get("fail_class", "ok"),
                    "tags": n.get("tags", []),
                },
            })
    return traces


def traces_to_world(traces: list[dict], world_id: str, max_parallelism: int = 3) -> ReplayWorld:
    tree = DiscoveryTree()
    tree.build_from_traces(traces)
    return ReplayWorld.from_discovery_tree(tree, world_id=world_id, max_parallelism=max_parallelism)


def verify_world(
    world,
    seed_world=None,
    effort: str = "high",
) -> VerifierResult:
    """Oracle + invalid-test + quality gates (arXiv:2609.04128 adapted)."""
    res = VerifierResult()
    nodes = _nodes_to_traces(world)
    if not nodes and hasattr(world, "_full"):
        nodes = [v for k, v in world._full.items() if k != "root"]
    if not nodes:
        res.reasons.append("invalid_empty_world")
        return res

    # Invalid-test: empty / all-unknown / all-success fake-easy must fail
    outcomes = [str((n.get("outcome") if isinstance(n, dict) else "") or "").lower() for n in nodes]
    if all((not o) or o == "unknown" for o in outcomes):
        res.reasons.append("invalid_all_unknown_outcomes")
        return res
    success_n = sum(1 for o in outcomes if "success" in o)
    if success_n == len(outcomes) and len(outcomes) > 1:
        res.reasons.append("invalid_all_success_fake_easy")
        return res
    res.invalid_rejected = True

    # Oracle: at least one success-like scored node
    successish = False
    for n in nodes:
        score = float(n.get("effect", n.get("score", 0)) or 0) if isinstance(n, dict) else 0.0
        oc = str((n.get("outcome") if isinstance(n, dict) else "") or "").lower()
        if "success" in oc or score >= 0.5:
            successish = True
            break
    if not successish:
        res.reasons.append("oracle_no_success_path")
    else:
        res.oracle_ok = True

    # Quality
    if len(nodes) < 3:
        res.reasons.append("quality_too_few_nodes")
        res.quality_ok = False
    elif len(nodes) > 400:
        res.reasons.append("quality_too_many_nodes")
        res.quality_ok = False
    else:
        res.quality_ok = True

    # P2-1: Red Queen — block difficulty collapse (overnight: L 63→27).
    # high: forbid >5% d_t drop OR require structural growth (L / novelty / nodes)
    # max: require non-decreasing d_t or strictly longer horizon
    if effort in ("high", "max") and seed_world is not None:
        try:
            seed_d = compute_env_difficulty(seed_world)
            child_d = compute_env_difficulty(world)
            seed_n = len(_nodes_to_traces(seed_world) or [])
            child_n = len(nodes)
            novelty_up = (child_d.scenario_novelty + child_d.skill_rarity) >= (
                seed_d.scenario_novelty + seed_d.skill_rarity
            )
            structural_up = child_d.L >= seed_d.L or child_n > seed_n or novelty_up
            if effort == "max":
                ok = child_d.d_t + 1e-6 >= seed_d.d_t or child_d.L > seed_d.L
            else:
                ok = child_d.d_t + 1e-6 >= seed_d.d_t * 0.95 or structural_up
            if not ok:
                res.reasons.append("difficulty_collapse_red_queen")
                res.quality_ok = False
            if child_d.d_t + 1e-6 < seed_d.d_t * 0.85:
                res.reasons.append("difficulty_dropped_too_much")
                res.quality_ok = False
        except Exception as e:
            res.reasons.append(f"difficulty_check_error:{e}")
            res.quality_ok = False
    return res


def _effort_span(effort: str, n: int) -> tuple[int, int]:
    """Return (start, end) edit window in sequence."""
    if n <= 0:
        return 0, 0
    if effort == "low":
        i = random.randrange(n)
        return i, i + 1
    if effort == "max":
        return 0, n
    # high: contiguous span ~40-70%
    span = max(1, int(n * random.uniform(0.4, 0.7)))
    start = random.randrange(0, max(1, n - span + 1))
    return start, min(n, start + span)


def evolve_traces(
    seed_traces: list[dict],
    direction: str,
    effort: str = "high",
    rng: Optional[random.Random] = None,
) -> list[dict]:
    rng = rng or random.Random()
    if not seed_traces:
        return []
    seq = [copy.deepcopy(t) for t in seed_traces]
    n = len(seq)
    a, b = _effort_span(effort, n)
    b = max(a + 1, b)

    if direction == "scenario":
        for i in range(a, b):
            if i < n:
                seq[i]["agent_id"] = rng.choice(SCENARIO_POOL)
        # P2-1: inject MORE novel scenario steps for high/max (raise L + novelty)
        inserts = {"low": 1, "high": max(2, n // 8), "max": max(3, n // 5)}.get(effort, 2)
        for _ in range(inserts):
            seq.insert(min(b, len(seq)), {
                "action": rng.choice(SKILL_POOL),
                "agent_id": rng.choice(SCENARIO_POOL),
                "outcome": "success",
                "effect": 0.55,
                "params": {"evolved": "scenario_insert", "fail_class": "ok"},
            })
    elif direction == "skill":
        for i in range(a, b):
            if i < n:
                seq[i]["action"] = rng.choice(SKILL_POOL)
                params = dict(seq[i].get("params") or {})
                fc = rng.choice(FAIL_CLASSES)
                params["fail_class"] = fc
                params["evolved"] = "skill_swap"
                seq[i]["params"] = params
                # fail_class≠ok must not remain a free success (Invalid-test)
                if fc != "ok":
                    seq[i]["outcome"] = rng.choice(["failure", "partial"])
                    seq[i]["effect"] = float(rng.uniform(-0.3, 0.45))
        # P2-1: always add rare tail for high/max (skill_rarity↑)
        if effort in ("high", "max"):
            tails = 2 if effort == "max" else 1
            for _ in range(tails):
                seq.append({
                    "action": rng.choice(["repair", "verify", "analyze_trace", "deploy_skill"]),
                    "agent_id": seq[-1].get("agent_id", "unknown") if seq else "unknown",
                    "outcome": "partial",
                    "effect": 0.4,
                    "params": {"evolved": "skill_tail", "fail_class": rng.choice(FAIL_CLASSES)},
                })
    elif direction == "length":
        inserts = {"low": 1, "high": max(3, n // 3), "max": max(4, n // 2)}.get(effort, 3)
        for _ in range(inserts):
            pos = rng.randrange(1, max(2, len(seq)))
            seq.insert(pos, {
                "action": rng.choice(SKILL_POOL),
                "agent_id": seq[pos - 1].get("agent_id", "unknown") if pos > 0 else "unknown",
                "outcome": rng.choice(["success", "partial", "failure"]),
                "effect": rng.uniform(0.2, 0.85),
                "params": {"evolved": "length_insert", "fail_class": rng.choice(FAIL_CLASSES)},
            })
    else:
        raise ValueError(f"unknown direction: {direction}")

    # keep at least one success anchor for Oracle
    if not any("success" in str(t.get("outcome", "")).lower() for t in seq):
        seq[0] = dict(seq[0])
        seq[0]["outcome"] = "success"
        seq[0]["effect"] = max(0.6, float(seq[0].get("effect") or 0.6))
        seq[0]["params"] = {**(seq[0].get("params") or {}), "evolved": "oracle_anchor"}
    return seq


class WorldEvolver:
    """Generate evolved child worlds from a seed world/traces."""

    def __init__(self, reference: Optional[ReferenceCorpus] = None, max_parallelism: int = 3):
        self.reference = reference
        self.max_parallelism = max_parallelism
        self._rng = random.Random()

    def evolve(
        self,
        seed_world,
        lineage_id: Optional[str] = None,
        generation: int = 1,
        direction: Optional[str] = None,
        effort: str = "high",
        seed_id: str = "",
    ) -> EvolutionResult:
        seed_traces = _nodes_to_traces(seed_world)
        seed_id = seed_id or getattr(seed_world, "world_id", None) or f"seed_{uuid.uuid4().hex[:8]}"
        lineage_id = lineage_id or f"lin_{uuid.uuid4().hex[:8]}"
        direction = direction or self._rng.choice(DIRECTIONS)
        if effort not in EFFORTS:
            effort = "high"

        seed_d = compute_env_difficulty(seed_world, reference=self.reference)
        evolved_traces = evolve_traces(seed_traces, direction, effort=effort, rng=self._rng)
        child_id = f"{seed_id}_g{generation}_{direction}_{uuid.uuid4().hex[:6]}"
        child = traces_to_world(evolved_traces, child_id, max_parallelism=self.max_parallelism)
        child_d = compute_env_difficulty(child, reference=self.reference)

        # attach lineage meta onto world object for pool
        child.lineage_id = lineage_id
        child.generation = generation
        child.direction = direction
        child.effort = effort
        child.parent_world_id = seed_id
        child.env_difficulty = child_d.to_dict()
        child.seed_difficulty = seed_d.to_dict()
        child.evolved = True
        child.source = "env_evolution"
        child.sealed_eligible = False

        verifier = verify_world(child, seed_world=seed_world, effort=effort)
        accepted = verifier.passed
        result = EvolutionResult(
            seed_id=seed_id,
            child_id=child_id,
            lineage_id=lineage_id,
            generation=generation,
            direction=direction,
            effort=effort,
            accepted=accepted,
            verifier=verifier,
            seed_difficulty=seed_d.to_dict(),
            child_difficulty=child_d.to_dict(),
            node_count=len(_nodes_to_traces(child)),
            note="evolved" if accepted else "rejected",
        )
        result.world = child if accepted else None
        return result

    def evolve_batch(
        self,
        seed_world,
        n: int = 2,
        lineage_id: Optional[str] = None,
        generation: int = 1,
        effort: str = "high",
        directions: Optional[list[str]] = None,
    ) -> list[EvolutionResult]:
        dirs = directions or list(DIRECTIONS)
        out = []
        for i in range(n):
            d = dirs[i % len(dirs)]
            out.append(
                self.evolve(
                    seed_world,
                    lineage_id=lineage_id,
                    generation=generation,
                    direction=d,
                    effort=effort,
                )
            )
        return out
