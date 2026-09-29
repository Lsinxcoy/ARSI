"""GRPO prerequisite data plane — export only (no training, no weights).

Turns ARSI task-level trajectories into:
  1) **groups**   — same task/prompt key, ≥2 members (completions / variants)
  2) **reward table** — task-level reward with fail_class + claim penalties
  3) **holdout-aligned split** — selection / private / blacklist (AIDE²)

Rules:
- never invent rewards from gold answers (blacklist already in c9 bridge)
- iron laws / sealed never enter selection (holdout)
- grpo_live=False always here — API hosts have no weights
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from arsi.harness.holdout import TaskSplit, split_tasks
from arsi.harness.training_bridge import TrainingExample, is_blacklisted, trace_to_example

SCHEMA = "arsi.grpo.data_plane.v1"
GRPO_LIVE = False

# ARSI default reward shaping (illustrative — not paper_confirmed hyperparams)
FAIL_CLASS_PENALTY = {
    "gate_reject": 0.15,
    "tool_loop": 0.25,
    "tool_error": 0.10,
    "timeout": 0.10,
    "compile_other": 0.20,
    "premature_complete": 0.30,
    "runtime_error": 0.10,
    "resource_limit": 0.05,
}


@dataclass
class RewardRow:
    example_id: str
    task_id: str
    group_id: str
    reward: float
    base: float
    penalty_fail: float
    penalty_claim: float
    success: bool
    fail_class: str
    agent_id: str
    harness_variant: str
    split: str = ""  # selection | private | blacklist
    source: str = "arsi.grpo.data_plane"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroupRow:
    group_id: str
    task_key: str
    n_members: int
    member_ids: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    advantage_ready: bool = False  # n>=2 and reward variance > 0
    note: str = "group_relative_policy_optimization_input"

    def to_dict(self) -> dict:
        return asdict(self)


def group_key_for(trace: dict, example: Optional[TrainingExample] = None) -> str:
    """Same prompt/task family ⇒ same GRPO group (relative advantage within group)."""
    t = dict(trace or {})
    task_id = str(t.get("task_id") or t.get("trace_id") or t.get("id") or "")
    if example is not None and example.task_id:
        task_id = example.task_id
    action = str(t.get("action") or (example.action if example else "") or "")
    params = t.get("params") or {}
    brief_id = str(params.get("brief_id") or "")
    # prefer brief_id (same brief = same prompt context), else task_id, else action
    if brief_id:
        raw = f"brief:{brief_id}"
    elif task_id:
        raw = f"task:{task_id}"
    else:
        raw = f"act:{action}"
    return "G-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def compute_reward(
    *,
    effect: float,
    success: bool,
    fail_class: str = "",
    claim_rejected: bool = False,
    cost: float = 0.0,
    beta_cost: float = 0.0,
) -> tuple[float, dict]:
    """Task-level shaped reward. claim-rejected → 0 (negative epistemology)."""
    base = float(effect) if effect is not None else (1.0 if success else 0.0)
    if not success:
        base = min(base, 0.0) if base > 0 and claim_rejected else base
    pen_f = float(FAIL_CLASS_PENALTY.get(str(fail_class or ""), 0.0))
    pen_c = 0.5 if claim_rejected else 0.0
    r = base - pen_f - pen_c - float(beta_cost) * max(0.0, float(cost))
    if claim_rejected:
        r = min(r, 0.0)
    detail = {
        "base": round(base, 6),
        "penalty_fail": pen_f,
        "penalty_claim": pen_c,
        "reward": round(r, 6),
    }
    return round(r, 6), detail


def build_groups(examples: Sequence[TrainingExample], traces: Optional[Sequence[dict]] = None) -> list[GroupRow]:
    traces = list(traces or [])
    by_id: dict[str, dict] = {}
    for i, tr in enumerate(traces):
        ex = examples[i] if i < len(examples) else trace_to_example(tr)
        by_id[ex.example_id] = dict(tr or {})
    buckets: dict[str, list[TrainingExample]] = defaultdict(list)
    for ex in examples:
        tr = by_id.get(ex.example_id) or {}
        gid = group_key_for(tr, ex)
        buckets[gid].append(ex)
    out: list[GroupRow] = []
    for gid, members in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        rewards = [float(m.reward) for m in members]
        var_ok = (max(rewards) - min(rewards)) > 1e-9 if len(rewards) >= 2 else False
        out.append(
            GroupRow(
                group_id=gid,
                task_key=members[0].task_id or members[0].action,
                n_members=len(members),
                member_ids=[m.example_id for m in members],
                rewards=[round(r, 6) for r in rewards],
                advantage_ready=len(members) >= 2 and var_ok,
            )
        )
    return out


def reward_table(
    traces: Iterable[dict],
    *,
    split: Optional[TaskSplit] = None,
    harness_variant: str = "default",
) -> list[RewardRow]:
    rows: list[RewardRow] = []
    for tr in traces or []:
        t = dict(tr or {})
        ex = trace_to_example(t, harness_variant=harness_variant)
        if ex.exclude_reason:
            continue
        params = t.get("params") or {}
        claim_gate = params.get("claim_gate") or {}
        claim_rejected = bool(claim_gate.get("accepted") is False)
        fail_class = str(params.get("fail_class") or t.get("fail_class") or "")
        r, detail = compute_reward(
            effect=float(ex.reward if ex.reward is not None else 0.0),
            success=ex.success,
            fail_class=fail_class,
            claim_rejected=claim_rejected,
        )
        tid = ex.task_id
        if split is not None:
            if tid in set(split.blacklist):
                sp = "blacklist"
            elif tid in set(split.private) or tid in set(split.ood):
                sp = "private"
            else:
                sp = "selection"
        else:
            sp = ""
        rows.append(
            RewardRow(
                example_id=ex.example_id,
                task_id=tid,
                group_id=group_key_for(t, ex),
                reward=detail["reward"],
                base=detail["base"],
                penalty_fail=detail["penalty_fail"],
                penalty_claim=detail["penalty_claim"],
                success=ex.success,
                fail_class=fail_class,
                agent_id=ex.agent_id,
                harness_variant=harness_variant,
                split=sp,
            )
        )
    return rows


def build_split_from_traces(traces: Iterable[dict], private_ratio: float = 0.2) -> TaskSplit:
    """Holdout-aligned split (AIDE²): sealed/i6/eval_loop stay blacklisted."""
    ids = []
    blacklist = []
    for tr in traces or []:
        t = dict(tr or {})
        tid = str(t.get("task_id") or t.get("trace_id") or t.get("id") or "")
        if not tid:
            continue
        ids.append(tid)
        bad, _why = is_blacklisted(t)
        if bad:
            blacklist.append(tid)
    return split_tasks(ids, private_ratio=private_ratio, blacklist=blacklist, seed=17)


def policy_matrix_groups(rows: Sequence[dict]) -> list[GroupRow]:
    """Build GRPO groups from a policy×world score matrix.

    rows: [{group_key/world_id, policy_id, reward, ...}, ...]
    Same world_id = same prompt; members = different policies (completions).
    """
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows or []:
        r = dict(row or {})
        gid = str(r.get("group_id") or r.get("world_id") or r.get("task_id") or "")
        pid = str(r.get("policy_id") or r.get("member_id") or "policy")
        rew = float(r.get("reward") if r.get("reward") is not None else r.get("score") or 0.0)
        if not gid:
            continue
        buckets[gid].append((pid, rew))
    out: list[GroupRow] = []
    for gid, members in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        rewards = [m[1] for m in members]
        var_ok = (max(rewards) - min(rewards)) > 1e-9 if len(rewards) >= 2 else False
        out.append(
            GroupRow(
                group_id="G-" + hashlib.sha256(gid.encode("utf-8")).hexdigest()[:12],
                task_key=gid,
                n_members=len(members),
                member_ids=[m[0] for m in members],
                rewards=[round(m[1], 6) for m in members],
                advantage_ready=len(members) >= 2 and var_ok,
                note="policy_matrix_same_world_multi_completion",
            )
        )
    return out


def collect_policy_matrix_from_pool(
    world_pool,
    policies: dict,
    max_worlds: int = 20,
    max_rounds: int = 8,
) -> list[dict]:
    """Evaluate each policy_fn on the same worlds → multi-completion matrix.

    policies: {policy_id: policy_fn}
    """
    rows: list[dict] = []
    worlds = list(getattr(world_pool, "worlds", []) or [])[:max_worlds]
    for w in worlds:
        wid = getattr(w, "world_id", None) or "world"
        for pid, fn in (policies or {}).items():
            try:
                ev = w.replay(fn, policy_name=str(pid), max_rounds=max_rounds)
                score = float(getattr(ev, "replay_score", 0.0) or 0.0)
                quality = float(getattr(ev, "quality", getattr(ev, "discovery_quality", 0.0)) or 0.0)
                probes = float(getattr(ev, "probes", 0) or 0)
            except Exception:
                score, quality, probes = 0.0, 0.0, 0.0
            # composite: quality + efficiency (break ties when quality saturates)
            effect = 0.7 * quality + 0.3 * max(0.0, 1.0 - probes / 20.0) + 0.05 * max(0.0, score)
            reward, _ = compute_reward(
                effect=effect,
                success=bool(quality >= 0.3 or score > 0),
                fail_class="",
            )
            rows.append(
                {
                    "group_id": str(wid),
                    "world_id": str(wid),
                    "policy_id": str(pid),
                    "score": score,
                    "quality": quality,
                    "probes": probes,
                    "reward": reward,
                }
            )
    return rows


def default_policy_matrix(n: int = 5) -> dict:
    """K structurally different exploration policies (not just beta)."""
    from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn
    from arsi.meta.eval_loop import fixed_exploration_fn

    specs = [
        ("beta_0.2_w1", 0.2, 1),
        ("beta_0.5_w2", 0.5, 2),
        ("beta_0.8_w3", 0.8, 3),
        ("beta_1.0_w4", 1.0, 4),
        ("fixed_w2", None, 2),
    ]
    pols = {}
    for name, beta, workers in specs[: max(2, int(n))]:
        if beta is None:
            pols[name] = fixed_exploration_fn(max_workers=workers)
        else:
            pols[name] = build_policy_fn(
                PortfolioPolicy(beta=beta, max_workers=workers, name=name)
            )
    return pols


def multisample_groups_from_traces(
    traces: Sequence[dict],
    *,
    variants: Sequence[str] = (),
) -> list[GroupRow]:
    """Group real multi-host / multi-variant outcomes for the same action key.

    Only groups with **actual reward variance** are advantage_ready —
    never invent scores (negative epistemology).
    """
    buckets: dict[str, dict[str, float]] = defaultdict(dict)
    for t in traces or []:
        if not isinstance(t, dict):
            continue
        key = str(t.get("action") or t.get("task_id") or "")
        if not key:
            continue
        vid = str((t.get("params") or {}).get("harness_variant") or t.get("agent_id") or "default")
        ex = trace_to_example(t)
        if ex.exclude_reason:
            continue
        buckets[key][vid] = float(ex.reward)
    out = []
    for key, vmap in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        mem = list(vmap.items())
        if len(mem) < 2:
            continue
        rewards = [r for _, r in mem]
        var_ok = (max(rewards) - min(rewards)) > 1e-9
        out.append(
            GroupRow(
                group_id="G-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
                task_key=key,
                n_members=len(mem),
                member_ids=[m[0] for m in mem],
                rewards=[round(r, 6) for r in rewards],
                advantage_ready=var_ok,
                note="real_multivariant_outcomes",
            )
        )
    return out


def load_world_pool_snapshot(path: str | Path | None = None):
    """Prefer full daemon snapshot when live ARSI.pool is tiny."""
    try:
        from arsi.foundation.paths import archive_dir
        from arsi.world_model.world_pool import WorldPool

        p = Path(path) if path else archive_dir() / "world_pool_snapshot.json"
        if p.exists():
            return WorldPool.load_from(p)
    except Exception:
        pass
    return None


def worlds_from_trace_chunks(
    traces: Sequence[dict],
    *,
    n_worlds: int = 8,
    min_chunk: int = 20,
) -> list:
    """Slice production traces into chunk worlds for multi-completion matrix."""
    from arsi.world_model.discovery_tree import DiscoveryTree
    from arsi.world_model.replay_world import ReplayWorld

    traces = [dict(t or {}) for t in (traces or [])]
    if not traces:
        return []
    n_worlds = max(1, int(n_worlds))
    chunk = max(min_chunk, len(traces) // n_worlds)
    worlds = []
    for i in range(n_worlds):
        sl = traces[i * chunk : (i + 1) * chunk] if i < n_worlds - 1 else traces[i * chunk :]
        if len(sl) < 5:
            continue
        tree = DiscoveryTree()
        try:
            tree.build_from_traces(sl)
            w = ReplayWorld.from_discovery_tree(tree, world_id=f"CHUNK-{i}")
            worlds.append(w)
        except Exception:
            continue
    return worlds


class _ChunkPool:
    """Minimal pool facade so collect_policy_matrix_from_pool can run."""

    def __init__(self, worlds: list):
        self.worlds = list(worlds or [])

    @property
    def size(self) -> int:
        return len(self.worlds)


def export_grpo_data_plane(
    traces: Iterable[dict],
    out_dir: str | Path,
    *,
    harness_variant: str = "default",
    private_ratio: float = 0.2,
    world_pool=None,
    policies: Optional[dict] = None,
    max_worlds: int = 20,
) -> dict:
    """Write groups.jsonl + rewards.jsonl + split.json + manifest. Export only.

    If world_pool+policies given, also fold policy-matrix groups (multi-completion
    on the same world) so advantage_ready > 0 without live multi-rollout.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces = [dict(t or {}) for t in (traces or [])]
    examples = [trace_to_example(t, harness_variant=harness_variant) for t in traces]
    kept = [(t, e) for t, e in zip(traces, examples) if not e.exclude_reason]
    kept_traces = [t for t, _ in kept]
    kept_examples = [e for _, e in kept]
    split = build_split_from_traces(kept_traces, private_ratio=private_ratio)
    rewards = reward_table(kept_traces, split=split, harness_variant=harness_variant)
    groups = build_groups(kept_examples, kept_traces)
    try:
        groups = groups + multisample_groups_from_traces(kept_traces)
    except Exception:
        pass
    matrix_rows: list[dict] = []
    if world_pool is not None and policies:
        matrix_rows = collect_policy_matrix_from_pool(world_pool, policies, max_worlds=max_worlds)
        groups = groups + policy_matrix_groups(matrix_rows)

    gpath = out / "grpo_groups.jsonl"
    with open(gpath, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
    rpath = out / "grpo_rewards.jsonl"
    with open(rpath, "w", encoding="utf-8") as f:
        for r in rewards:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    spath = out / "grpo_split.json"
    spath.write_text(json.dumps(split.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    n_ready = sum(1 for g in groups if g.advantage_ready)
    n_sel = sum(1 for r in rewards if r.split == "selection")
    n_priv = sum(1 for r in rewards if r.split == "private")
    manifest = {
        "schema": SCHEMA,
        "grpo_live": GRPO_LIVE,
        "mode": "export_only_no_training",
        "harness_variant": harness_variant,
        "n_traces": len(traces),
        "n_exported_rewards": len(rewards),
        "n_groups": len(groups),
        "n_advantage_ready_groups": n_ready,
        "n_policy_matrix_rows": len(matrix_rows),
        "split": {"selection": n_sel, "private": n_priv, "blacklist_ids": len(split.blacklist)},
        "fail_penalty_table": FAIL_CLASS_PENALTY,
        "paths": {
            "groups": str(gpath),
            "rewards": str(rpath),
            "split": str(spath),
        },
        "note": "grpo_pre_data_plane_holdout_aligned",
        "created_at": datetime.now().isoformat(),
    }
    mpath = out / "grpo_data_plane.manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(mpath)
    return manifest
