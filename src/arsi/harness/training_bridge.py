"""c9 Training Bridge — trajectories → reusable training signal (NOT live GRPO).

HarnessX closes harness–model loop via Cross-Harness GRPO. ARSI hosts API LLMs
(no weights), so this module **exports** task-level aligned examples instead:

  {task_id, harness_variant, outcome, reward, prompt_facts, trajectory_digest, …}

Use later for SFT / DPO / GRPO offline. Never invents rewards; evolve-blacklist
tasks (sealed_tasks / I6 gates) stay out of the training set.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = "arsi.c9.training_bridge.v1"


@dataclass
class TrainingExample:
    example_id: str
    task_id: str
    harness_variant: str
    outcome: str
    reward: float
    success: bool
    action: str
    agent_id: str
    trajectory_digest: str
    prompt_facts: dict = field(default_factory=dict)
    exclude_reason: str = ""
    source: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def is_blacklisted(trace: dict) -> tuple[bool, str]:
    """Evolve / train blacklist: sealed eval, I6 gates, eval_loop internals."""
    blob = json.dumps(trace, ensure_ascii=False, default=str).lower()
    for bad in ("sealed", "i6_gate", "eval_loop", "gold_answer", "answer_key"):
        if bad in blob:
            return True, f"blacklist:{bad}"
    return False, ""


def trace_to_example(trace: dict, harness_variant: str = "default") -> TrainingExample:
    t = dict(trace or {})
    task_id = str(t.get("task_id") or t.get("trace_id") or t.get("id") or "")
    outcome = str(t.get("outcome") or "")
    success = outcome.lower() in ("success", "ok", "true") or bool(t.get("success"))
    try:
        reward = float(t.get("effect") if t.get("effect") is not None else (1.0 if success else 0.0))
    except Exception:
        reward = 1.0 if success else 0.0
    excluded, reason = is_blacklisted(t)
    facts = {
        k: t.get(k)
        for k in ("action", "fail_class", "params", "note", "error")
        if t.get(k) is not None
    }
    digest = _digest({k: t.get(k) for k in ("action", "outcome", "effect", "params", "task_id", "agent_id")})
    return TrainingExample(
        example_id=f"TX-{digest}",
        task_id=task_id or digest,
        harness_variant=harness_variant,
        outcome=outcome,
        reward=reward,
        success=success,
        action=str(t.get("action") or "unknown"),
        agent_id=str(t.get("agent_id") or ""),
        trajectory_digest=digest,
        prompt_facts=facts,
        exclude_reason=reason if excluded else "",
        source="behavior_trace",
        created_at=datetime.now().isoformat(),
    )


def export_training_set(
    traces: Iterable[dict],
    out_path: str | Path,
    harness_variant: str = "default",
    include_excluded: bool = False,
) -> dict:
    """Write JSONL training examples + manifest. Blacklisted rows dropped unless flagged."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_ok = n_ex = 0
    with open(out, "w", encoding="utf-8") as f:
        for tr in traces or []:
            n_in += 1
            ex = trace_to_example(tr, harness_variant=harness_variant)
            if ex.exclude_reason and not include_excluded:
                n_ex += 1
                continue
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n_ok += 1
    manifest = {
        "schema": SCHEMA,
        "path": str(out),
        "n_in": n_in,
        "n_exported": n_ok,
        "n_excluded": n_ex,
        "harness_variant": harness_variant,
        "alignment": "task_level_not_action_level",
        "grpo": False,
        "note": "export_only_api_host_no_weights",
        "created_at": datetime.now().isoformat(),
    }
    man_path = out.with_suffix(out.suffix + ".manifest.json")
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    return manifest
