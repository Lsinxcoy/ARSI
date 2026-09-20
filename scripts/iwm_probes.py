r"""IWM behavioral probes — Q1–Q6 acceptance gate on live/demo data.

Usage:
    set PYTHONPATH=E:/ARSI/src
    python scripts/iwm_probes.py
    python scripts/iwm_probes.py --json
    python scripts/iwm_probes.py --steps 8

Exit code 0 if Q1–Q3 all pass (introspective_v1_candidate);
exit 2 if skeleton (Q1–Q3 any fail); exit 1 on runner error.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.schema import BehaviorTrace
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.verified import verified_from_pytest
from arsi.governor.core import AutopoieticGovernor
from arsi.iwm import IWM
from arsi.meta.live_manifest import ManifestStore
from arsi.mnemosyne.core import Mnemosyne
from arsi.pipelines.dream import DreamPipeline
from arsi.world_model.siwm import SIWM

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_arsi(archive: Path) -> ARSI:
    store = MnemosyneStore(str(archive / "arsi.db"))
    mnemosyne = Mnemosyne(store)
    siwm = SIWM(store)
    laws_p = archive / "laws.yaml"
    if not laws_p.exists():
        laws_p.write_text("laws: []\n", encoding="utf-8")
    laws = IronLaws(laws_p)
    dream = DreamPipeline(siwm, mnemosyne, llm=None)
    arsi = ARSI(
        store=store,
        mnemosyne=mnemosyne,
        siwm=siwm,
        governor=AutopoieticGovernor(siwm, store, laws),
        empowerment=EmpowermentEngine(mnemosyne, siwm, NullAdapter()),
        dream=dream,
        iron_laws=laws,
        llm=None,
    )
    arsi.iwm = IWM(archive_dir=archive / "iwm")
    arsi.dream.iwm = arsi.iwm
    arsi.governor.iwm = arsi.iwm
    arsi.manifest_store = ManifestStore(archive / "trace_pool")
    return arsi


def seed_traces(arsi: ARSI, n: int = 40) -> None:
    actions = ["learn", "remember", "dream", "maintain", "evolve"]
    for i in range(n):
        arsi.mnemosyne.ingest_trace(
            BehaviorTrace(
                agent_id=["mimo", "hermes", "synthex"][i % 3],
                action=actions[i % len(actions)],
                outcome="success" if i % 3 else "failure",
                effect=0.75 if i % 3 else 0.2,
            )
        )


def run_probes(arsi: ARSI, steps: int = 6) -> dict:
    arsi.siwm.train_from_history()
    arsi.siwm.eta.eta_smooth = 0.42
    for _ in range(steps):
        arsi.step()
    arsi.dream.execute(arsi.siwm.refresh_state())
    qs = arsi.iwm.q_scores()
    ok, gate = arsi.iwm.q_gate()
    health = arsi.iwm.health()
    claim = verified_from_pytest("arsi.iwm.suite", PROJECT_ROOT, pytest_args=["-q", "tests/test_iwm.py"])
    advice = arsi.iwm.governor_advice(arsi.siwm.get_state())
    return {
        "q_gate_ok": ok,
        "claim": gate["claim"],
        "failed": gate["failed"],
        "q_pass": {k: bool(v.get("pass")) for k, v in qs.items()},
        "q_scores": qs,
        "health": health,
        "eta": arsi.siwm.eta.value,
        "eta_policy": getattr(arsi.dream, "_last_eta_policy", ""),
        "provenance": arsi.iwm.provenance.count,
        "ledger_size": arsi.iwm.ledger.size,
        "hooks_applied": health.get("iwm", {}).get("hooks_applied", 0),
        "suite_verified": claim.to_dict(),
        "memory_trust": advice.get("memory_trust"),
        "trust_memory_for_learn": advice.get("trust_memory_for_learn"),
        "downweight_memory_ops": advice.get("downweight_memory_ops"),
        "organ_trust": advice.get("organ_trust"),
        "world_pool_size": arsi.world_pool.size,
        "layer1": arsi.get_stats().get("layer1"),
        "live_data_ready": arsi.world_pool.size >= 5,
        "note": (
            "live I6 requires pool>=5 + organ evidence on daemon-fed data; "
            "empty/thin organs → document as skeleton"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ARSI IWM Q1–Q6 probes")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--archive", default="")
    args = parser.parse_args()

    archive = Path(args.archive) if args.archive else Path(tempfile.mkdtemp(prefix="arsi_iwm_probe_"))
    archive.mkdir(parents=True, exist_ok=True)
    try:
        arsi = build_arsi(archive)
        seed_traces(arsi)
        report = run_probes(arsi, steps=args.steps)
        report["archive"] = str(archive)
        out = PROJECT_ROOT / "archive" / "eval" / "iwm_probe_latest.json"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(out)
        except Exception as e:
            report["report_error"] = str(e)

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print("IWM Q1–Q6 PROBE")
            print(f"  claim: {report['claim']}  failed={report['failed']}")
            print(f"  q_pass: {report['q_pass']}")
            print(f"  eta: {report['eta']:.4f}  policy: {report['eta_policy']}")
            print(f"  provenance: {report['provenance']}  ledger: {report['ledger_size']}")
            print(f"  hooks_applied: {report['hooks_applied']}")
            print(f"  suite_verified: {report['suite_verified'].get('verified')} "
                  f"({report['suite_verified'].get('reason')})")
            print(f"  report: {report.get('report_path', '')}")
            print("  boundary: empty/thin live organs → document as skeleton until daemon data thickens")

        return 0 if report["q_gate_ok"] and report["q_pass"].get("Q1_organs") and report["q_pass"].get("Q3_loop") else 2
    except Exception as e:
        print(f"probe_error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
