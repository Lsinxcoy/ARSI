"""ARSI Human Supervision CLI v2 — enhanced monitoring and control.

Usage:
    set ARSI_API_KEY=sk-...
    python scripts/human_cli.py

Commands:
    dashboard       — full system dashboard (state + dims + dynamics + cost + log)
    stats           — show system stats
    state           — show world state (Φ+Ψ+η)
    step            — run one step
    term [N]        — run a term with N steps (default 10)
    dream           — force dream cycle
    empower ID      — empower an agent
    dims            — run all 6 empowerment dimension analyses
    dim NAME        — run one dimension (knowledge/decomposition/calibration/attention/metacognition/environment)
    dynamics        — show dynamics model status
    preenact        — show pre-enactment evaluation of candidates
    traces [N]      — show recent traces (default 10)
    beliefs         — show current beliefs
    laws            — show iron laws
    eval            — run evaluation
    log [N]         — show recent dream sessions and term reports (default 5)
    manifests [N]   — list live cycle manifests (default 5)
    grid            — show current / planned GridPlan + next plan preview
    beta            — show beta, sweep history, params
    pool            — world pool stats
    dreamrsi        — run Dream-RSI meta cycle + eval loop
    compare         — run fixed vs dream_rsi evaluation compare
    params          — show dream_rsi hyperparams + official status
    iwm             — IWM health + Q-gate
    organs          — organ reliability snapshot
    qgate           — Q1–Q6 scores and claim
    frontier        — knowledge frontier + explore bias
    help            — show this help
    quit            — exit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.foundation.schema import EmpowermentDimension


def format_dict(d: dict, indent: int = 2) -> str:
    return json.dumps(d, indent=indent, ensure_ascii=False, default=str)


def cmd_dashboard(arsi: ARSI) -> None:
    """Full system dashboard — one-screen overview."""
    stats = arsi.get_stats()
    state = arsi.siwm.refresh_state()

    print("\n" + "═" * 56)
    print("  ARSI 系统仪表盘")
    print("═" * 56)

    # World state
    print("\n  ── 世界状态 ──")
    print(f"  Φ: 代数={state.phi.generation}  存储={state.phi.storage_stats}")
    print(f"  Ψ: 信念={len(state.psi.beliefs)}  情绪={state.psi.affect}")
    eta = state.eta
    eta_bar = "█" * int(eta * 20) + "░" * (20 - int(eta * 20))
    print(f"  η: [{eta_bar}] {eta:.3f}")

    # Core metrics
    print("\n  ── 核心指标 ──")
    print(f"  步数: {stats.get('step_count', 0)}  Term: {stats.get('term_count', 0)}")
    print(f"  轨迹: {stats.get('trace_count', 0)}  经验: {stats.get('experience_count', 0)}")
    print(f"  梦境: {stats.get('dreams', 0)}  预演: {stats.get('pre_enactment_count', 0)}")

    # LLM usage
    print("\n  ── LLM 使用 ──")
    print(f"  可用: {stats.get('llm_available', False)}")
    print(f"  Governor: {stats.get('llm_governor_calls', 0)}  MindZero: {stats.get('llm_mindzero_calls', 0)}")
    print(f"  Token: {stats.get('cost_tokens', 0)}  验证查询: {stats.get('cost_verifier_queries', 0)}")

    # Dynamics
    print("\n  ── 动力学 ──")
    print(f"  已训练: {stats.get('dynamics_trained', False)}  动作数: {stats.get('dynamics_actions', 0)}")

    # Iron laws
    laws = stats.get("iron_laws", [])
    print(f"\n  ── 铁律: {len(laws)} 条 ──")

    # Recent log
    dreams = arsi.store.get_self_records("dream_session", limit=2)
    terms = arsi.store.get_self_records("term_report", limit=1)
    if dreams:
        d = json.loads(dreams[0].get("data", "{}"))
        print(f"\n  ── 最近梦境 ──")
        print(f"  η: {d.get('eta_before', '?')} → {d.get('eta_after', '?')}")
    if terms:
        t = json.loads(terms[0].get("data", "{}"))
        gd = t.get("gain_decomposition", {})
        print(f"\n  ── 最近 Term ──")
        print(f"  {t.get('term_id', '?')}: 增益={gd.get('total', '?')} "
              f"(放大={gd.get('amplified', '?')} 进口={gd.get('imported', '?')} 自组织={gd.get('self_organized', '?')})")

    print("\n" + "═" * 56)


def cmd_stats(arsi: ARSI) -> None:
    stats = arsi.get_stats()
    print("\n── 系统状态 ──")
    for k, v in stats.items():
        if k == "iron_laws":
            print(f"  {k}: {len(v)} laws")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def cmd_state(arsi: ARSI) -> None:
    state = arsi.siwm.refresh_state()
    print("\n── 联合世界状态 ──")
    print(f"  Φ (物理):")
    print(f"    代数: {state.phi.generation}")
    print(f"    存储: {state.phi.storage_stats}")
    print(f"  Ψ (心智):")
    print(f"    信念数: {len(state.psi.beliefs)}")
    print(f"    情绪: {state.psi.affect}")
    for i, b in enumerate(state.psi.beliefs[:5]):
        print(f"    信念[{i}]: {b.content[:60]} (conf={b.confidence:.2f})")
    print(f"  η (失配度): {state.eta:.4f}")
    depth = arsi.siwm.eta.adaptive_depth()
    print(f"  仿真深度: {depth}")
    if arsi.siwm.eta.should_dream():
        print(f"  ⚠ η 超阈值，建议 dream")


def cmd_step(arsi: ARSI) -> None:
    result = arsi.step()
    print(f"\n── Step {result['step']} ──")
    d = result["decision"]
    print(f"  动作: {d['action']}")
    print(f"  理由: {d['reason'][:100]}")
    print(f"  来源: {d['source']}")
    print(f"  置信度: {d.get('confidence', 0):.3f}")
    for a in result["actions"]:
        print(f"  执行: {a.get('type', '?')} — {str(a)[:80]}")


def cmd_term(arsi: ARSI, n: int = 10) -> None:
    print(f"\n── 运行 term ({n} steps) ──")
    result = arsi.run_term(n_steps=n)
    for s in result["steps"]:
        d = s["decision"]
        print(f"  Step {s['step']}: {d['action']} [{d.get('source', '?')}]")
    print(f"\n  评估: {format_dict(result['evaluation'])}")


def cmd_dream(arsi: ARSI) -> None:
    state = arsi.siwm.refresh_state()
    eta_before = state.eta
    new_state = arsi.dream.execute(state)
    print(f"\n── 梦境周期 ──")
    print(f"  η: {eta_before:.4f} → {new_state.eta:.4f}")
    print(f"  信念数: {len(state.psi.beliefs)} → {len(new_state.psi.beliefs)}")
    print(f"  LLM 调和次数: {arsi.dream._llm_reconcile_count}")


def cmd_empower(arsi: ARSI, agent_id: str) -> None:
    print(f"\n── 赋能 {agent_id} ──")
    result = arsi.empower_agent(agent_id)
    print(f"  诊断维度: {result['diagnosis'].get('dimension', '?')}")
    print(f"  诊断理由: {result['diagnosis'].get('reason', '?')[:100]}")
    print(f"  验证状态: {result['verification']}")


def cmd_dims(arsi: ARSI) -> None:
    print(f"\n── 六维度分析 ──")
    traces = arsi.store.get_recent_traces(n=50)
    state = arsi.siwm.refresh_state()
    orch = DimensionOrchestrator(arsi.store, llm=arsi.llm)
    results = orch.run_all(traces, state)

    for dim_name, result in results.items():
        if "error" in result:
            print(f"  {dim_name}: 错误 - {result['error']}")
            continue
        sense = result.get("sense", {})
        gen = result.get("generate", {})
        val = result.get("validation_status", "?")
        gap = sense.get("gap_detected", False)
        action = gen.get("action", "none")
        print(f"  {dim_name}: gap={gap} → {action} → {val}")


def cmd_dim(arsi: ARSI, dim_name: str) -> None:
    dim_map = {
        "knowledge": EmpowermentDimension.KNOWLEDGE,
        "decomposition": EmpowermentDimension.DECOMPOSITION,
        "calibration": EmpowermentDimension.CALIBRATION,
        "attention": EmpowermentDimension.ATTENTION,
        "metacognition": EmpowermentDimension.METACOGNITION,
        "environment": EmpowermentDimension.ENVIRONMENT,
    }
    if dim_name not in dim_map:
        print(f"  未知维度: {dim_name}")
        print(f"  可用: {', '.join(dim_map.keys())}")
        return

    print(f"\n── {dim_name.upper()} 维度分析 ──")
    traces = arsi.store.get_recent_traces(n=50)
    state = arsi.siwm.refresh_state()
    orch = DimensionOrchestrator(arsi.store, llm=arsi.llm)
    result = orch.run_dimension(dim_map[dim_name], traces, state)

    sense = result.get("sense", {})
    gen = result.get("generate", {})
    val = result.get("validation_status", "?")
    evidence = result.get("validation_evidence", {})

    print(f"  感知: gap_detected={sense.get('gap_detected')}")
    for k, v in sense.items():
        if k != "gap_detected":
            print(f"    {k}: {str(v)[:80]}")
    print(f"  生成: {gen.get('action', '?')}")
    for k, v in gen.items():
        if k != "action":
            print(f"    {k}: {str(v)[:80]}")
    print(f"  验证: {val}")
    if evidence:
        print(f"    证据: {str(evidence)[:100]}")


def cmd_dynamics(arsi: ARSI) -> None:
    print(f"\n── 动力学模型状态 ──")
    stats = arsi.dynamics.stats
    tm = stats.get("transition_model", {})
    print(f"  已训练: {tm.get('trained', False)}")
    print(f"  动作数: {tm.get('action_count', 0)}")
    print(f"  总转移: {tm.get('total_transitions', 0)}")
    print(f"  每动作: {tm.get('actions', {})}")
    print(f"  预测次数: {stats.get('prediction_count', 0)}")
    print(f"  LLM 规则: {stats.get('llm_extractor', {}).get('rule_count', 0)}")


def cmd_preenact(arsi: ARSI) -> None:
    print(f"\n── 预演评估 ──")
    state = arsi.siwm.refresh_state()
    candidates = ["dream", "learn", "evolve", "maintain", "remember"]
    results = arsi.pre_enactment.evaluate_candidates(state, candidates)
    for r in results:
        print(f"  {r['action']}: score={r['score']:.4f} conf={r['confidence']:.3f} src={r['source']}")
    pe_stats = arsi.pre_enactment.stats
    print(f"\n  预演总次数: {pe_stats.get('pre_enactment_count', 0)}")


def cmd_traces(arsi: ARSI, n: int = 10) -> None:
    traces = arsi.store.get_recent_traces(n=n)
    print(f"\n── 最近 {len(traces)} 条轨迹 ──")
    for t in traces:
        ts = t.get("timestamp", "?")[:19] if isinstance(t.get("timestamp"), str) else "?"
        print(f"  [{ts}] {t.get('agent_id', '?')} "
              f"{t.get('action', '?')} → {t.get('outcome', '?')} "
              f"(effect={t.get('effect', 0):.2f})")


def cmd_beliefs(arsi: ARSI) -> None:
    state = arsi.siwm.refresh_state()
    print(f"\n── 当前信念 ({len(state.psi.beliefs)}) ──")
    for i, b in enumerate(state.psi.beliefs):
        print(f"  [{i}] {b.content}")
        print(f"      conf={b.confidence:.2f} source={b.source}")


def cmd_laws(arsi: ARSI) -> None:
    print(f"\n── 铁律 ({len(arsi.iron_laws.law_ids)}) ──")
    for law_id in arsi.iron_laws.law_ids:
        law = arsi.iron_laws.get_law(law_id)
        if law:
            print(f"  {law.id}: {law.name} [{law.severity}]")
            print(f"      {law.description}")


def cmd_eval(arsi: ARSI) -> None:
    result = arsi.run_evaluation()
    print(f"\n── 评估 ──")
    print(format_dict(result))


def cmd_log(arsi: ARSI, n: int = 5) -> None:
    print(f"\n── 最近日志 ──")
    dreams = arsi.store.get_self_records("dream_session", limit=n)
    terms = arsi.store.get_self_records("term_report", limit=n)
    print(f"  梦境会话 ({len(dreams)}):")
    for d in dreams:
        data = json.loads(d.get("data", "{}"))
        print(f"    η: {data.get('eta_before', '?')} → {data.get('eta_after', '?')} "
              f"(改善: {data.get('eta_improved', '?')})")
    print(f"  Term 报告 ({len(terms)}):")
    for t in terms:
        data = json.loads(t.get("data", "{}"))
        print(f"    {data.get('term_id', '?')}: steps={data.get('steps', '?')} "
              f"η={data.get('final_stats', {}).get('eta', '?')}")


def cmd_manifests(arsi: ARSI, n: int = 5) -> None:
    print("\n── Live Cycle Manifests ──")
    rows = arsi.manifest_store.recent(n)
    if not rows:
        print("  (empty — run dreamrsi/term to produce manifests)")
        return
    for m in rows:
        print(
            f"  #{m.cycle_id} {m.kind}: best={m.best_score:.4f} β={m.beta:.2f} "
            f"W/R plan={m.planned_grid.branch_count}/{m.planned_grid.refine_count} "
            f"eff={m.effective_grid.branch_count}/{m.effective_grid.refine_count} "
            f"probes={m.probe_work} pool={m.pool_size_after}"
        )
        print(f"      reason={m.planned_grid.reason} gate={m.quality_gate.to_dict()}")
    print(f"  store_root: {arsi.manifest_store.root}")


def cmd_grid(arsi: ARSI) -> None:
    print("\n── GridPlan ──")
    print(f"  current: {format_dict(arsi.current_grid_plan.to_dict())}")
    nxt = arsi.plan_next_grid()
    print(f"  next:    {format_dict(nxt.to_dict())}")
    print(f"  note: W = parallel focus count (dimensions), not OS threads")


def cmd_beta(arsi: ARSI) -> None:
    print("\n── Beta / Sweep ──")
    print(f"  portfolio.beta = {arsi.portfolio_policy.beta:.3f}  workers={arsi.portfolio_policy.max_workers}")
    print(f"  default_uncertain = {arsi.dream_rsi_params.beta_default_uncertain}")
    print(f"  sweep_grid = {arsi.dream_rsi_params.beta_sweep_grid}")
    sweeps = arsi.manifest_store.recent_beta_sweeps(3)
    if not sweeps:
        print("  (no beta_sweep.json yet)")
    for s in sweeps:
        print(f"  cycle={s.get('cycle_id')}: selected={s.get('selected_default_beta')} reason={s.get('reason')}")
        print(f"    non_degenerate={s.get('non_degenerate')} points={len(s.get('points') or [])}")


def cmd_pool(arsi: ARSI) -> None:
    print("\n── World Pool ──")
    print(f"  size={arsi.world_pool.size}")
    print(f"  stats={format_dict(arsi.world_pool.stats())}")


def cmd_dreamrsi(arsi: ARSI) -> None:
    print("\n── Dream-RSI meta cycle ──")
    result = arsi.dream_rsi_cycle()
    print(format_dict({k: v for k, v in result.items() if k != "feedback_excerpt"}))
    if result.get("eval_loop"):
        print("\n── Eval Loop ──")
        print(format_dict(result["eval_loop"]))


def cmd_compare(arsi: ARSI) -> None:
    from arsi.meta.eval_loop import run_eval_loop
    print("\n── Fixed vs Dream-RSI compare ──")
    result = run_eval_loop(arsi, params=arsi.dream_rsi_params)
    print(format_dict(result.to_dict()))


def cmd_params(arsi: ARSI) -> None:
    print("\n── Dream-RSI Hyperparams ──")
    print(format_dict(arsi.dream_rsi_params.to_dict()))
    print("  source: config/dream_rsi_params.yaml (official code not released → ARSI defaults)")


def cmd_iwm(arsi: ARSI) -> None:
    print("\n── IWM Health ──")
    if getattr(arsi, "iwm", None) is None:
        print("  IWM not attached")
        return
    print(format_dict(arsi.iwm.health()))
    ok, gate = arsi.iwm.q_gate()
    print(f"\n  q_gate_ok={ok} claim={gate.get('claim')} failed={gate.get('failed')}")


def cmd_organs(arsi: ARSI) -> None:
    print("\n── Organ Reliability ──")
    if getattr(arsi, "iwm", None) is None:
        print("  IWM not attached")
        return
    snap = arsi.iwm.organ.snapshot()
    for name, rec in snap.items():
        print(f"  {name:20s} status={rec['status']:12s} rel={rec['reliability']:.3f} "
              f"n={rec['samples']} hint={rec.get('control_hint','')}")
    print(f"\n  hooks: {format_dict(arsi.iwm.organ.control_hooks())}")


def cmd_qgate(arsi: ARSI) -> None:
    print("\n── IWM Q Gate ──")
    if getattr(arsi, "iwm", None) is None:
        print("  IWM not attached")
        return
    ok, gate = arsi.iwm.q_gate()
    print(format_dict(gate))
    print(f"\n  claim={gate.get('claim')}  v1_ok={ok}  failed={gate.get('failed')}")


def cmd_frontier(arsi: ARSI) -> None:
    print("\n── Knowledge Frontier ──")
    if getattr(arsi, "iwm", None) is None:
        print("  IWM not attached")
        return
    print(format_dict(arsi.iwm.frontier.report()))


def main():
    print("=" * 60)
    print("ARSI 人类监督界面 v2")
    print("=" * 60)
    print("输入 help 查看命令列表\n")

    arsi = ARSI.from_config("config/arsi.yaml")

    while True:
        try:
            raw = input("arsi> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(__doc__)
            elif cmd == "dashboard":
                cmd_dashboard(arsi)
            elif cmd == "stats":
                cmd_stats(arsi)
            elif cmd == "state":
                cmd_state(arsi)
            elif cmd == "step":
                cmd_step(arsi)
            elif cmd == "term":
                n = int(args[0]) if args else 10
                cmd_term(arsi, n)
            elif cmd == "dream":
                cmd_dream(arsi)
            elif cmd == "empower":
                agent = args[0] if args else "agent_alpha"
                cmd_empower(arsi, agent)
            elif cmd == "dims":
                cmd_dims(arsi)
            elif cmd == "dim":
                dim = args[0] if args else "knowledge"
                cmd_dim(arsi, dim)
            elif cmd == "dynamics":
                cmd_dynamics(arsi)
            elif cmd == "preenact":
                cmd_preenact(arsi)
            elif cmd == "traces":
                n = int(args[0]) if args else 10
                cmd_traces(arsi, n)
            elif cmd == "beliefs":
                cmd_beliefs(arsi)
            elif cmd == "laws":
                cmd_laws(arsi)
            elif cmd == "eval":
                cmd_eval(arsi)
            elif cmd == "log":
                n = int(args[0]) if args else 5
                cmd_log(arsi, n)
            elif cmd == "manifests":
                n = int(args[0]) if args else 5
                cmd_manifests(arsi, n)
            elif cmd == "grid":
                cmd_grid(arsi)
            elif cmd == "beta":
                cmd_beta(arsi)
            elif cmd == "pool":
                cmd_pool(arsi)
            elif cmd == "dreamrsi":
                cmd_dreamrsi(arsi)
            elif cmd == "compare":
                cmd_compare(arsi)
            elif cmd == "params":
                cmd_params(arsi)
            elif cmd == "iwm":
                cmd_iwm(arsi)
            elif cmd == "organs":
                cmd_organs(arsi)
            elif cmd == "qgate":
                cmd_qgate(arsi)
            elif cmd == "frontier":
                cmd_frontier(arsi)
            else:
                print(f"未知命令: {cmd}。输入 help 查看帮助。")
        except Exception as e:
            print(f"错误: {e}")

    arsi.close()


if __name__ == "__main__":
    main()
