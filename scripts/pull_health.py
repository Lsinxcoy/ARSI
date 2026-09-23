import json
from pathlib import Path
from datetime import datetime

rows = []
for l in Path(r"E:\ARSI\archive\arsi_health.jsonl").read_text(encoding="utf-8", errors="ignore").splitlines():
    l = l.strip()
    if l.startswith("{"):
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
print("rows", len(rows), "now", datetime.now().isoformat())
for d in rows[-10:]:
    print("---")
    print(d.get("tick"), "|", d.get("status"), "|", d.get("phase"), "|", d.get("timestamp"))
    print(
        "  traces", d.get("trace_count"),
        "exp", d.get("experience_count"),
        "pool", d.get("world_pool_size"),
        "cycles", d.get("manifest_cycles"),
        "beta", d.get("beta"),
        "eta", d.get("eta"),
        "elapsed", d.get("elapsed_s"),
        "new", d.get("new_traces"),
    )
    h = d.get("harvest")
    if isinstance(h, dict) and h:
        print("  harvest", {k: h.get(k) for k in ("pool_size", "traces_kept", "warn_admitted", "pass_admitted")})
        evo = h.get("evolution") or {}
        if evo:
            print("  harvest_evolution", {k: evo.get(k) for k in ("attempted", "accepted_count", "error")})
    ee = d.get("env_evolution")
    if isinstance(ee, dict) and ee:
        diff = ee.get("difficulty") or (ee.get("pool_difficulty") or {})
        elh = ee.get("el") or ee.get("el_health") or {}
        print(
            "  env_evolution",
            "d_t", diff.get("d_t_mean"), "spread", diff.get("d_t_spread"),
            "homog", diff.get("homogeneous"),
            "el_adv", elh.get("advances"), "lin", len(elh.get("lineages") or {}),
        )
    cf = d.get("capability_flow")
    if isinstance(cf, dict) and cf:
        print(
            "  capability_flow",
            "n", cf.get("n_samples"), "v_upd", cf.get("n_v_updates"),
            "mse", cf.get("field_mse_ewma"),
            "neg_organs", cf.get("last_negative_organs"),
        )
        rk = cf.get("last_rankme") or {}
        zw = rk.get("z_window") or {}
        if zw:
            print("  rankme_z", zw.get("effective_rank"), "collapse", zw.get("collapse"), "dim", zw.get("dim"))
    dg = d.get("difficulty_flow_gate") or {}
    if isinstance(dg, dict) and dg:
        print(
            "  d×flow_gate",
            "allow", dg.get("allow_promote"),
            "d_t_rising", dg.get("d_t_rising"),
            "prog", (dg.get("progress") or {}).get("score"),
            "reason", (dg.get("reason") or "")[:80],
        )
    iw = d.get("iwm")
    if isinstance(iw, dict):
        inner = iw.get("iwm") if isinstance(iw.get("iwm"), dict) else iw
        if isinstance(inner, dict):
            print(
                "  mem", inner.get("memory_trust"), inner.get("memory_status"),
                "trust_learn", inner.get("trust_memory_for_learn"),
                "L1", inner.get("layer1_holdout"),
            )
            print("  organs", inner.get("organ_trust"))
            dl = inner.get("dream_loop")
            if dl:
                print("  dream", {k: dl.get(k) for k in ("count", "helped", "last_verdict", "allowed_default")})
            qg = inner.get("q_gate") or {}
            if qg:
                print("  q_claim", qg.get("claim"), "failed", qg.get("failed"))
    if d.get("layer1"):
        print("  layer1", d.get("layer1"))
    ma = d.get("multi_agent")
    if isinstance(ma, dict) and ma:
        print("  multi_agent n", ma.get("agent_count"), "unmeasured", ma.get("unmeasured_agents"))
        for k, v in (ma.get("agents") or {}).items():
            print("   ", k, v.get("organ_status"), "done", v.get("tasks_completed"), "rate", v.get("success_rate"))
    hl = d.get("host_loop")
    if isinstance(hl, dict) and hl:
        print("  host_loop", {k: hl.get(k) for k in list(hl)[:5]})
    v = d.get("verified")
    if isinstance(v, dict):
        print("  verified", v.get("verified"), v.get("reason"))
    gp = d.get("grid_plan")
    if isinstance(gp, dict):
        print("  grid", gp.get("branch_count"), gp.get("refine_count"), (gp.get("reason") or "")[:70])

p = Path(r"E:\ARSI\archive\world_pool_snapshot.json")
d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
print("pool_snapshot_worlds", len(d.get("worlds") or []))
el_snap = d.get("el") or {}
if el_snap:
    print("pool_el tau", el_snap.get("tau"), "advances", el_snap.get("advances"), "lineages", len(el_snap.get("lineages") or {}))
env_cfg = d.get("env_cfg") or {}
if env_cfg:
    print("pool_env_cfg", {k: env_cfg.get(k) for k in ("enabled", "effort", "evolve_every_n", "max_evolved_per_harvest")})
w_dts = []
for w in d.get("worlds") or []:
    ed = w.get("env_difficulty") or {}
    if ed.get("d_t") is not None:
        w_dts.append((w.get("world_id"), ed.get("d_t"), w.get("generation"), w.get("evolved")))
if w_dts:
    print("pool_D_T sample", w_dts[-8:])

print("=== manifests tail ===")
iters = sorted(Path(r"E:\ARSI\archive\trace_pool").glob("iter*"), key=lambda x: x.name)[-5:]
for iid in iters:
    mp = iid / "live_cycle_manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        print(
            iid.name,
            "cycle", m.get("cycle_id"),
            m.get("kind"),
            "best", m.get("best_score"),
            "live_cap", m.get("live_capability_score"),
            "beta", m.get("beta"),
            "pool", m.get("pool_size_after"),
            "policy", m.get("deployed_policy"),
            "notes", m.get("notes"),
        )
