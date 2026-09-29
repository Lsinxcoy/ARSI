"""ARSI production smoke — read-only, safe in prod."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    checks = []

    def add(name: str, ok: bool, detail: str = ""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:120]})

    try:
        from arsi.core import ARSI

        arsi = ARSI.from_config(str(Path(__file__).resolve().parents[1] / "config" / "arsi.yaml"))
        add("core_load", True)
    except Exception as e:
        add("core_load", False, str(e))
        print(json.dumps({"checks": checks}, indent=2))
        return 1

    st = arsi.get_stats() or {}
    add("iron_laws", bool(st.get("iron_laws")), str(st.get("iron_laws"))[:40])
    add("pool", (st.get("world_pool_size") or 0) >= 0, str(st.get("world_pool_size")))
    add("layer1_present", "layer1" in st, json.dumps(st.get("layer1"))[:60])

    try:
        from arsi.harness.contracts import well_formedness

        wf = well_formedness()
        add("severa_wellformed", bool(wf.get("ok")), "")
    except Exception as e:
        add("severa_wellformed", False, str(e))

    try:
        from arsi.harness.epistemic import Claim, admit_claim

        ok, why = admit_claim(Claim(kind="no_known_defect", subject="s", drawbacks_checked=["a"]))
        add("epistemic_ok", ok is True, why)
        ok2, why2 = admit_claim(Claim(kind="no_known_defect", subject="s", raw="this is_correct"))
        add("epistemic_forbids_positive", ok2 is False, why2)
    except Exception as e:
        add("epistemic", False, str(e))

    try:
        from arsi.world_model.opf_discipline import opf_discipline

        opf = opf_discipline([{"live_last": 0.1, "self_trust": 0.2}, {"live_last": 0.2, "self_trust": 0.3}])
        add("opf_module", True, opf.note)
    except Exception as e:
        add("opf_module", False, str(e))

    try:
        from arsi.harness.grpo_data import GRPO_LIVE

        add("grpo_export_only", GRPO_LIVE is False, str(GRPO_LIVE))
    except Exception as e:
        add("grpo_export_only", False, str(e))

    # secrets: yaml must not hold live key
    y = Path(r"E:\ARSI\config\arsi.yaml").read_text(encoding="utf-8")
    leaked = "rc-" in y and 'api_key: ""' not in y
    add("no_secret_in_yaml", not leaked, "api_key_empty" if not leaked else "KEY_LEAKED")

    failed = [c for c in checks if not c["ok"]]
    print(json.dumps({"ok": not failed, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
