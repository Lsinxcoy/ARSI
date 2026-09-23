"""ODEWorld P1 — negative-velocity focus + z subgoal planning."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from arsi.iwm.organ_self import ORGAN_MEMORY
from arsi.iwm.host_strategy import build_host_strategy, strategy_for_skill_content
from arsi.world_model.capability_flow import CapabilityFlowTracker
from arsi.world_model.dynamics import DynamicsModel
from arsi.governor.pre_enactment import PreEnactmentEngine
from arsi.foundation.schema import WorldState


def _stats(eta=0.1, self_t=0.8, mem=0.7, live=0.5, pool=0.2, d_t=2.0):
    return {
        "eta": eta,
        "beta": 0.6,
        "world_pool_size": 4,
        "iwm": {"self_trust": self_t, "memory_trust": mem, "layer1_holdout": 0.8},
        "env_evolution": {
            "difficulty": {"d_t_mean": d_t, "d_t_spread": 1.0},
            "el": {"lineages": {"a": {"pass_rate": 0.5}}},
        },
        "live_history": [live],
        "pool_history": [pool],
        "grid_plan": {"branch_count": 2},
        "multi_agent": {"agents": {}},
    }


class TestFlowGuidance:
    def test_memory_drop_focus_reingest(self):
        tr = CapabilityFlowTracker()
        tr.observe(_stats(mem=0.8), t_wall=1000.0)
        tr.observe(_stats(mem=0.2), t_wall=1020.0)  # memory_trust falling
        g = tr.flow_guidance()
        assert "memory_trust" in g["negative_organs"]
        assert g["focus"] == "reingest"
        assert g["z_subgoal"] or g["n_v_updates"] >= 1

    def test_predictor_drop_focus_calibrate(self):
        tr = CapabilityFlowTracker()
        s1 = _stats()
        s2 = _stats()
        s1["iwm"]["layer1_holdout"] = 0.9
        s2["iwm"]["layer1_holdout"] = 0.1
        s2["iwm"]["self_trust"] = 0.8  # keep self stable
        tr.observe(s1, t_wall=100.0)
        tr.observe(s2, t_wall=130.0)
        g = tr.flow_guidance()
        assert "behavior_predictor_trust" in g["negative_organs"]
        assert g["focus"] == "calibrate"

    def test_no_negative_default_focus(self):
        tr = CapabilityFlowTracker()
        tr.observe(_stats(mem=0.5), t_wall=10.0)
        tr.observe(_stats(mem=0.6), t_wall=40.0)  # rising
        g = tr.flow_guidance()
        assert g["focus"] == "execute_with_evidence"
        assert g["negative_organs"] == [] or "memory_trust" not in g["negative_organs"]


class _FakeFlow:
    def flow_guidance(self, horizon_s: float = 600.0) -> dict:
        return {
            "focus": "reingest",
            "negative_organs": ["memory_trust"],
            "v_hat": {"memory_trust": -0.01},
            "z_now": {"memory_trust": 0.3},
            "z_subgoal": {"memory_trust": 0.5, "self_trust": 0.8},
            "horizon_s": horizon_s,
            "n_v_updates": 5,
            "source": "fake",
        }


class _FakeARSI:
    def __init__(self, flow):
        self.capability_flow = flow
        self.iwm = None


class TestHostStrategyP1:
    def test_negative_memory_overrides_learn_to_reingest(self):
        # IWM absent → default execute; flow says memory falling → reingest
        arsi = _FakeARSI(_FakeFlow())
        st = build_host_strategy(arsi, agent_id="hermes")
        assert st.focus == "reingest"
        assert "memory_trust" in st.negative_organs
        assert st.z_subgoal
        assert st.control_flags.get("use_z_subgoal") is True
        assert any("negative_velocity" in w for w in st.operational_warnings)
        block = st.as_structured_block()
        assert any("negative_velocity_organs" in ln for ln in block)
        assert any("z_subgoal" in ln for ln in block)

    def test_skill_content_mentions_flow(self):
        arsi = _FakeARSI(_FakeFlow())
        st = build_host_strategy(arsi, agent_id="mimo-desktop")
        s = strategy_for_skill_content(st)
        assert "negative_velocity_organs" in s
        assert "z_subgoal_keys" in s


class TestPreEnactmentP1:
    def _dyn(self):
        from arsi.foundation.store import MnemosyneStore
        import tempfile
        root = Path(tempfile.mkdtemp())
        return DynamicsModel(MnemosyneStore(str(root / "d.db")))

    def test_flow_bonus_boosts_learn_when_memory_negative(self):
        dyn = self._dyn()
        pe = PreEnactmentEngine(dyn)
        pe.flow_guide = _FakeFlow().flow_guidance()
        bonuses = pe._flow_action_bonuses()
        assert bonuses.get("learn", 0) > 0
        assert bonuses.get("remember", 0) >= 0

        evaluated = pe.evaluate_candidates(WorldState(), ["learn", "maintain", "dream"])
        by = {e["action"]: e for e in evaluated}
        assert "flow_bonus" in by["learn"]
        assert by["learn"]["flow_bonus"] >= by["maintain"]["flow_bonus"]
        assert by["learn"]["z_subgoal"] is True

    def test_bind_capability_flow_from_arsi(self):
        dyn = self._dyn()
        pe = PreEnactmentEngine(dyn)
        arsi = _FakeARSI(_FakeFlow())
        g = pe.bind_capability_flow(arsi)
        assert g.get("negative_organs") == ["memory_trust"]
        assert pe.flow_guide.get("focus") == "reingest"


class TestDreamCycleP1:
    @pytest.fixture
    def arsi(self, tmp_path, monkeypatch):
        import yaml
        from arsi.core import ARSI
        from arsi.foundation import paths as arsi_paths

        arch = tmp_path / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(arsi_paths, "archive_dir", lambda: arch)
        config = {
            "llm": {
                "provider": "openai",
                "model": "test",
                "api_key": "sk-test-disabled",
                "use_proxy": False,
                "fallback_to_heuristic": True,
                "timeout": 1,
            },
            "db_path": ":memory:",
            "iron_laws_path": str(tmp_path / "iron_laws.yaml"),
            "sealed_tasks_path": str(tmp_path / "sealed_tasks.yaml"),
        }
        (tmp_path / "iron_laws.yaml").write_text(
            "laws:\n  - id: G10\n    name: test\n    description: test\n    severity: block\n",
            encoding="utf-8",
        )
        (tmp_path / "arsi.yaml").write_text(yaml.dump(config), encoding="utf-8")
        instance = ARSI.from_config(tmp_path / "arsi.yaml")
        if instance.llm:
            instance.llm._client = None
        yield instance
        instance.close()

    def test_dream_returns_flow_guidance_and_z_subgoal(self, arsi):
        for i in range(14):
            arsi.ingest_trace(
                agent_id=["hermes", "mimo"][i % 2],
                action=["learn", "empower"][i % 2],
                outcome="success",
                effect=0.65,
                params={"fail_class": "ok"},
            )
        result = arsi.dream_rsi_cycle(skip_llm=True)
        assert result.get("ran") is True
        assert "flow_guidance" in result
        assert "z_subgoal" in result
        # pre-enactment should be bound after cycle
        assert arsi.pre_enactment.flow_guide is not None
