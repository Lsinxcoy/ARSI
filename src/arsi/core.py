"""ARSI Core — the main system orchestrator.

Assembles all modules into a runnable autopoietic system:
  Mnemosyne (memory) + SIWM (world model) + Governor (decisions)
  + Empowerment (agent improvement) + Dream (self-refresh)
  + Sealed Evaluator (honest measurement) + LLM (intelligence)

Usage:
    from arsi.core import ARSI
    arsi = ARSI.from_config("config/arsi.yaml")
    arsi.ingest_trace(agent_id="a1", action="learn", outcome="success", effect=0.8)
    action = arsi.step()  # one full Governor decision + execution cycle
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from arsi.foundation.config import ARSIConfig, load_config
from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.llm import LLMClient, LLMConfig
from arsi.foundation.schema import (
    BehaviorTrace,
    CoupledAction,
    EmpowermentDimension,
    EmpowermentOp,
    VerificationStatus,
    WorldState,
)
from arsi.foundation.store import MnemosyneStore
from arsi.mnemosyne.core import Mnemosyne
from arsi.mnemosyne.memory_proxy import MemoryProxy, NullNativeMemory
from arsi.world_model.siwm import SIWM
from arsi.governor.core import AutopoieticGovernor, DimensionManager
from arsi.governor.pre_enactment import PreEnactmentEngine
from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.empowerment.lifecycle_integration import DimensionLifecycleIntegrator
from arsi.pipelines.dream import DreamPipeline
from arsi.llm_brain import LLMPoweredGovernor, LLMPoweredMindZero, LLMPoweredDiagnosis, LLMPoweredDream
from arsi.world_model.dynamics import DynamicsModel
from arsi.world_model.counterfactual import CounterfactualSimulator
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.world_pool import WorldPool
from arsi.world_model.replay_world import ReplayWorld
from arsi.world_model.dream_rsi_deep import AdaptiveBehaviorController
from arsi.governor.exploration_policy import ExplorationPolicy, PolicyDevelopmentAgent
from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn, default_beta_from_live
from arsi.governor.operator_scheduler import OperatorScheduler
from arsi.foundation.quality_gate import TraceQualityGate
from arsi.meta.live_manifest import LiveCycleManifest, ManifestStore, GridSnapshot, QualityGateStats
from arsi.meta.grid_plan import GridPlan, GridPlanningContext, plan_grid
from arsi.meta.beta_sweep import sweep_beta
from arsi.meta.dream_rsi_params import DreamRSIParams, load_dream_rsi_params
from arsi.meta.eval_loop import run_eval_loop
from arsi.iwm import IWM
from arsi.sealed_eval.gain_decomposition import GainDecomposer
from arsi.foundation.cost_ledger import CostLedger

logger = logging.getLogger(__name__)


class ARSI:
    """The complete ARSI system.

    One object that owns everything: memory, world model, governor,
    empowerment, dream pipeline, sealed evaluation, and LLM.
    """

    def __init__(
        self,
        store: MnemosyneStore,
        mnemosyne: Mnemosyne,
        siwm: SIWM,
        governor: AutopoieticGovernor,
        empowerment: EmpowermentEngine,
        dream: DreamPipeline,
        iron_laws: IronLaws,
        llm: Optional[LLMClient] = None,
    ):
        self.store = store
        self.mnemosyne = mnemosyne
        self.siwm = siwm
        self.governor = governor
        self.empowerment = empowerment
        self.dream = dream
        self.iron_laws = iron_laws
        self.llm = llm
        self._step_count = 0
        self._term_count = 0
        self._memory_proxies: dict[str, MemoryProxy] = {}

        # LLM brain (wraps heuristic modules with LLM intelligence)
        self.llm_governor = LLMPoweredGovernor(llm=llm, heuristic_governor=governor)
        self.llm_mindzero = LLMPoweredMindZero(llm=llm, heuristic_mindzero=siwm.mindzero)
        self.llm_diagnosis = LLMPoweredDiagnosis(llm=llm)
        self.llm_dream = LLMPoweredDream(llm=llm)

        # SIWM Layer 2 + Pre-enactment
        self.dynamics = DynamicsModel(store, llm=llm)
        self.pre_enactment = PreEnactmentEngine(self.dynamics)
        self._dynamics_trained = False

        # SIWM Layer 3: Counterfactual simulator
        self.counterfactual = CounterfactualSimulator(self.dynamics, llm=llm)

        # Dream-RSI: Discovery tree + programmable exploration policy
        # Deep mechanisms from arXiv:2609.14858 Appendix B.2 + §3
        self.discovery_tree = DiscoveryTree()
        self.exploration_policy = ExplorationPolicy(name="arsi_default")
        self.policy_developer = PolicyDevelopmentAgent(llm=llm)
        self.world_pool = WorldPool()
        self.portfolio_policy = PortfolioPolicy(beta=0.6, max_workers=3)
        self._live_cycle_history: list[dict] = []
        self._term_trees_built = 0
        self._dream_rsi_cycles = 0
        # Phase D: meta-layer (manifest / grid / gate / scheduler / adaptive)
        self.manifest_store = ManifestStore()
        self.quality_gate = TraceQualityGate(llm=llm)
        self.operator_scheduler = OperatorScheduler()
        self.adaptive_controller = AdaptiveBehaviorController()
        self.current_grid_plan = GridPlan(reason="unplanned")
        self._world_min_verdict = "WARN"  # PASS | WARN
        self._term_best_scores: list[float] = []
        # Hyperparams: official unpublished → config/dream_rsi_params.yaml
        self.dream_rsi_params = load_dream_rsi_params()
        self._world_min_verdict = self.dream_rsi_params.world_min_verdict
        # QualityGate quota + replay score mode from params
        if hasattr(self.quality_gate, "max_warn_ratio"):
            self.quality_gate.max_warn_ratio = float(getattr(self.dream_rsi_params, "max_warn_ratio", 0.25) or 0.25)
            self.quality_gate.max_admitted = int(getattr(self.dream_rsi_params, "max_admitted", 120) or 120)
        self.portfolio_policy = PortfolioPolicy(
            beta=self.dream_rsi_params.beta_default_uncertain,
            max_workers=self.dream_rsi_params.bootstrap_branch_count,
        )
        self._last_eval_loop: Optional[dict] = None

        # IWM — introspective world model (Q1–Q6); organ trust drives control
        default_iwm_dir = Path(__file__).resolve().parents[2] / "archive" / "iwm"
        try:
            self.iwm = IWM(archive_dir=default_iwm_dir)
        except Exception:
            self.iwm = IWM()
        self.dream.iwm = self.iwm
        self.governor.iwm = self.iwm

        # N1: Gain decomposition
        self.gain_decomposer = GainDecomposer(store, llm=llm)

        # N2: Cost ledger
        self.cost_ledger = CostLedger()

        # N3: Dimension lifecycle integration
        self.dim_orchestrator = DimensionOrchestrator(store, llm=llm)
        self.dim_lifecycle = DimensionLifecycleIntegrator(
            store, self.dim_orchestrator, governor.dims, llm=llm
        )

        # Enable semantic edge discovery
        self.mnemosyne.edge_discovery.enable_semantic()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "ARSI":
        """Build ARSI from a YAML config file."""
        config_path = Path(config_path)
        config = load_config(config_path)
        config_dir = config_path.parent

        # Store
        store = MnemosyneStore(config.db_path)

        # Iron laws
        laws_path = config_dir / Path(config.iron_laws_path).name
        iron_laws = IronLaws(laws_path)

        # LLM
        llm = cls._build_llm(config)

        # Core modules
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store, config.eta)
        dim_mgr = DimensionManager(
            max_per_term=config.dimensions.max_per_term,
            min_data_sufficiency=config.dimensions.min_data_sufficiency,
        )
        governor = AutopoieticGovernor(siwm, store, iron_laws, dim_mgr)
        empowerment = EmpowermentEngine(mnemosyne, siwm, NullAdapter())
        dream = DreamPipeline(siwm, mnemosyne, llm=llm)

        logger.info(f"ARSI initialized (db={config.db_path}, llm={'yes' if llm and llm.available else 'no'})")

        return cls(
            store=store,
            mnemosyne=mnemosyne,
            siwm=siwm,
            governor=governor,
            empowerment=empowerment,
            dream=dream,
            iron_laws=iron_laws,
            llm=llm,
        )

    @staticmethod
    def _build_llm(config: ARSIConfig) -> Optional[LLMClient]:
        """Build LLM client from config."""
        api_key = os.environ.get("ARSI_API_KEY", "")
        if not api_key and config.llm.api_key:
            api_key = config.llm.api_key

        llm_config = LLMConfig(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=api_key,
            api_base=config.llm.api_base or "",
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
            timeout=config.llm.timeout,
            use_proxy=config.llm.use_proxy,
            proxy_url=config.llm.proxy_url,
            extra_headers=config.llm.extra_headers or {},
            fallback_to_heuristic=config.llm.fallback_to_heuristic,
        )
        return LLMClient(llm_config)

    # ── Trace Ingestion ─────────────────────────────────────────

    def ingest_trace(
        self,
        agent_id: str,
        action: str,
        outcome: str,
        effect: float = 0.0,
        params: Optional[dict] = None,
        capture_state: bool = False,
    ) -> BehaviorTrace:
        """Ingest a behavior trace from a host agent.

        Set capture_state=True to record state snapshots (slower,
        only for ARSI's own execution traces, not bulk adapter ingestion).
        """
        state_before = self.siwm.get_state() if capture_state else WorldState()

        trace = BehaviorTrace(
            agent_id=agent_id,
            action=action,
            action_params=params or {},
            outcome=outcome,
            effect=effect,
            generation=self.store.current_generation,
            state_before=state_before,
        )
        self.mnemosyne.ingest_trace(trace)

        if capture_state:
            trace.state_after = self.siwm.refresh_state()

        return trace

    # ── Memory Proxy ────────────────────────────────────────────

    def get_memory_proxy(self, agent_id: str, native=None) -> MemoryProxy:
        """Get or create a memory proxy for an agent."""
        if agent_id not in self._memory_proxies:
            self._memory_proxies[agent_id] = MemoryProxy(
                self.mnemosyne, native or NullNativeMemory(), agent_id
            )
        return self._memory_proxies[agent_id]

    # ── Core Step ───────────────────────────────────────────────

    def step(self) -> dict:
        """Execute one full ARSI cycle.

        Decision priority:
          1. Pre-enactment (Layer 2 dynamics prediction)
          2. LLM (if pre-enactment confidence too low)
          3. Heuristic (fallback)
        """
        self._step_count += 1
        result = {"step": self._step_count, "actions": [], "decision_source": "heuristic"}

        # 0. Train dynamics model periodically
        if self._step_count % 10 == 1 or not self._dynamics_trained:
            train_result = self.dynamics.train()
            self._dynamics_trained = train_result["statistical"]["trained"]
            result["dynamics_trained"] = self._dynamics_trained

        # 1. Refresh world state
        state = self.siwm.refresh_state()
        result["state"] = {
            "generation": state.phi.generation,
            "eta": round(state.eta, 4),
            "belief_count": len(state.psi.beliefs),
            "storage": state.phi.storage_stats,
        }

        # 3. Three-layer decision — skip untrusted pre-enactment when IWM says so
        candidates = ["dream", "learn", "evolve", "maintain", "remember"]
        advice = self.iwm.governor_advice(state) if self.iwm is not None else {}
        result["iwm_advice"] = {
            k: advice.get(k)
            for k in (
                "self_trust",
                "degrade_to_baseline",
                "downweight_pre_enactment",
                "forbid_default_dream",
                "prefer_learn",
                "unreliable_organs",
            )
            if k in advice
        }

        # Layer A: Pre-enactment (predict consequences) — downweight if dynamics untrusted
        use_pre = not (
            advice.get("degrade_to_baseline") or advice.get("downweight_pre_enactment")
        )
        if use_pre:
            pre_result = self.pre_enactment.select_best(state, candidates)
        else:
            pre_result = {"confidence": 0.0, "action": "", "reason": "iwm_dynamics_untrusted"}
        if pre_result["confidence"] > 0.1:
            decision = pre_result
            result["decision_source"] = "pre_enactment"
        else:
            # Layer B: LLM decision
            llm_calls_before = self.llm_governor._llm_calls
            llm_decision = self.llm_governor.decide(state, candidates)
            decision = llm_decision
            result["decision_source"] = "llm" if self.llm_governor._llm_calls > llm_calls_before else "heuristic"
            # Track LLM cost
            if self.llm_governor._llm_calls > llm_calls_before:
                self.cost_ledger.record_llm_call(input_tokens=400, output_tokens=150)

        # IWM forbid_default_dream: override dream → learn
        if advice.get("forbid_default_dream") and decision.get("action") == "dream":
            decision = dict(decision)
            decision["action"] = "learn"
            decision["reason"] = (
                decision.get("reason", "") + " [IWM: dream 器官无证据帮助，改 learn]"
            ).strip()
            result["decision_source"] = f"{result['decision_source']}+iwm_override"
            result["iwm_override"] = "dream→learn"

        result["decision"] = {
            "action": decision["action"],
            "reason": decision.get("reason", ""),
            "risk": decision.get("risk", "low"),
            "source": result["decision_source"],
            "confidence": decision.get("confidence", 0.0),
        }

        # Provenance for core.step path
        if self.iwm is not None:
            try:
                rec = self.iwm.record_decision(
                    action=decision["action"],
                    reason=decision.get("reason", ""),
                    decision_source=result["decision_source"],
                    candidates=candidates,
                    state_digest={
                        "eta": state.eta,
                        "generation": state.phi.generation,
                        "belief_count": len(state.psi.beliefs),
                    },
                    iwm_snapshot=advice,
                    confidence=float(decision.get("confidence", 0.0) or 0.0),
                )
                result["decision"]["provenance_id"] = rec.decision_id
            except Exception:
                pass

        # 3b. Execute the action
        exec_result = self._execute_action_str(decision["action"], state)
        result["actions"].append(exec_result)
        exec_ok = not (
            isinstance(exec_result, dict)
            and exec_result.get("type") in ("freeze", "error")
        ) and not (isinstance(exec_result, dict) and exec_result.get("note") == "unknown action")
        # Stricter meaningful success for IWM calibrator
        exec_note = ""
        if isinstance(exec_result, dict):
            exec_note = str(exec_result.get("note") or exec_result.get("type") or "")
            if decision["action"] == "learn" and int(exec_result.get("distilled", 0) or 0) <= 0:
                exec_ok = False
                exec_note = exec_note or "learn_no_distill"
            if decision["action"] == "remember":
                exec_ok = False
                exec_note = "waiting_new_traces"

        # 4. Update η (predict what we did) + IWM observation
        pred = self.siwm.predict_and_update(decision["action"])
        result["prediction"] = {
            k: pred.get(k) for k in (
                "predicted_category", "actual_category", "correct", "eta",
                "live_accuracy", "holdout_accuracy",
            )
        }
        if self.iwm is not None:
            self.iwm.observe_behavior(
                action=decision["action"],
                outcome=exec_result.get("type", decision["action"]) if isinstance(exec_result, dict) else "",
                effect=0.6 if exec_ok else 0.2,
                predicted=pred.get("predicted_category"),
                predicted_correct=pred.get("correct"),
            )
            # Bind measured Layer1 holdout to organ every few steps
            if self._step_count % 3 == 0 or self._step_count == 1:
                try:
                    train_res = self.siwm.train_from_history()
                    holdout = float(train_res.get("holdout_accuracy") or pred.get("holdout_accuracy") or 0.0)
                    self.iwm.observe_layer1_holdout(holdout, note=f"live={pred.get('live_accuracy')}")
                    result["layer1_train"] = train_res
                except Exception as e:
                    logger.warning(f"layer1 holdout bind failed: {e}")
            # Dynamics organ: pre-enactment predicted delta vs post-step state
            if result.get("decision_source", "").startswith("pre_enactment"):
                try:
                    after = self.siwm.refresh_state()
                    predicted_delta = {
                        "eta": state.eta,
                        "trace_count": state.phi.storage_stats.get("trace_count", 0),
                        "experience_count": state.phi.storage_stats.get("experience_count", 0),
                        "generation": state.phi.generation,
                    }
                    actual_delta = {
                        "eta": after.eta,
                        "trace_count": after.phi.storage_stats.get("trace_count", 0),
                        "experience_count": after.phi.storage_stats.get("experience_count", 0),
                        "generation": after.phi.generation,
                    }
                    self.iwm.observe_dynamics(
                        action=decision["action"],
                        predicted_state=predicted_delta,
                        actual_state=actual_delta,
                    )
                except Exception:
                    pass
            # Memory organ: learn distill success
            if decision["action"] == "learn":
                distilled = 0
                if isinstance(exec_result, dict):
                    distilled = int(exec_result.get("distilled", 0) or 0)
                self.iwm.observe_memory(
                    helped=distilled > 0,
                    note=f"distilled={distilled}",
                )
            if result.get("decision", {}).get("provenance_id"):
                # Periodically record baseline arm (no IWM) for Q6 comparison
                used_iwm = bool(advice) and not (
                    advice.get("degrade_to_baseline") or advice.get("downweight_pre_enactment")
                )
                # every 5th decision treated as baseline sample when hooks idle
                if self._step_count % 5 == 0 and not advice.get("forbid_default_dream"):
                    used_iwm = False
                self.iwm.apply_outcome(
                    result["decision"]["provenance_id"],
                    used_iwm=used_iwm,
                    success=bool(exec_ok),
                    note=f"{result['decision_source']}|{exec_note}",
                )

        # 5. Governor metabolism (distill policy)
        if self._step_count % 5 == 0:
            policy = self.governor.distill_policy()
            result["policy_distilled"] = bool(policy)

        # 6. Check if dream is needed — honor IWM forbid_default_dream
        if self.siwm.eta.should_dream() and not advice.get("forbid_default_dream"):
            dream_result = self.dream.execute(self.siwm.refresh_state())
            result["actions"].append({
                "type": "dream",
                "eta_before": state.eta,
                "eta_after": dream_result.eta,
                "loop_trial": getattr(self.dream, "_last_loop_trial", {}),
                "eta_policy": getattr(self.dream, "_last_eta_policy", ""),
            })
        elif self.siwm.eta.should_dream() and advice.get("forbid_default_dream"):
            result["actions"].append({
                "type": "dream_skipped",
                "reason": "iwm_forbid_default_dream",
                "eta": state.eta,
            })

        # 7. Dream-RSI meta-exploration: periodically dream over history pool
        # Paper: improved policy is redeployed online; history is a replay
        # simulator, NOT semantic guidance injected into prompts.
        every_n = getattr(self.dream_rsi_params, "dream_every_n_steps", 8) or 8
        if self._step_count % every_n == 0:
            dream_rsi = self.dream_rsi_cycle()
            result["dream_rsi"] = dream_rsi

        return result

    def _execute_action_str(self, action_name: str, state: WorldState) -> dict:
        """Execute an action by name."""
        a = action_name

        if a == "dream":
            new_state = self.dream.execute(state)
            return {"type": "dream", "eta_after": new_state.eta}

        elif a == "learn":
            return self._execute_learn(state)

        elif a == "evolve":
            return self._execute_evolve(state)

        elif a == "maintain":
            return self._execute_maintain()

        elif a == "remember":
            return {"type": "remember", "note": "waiting for new traces"}

        elif a == "freeze":
            return {"type": "freeze", "reason": action.a_ment.reason}

        else:
            return {"type": a, "note": "unknown action"}

    def _execute_learn(self, state: WorldState) -> dict:
        """Learn: distill PROXY traces into EXPERIENCE."""
        proxy_count = state.phi.storage_stats.get("proxy_count", 0)
        exp_count = state.phi.storage_stats.get("experience_count", 0)

        if proxy_count <= exp_count:
            return {"type": "learn", "note": "no new traces to distill"}

        # Distill high-importance traces
        proxy_traces = self.store.search_memories(
            zone=__import__("arsi.foundation.schema", fromlist=["MemoryZone"]).MemoryZone.PROXY,
            limit=10,
        )
        distilled = 0
        for t in proxy_traces:
            if t.importance >= 0.5 and t.status.value == "active":
                self.mnemosyne._distill(t)
                distilled += 1

        return {"type": "learn", "distilled": distilled}

    def _execute_evolve(self, state: WorldState) -> dict:
        """Evolve: attempt mechanism mutation (placeholder)."""
        exp_count = state.phi.storage_stats.get("experience_count", 0)
        if exp_count < 5:
            return {"type": "evolve", "note": f"insufficient experience ({exp_count}/5)"}

        # Record an evolution attempt
        self.store.record_effect("evolve_attempt", 0.5, "placeholder_mutation")
        return {"type": "evolve", "note": "mutation attempted", "effect": 0.5}

    def _execute_maintain(self) -> dict:
        """Maintain: memory consolidation + decay."""
        result = self.mnemosyne.consolidate()
        return {"type": "maintain", **result}

    # ── Empowerment ─────────────────────────────────────────────

    def empower_agent(self, agent_id: str) -> dict:
        """Empower a host agent with LLM-powered diagnosis."""
        state = self.siwm.refresh_state()
        traces = self.store.get_recent_traces(n=20, agent_id=agent_id)

        # LLM diagnosis
        diagnosis = self.llm_diagnosis.diagnose(traces, state)

        # Execute empowerment
        op = self.empowerment.empower(agent_id, state)

        return {
            "empowerment_id": op.id,
            "agent_id": agent_id,
            "diagnosis": diagnosis,
            "verification": op.verification.value,
            "evidence": op.verification_evidence,
        }

    # ── Sealed Evaluation ───────────────────────────────────────

    def run_evaluation(self, task_set_path: Optional[str] = None) -> dict:
        """Run sealed evaluation (requires agent adapter)."""
        # Simplified: report current stats as a proxy for evaluation
        stats = self.get_stats()
        return {
            "capability_proxy": stats["experience_count"] / max(stats["trace_count"], 1),
            "self_model_eta": self.siwm.eta.value,
            "generation": stats["generation"],
        }

    # ── Stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get comprehensive system stats."""
        m_stats = self.mnemosyne.get_stats()
        g_stats = self.governor.stats
        e_stats = self.empowerment.stats
        d_stats = self.dream.stats
        pe_stats = self.pre_enactment.stats
        dyn_stats = self.dynamics.stats

        return {
            **m_stats,
            "step_count": self._step_count,
            "term_count": self._term_count,
            "eta": round(self.siwm.eta.value, 4),
            "governor_decisions": g_stats.get("decision_count", 0),
            "empowerments": e_stats.get("empowerment_count", 0),
            "dreams": d_stats.get("dream_count", 0),
            "llm_available": self.llm.available if self.llm else False,
            "llm_governor_calls": self.llm_governor._llm_calls,
            "llm_mindzero_calls": self.llm_mindzero._llm_calls,
            "llm_diagnosis_calls": self.llm_diagnosis._llm_calls,
            "pre_enactment_count": pe_stats.get("pre_enactment_count", 0),
            "dynamics_trained": dyn_stats.get("transition_model", {}).get("trained", False),
            "dynamics_actions": dyn_stats.get("transition_model", {}).get("action_count", 0),
            "cost_tokens": self.cost_ledger.snapshot()["tokens"]["total"],
            "cost_llm_calls": self.cost_ledger.snapshot()["tokens"]["llm_calls"],
            "cost_verifier_queries": self.cost_ledger.snapshot()["verifier"]["queries"],
            "iron_laws": self.iron_laws.law_ids,
            "world_pool_size": self.world_pool.size,
            "manifest_cycles": self.manifest_store.size,
            "beta": self.portfolio_policy.beta,
            "grid_plan": self.current_grid_plan.to_dict(),
            "quality_gate": self.quality_gate.stats,
            "adaptive_trend": self.adaptive_controller.current_trend,
            "dream_rsi_params": self.dream_rsi_params.to_dict(),
            "last_eval_loop": self._last_eval_loop,
            "iwm": self.iwm.health() if self.iwm is not None else {},
            "iwm_q_gate": self.iwm.q_gate()[1] if self.iwm is not None else {},
            "layer1": {
                "live_accuracy": round(getattr(self.siwm.layer1, "live_accuracy", 0.0) or 0.0, 4),
                "holdout_accuracy": round(self.siwm.last_holdout_accuracy, 4),
                "rule_count": len(self.siwm.layer1.rules),
            },
        }

    # ── Lifecycle ───────────────────────────────────────────────

    def run_term(self, n_steps: int = 10) -> dict:
        """Run a full improvement term with gain decomposition and cost tracking."""
        self._term_count += 1
        term_id = f"term_{self._term_count}"

        # Phase D: plan grid + schedule operators for this term
        grid_plan = self.plan_next_grid()
        scheduled = self.operator_scheduler.schedule(
            budget=max(1.0, float(self.current_grid_plan.branch_count)),
            max_operators=self.current_grid_plan.branch_count,
        )
        scheduled_dims = [op.dimension.value for op in scheduled]
        n_steps = min(n_steps, max(1, self.current_grid_plan.branch_count * self.current_grid_plan.refine_count))

        # Reset cost ledger for this term
        self.cost_ledger.reset()

        # Record before-evaluation
        before_eval = self.run_evaluation()
        traces_before = len(self.store.get_recent_traces(n=1000))

        results = []
        for i in range(n_steps):
            result = self.step()
            results.append(result)
            # Track LLM costs (approximate)
            if result.get("decision_source") == "llm":
                self.cost_ledger.record_llm_call(input_tokens=500, output_tokens=200)
            logger.info(f"  Step {i+1}/{n_steps}: {result['decision']['action']}")

        # End-of-term evaluation
        after_eval = self.run_evaluation()
        self.cost_ledger.record_verifier_query(1)

        # Gain decomposition
        term_traces = self.store.get_recent_traces(n=100)
        decomposition = self.gain_decomposer.decompose(before_eval, after_eval, term_traces)
        self.gain_decomposer.record(decomposition, term_id)

        # Cost-benefit
        cost_benefit = self.cost_ledger.cost_benefit(decomposition.get("total", 0))

        # Dimension lifecycle integration
        state = self.siwm.refresh_state()
        lifecycle_result = self.dim_lifecycle.run_and_integrate(term_traces, state)

        # Adaptive controller: feed performance
        best_score = float(after_eval.get("score", after_eval.get("success_rate", 0.0)) or 0.0)
        if isinstance(after_eval, dict):
            for key in ("score", "total", "success_rate", "avg_score"):
                if key in after_eval and isinstance(after_eval[key], (int, float)):
                    best_score = float(after_eval[key])
                    break
        self.adaptive_controller.record_performance(best_score)
        self._term_best_scores.append(best_score)

        # MetaRSI Law 2: capability change invalidates operator signals
        self.operator_scheduler.mark_capability_change()

        # Live cycle manifest for this term
        probe_work = max(0, traces_before)  # approximation: traces available this term
        manifest = self.manifest_store.next_manifest(
            kind="run_term",
            planned_grid=GridSnapshot(
                branch_count=self.current_grid_plan.branch_count,
                refine_count=self.current_grid_plan.refine_count,
                reason=self.current_grid_plan.reason,
                selected_dimensions=scheduled_dims,
                selected_operators=scheduled_dims,
            ),
            effective_grid=GridSnapshot(
                branch_count=len(scheduled_dims) or self.current_grid_plan.branch_count,
                refine_count=max(1, n_steps // max(1, len(scheduled_dims) or 1)),
                reason="effective_from_run",
                selected_dimensions=scheduled_dims,
                selected_operators=scheduled_dims,
                opened_width=len(scheduled_dims),
                opened_depth=max(1, n_steps // max(1, len(scheduled_dims) or 1)),
            ),
            probe_work=n_steps,
            decision_rounds=n_steps,
            best_score=best_score,
            avg_score=best_score,
            beta=self.portfolio_policy.beta,
            deployed_policy=self.portfolio_policy.name,
            pool_size_after=self.world_pool.size,
            agents=sorted({t.get("agent_id", "") for t in term_traces if t.get("agent_id")}),
            sealed_delta=None,
            gain_decomposition={
                "total": decomposition.get("total", 0),
                "amplified": decomposition.get("amplified", 0),
                "imported": decomposition.get("imported", 0),
                "self_organized": decomposition.get("self_organized", 0),
            } if isinstance(decomposition, dict) else {},
            notes=term_id,
        )
        self.manifest_store.append(manifest)
        self._live_cycle_history.append(manifest.beta_history_row())

        # Record term report
        self.mnemosyne.write_self_record("term_report", {
            "term_id": term_id,
            "steps": n_steps,
            "final_stats": self.get_stats(),
            "evaluation": after_eval,
            "gain_decomposition": decomposition,
            "cost_benefit": cost_benefit,
            "lifecycle_recommendation": lifecycle_result.get("recommendation", {}),
            "grid_plan": self.current_grid_plan.to_dict(),
            "manifest_cycle": manifest.cycle_id,
        })

        return {
            "term_id": term_id,
            "steps": results,
            "evaluation": after_eval,
            "gain_decomposition": decomposition,
            "cost_benefit": cost_benefit,
            "lifecycle": lifecycle_result.get("lifecycle_updates", {}),
            "final_stats": self.get_stats(),
            "grid_plan": self.current_grid_plan.to_dict(),
            "scheduled_operators": scheduled_dims,
            "manifest_cycle": manifest.cycle_id,
            "beta": self.portfolio_policy.beta,
        }

    # ── Dream-RSI: Discovery Tree + Policy Development ─────────

    def build_discovery_tree(self) -> dict:
        """Build discovery tree from all behavior traces.

        Dream-RSI mechanism: converts flat traces into structured tree
        that enables replay simulation.
        """
        traces = self.store.get_recent_traces(n=500)
        stats = self.discovery_tree.build_from_traces(traces)
        return stats

    def harvest_term_tree(self, world_id: str | None = None) -> dict:
        """Dream-RSI online phase: append current traces as a new world.

        Paper §3: after a rollout, tree T_t is appended to history
        H_t = H_{t-1} ∪ {T_t}. Policy improvement then dreams across ALL worlds.
        Phase D5a: only QualityGate PASS/WARN traces enter the replay pool.
        """
        traces = self.store.get_recent_traces(n=300)
        if not traces:
            return {"harvested": False, "reason": "no_traces"}

        filtered, gate_stats = self._filter_traces_for_world(traces)
        if not filtered:
            return {
                "harvested": False,
                "reason": "all_traces_rejected_by_gate",
                "quality_gate": gate_stats,
            }

        world = self.world_pool.append_from_traces(
            filtered,
            world_id=world_id,
            max_parallelism=self.portfolio_policy.max_workers,
        )
        # Apply configured replay score mode to new worlds
        try:
            mode = getattr(self.dream_rsi_params, "score_mode", "quality_anchored") or "quality_anchored"
            world.score_mode = mode
            world.min_quality_signal = float(getattr(self.dream_rsi_params, "min_quality_signal", 0.08) or 0.08)
            world.weak_quality_cost_scale = float(getattr(self.dream_rsi_params, "weak_quality_cost_scale", 0.2) or 0.2)
        except Exception:
            pass
        # Keep latest tree object in sync for legacy APIs
        self.discovery_tree = DiscoveryTree()
        self.discovery_tree.build_from_traces(filtered)
        self._term_trees_built += 1
        return {
            "harvested": True,
            "world_id": world.world_id,
            "pool_size": self.world_pool.size,
            "node_count": len(world._full),
            "traces_in": len(traces),
            "traces_kept": len(filtered),
            "quality_gate": gate_stats,
        }

    @staticmethod
    def _verdict_from_gate2(g2: dict) -> str:
        if not isinstance(g2, dict):
            return "WARN"
        vals = [str(v).upper() for v in g2.values() if v]
        if "FAIL" in vals:
            return "FAIL"
        if "PASS" in vals and "WARN" not in vals:
            return "PASS"
        return "WARN"

    def _filter_traces_for_world(self, traces: list[dict]) -> tuple[list[dict], dict]:
        """NeoHorse quality gate → only clean traces become replay worlds.

        PASS-first: always prefer PASS; WARN admitted only up to quota.
        """
        min_v = (self._world_min_verdict or "PASS").upper()
        keep_order = {"PASS": 2, "WARN": 1, "FAIL": 0, "NOT_EVALUATED": 0}
        min_rank = keep_order.get(min_v, 2)
        max_warn_ratio = float(getattr(self.quality_gate, "max_warn_ratio", 0.25) or 0.25)
        max_admitted = int(getattr(self.quality_gate, "max_admitted", 120) or 120)

        scored: list[tuple[str, float, dict]] = []
        stats = {"pass": 0, "warn": 0, "fail": 0, "not_evaluated": 0}
        for t in traces:
            result = self.quality_gate.evaluate(dict(t))
            if not result.get("accepted"):
                stats["fail"] += 1
                continue
            verdict = self._verdict_from_gate2(result.get("gate2", {}))
            if verdict == "PASS":
                stats["pass"] += 1
            elif verdict == "WARN":
                stats["warn"] += 1
            elif verdict == "FAIL":
                stats["fail"] += 1
            else:
                stats["not_evaluated"] += 1
            rank = keep_order.get(verdict, 0)
            if rank < min_rank and verdict != "PASS":
                # still allow WARN only if min_rank allows and quota later permits
                if rank < keep_order.get("WARN", 1):
                    continue
            effect = float(t.get("effect", 0.0) or 0.0)
            nt = dict(t)
            nt.setdefault("params", {})
            if isinstance(nt["params"], dict):
                nt["params"]["difficulty"] = result.get("gate3_difficulty")
                nt["params"]["gate_verdict"] = verdict
            nt["fail_class"] = "ok" if verdict != "FAIL" else "unknown"
            # sort key: PASS first, then higher effect
            scored.append((verdict, effect, nt))

        pass_items = [x for x in scored if x[0] == "PASS"]
        warn_items = [x for x in scored if x[0] == "WARN"]
        # Prefer high-effect PASS
        pass_items.sort(key=lambda x: x[1], reverse=True)
        warn_items.sort(key=lambda x: x[1], reverse=True)

        kept: list[dict] = [x[2] for x in pass_items]
        warn_quota = max(0, int(len(kept) * max_warn_ratio / max(1e-6, 1 - max_warn_ratio)))
        if min_rank <= 1:
            kept.extend(x[2] for x in warn_items[:warn_quota])
        if len(kept) > max_admitted:
            kept = kept[:max_admitted]
        stats["warn_quota"] = warn_quota
        stats["warn_admitted"] = min(len(warn_items), warn_quota) if min_rank <= 1 else 0
        stats["pass_admitted"] = min(len(pass_items), max_admitted)
        stats["kept"] = len(kept)
        stats["min_verdict"] = min_v
        return kept, stats

    def plan_next_grid(self) -> GridPlan:
        """Phase D2: evidence-driven W/R from live manifests + adaptive force."""
        history = self.manifest_store.beta_history(n=8)
        if not history:
            history = self._live_cycle_history[-8:]
        all_dims = [d.value for d in EmpowermentDimension]
        covered = list(self.current_grid_plan.to_dict().get("evidence", {}).get("covered", []))
        if not covered:
            covered = [
                m.effective_grid.selected_dimensions[-1]
                for m in self.manifest_store.recent(3)
                if m.effective_grid.selected_dimensions
            ]
            covered = covered[-1] if covered else []
            if isinstance(covered, str):
                covered = [covered]
            # union of recent selected dims
            union = []
            for m in self.manifest_store.recent(5):
                union.extend(m.effective_grid.selected_dimensions or m.planned_grid.selected_dimensions or [])
            covered = list(dict.fromkeys(union))

        force = 1.0
        if len(self._term_best_scores) >= 3:
            self.adaptive_controller.record_performance(self._term_best_scores[-1])
            trend = self.adaptive_controller.current_trend
            if trend == "rising":
                force = 0.75
            elif trend == "plateau":
                force = 1.25
            elif trend == "declining":
                force = 0.9

        ctx = GridPlanningContext(
            history=history,
            hard_max_branch_count=self.dream_rsi_params.hard_max_branch_count,
            hard_max_refine_count=self.dream_rsi_params.hard_max_refine_count,
            covered_dimensions=covered,
            all_dimensions=all_dims,
            budget_force=force,
            cost_budget=self.dream_rsi_params.default_cost_budget,
        )
        self.current_grid_plan = plan_grid(ctx)
        # Sync portfolio width with plan
        self.portfolio_policy = PortfolioPolicy(
            beta=self.portfolio_policy.beta,
            max_workers=max(1, self.current_grid_plan.branch_count),
            name=self.portfolio_policy.name,
        )
        if self.governor and hasattr(self.governor, "dims"):
            self.governor.dims.max_per_term = self.current_grid_plan.branch_count
        return self.current_grid_plan

    def dream_rsi_cycle(self, num_revisions: int = 2) -> dict:
        """Full Dream-RSI offline phase: dream over world pool, redeploy best policy.

        Phase D: plans grid, sweeps beta, writes live_cycle_manifest + beta_sweep.
        """
        grid_plan = self.plan_next_grid()
        harvest = self.harvest_term_tree()
        if self.world_pool.size == 0:
            return {"ran": False, "reason": "empty_pool", "harvest": harvest, "grid_plan": grid_plan.to_dict()}

        current_fn = build_policy_fn(self.portfolio_policy)
        current_eval = self.world_pool.evaluate_policy_across_pool(
            current_fn, policy_name=self.portfolio_policy.name
        )

        # LLM policy revision using structured replay feedback (NOT semantic guidance)
        feedback = self._pool_feedback_for_llm(current_eval)
        new_policy = ExplorationPolicy(
            name=f"portfolio_r{self._dream_rsi_cycles + 1}",
            code=self.exploration_policy.code,
        )
        revised = new_policy.revise(feedback, self.llm) if self.llm and self.llm.available else False

        # Phase D3: beta sweep on pool + live history rule
        live_hist = self.manifest_store.beta_history(n=3) or self._live_cycle_history[-3:]

        def _factory(beta: float):
            pol = PortfolioPolicy(
                beta=beta,
                max_workers=max(1, grid_plan.branch_count),
                name=f"beta_{beta:.2f}",
            )
            return build_policy_fn(pol)

        sweep = sweep_beta(
            self.world_pool,
            _factory,
            grid=self.dream_rsi_params.beta_sweep_grid,
            parallel_lambda=self.dream_rsi_params.parallel_lambda,
            live_history=live_hist,
        )

        candidates = []
        candidate_names = []
        if revised:
            candidates.append(self._exploration_policy_fn(new_policy))
            candidate_names.append(new_policy.name)

        # Also try a slightly different beta (cross-cycle knob, not in-episode)
        alt_beta = sweep.selected_default_beta
        if abs(alt_beta - self.portfolio_policy.beta) > 0.05:
            alt = PortfolioPolicy(
                beta=alt_beta,
                max_workers=max(1, grid_plan.branch_count),
                name=f"beta_{alt_beta:.2f}",
            )
            candidates.append(build_policy_fn(alt))
            candidate_names.append(alt.name)

        # Explicit fixed-exploration baseline candidate (D7)
        from arsi.meta.eval_loop import fixed_exploration_fn
        candidates.append(fixed_exploration_fn(max_workers=max(1, grid_plan.branch_count)))
        candidate_names.append("fixed_baseline")

        if candidates:
            selection = self.world_pool.select_best_policy(
                current_fn,
                candidates,
                current_name=self.portfolio_policy.name,
                candidate_names=candidate_names,
            )
        else:
            selection = {
                "best_name": self.portfolio_policy.name,
                "best_fn": current_fn,
                "best_eval": current_eval,
                "monotone_ok": True,
                "all": [{"name": self.portfolio_policy.name, "avg_score": current_eval.get("avg_score", 0.0)}],
            }

        # Redeploy — freeze β when sweep is degenerate (no information)
        deployed = selection["best_name"]
        deployed_score = selection["best_eval"].get("avg_score", 0.0)
        sweep_reason = getattr(sweep, "reason", "") or ""
        degenerate = (not getattr(sweep, "non_degenerate", True)) or sweep_reason.startswith("degenerate_freeze")
        if degenerate:
            deployed = f"frozen_plateau_{self.portfolio_policy.beta:.2f}"
            deployed_score = float(current_eval.get("avg_score", 0.0) or 0.0)
            # do NOT thrash portfolio beta names
        else:
            if revised and deployed == new_policy.name:
                self.exploration_policy = new_policy
            if deployed.startswith("beta_") and not degenerate:
                try:
                    self.portfolio_policy = PortfolioPolicy(
                        beta=float(deployed.split("_")[1]),
                        max_workers=max(1, grid_plan.branch_count),
                        name=deployed,
                    )
                except ValueError:
                    pass
            elif (
                not degenerate
                and abs(sweep.selected_default_beta - self.portfolio_policy.beta) > 0.05
                and sweep_reason.startswith("plateau")
            ):
                self.portfolio_policy = PortfolioPolicy(
                    beta=sweep.selected_default_beta,
                    max_workers=max(1, grid_plan.branch_count),
                    name=f"portfolio_beta_{sweep.selected_default_beta:.2f}",
                )

        # Adaptive controller + performance
        self.adaptive_controller.record_performance(deployed_score)
        self._term_best_scores.append(deployed_score)

        gate_stats = harvest.get("quality_gate") or {}
        qstats = QualityGateStats(
            pass_=int(gate_stats.get("pass", 0)),
            warn=int(gate_stats.get("warn", 0)),
            fail=int(gate_stats.get("fail", 0)),
        )

        # Live cycle manifest (Phase D1)
        cycle_id = self.manifest_store.next_cycle_id
        manifest = self.manifest_store.next_manifest(
            kind="dream_rsi_cycle",
            planned_grid=GridSnapshot(
                branch_count=grid_plan.branch_count,
                refine_count=grid_plan.refine_count,
                reason=grid_plan.reason,
                selected_operators=[
                    op.dimension.value
                    for op in self.operator_scheduler.schedule(
                        budget=float(grid_plan.branch_count),
                        max_operators=grid_plan.branch_count,
                    )
                ],
            ),
            effective_grid=GridSnapshot(
                branch_count=self.portfolio_policy.max_workers,
                refine_count=max(1, int(current_eval.get("avg_probes", grid_plan.refine_count) or grid_plan.refine_count)),
                reason="from_replay",
                opened_width=self.portfolio_policy.max_workers,
                opened_depth=grid_plan.refine_count,
            ),
            probe_work=int(current_eval.get("avg_probes", 0) or 0),
            decision_rounds=int(self.world_pool.size),
            best_score=float(deployed_score),
            avg_score=float(current_eval.get("avg_score", 0.0)),
            beta=self.portfolio_policy.beta,
            deployed_policy=deployed,
            pool_size_after=self.world_pool.size,
            harvested_world_id=str(harvest.get("world_id", "")),
            quality_gate=qstats,
            agents=list(harvest.get("agents", [])),
            notes=f"sweep={sweep.reason}; revised={revised}",
        )
        path = self.manifest_store.append(manifest)
        sweep_path = self.manifest_store.save_beta_sweep(manifest.cycle_id, sweep.to_dict())

        self._live_cycle_history.append(manifest.beta_history_row())
        self._dream_rsi_cycles += 1
        self.operator_scheduler.mark_capability_change()

        # Phase D7: fixed vs dream compare + live-regression auto-rollback
        eval_loop = run_eval_loop(self, params=self.dream_rsi_params)
        self._last_eval_loop = eval_loop.to_dict() if hasattr(eval_loop, "to_dict") else dict(eval_loop)
        if self.iwm is not None:
            try:
                self.iwm.observe_eval_loop(eval_loop)
            except Exception as e:
                logger.warning(f"IWM observe_eval_loop failed: {e}")
        self._last_eval_loop = eval_loop.to_dict()

        return {
            "ran": True,
            "harvest": harvest,
            "pool_size": self.world_pool.size,
            "current_score": current_eval.get("avg_score", 0.0),
            "deployed": deployed,
            "deployed_score": deployed_score,
            "revised": revised,
            "monotone_ok": selection.get("monotone_ok", True),
            "all_candidates": selection.get("all", []),
            "beta": self.portfolio_policy.beta,
            "grid_plan": grid_plan.to_dict(),
            "beta_sweep": sweep.to_dict(),
            "manifest_cycle": manifest.cycle_id,
            "manifest_path": str(path),
            "beta_sweep_path": str(sweep_path),
            "eval_loop": self._last_eval_loop,
            "params": self.dream_rsi_params.to_dict(),
            "feedback_excerpt": feedback[:200],
        }

    def develop_exploration_policy(self, num_revisions: int = 3) -> dict:
        """Develop improved exploration policy via Dream-RSI loop.

        Multi-world: dreams across WorldPool when available; falls back
        to single discovery tree for backward compatibility.
        """
        if self.world_pool.size == 0:
            self.harvest_term_tree()

        if self.world_pool.size > 0:
            return self.dream_rsi_cycle(num_revisions=num_revisions)

        if not self.discovery_tree.nodes:
            self.build_discovery_tree()

        best_policy = self.policy_developer.develop(
            self.exploration_policy,
            self.discovery_tree,
            num_revisions=num_revisions,
        )

        if best_policy.best_score > self.exploration_policy.best_score:
            self.exploration_policy = best_policy
            logger.info(f"Policy improved: {best_policy.best_score:.4f}")

        return {
            "policy_name": best_policy.name,
            "revisions": num_revisions,
            "best_score": best_policy.best_score,
            "avg_score": best_policy.avg_score,
            "policy_stats": best_policy.stats,
            "tree_stats": self.discovery_tree.get_stats(),
        }

    def replay_policy(self, policy_name: str = None) -> dict:
        """Replay portfolio/exploration policy through world pool (or single tree)."""
        if self.world_pool.size > 0:
            return self.world_pool.evaluate_policy_across_pool(
                build_policy_fn(self.portfolio_policy),
                policy_name=policy_name or self.portfolio_policy.name,
            )

        if not self.discovery_tree.nodes:
            self.build_discovery_tree()

        result = self.discovery_tree.replay_alternative(
            self.exploration_policy.select,
            max_rounds=10,
        )
        return result

    def _pool_feedback_for_llm(self, eval_result: dict) -> str:
        """Structured replay feedback for policy-development agent.

        Dream-RSI §5.1: history must be structured simulator feedback,
        NOT high-level directional semantic guidance injected as advice.
        """
        per = eval_result.get("per_world", [])
        avg_score = eval_result.get("avg_score", 0.0)
        avg_quality = eval_result.get("avg_quality", 0.0)
        avg_probes = eval_result.get("avg_probes", 0.0)
        lines = [
            "回放模拟器反馈（结构化数据，不是方向建议）：",
            f"- 世界数: {eval_result.get('world_count', 0)}",
            f"- 平均回放分: {avg_score:.4f}",
            f"- 平均最佳发现: {avg_quality:.4f}",
            f"- 平均探针数: {avg_probes:.2f}",
            "各世界：",
        ]
        for w in per[:8]:
            lines.append(
                f"  {w['world_id']}: score={w['score']}, quality={w['quality']}, "
                f"probes={w['probes']}, rounds={w['rounds']}, "
                f"anchors={w['anchors']}, repairables={w['repairables']}"
            )
        lines.append(
            "改进约束：保持 select_nodes(eligible, observed, tree_stats) 签名；"
            "不要输出方向性建议文本，只输出可执行策略代码。"
            "优先：在质量不降时降低探针数；提高有效并行；对可修复失败保留机会。"
        )
        return "\n".join(lines)

    @staticmethod
    def _exploration_policy_fn(policy: ExplorationPolicy):
        def _fn(observed, legal, max_parallelism=3):
            stats = {"legal": legal, "observed": list(observed.keys()) if isinstance(observed, dict) else list(observed)}
            selected = policy.select(legal if isinstance(legal, list) else list(legal), set(observed) if not isinstance(observed, dict) else set(observed.keys()), stats)
            return selected[:max_parallelism]
        return _fn

    def close(self) -> None:
        """Clean shutdown."""
        self.store.close()
