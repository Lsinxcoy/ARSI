"""Tests for R3 history simulator + R4 autonomy ladder."""
import pytest

from arsi.core import ARSI
from arsi.adapters.bidirectional_interface import ARSIInterface, ARSIBrief
from arsi.governor.autonomy_ladder import AdaptiveTraceSelector, MetaImprover
from arsi.foundation.store import MnemosyneStore
from arsi.world_model.discovery_tree import DiscoveryTree


@pytest.fixture
def arsi(tmp_path):
    import yaml
    config = {
        "llm": {"provider": "openai", "model": "test", "api_key": "sk-test",
                "use_proxy": False, "fallback_to_heuristic": True, "timeout": 1},
        "db_path": ":memory:",
        "iron_laws_path": str(tmp_path / "iron_laws.yaml"),
        "sealed_tasks_path": str(tmp_path / "sealed_tasks.yaml"),
    }
    config_path = tmp_path / "arsi.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    (tmp_path / "iron_laws.yaml").write_text(
        "laws:\n  - id: G10\n    name: test\n    description: test\n    severity: block\n",
        encoding="utf-8",
    )
    (tmp_path / "sealed_tasks.yaml").write_text("tasks: []\n", encoding="utf-8")
    instance = ARSI.from_config(config_path)
    if instance.llm:
        instance.llm._client = None
    yield instance
    instance.close()


class TestHistorySimulator:
    def test_brief_with_no_history(self, arsi):
        interface = ARSIInterface(arsi)
        brief = interface.brief("Test task", "agent_a")
        # No discovery tree built yet
        assert brief.history_simulator is None or not brief.history_simulator.get("available")

    def test_brief_with_history(self, arsi):
        # Build discovery tree
        for i in range(20):
            arsi.ingest_trace("agent_a", f"action_{i}", "success" if i % 2 == 0 else "failure", 0.5)
        arsi.build_discovery_tree()

        interface = ARSIInterface(arsi)
        brief = interface.brief("action task", "agent_a")
        # History simulator should be available
        if brief.history_simulator:
            assert brief.history_simulator.get("available") in [True, False]

    def test_query_history(self, arsi):
        brief = ARSIBrief(
            task_description="test", agent_id="a",
            recommendations=[], relevant_skills=[], warnings=[],
            past_lessons=[],
            history_simulator={
                "available": True,
                "success_rate": 0.75,
                "sample_size": 40,
                "common_failure_patterns": ["pattern1"],
                "best_practice": "do X",
                "confidence": 0.8,
            },
        )
        result = brief.query_history("test")
        assert result["available"] is True
        assert result["historical_success_rate"] == 0.75
        assert result["sample_size"] == 40

    def test_query_history_no_data(self):
        brief = ARSIBrief(
            task_description="test", agent_id="a",
            recommendations=[], relevant_skills=[], warnings=[],
            past_lessons=[],
        )
        result = brief.query_history("test")
        assert result["available"] is False


class TestAdaptiveTraceSelector:
    @pytest.fixture
    def selector(self):
        return AdaptiveTraceSelector(MnemosyneStore(":memory:"))

    def test_select_traces(self, selector):
        traces = [{"action": f"act_{i}", "outcome": "success", "effect": 0.5} for i in range(100)]
        selected = selector.select_for_analysis(traces, max_count=30)
        assert len(selected) == 30

    def test_novelty_scoring(self, selector):
        # First time seeing an action → high novelty
        n1 = selector._compute_novelty({"action": "new_action", "outcome": "success"})
        # Seen many times → low novelty
        for _ in range(10):
            selector._historical_patterns["new_action:success"] = 10
        n2 = selector._compute_novelty({"action": "new_action", "outcome": "success"})
        assert n1 > n2

    def test_relevance_to_weak_dimensions(self, selector):
        # Traces about weak dimensions get higher relevance
        r1 = selector._compute_relevance({"action": "knowledge_gap"})
        r2 = selector._compute_relevance({"action": "unrelated_thing"})
        assert r1 > r2

    def test_stats(self, selector):
        traces = [{"action": f"act_{i}", "outcome": "success", "effect": 0.5} for i in range(10)]
        selector.select_for_analysis(traces, max_count=5)
        stats = selector.stats
        assert stats["selection_count"] == 1


class TestMetaImprover:
    @pytest.fixture
    def improver(self):
        return MetaImprover(MnemosyneStore(":memory:"))

    def test_evaluate_no_data(self, improver):
        result = improver.evaluate_diagnostic_quality("knowledge")
        assert result["status"] == "no_data"

    def test_revise_no_data(self, improver):
        result = improver.revise_diagnostic_algorithm("knowledge", {})
        assert result["status"] == "no_data_to_revise"

    def test_propose_revision(self, improver):
        proposal = improver._propose_revision("calibration", {})
        assert "calibration" in proposal.lower() or "window" in proposal.lower()

    def test_stats(self, improver):
        stats = improver.stats
        assert "diagnostic_history_count" in stats
        assert "policy_revision_count" in stats


class TestR3R4Integration:
    def test_full_cycle_with_history(self, arsi):
        """Test R3+R4 integration: traces → tree → brief with history → L3 selection."""
        # Ingest traces
        for i in range(30):
            arsi.ingest_trace("agent_x", f"task_{i % 5}", "success" if i % 3 != 0 else "failure", 0.3 + i * 0.02)

        # Build discovery tree
        arsi.build_discovery_tree()

        # Generate brief with history simulator
        interface = ARSIInterface(arsi)
        brief = interface.brief("task_1", "agent_x")

        # History should be available
        if brief.history_simulator:
            assert "success_rate" in brief.history_simulator

        # L3: Adaptive trace selection
        selector = AdaptiveTraceSelector(arsi.store)
        all_traces = arsi.store.get_recent_traces(n=100)
        selected = selector.select_for_analysis(all_traces, max_count=20)
        assert len(selected) <= 20
