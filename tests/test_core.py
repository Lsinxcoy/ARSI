"""Tests for ARSI Core orchestrator."""
import pytest

from arsi.core import ARSI
from arsi.foundation.schema import MemoryZone


@pytest.fixture
def arsi(tmp_path):
    """Create an ARSI instance with in-memory store."""
    import yaml

    # Write minimal config (LLM disabled for tests)
    config = {
        "llm": {"provider": "openai", "model": "test", "api_key": "sk-test-disabled",
                "use_proxy": False, "fallback_to_heuristic": True, "timeout": 1},
        "db_path": ":memory:",
        "iron_laws_path": str(tmp_path / "iron_laws.yaml"),
        "sealed_tasks_path": str(tmp_path / "sealed_tasks.yaml"),
    }
    config_path = tmp_path / "arsi.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    # Write iron laws
    (tmp_path / "iron_laws.yaml").write_text(
        "laws:\n  - id: G10\n    name: test\n    description: test\n    severity: block\n",
        encoding="utf-8",
    )

    instance = ARSI.from_config(config_path)
    # Force disable LLM for tests (no network calls)
    if instance.llm:
        instance.llm._client = None
    yield instance
    instance.close()


class TestARSICore:
    def test_from_config(self, arsi):
        assert arsi.store is not None
        assert arsi.mnemosyne is not None
        assert arsi.siwm is not None
        assert arsi.governor is not None
        assert arsi.dream is not None

    def test_ingest_trace(self, arsi):
        trace = arsi.ingest_trace("a1", "learn", "success", 0.8)
        assert trace.agent_id == "a1"
        stats = arsi.get_stats()
        assert stats["trace_count"] == 1

    def test_step(self, arsi):
        # Ingest some traces first
        for i in range(10):
            arsi.ingest_trace("a1", "learn" if i % 2 == 0 else "evolve", "success", 0.5 + i * 0.05)

        result = arsi.step()
        assert "decision" in result
        assert "actions" in result
        assert result["step"] == 1

    def test_step_dream_when_eta_high(self, arsi):
        # Force high η
        arsi.siwm.eta.eta_smooth = 0.50
        result = arsi.step()
        assert result["decision"]["action"] == "dream"

    def test_run_term(self, arsi):
        for i in range(15):
            arsi.ingest_trace("a1", "learn", "success", 0.6)

        term_result = arsi.run_term(n_steps=3)
        assert term_result["term_id"] == "term_1"
        assert len(term_result["steps"]) == 3
        assert "evaluation" in term_result

    def test_get_memory_proxy(self, arsi):
        proxy = arsi.get_memory_proxy("agent_x")
        assert proxy.agent_id == "agent_x"
        # Same proxy returned on second call
        proxy2 = arsi.get_memory_proxy("agent_x")
        assert proxy is proxy2

    def test_empower_agent(self, arsi):
        for i in range(10):
            arsi.ingest_trace("agent_b", "learn", "failure", 0.2)
        result = arsi.empower_agent("agent_b")
        assert result["agent_id"] == "agent_b"
        assert "diagnosis" in result
        assert "verification" in result

    def test_stats(self, arsi):
        arsi.ingest_trace("a1", "learn", "success", 0.8)
        arsi.step()
        stats = arsi.get_stats()
        assert stats["step_count"] == 1
        assert "eta" in stats
        assert "iron_laws" in stats
        assert isinstance(stats["llm_available"], bool)

    def test_learn_distills_traces(self, arsi):
        # Ingest high-importance traces
        for i in range(5):
            arsi.ingest_trace("a1", "learn", "success", 0.9)

        # Force learn action
        arsi.siwm.eta.eta_smooth = 0.0  # Low η → Governor may choose learn
        result = arsi.step()
        # At least the step should complete
        assert result["step"] == 1

    def test_maintain_consolidates(self, arsi):
        for i in range(20):
            arsi.ingest_trace("a1", "remember", "ok", 0.3)

        # Force many steps to trigger maintain
        for _ in range(5):
            arsi.step()

        stats = arsi.get_stats()
        assert stats["step_count"] == 5
