"""Tests for bidirectional interface."""
import pytest

from arsi.core import ARSI
from arsi.adapters.bidirectional_interface import ARSIInterface, ARSIBrief, ARSIReport


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


@pytest.fixture
def interface(arsi):
    return ARSIInterface(arsi)


class TestBrief:
    def test_generate_brief(self, interface):
        brief = interface.brief("Fix the parser bug", "hermes")
        assert brief.agent_id == "hermes"
        assert brief.task_description == "Fix the parser bug"
        assert len(brief.brief_id) > 0

    def test_brief_format(self, interface):
        brief = interface.brief("Write a test", "mimo")
        text = brief.format_for_agent()
        assert "ARSI Brief" in text
        assert "mimo" in text

    def test_brief_recommendations(self, interface):
        brief = interface.brief("Fix the bug in code", "hermes")
        assert len(brief.recommendations) > 0

    def test_brief_skills(self, interface):
        brief = interface.brief("Debug the error", "hermes")
        assert "arsi-metacognition" in brief.relevant_skills

    def test_brief_stats(self, interface):
        interface.brief("Task 1", "agent_a")
        interface.brief("Task 2", "agent_b")
        assert interface.stats["brief_count"] == 2


class TestReport:
    def test_process_report(self, arsi, interface):
        brief = interface.brief("Fix bug", "hermes")
        report = ARSIReport(
            brief_id=brief.brief_id,
            agent_id="hermes",
            task_description="Fix bug",
            outcome="success",
            effect=0.8,
            skills_used=["arsi-metacognition"],
            recommendations_followed=["先复现问题"],
            recommendations_ignored=[],
        )
        result = interface.report(report)
        assert result["status"] == "processed"
        assert "effectiveness" in result

    def test_report_effectiveness_followed_success(self, arsi, interface):
        brief = interface.brief("Fix bug", "hermes")
        report = ARSIReport(
            brief_id=brief.brief_id,
            agent_id="hermes",
            task_description="Fix bug",
            outcome="success",
            effect=0.8,
            skills_used=["arsi-metacognition"],
            recommendations_followed=["rec1", "rec2"],
            recommendations_ignored=[],
        )
        result = interface.report(report)
        eff = result["effectiveness"]
        assert eff["follow_rate"] == 1.0
        assert eff["verdict"] == "recommendations_seemed_helpful"

    def test_report_effectiveness_ignored_failure(self, arsi, interface):
        brief = interface.brief("Fix bug", "hermes")
        report = ARSIReport(
            brief_id=brief.brief_id,
            agent_id="hermes",
            task_description="Fix bug",
            outcome="failure",
            effect=0.2,
            skills_used=[],
            recommendations_followed=[],
            recommendations_ignored=["rec1", "rec2"],
        )
        result = interface.report(report)
        eff = result["effectiveness"]
        assert eff["verdict"] == "ignoring_recommendations_hurt"

    def test_report_ingests_trace(self, arsi, interface):
        brief = interface.brief("Test task", "hermes")
        report = ARSIReport(
            brief_id=brief.brief_id,
            agent_id="hermes",
            task_description="Test task",
            outcome="success",
            effect=0.7,
            skills_used=[],
            recommendations_followed=[],
            recommendations_ignored=[],
        )
        interface.report(report)
        traces = arsi.store.get_recent_traces(n=10, agent_id="hermes")
        assert len(traces) > 0

    def test_report_stats(self, interface):
        brief = interface.brief("Task", "agent")
        report = ARSIReport(
            brief_id=brief.brief_id,
            agent_id="agent",
            task_description="Task",
            outcome="success",
            effect=0.5,
            skills_used=[],
            recommendations_followed=[],
            recommendations_ignored=[],
        )
        interface.report(report)
        assert interface.stats["report_count"] == 1


class TestProtocol:
    def test_full_brief_report_cycle(self, arsi, interface):
        """Test the full bidirectional protocol."""
        # 1. Agent asks for brief
        brief = interface.brief("Analyze Hermes session data", "mimo")
        assert brief.agent_id == "mimo"

        # 2. Agent executes task (simulated)
        # 3. Agent reports result
        report = ARSIReport(
            brief_id=brief.brief_id,
            agent_id="mimo",
            task_description="Analyze Hermes session data",
            outcome="success",
            effect=0.75,
            skills_used=["arsi-calibration"],
            recommendations_followed=["Past experience recommendation"],
            recommendations_ignored=[],
        )
        result = interface.report(report)

        # 4. Verify learning happened
        assert result["status"] == "processed"
        assert result["effectiveness"]["verdict"] == "recommendations_seemed_helpful"

        # 5. Verify trace was recorded
        traces = arsi.store.get_recent_traces(n=10, agent_id="mimo")
        assert len(traces) > 0
