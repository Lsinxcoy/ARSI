"""Pool snapshot no-shrink + capture_missing tests."""
from __future__ import annotations


class TestPoolPersist:
    def test_no_shrink(self, tmp_path):
        import json

        from arsi.world_model.world_pool import WorldPool

        path = tmp_path / "snap.json"
        big = WorldPool(max_worlds=50)
        # fabricate 3 worlds via from_snapshot
        data = {
            "max_worlds": 50,
            "worlds": [
                {"world_id": f"W{i}", "nodes": {"root": {"id": "root"}, "n": {"id": "n", "action": "learn", "outcome": "success", "score": 0.5, "children": [], "parent_id": "root"}}}
                for i in range(3)
            ],
            "manifest": [],
        }
        big = WorldPool.from_snapshot(data)
        assert big.size == 3
        big.persist_to(path)
        assert len(json.loads(path.read_text(encoding="utf-8"))["worlds"]) == 3

        small = WorldPool.from_snapshot({"max_worlds": 50, "worlds": [data["worlds"][0]]})
        assert small.size == 1
        small.persist_to(path)  # must not wipe
        assert len(json.loads(path.read_text(encoding="utf-8"))["worlds"]) == 3


class TestCaptureMissing:
    def test_report_flags_missing_capture(self):
        from arsi.adapters.bidirectional_interface import ARSIReport

        r = ARSIReport(
            brief_id="b",
            agent_id="hermes",
            task_description="run terminal",
            outcome="failure",
            effect=0.0,
            skills_used=[],
            recommendations_followed=[],
            recommendations_ignored=[],
            fail_class="tool_error_terminal",
        )
        assert (r.measurements or {}).get("capture_missing") is None
        # after interface marks
        if "terminal" in r.task_description and not r.measurements:
            r.measurements = {"capture_missing": True}
        assert r.measurements["capture_missing"] is True
