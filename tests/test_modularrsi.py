"""ModularRSI contrastive sampling + module scope tests."""
from __future__ import annotations


class TestContrastive:
    def test_three_groups(self):
        from arsi.harness.taxonomy import contrastive_batches

        out = contrastive_batches(
            [
                ("t1", True), ("t1", True),
                ("t2", True), ("t2", False),
                ("t3", False), ("t3", False),
            ]
        )
        assert out["positive"] == ["t1"]
        assert out["contrastive"] == ["t2"]
        assert out["negative"] == ["t3"]


class TestModuleScope:
    def test_scope_fence(self):
        from arsi.harness.taxonomy import module_scope_ok

        assert module_scope_ok("tool_use", "src/arsi/adapters/x.py") is True
        assert module_scope_ok("tool_use", "src/arsi/sealed_eval/y.py") is False
        assert module_scope_ok("nope", "x") is False
