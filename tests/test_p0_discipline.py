"""P0/P2 discipline tests — verified claims + probe script importability."""
from __future__ import annotations

from pathlib import Path

from arsi.foundation.verified import (
    VerifiedClaim,
    unverified,
    verified_from_exit_code,
    verified_from_pytest,
)


class TestVerifiedClaims:
    def test_unverified_never_true(self):
        c = unverified("demo", "no_evidence")
        assert c.verified is False
        assert c.reason == "no_evidence"
        assert c.timestamp

    def test_exit_code_zero_is_verified(self):
        c = verified_from_exit_code("x", 0, command="true")
        assert c.verified is True
        assert c.exit_code == 0

    def test_exit_code_nonzero_not_verified(self):
        c = verified_from_exit_code("x", 2)
        assert c.verified is False
        assert c.reason == "exit_2"

    def test_pytest_helper_shape(self, tmp_path):
        # empty project with no tests may still exit 5 (no tests collected) → not verified True
        (tmp_path / "tests").mkdir()
        c = verified_from_pytest("empty", tmp_path, pytest_args=["-q"], timeout=30)
        assert isinstance(c, VerifiedClaim)
        assert c.command
        assert c.timestamp
        if c.exit_code != 0:
            assert c.verified is False


class TestScripts:
    def test_iwm_probes_module_loads(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "iwm_probes.py"
        assert path.exists()
        spec = importlib.util.spec_from_file_location("iwm_probes", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_probes")
        assert hasattr(mod, "build_arsi")
