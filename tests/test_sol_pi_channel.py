"""B-line host empowerment channel tests (SoL-Pi absorption)."""
from __future__ import annotations

import json
from pathlib import Path

from arsi.foundation.efficiency_gate import dual_gate, host_channel_gate
from arsi.foundation.evidence_receipt import (
    EvidenceReceipt,
    build_receipt_from_text,
    fusion_write_and_verify,
)
from arsi.foundation.observation_pack import ObservationPack, get_observation, pack_for_host


class TestEvidenceReceipt:
    def test_grounded_receipt(self, tmp_path):
        text = "Rule: always verify skill file after write.\nEvidence: Hermes skills dir exists.\n"
        rec = build_receipt_from_text(text, source_kind="unit", fields={"k": 1}, persist=True)
        assert rec.verified is True
        assert rec.quotes
        assert rec.source_hash
        assert rec.fallback_used in (True, False)
        assert rec.brief_snippet().startswith("# ARSI Evidence Receipt")

    def test_fusion_write_and_verify(self, tmp_path):
        p = tmp_path / "skill" / "SKILL.md"
        content = "# Skill\n\nEmpower: calibrate before high-risk acts.\n"
        r = fusion_write_and_verify(p, content, receipt_kind="skill")
        assert r["write_ok"] is True
        assert p.exists()
        assert r["receipt"]["verified"] is True
        assert r["receipt"]["source_hash"]

    def test_quote_not_invented(self):
        rec = EvidenceReceipt(quotes=["not in any file"], source_path="", source_hash="abc")
        rec.verify()
        # no source path → quotes ungrounded → not verified
        assert rec.verified is False


class TestObservationPack:
    def test_small_payload_unchanged(self):
        out = pack_for_host("tiny", kind="brief")
        assert out == "tiny"

    def test_large_payload_handle_roundtrip(self):
        big = "X" * 5000 + "\nEND_MARK_ARSI"
        out = pack_for_host(big, kind="brief_test")
        assert out.startswith("# ARSI Observation Handle")
        assert "handle=obs_brief_test_" in out
        # retrieve
        handle_line = [ln for ln in out.splitlines() if ln.startswith("- handle=")]
        assert handle_line
        hid = handle_line[0].split("=", 1)[1].strip()
        back = get_observation(hid)
        assert back is not None
        assert "END_MARK_ARSI" in back
        assert len(back) == len(big)

    def test_pack_threshold(self, tmp_path):
        pack = ObservationPack(threshold=50, root=tmp_path / "pk")
        small = pack.pack("abc", kind="t")
        assert small == "abc"
        big = pack.pack("y" * 80, kind="t")
        assert not isinstance(big, str)
        assert big.size >= 80


class TestEfficiencyGate:
    def test_accept_when_capability_stable_and_cost_down(self):
        r = dual_gate(0.80, 0.80, 1000, 700)
        assert r.accepted is True
        assert r.capability_ok is True
        assert r.efficiency_ok is True

    def test_reject_capability_regression(self):
        r = dual_gate(0.80, 0.60, 1000, 500)
        assert r.accepted is False
        assert "capability" in r.reason

    def test_hold_when_no_efficiency_gain(self):
        r = dual_gate(0.80, 0.81, 1000, 1000)
        assert r.accepted is False
        assert r.efficiency_ok is False

    def test_host_channel_gate(self):
        r = host_channel_gate(0.9, 0.92, 500, 300)
        assert r.accepted is True
        r2 = host_channel_gate(0.9, 0.7, 500, 200)
        assert r2.accepted is False
