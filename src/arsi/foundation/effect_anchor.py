"""External effect anchor — SYNTHEX P0/S5 + G5 hardening.

Effects used for evolution/policy claims must come from outside the
mechanism being scored (eval_loop, sealed task, host outcome), not
from the mechanism declaring its own value.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

from arsi.foundation.verified import VerifiedClaim

logger = logging.getLogger(__name__)

EXTERNAL_SOURCES = frozenset({
    "eval_loop",
    "sealed_eval",
    "host_outcome",
    "quality_gate",
    "manifest_live",
    "behavior_trace",
    "layer1_holdout",
    "loop_trial",
})

INTERNAL_PLACEHOLDERS = frozenset({
    "placeholder_mutation",
    "self_eval",
    "internal_score",
    "",
})


@dataclass
class EffectAnchor:
    mechanism: str
    effect: float
    source: str = ""
    external: bool = False
    evidence_ref: str = ""
    claim: Optional[VerifiedClaim] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.claim is not None:
            d["claim"] = self.claim.to_dict()
        return d


def is_external_source(source: str) -> bool:
    s = (source or "").strip().lower()
    if s in INTERNAL_PLACEHOLDERS:
        return False
    if s in EXTERNAL_SOURCES:
        return True
    # prefix forms: eval_loop:..., host_outcome:...
    head = s.split(":", 1)[0]
    return head in EXTERNAL_SOURCES


def build_effect_anchor(
    mechanism: str,
    effect: float,
    source: str = "",
    evidence_ref: str = "",
    store=None,
) -> EffectAnchor:
    """Build + validate an effect claim. Prefer external sources."""
    external = is_external_source(source)
    if not external:
        claim = VerifiedClaim(
            claim=f"effect:{mechanism}",
            verified=False,
            reason=f"non_external_source:{source or 'empty'}",
            command="arsi.foundation.effect_anchor.build_effect_anchor",
            detail={"effect": effect, "source": source},
        )
        logger.warning(
            "G5 effect not externally anchored: mechanism=%s source=%s effect=%s",
            mechanism, source, effect,
        )
        return EffectAnchor(
            mechanism=mechanism,
            effect=float(effect),
            source=source or "",
            external=False,
            evidence_ref=evidence_ref,
            claim=claim,
        )

    # External: still not "verified capability" without a probe id
    verified = bool(evidence_ref) and external
    claim = VerifiedClaim(
        claim=f"effect:{mechanism}",
        verified=verified,
        reason="external_source+evidence_ref" if verified else "external_source_no_evidence_ref",
        exit_code=0 if verified else 3,
        command=f"source={source};ref={evidence_ref}",
        detail={"effect": effect, "source": source, "evidence_ref": evidence_ref},
    )
    anchor = EffectAnchor(
        mechanism=mechanism,
        effect=float(effect),
        source=source,
        external=True,
        evidence_ref=evidence_ref,
        claim=claim,
    )
    if store is not None and hasattr(store, "record_effect"):
        try:
            store.record_effect(mechanism, float(effect), context=f"{source}|{evidence_ref}")
        except Exception as e:
            logger.warning(f"effect store write failed: {e}")
    return anchor


def record_external_effect(
    store,
    mechanism: str,
    effect: float,
    source: str,
    evidence_ref: str = "",
) -> EffectAnchor:
    """Canonical write path for effects (single source, S3+S5)."""
    return build_effect_anchor(
        mechanism=mechanism,
        effect=effect,
        source=source,
        evidence_ref=evidence_ref,
        store=store,
    )
