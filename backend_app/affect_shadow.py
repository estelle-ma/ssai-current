"""Shadow comparison layer for the affect-state migration.

This module does not change Current's production interpretation or ranking.
It runs the additive affect assessment beside the legacy ``mood_id`` result and
emits a privacy-safe comparison event. The output is for regression analysis
and development diagnostics, not for user-facing diagnosis.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from .affect import (
    AffectAssessment,
    PhysiologySignal,
    RegulationGoal,
    assess_affect,
    build_interpretation_trace,
    fuse_physiology,
)


SHADOW_VERSION = "affect-shadow-v0.1"


class ShadowStatus(str, Enum):
    exact = "exact"
    compatible = "compatible"
    material_disagreement = "material_disagreement"
    insufficient_signal = "insufficient_signal"


class LegacyStateSnapshot(BaseModel):
    mood_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None


class AffectShadowResult(BaseModel):
    version: str = SHADOW_VERSION
    status: ShadowStatus
    reasons: list[str] = Field(default_factory=list, max_length=12)
    legacy: LegacyStateSnapshot
    assessment: AffectAssessment
    trace: dict[str, Any]


_COMPATIBLE_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"bright", "okay"}),
    frozenset({"tight", "noisy"}),
    frozenset({"low", "empty"}),
    frozenset({"near", "empty"}),
    frozenset({"spark", "fresh"}),
    frozenset({"tired", "quiet"}),
)


def _read_field(value: Any, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, BaseModel):
        return getattr(value, field, default)
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def snapshot_legacy_state(value: Any, *, source: str | None = None) -> LegacyStateSnapshot:
    """Normalize a NeedState, InterpretResponse, mapping, or compatible object."""

    candidate = value
    if _read_field(candidate, "mood_id") is None:
        nested = _read_field(candidate, "state")
        if nested is not None:
            candidate = nested

    confidence = _read_field(candidate, "confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    mood_id = _read_field(candidate, "mood_id")
    resolved_source = source or _read_field(value, "source")
    return LegacyStateSnapshot(
        mood_id=str(mood_id) if mood_id else None,
        confidence=confidence,
        source=str(resolved_source) if resolved_source else None,
    )


def _same_family(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return any(left in family and right in family for family in _COMPATIBLE_FAMILIES)


def compare_legacy_to_affect(
    legacy: LegacyStateSnapshot,
    assessment: AffectAssessment,
) -> tuple[ShadowStatus, list[str]]:
    """Compare outputs without treating the new fallback as ground truth.

    A material disagreement means the two pipelines imply meaningfully
    different recommendation directions. It is a debugging signal, not a
    claim that the affect result is clinically correct.
    """

    reasons: list[str] = []
    legacy_mood = legacy.mood_id
    affect_mood = assessment.legacy_mood_id
    state = assessment.state
    goal = assessment.primary_goal

    no_direction_or_affect_signal = (
        state.valence is None
        and state.arousal is None
        and state.fatigue is None
        and not state.emotion_probs
        and goal is None
    )
    if no_direction_or_affect_signal:
        if legacy_mood in {"low", "tired"}:
            reasons.append("legacy_assumed_negative_state_without_signal")
        return ShadowStatus.insufficient_signal, reasons

    if legacy_mood == affect_mood:
        status = ShadowStatus.exact
    elif _same_family(legacy_mood, affect_mood):
        status = ShadowStatus.compatible
        reasons.append("same_broad_affect_family")
    else:
        status = ShadowStatus.material_disagreement
        reasons.append("legacy_and_affect_mood_diverge")

    positive = state.valence is not None and state.valence >= 0.25
    low_fatigue = state.fatigue is None or state.fatigue < 0.55
    outward_goal = goal in {
        RegulationGoal.activate,
        RegulationGoal.move_discharge,
        RegulationGoal.connect,
        RegulationGoal.broaden,
        RegulationGoal.focus,
        RegulationGoal.savor,
    }

    if positive and legacy_mood in {"low", "tired"}:
        status = ShadowStatus.material_disagreement
        reasons.append("positive_state_mapped_to_low_or_tired")

    if legacy_mood == "tired" and low_fatigue:
        status = ShadowStatus.material_disagreement
        reasons.append("tired_without_fatigue_support")

    if legacy_mood in {"tired", "quiet"} and outward_goal:
        status = ShadowStatus.material_disagreement
        reasons.append("rest_label_conflicts_with_outward_goal")

    if goal is RegulationGoal.savor and legacy_mood not in {"bright", "okay"}:
        status = ShadowStatus.material_disagreement
        reasons.append("savor_goal_not_reflected_by_legacy_state")

    return status, list(dict.fromkeys(reasons))[:12]


def run_affect_shadow(
    text: str,
    *,
    legacy_state: Any = None,
    rule_state: Any = None,
    model_state: Any = None,
    context: Sequence[str] = (),
    physiology: PhysiologySignal | None = None,
    legacy_source: str | None = None,
    fallback_reason: str | None = None,
) -> AffectShadowResult:
    """Run the new assessment beside the existing interpretation result.

    The returned event contains only a digest and structured fields. It never
    includes ``text`` or raw conversation context.
    """

    assessment = assess_affect(text, context=context)
    if physiology is not None:
        assessment = fuse_physiology(assessment, physiology)

    legacy = snapshot_legacy_state(legacy_state, source=legacy_source)
    status, reasons = compare_legacy_to_affect(legacy, assessment)
    trace = build_interpretation_trace(
        text,
        rule_state=rule_state,
        model_state=model_state,
        final_assessment=assessment,
        source=legacy_source,
        fallback_reason=fallback_reason,
    )
    trace.update(
        {
            "shadow_version": SHADOW_VERSION,
            "comparison_status": status.value,
            "comparison_reasons": reasons,
            "legacy_final": legacy.model_dump(mode="json"),
        }
    )
    return AffectShadowResult(
        status=status,
        reasons=reasons,
        legacy=legacy,
        assessment=assessment,
        trace=trace,
    )
