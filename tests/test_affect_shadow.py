from __future__ import annotations

import json

from backend_app.affect_shadow import (
    ShadowStatus,
    run_affect_shadow,
    snapshot_legacy_state,
)


def test_shadow_flags_positive_state_collapsed_to_tired():
    result = run_affect_shadow(
        "项目终于过了，我现在特别兴奋",
        legacy_state={"mood_id": "tired", "confidence": 0.82},
        legacy_source="model",
    )

    assert result.status is ShadowStatus.material_disagreement
    assert "positive_state_mapped_to_low_or_tired" in result.reasons
    assert "tired_without_fatigue_support" in result.reasons
    assert result.assessment.primary_goal.value == "savor"


def test_shadow_accepts_genuinely_tired_state():
    result = run_affect_shadow(
        "昨晚没睡，今天累坏了，只想坐着休息",
        legacy_state={"mood_id": "tired", "confidence": 0.8},
        legacy_source="rules",
    )

    assert result.status is ShadowStatus.exact
    assert result.assessment.state.fatigue is not None
    assert result.assessment.state.fatigue >= 0.70
    assert "tired_without_fatigue_support" not in result.reasons


def test_shadow_marks_related_high_arousal_labels_as_compatible():
    result = run_affect_shadow(
        "明天要答辩，我现在很紧张",
        legacy_state={"mood_id": "noisy", "confidence": 0.6},
        legacy_source="rules",
    )

    assert result.assessment.legacy_mood_id == "tight"
    assert result.status is ShadowStatus.compatible
    assert result.reasons == ["same_broad_affect_family"]


def test_unknown_input_does_not_validate_a_negative_legacy_default():
    result = run_affect_shadow(
        "我想吃烤串",
        legacy_state={"mood_id": "low", "confidence": 0.5},
        legacy_source="rules",
    )

    assert result.status is ShadowStatus.insufficient_signal
    assert result.assessment.state.valence is None
    assert "legacy_assumed_negative_state_without_signal" in result.reasons


def test_explicit_direction_is_compared_even_when_affect_is_unknown():
    result = run_affect_shadow(
        "想换个地方工作，最好能坐久一点",
        legacy_state={"mood_id": "low", "confidence": 0.5},
        legacy_source="rules",
    )

    assert result.assessment.primary_goal.value == "focus"
    assert result.status is ShadowStatus.material_disagreement


def test_shadow_event_contains_no_raw_text_or_context():
    text = "刚吵完架，一肚子火，想出去走走"
    context = ["同事在会上否定了我的方案"]
    result = run_affect_shadow(
        text,
        context=context,
        legacy_state={"mood_id": "tired", "confidence": 0.7},
        rule_state={"mood_id": "tired", "need_keys": ["walk"]},
        model_state={"mood_id": "noisy", "confidence": 0.65},
        legacy_source="model",
    )
    rendered = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)

    assert text not in rendered
    assert context[0] not in rendered
    assert "一肚子火" not in rendered
    assert result.trace["input_length"] == len(text)
    assert len(result.trace["input_digest"]) == 16


def test_snapshot_accepts_nested_interpret_response_shape():
    snapshot = snapshot_legacy_state(
        {
            "state": {"mood_id": "bright", "confidence": 0.74},
            "source": "model",
        }
    )

    assert snapshot.mood_id == "bright"
    assert snapshot.confidence == 0.74
    assert snapshot.source == "model"


def test_legacy_snapshot_clamps_malformed_confidence():
    high = snapshot_legacy_state({"mood_id": "bright", "confidence": 4})
    invalid = snapshot_legacy_state({"mood_id": "okay", "confidence": "unknown"})

    assert high.confidence == 1.0
    assert invalid.confidence is None
