from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend_app.affect import (
    PhysiologySignal,
    RegulationGoal,
    assess_affect,
    build_interpretation_trace,
    fuse_physiology,
)


CASES = json.loads((Path(__file__).parent / "fixtures" / "affect_regression_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_affect_regression_corpus(case):
    result = assess_affect(case["text"])
    primary = result.primary_goal.value if result.primary_goal else None

    allowed = case.get("allowed_primary_goals", [])
    if allowed:
        assert primary in allowed
    else:
        assert primary is None

    assert primary not in case.get("forbidden_goals", [])
    assert result.legacy_mood_id not in case.get("forbidden_legacy_moods", [])

    if case.get("valence") == "positive":
        assert result.state.valence is not None and result.state.valence > 0
    elif case.get("valence") == "negative":
        assert result.state.valence is not None and result.state.valence < 0

    if case.get("arousal") == "positive":
        assert result.state.arousal is not None and result.state.arousal > 0
    elif case.get("arousal") == "negative":
        assert result.state.arousal is not None and result.state.arousal < 0

    if case.get("fatigue") == "high":
        assert result.state.fatigue is not None and result.state.fatigue >= 0.70

    if case.get("expect_unknown"):
        assert result.state.valence is None
        assert result.state.arousal is None
        assert result.state.fatigue is None

    if "expect_clarification" in case:
        assert result.needs_clarification is case["expect_clarification"]


def test_positive_cases_do_not_collapse_to_relaxation_or_tiredness():
    positive = [case for case in CASES if case["category"] == "positive"]
    results = [assess_affect(case["text"]) for case in positive]
    assert all(result.primary_goal is not RegulationGoal.downshift for result in results)
    assert all(result.legacy_mood_id not in {"tired", "low"} for result in results)


def test_unknown_is_not_silently_low_or_tired():
    result = assess_affect("我想吃烤串")
    assert result.state.valence is None
    assert result.legacy_mood_id == "okay"
    assert result.needs_clarification is True


def test_same_sentence_changes_with_context():
    tired = assess_affect("忙完了", context=["连续三天没睡，已经很困了"])
    celebratory = assess_affect("忙完了", context=["项目终于通过了，准备和朋友庆祝"])

    assert tired.state.fatigue is not None and tired.state.fatigue >= 0.70
    assert celebratory.state.valence is not None and celebratory.state.valence > 0
    assert celebratory.legacy_mood_id == "bright"


def test_physiology_adjusts_arousal_but_never_names_the_emotion():
    assessment = assess_affect("刚拿到 offer，我特别开心")
    before_valence = assessment.state.valence
    before_probs = dict(assessment.state.emotion_probs)

    fused = fuse_physiology(
        assessment,
        PhysiologySignal(
            hrv_z=-1.2,
            heart_rate_z=0.9,
            recent_activity="rest",
            sample_age_minutes=8,
            signal_quality=0.82,
        ),
    )

    assert fused.state.source == "fused"
    assert fused.state.valence == before_valence
    assert fused.state.emotion_probs == before_probs
    assert "physiology_arousal_hint" in fused.state.evidence_codes


def test_exercise_contamination_is_not_fused():
    assessment = assess_affect("我现在有点紧张")
    fused = fuse_physiology(
        assessment,
        PhysiologySignal(
            hrv_z=-2.0,
            heart_rate_z=2.0,
            recent_activity="exercise",
            sample_age_minutes=5,
            signal_quality=0.95,
        ),
    )
    assert fused == assessment


def test_trace_contains_digest_but_not_raw_text():
    text = "刚吵完架，一肚子火，想出去走走"
    assessment = assess_affect(text)
    trace = build_interpretation_trace(text, final_assessment=assessment, source="rules")
    rendered = json.dumps(trace, ensure_ascii=False)

    assert trace["input_length"] == len(text)
    assert len(trace["input_digest"]) == 16
    assert text not in rendered
    assert "一肚子火" not in rendered
