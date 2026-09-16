"""Affect-state foundation for Current's next recommendation pipeline.

This module deliberately lives beside the legacy ``mood_id`` pipeline first.
It separates four concepts that were previously compressed into one label:

1. continuous affect dimensions (valence, arousal, agency, fatigue),
2. a non-exclusive emotion distribution,
3. the regulation direction the user appears to want, and
4. a compatibility mapping back to the existing recommender.

The deterministic functions here are a testable fallback and a migration aid,
not a clinical classifier.  Explicit user requests always outrank inferred
state.  Physiology may adjust arousal confidence, but it never names an emotion
or changes valence by itself.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator


class RegulationGoal(str, Enum):
    downshift = "downshift"
    ground = "ground"
    activate = "activate"
    move_discharge = "move_discharge"
    connect = "connect"
    broaden = "broaden"
    focus = "focus"
    savor = "savor"


class AffectConfidence(BaseModel):
    valence: float = Field(default=0.0, ge=0.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    agency: float = Field(default=0.0, ge=0.0, le=1.0)
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0)


class AffectState(BaseModel):
    """Continuous state estimate.

    ``None`` means unknown.  Unknown is intentionally different from neutral:
    a missing signal must not silently become low mood or tiredness.
    """

    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)
    agency: float | None = Field(default=None, ge=-1.0, le=1.0)
    fatigue: float | None = Field(default=None, ge=0.0, le=1.0)
    emotion_probs: dict[str, float] = Field(default_factory=dict)
    confidence: AffectConfidence = Field(default_factory=AffectConfidence)
    source: Literal["rules", "model", "fused", "unknown"] = "unknown"
    evidence_codes: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("emotion_probs")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for key, probability in value.items():
            if not key or not isinstance(probability, (int, float)):
                continue
            cleaned[str(key)] = max(0.0, min(1.0, float(probability)))
        total = sum(cleaned.values())
        if total > 1.000001:
            cleaned = {key: round(probability / total, 6) for key, probability in cleaned.items()}
        return cleaned


class AffectAssessment(BaseModel):
    state: AffectState
    primary_goal: RegulationGoal | None = None
    secondary_goal: RegulationGoal | None = None
    goal_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_reason: str | None = Field(default=None, max_length=80)
    # Temporary bridge while the legacy recommender still consumes mood_id.
    legacy_mood_id: str = "okay"


class PhysiologySignal(BaseModel):
    """Relative, quality-gated wearable hints.

    Values are relative to this user's own baseline.  Absolute HRV is not used
    to compare people.  ``recent_activity=exercise`` makes the signal ineligible
    for affect fusion because movement can dominate the reading.
    """

    hrv_z: float | None = Field(default=None, ge=-4.0, le=4.0)
    heart_rate_z: float | None = Field(default=None, ge=-4.0, le=4.0)
    recent_activity: Literal["rest", "light", "exercise", "unknown"] = "unknown"
    sample_age_minutes: int = Field(default=0, ge=0, le=1440)
    signal_quality: float = Field(default=0.0, ge=0.0, le=1.0)


# Product heuristics, not diagnostic definitions.  They exist so fallback
# behavior is visible and regression-testable when the LLM is absent.
_EMOTION_PROTOTYPES: dict[str, tuple[float, float, float, float]] = {
    "joyful": (0.82, 0.35, 0.68, 0.15),
    "excited": (0.86, 0.84, 0.72, 0.12),
    "calm": (0.62, -0.48, 0.62, 0.18),
    "anxious": (-0.64, 0.82, -0.56, 0.30),
    "angry": (-0.76, 0.86, 0.30, 0.28),
    "sad": (-0.78, -0.38, -0.48, 0.48),
    "lonely": (-0.58, -0.08, -0.38, 0.36),
    "bored": (-0.34, -0.56, -0.16, 0.30),
    "tired": (-0.18, -0.52, -0.28, 0.90),
}

_EMOTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "excited": (
        "特别兴奋", "很兴奋", "太兴奋", "激动", "开心得不想回家", "兴奋得睡不着",
        "拿到offer", "拿到 offer", "项目终于过了", "终于通过", "好消息", "庆祝",
        "thrilled", "excited",
    ),
    "joyful": (
        "特别开心", "很开心", "太开心", "高兴", "快乐", "心情很好", "状态特别好",
        "状态很好", "很满足", "挺开心", "开心", "happy", "joyful",
    ),
    "calm": ("很平静", "平静", "安宁", "松弛", "很稳", "惬意", "calm"),
    "anxious": (
        "焦虑", "紧张", "心慌", "慌", "发紧", "绷着", "喘不过", "坐不住",
        "脑子一直转", "脑子停不下来", "一直担心", "担心", "害怕", "anxious", "nervous",
    ),
    "angry": (
        "很生气", "生气", "愤怒", "火大", "气死", "一肚子火", "憋着一股火",
        "吵完架", "想发泄", "气到了", "气到", "angry", "furious",
    ),
    "sad": (
        "很难过", "难过", "低落", "伤心", "委屈", "想哭", "心情很沉",
        "提不起劲", "不开心", "sad", "down",
    ),
    "lonely": (
        "很孤单", "孤单", "孤独", "一个人难受", "一个人有点难受", "想待在有人的地方", "有人的地方", "空落落", "想有人在", "陪我",
        "lonely",
    ),
    "bored": ("很无聊", "无聊", "麻木", "没意思", "脑子很空", "待腻", "bored"),
    "tired": (
        "很累", "好累", "累坏了", "精疲力尽", "疲惫", "很困", "困", "没睡",
        "没力气", "没什么力气", "累死了", "exhausted", "tired",
    ),
}

_NEGATE_TIRED = ("不是累", "我不累", "不累", "不是疲惫", "精神很好", "有的是力气")
_HIGH_ENERGY = ("很有劲", "有力气", "想跳舞", "想庆祝", "不想回家", "想动", "出去嗨")
_NOT_DOWNSHIFT = (
    "不想放松", "不想安静", "不要安静", "不想静下来", "不想慢下来", "想热闹",
    "热闹一点", "热闹点",
)

_GOAL_PATTERNS: dict[RegulationGoal, tuple[str, ...]] = {
    RegulationGoal.connect: (
        "想找人", "想有人", "有人在", "想聊天", "找朋友", "一起庆祝", "一起", "热闹",
        "生活气", "在人群边上", "周围有人", "有人的地方", "社交",
    ),
    RegulationGoal.move_discharge: (
        "出去走走", "走一走", "走走", "散步", "跑一跑", "跑步", "骑车", "运动",
        "打球", "打一场球", "篮球", "羽毛球", "网球", "动一动", "发泄", "跳舞",
    ),
    RegulationGoal.savor: (
        "看夕阳", "吹风", "慢慢享受", "延续", "保持这种状态", "把开心留住", "庆祝",
        "不想回家", "慢慢逛", "享受一下",
    ),
    RegulationGoal.broaden: (
        "新鲜", "没见过", "换个地方", "看展", "逛逛", "逛一逛", "找灵感", "手工",
        "陶艺", "没看过",
    ),
    RegulationGoal.focus: (
        "专注", "写东西", "工作", "学习", "创作", "做点事", "换个地方工作", "看书",
    ),
    RegulationGoal.downshift: (
        "安静", "放松", "慢下来", "缓一缓", "休息", "躺一会", "坐着休息", "能坐着", "坐一会", "不想动",
    ),
    RegulationGoal.ground: (
        "喘口气", "透气", "稳下来", "别慌", "安全一点", "可控", "缓口气",
    ),
    RegulationGoal.activate: (
        "晒晒太阳", "晒太阳", "动起来", "恢复精神", "别闷着", "提提精神", "出去一下",
    ),
}


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern.casefold() in text for pattern in patterns)


def _count_matches(text: str, patterns: Sequence[str]) -> int:
    return sum(1 for pattern in patterns if pattern.casefold() in text)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _weighted_text(current: str, context: Sequence[str]) -> tuple[str, str]:
    current_folded = current.casefold().strip()
    recent = " ".join(part.casefold().strip() for part in context[-3:] if part and part.strip())
    return current_folded, recent


def _emotion_scores(current: str, context: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    tired_negated = _contains_any(current, _NEGATE_TIRED)
    for emotion, patterns in _EMOTION_PATTERNS.items():
        current_hits = _count_matches(current, patterns)
        context_hits = _count_matches(context, patterns)
        score = current_hits + context_hits * 0.45
        if emotion == "tired" and tired_negated:
            score = 0.0
        if score > 0:
            scores[emotion] = score
    return scores


def _state_from_scores(current: str, scores: Mapping[str, float]) -> AffectState:
    if not scores:
        return AffectState(source="unknown")

    total = sum(scores.values())
    dimensions = [0.0, 0.0, 0.0, 0.0]
    for emotion, score in scores.items():
        prototype = _EMOTION_PROTOTYPES[emotion]
        for index, value in enumerate(prototype):
            dimensions[index] += value * score
    dimensions = [value / total for value in dimensions]

    if "tired" in scores:
        dimensions[3] = max(dimensions[3], 0.82)
    if _contains_any(current, _HIGH_ENERGY) and "tired" not in scores:
        dimensions[1] = max(dimensions[1], 0.65)
        dimensions[3] = min(dimensions[3], 0.18)

    probabilities = {key: round(value / total, 6) for key, value in scores.items()}
    signal_strength = min(1.0, 0.40 + total * 0.12)
    fatigue_strength = min(1.0, 0.35 + (scores.get("tired", 0.0) * 0.25))
    if _contains_any(current, _HIGH_ENERGY):
        fatigue_strength = max(fatigue_strength, 0.72)

    return AffectState(
        valence=round(_clamp(dimensions[0], -1.0, 1.0), 4),
        arousal=round(_clamp(dimensions[1], -1.0, 1.0), 4),
        agency=round(_clamp(dimensions[2], -1.0, 1.0), 4),
        fatigue=round(_clamp(dimensions[3], 0.0, 1.0), 4),
        emotion_probs=probabilities,
        confidence=AffectConfidence(
            valence=signal_strength,
            arousal=signal_strength,
            agency=max(0.25, signal_strength - 0.12),
            fatigue=fatigue_strength,
        ),
        source="rules",
        evidence_codes=sorted(scores, key=scores.get, reverse=True)[:6],
    )


def _goal_scores(current: str, context: str, state: AffectState) -> tuple[dict[RegulationGoal, float], bool]:
    scores: dict[RegulationGoal, float] = {goal: 0.0 for goal in RegulationGoal}
    explicit = False
    for goal, patterns in _GOAL_PATTERNS.items():
        current_hits = _count_matches(current, patterns)
        context_hits = _count_matches(context, patterns)
        if current_hits:
            explicit = True
        scores[goal] += current_hits * 4.0 + context_hits * 0.75

    downshift_blocked = _contains_any(current, _NOT_DOWNSHIFT)
    if downshift_blocked:
        scores[RegulationGoal.downshift] = -100.0

    probs = state.emotion_probs
    valence = state.valence
    arousal = state.arousal
    agency = state.agency
    fatigue = state.fatigue

    if probs.get("lonely", 0.0) >= 0.25:
        scores[RegulationGoal.connect] += 2.6
    if probs.get("bored", 0.0) >= 0.25:
        scores[RegulationGoal.broaden] += 2.4
    if probs.get("angry", 0.0) >= 0.25:
        scores[RegulationGoal.move_discharge] += 2.5
    if probs.get("anxious", 0.0) >= 0.25:
        scores[RegulationGoal.ground] += 2.8
        if not downshift_blocked:
            scores[RegulationGoal.downshift] += 1.1
    if probs.get("tired", 0.0) >= 0.25 and not downshift_blocked:
        scores[RegulationGoal.downshift] += 2.4
    if probs.get("sad", 0.0) >= 0.25:
        scores[RegulationGoal.activate] += 1.8

    if valence is not None and valence >= 0.35:
        scores[RegulationGoal.savor] += 2.8
        if arousal is not None and arousal >= 0.35:
            scores[RegulationGoal.connect] += 0.8
            scores[RegulationGoal.broaden] += 0.6
    if (
        valence is not None and valence <= -0.35
        and arousal is not None and arousal >= 0.35
        and agency is not None and agency <= 0.0
    ):
        scores[RegulationGoal.ground] += 1.8
    if (
        valence is not None and valence <= -0.25
        and arousal is not None and arousal <= -0.20
        and (fatigue or 0.0) < 0.75
    ):
        scores[RegulationGoal.activate] += 1.2

    return scores, explicit


def _legacy_mood(state: AffectState) -> str:
    """Compatibility mapping only; never use this as the new source of truth."""
    if not state.emotion_probs and state.valence is None:
        return "okay"
    dominant = max(state.emotion_probs, key=state.emotion_probs.get, default="")
    if dominant in {"joyful", "excited", "calm"} and (state.valence or 0.0) >= 0.25:
        return "bright"
    if dominant == "tired" and (state.fatigue or 0.0) >= 0.70:
        return "tired"
    if dominant == "anxious":
        return "tight"
    if dominant == "angry":
        return "noisy"
    if dominant == "sad":
        return "low"
    if dominant == "lonely":
        return "near"
    if dominant == "bored":
        return "fresh"
    return "okay"


def assess_affect(
    text: str,
    *,
    context: Sequence[str] = (),
) -> AffectAssessment:
    """Return a deterministic, privacy-safe fallback assessment.

    Context is optional and receives less weight than the current utterance.
    This lets the same sentence mean different things after different stories
    without allowing stale context to override what the user says now.
    """

    current, recent_context = _weighted_text(text, context)
    state = _state_from_scores(current, _emotion_scores(current, recent_context))
    scores, has_explicit_goal = _goal_scores(current, recent_context, state)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    positive = [(goal, score) for goal, score in ordered if score > 0.0]

    if not positive:
        return AffectAssessment(
            state=state,
            needs_clarification=True,
            clarification_reason="no_direction_signal",
            legacy_mood_id=_legacy_mood(state),
        )

    primary, top_score = positive[0]
    secondary = positive[1][0] if len(positive) > 1 and positive[1][1] >= max(1.5, top_score - 2.0) else None
    needs_clarification = False
    clarification_reason = None

    # Positive arousal can reasonably mean celebrating with people, exploring,
    # or savoring alone.  Ask only when the user has not already chosen.
    if (
        not has_explicit_goal
        and (state.valence or 0.0) >= 0.35
        and (state.arousal or 0.0) >= 0.35
        and secondary is not None
    ):
        needs_clarification = True
        clarification_reason = "positive_direction_ambiguous"

    confidence = min(1.0, 0.30 + max(0.0, top_score) / 8.0)
    return AffectAssessment(
        state=state,
        primary_goal=primary,
        secondary_goal=secondary,
        goal_confidence=round(confidence, 4),
        needs_clarification=needs_clarification,
        clarification_reason=clarification_reason,
        legacy_mood_id=_legacy_mood(state),
    )


def fuse_physiology(assessment: AffectAssessment, signal: PhysiologySignal) -> AffectAssessment:
    """Blend a fresh, resting physiology hint into arousal only.

    The function intentionally leaves valence and ``emotion_probs`` untouched.
    Wearable activation cannot tell excitement from anxiety without context.
    """

    if (
        signal.signal_quality < 0.50
        or signal.sample_age_minutes > 60
        or signal.recent_activity == "exercise"
        or (signal.hrv_z is None and signal.heart_rate_z is None)
    ):
        return assessment

    hints: list[float] = []
    if signal.hrv_z is not None:
        hints.append(_clamp(-signal.hrv_z / 2.5, -1.0, 1.0))
    if signal.heart_rate_z is not None:
        hints.append(_clamp(signal.heart_rate_z / 2.5, -1.0, 1.0))
    physiology_arousal = sum(hints) / len(hints)

    current = assessment.state.arousal
    blended = physiology_arousal if current is None else current * 0.72 + physiology_arousal * 0.28
    confidence = assessment.state.confidence.model_copy(
        update={"arousal": max(assessment.state.confidence.arousal, signal.signal_quality * 0.85)}
    )
    evidence = list(dict.fromkeys(assessment.state.evidence_codes + ["physiology_arousal_hint"]))
    state = assessment.state.model_copy(
        update={
            "arousal": round(_clamp(blended, -1.0, 1.0), 4),
            "confidence": confidence,
            "source": "fused",
            "evidence_codes": evidence[:12],
        }
    )
    return assessment.model_copy(update={"state": state})


_SAFE_LEGACY_FIELDS = (
    "mood_id", "need_keys", "place_types", "avoid_tags", "energy", "social_mode",
    "budget_level", "environment", "confidence", "risk_level",
)


def _safe_snapshot(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        raw = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {field: getattr(value, field) for field in _SAFE_LEGACY_FIELDS if hasattr(value, field)}
    return {field: raw[field] for field in _SAFE_LEGACY_FIELDS if field in raw}


def build_interpretation_trace(
    text: str,
    *,
    rule_state: Any = None,
    model_state: Any = None,
    final_assessment: AffectAssessment | None = None,
    source: str | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build a diagnostic event without storing the user's words."""

    return {
        "event": "affect_interpretation",
        "input_digest": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "input_length": len(text),
        "source": source,
        "fallback_reason": fallback_reason,
        "rule_state": _safe_snapshot(rule_state),
        "model_state": _safe_snapshot(model_state),
        "affect": final_assessment.model_dump(mode="json") if final_assessment else None,
    }
