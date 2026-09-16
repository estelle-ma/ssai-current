# Affect × Space V2 — foundation branch

This branch starts the migration away from a single, overloaded `mood_id` without changing production behavior yet.

## Why it is isolated first

The current enum mixes emotions (`low`, `bright`), body states (`tired`, `tight`), needs (`quiet`, `near`) and action directions (`fresh`, `spark`). A one-label output therefore collapses different situations into the safest generic story: “tired, needs to relax.”

The foundation module separates:

- `valence`, `arousal`, `agency`, `fatigue`;
- non-exclusive emotion probabilities;
- the user's regulation direction (`ground`, `connect`, `savor`, etc.);
- a temporary compatibility mapping back to legacy `mood_id`.

## Invariants

1. Unknown is represented as unknown, not silently converted to `low` or `tired`.
2. Explicit user direction outranks inferred emotion.
3. Positive affect has a `savor` path and is not defaulted to relaxation.
4. Physiology may refine arousal confidence, but never names an emotion or changes valence alone.
5. Diagnostic traces contain digests and structured fields, never the raw utterance.

## Fixed corpus

`tests/fixtures/affect_regression_cases.json` currently covers anxiety, anger, low energy, loneliness, positive affect, novelty/focus, unknown inputs, explicit fatigue and mixed states.

Run:

```bash
pytest -q tests/test_affect_foundation.py
python -m scripts.evaluate_affect_cases
```

## Next integration steps

1. Add an optional `affect` payload to `InterpretResponse` while retaining `NeedState.mood_id`.
2. Populate it in `interpretation.py` for both model and rules paths.
3. Log privacy-safe rule/model/final snapshots in development diagnostics.
4. Shadow-run the new assessment beside the legacy recommender.
5. Introduce regulation-goal → environmental-affordance scoring only after interpretation regression tests are stable.
