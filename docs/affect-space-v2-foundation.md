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

## Shadow comparison

`backend_app/affect_shadow.py` compares the additive affect assessment with a
legacy `NeedState` or `InterpretResponse` without changing production output.
It classifies each comparison as:

- `exact`;
- `compatible`;
- `material_disagreement`;
- `insufficient_signal`.

High-value disagreement reasons include positive affect being mapped to
`low`/`tired`, a `tired` label without fatigue support, and an outward-facing
goal such as `connect` or `savor` conflicting with a rest label.

The shadow event stores only an input digest, input length and structured
snapshots. It never stores the utterance or raw conversation context.

Run:

```bash
pytest -q tests/test_affect_shadow.py
```

The runner is intentionally not wired into `interpretation.py` yet. That final
wiring should happen after parallel work on the existing interpretation path is
reconciled, so the integration commit can be reviewed separately.

## Next integration steps

1. Reconcile concurrent changes to `interpretation.py` and `schemas.py`.
2. Add an optional `affect` payload to `InterpretResponse` while retaining `NeedState.mood_id`.
3. Invoke `run_affect_shadow()` after the legacy result is finalized.
4. Log only `AffectShadowResult.trace` in development diagnostics.
5. Keep recommendation output unchanged until disagreement rates are reviewed.
6. Introduce regulation-goal → environmental-affordance scoring only after interpretation regression tests are stable.
