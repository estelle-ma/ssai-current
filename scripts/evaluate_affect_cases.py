"""Print a compact report for the fixed affect regression corpus."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from backend_app.affect import assess_affect


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "fixtures" / "affect_regression_cases.json"


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        result = assess_affect(case["text"])
        primary = result.primary_goal.value if result.primary_goal else None
        allowed = case.get("allowed_primary_goals", [])
        passed = primary in allowed if allowed else primary is None
        passed = passed and result.legacy_mood_id not in case.get("forbidden_legacy_moods", [])
        rows.append((case, result, passed))

    failures = [row for row in rows if not row[2]]
    goal_counts = Counter((row[1].primary_goal.value if row[1].primary_goal else "unknown") for row in rows)
    legacy_counts = Counter(row[1].legacy_mood_id for row in rows)

    print(f"cases={len(rows)} passed={len(rows) - len(failures)} failed={len(failures)}")
    print("primary_goals:", dict(goal_counts))
    print("legacy_moods:", dict(legacy_counts))
    if failures:
        print("\nFailures:")
        for case, result, _ in failures:
            print(
                f"- {case['id']}: got goal={result.primary_goal} legacy={result.legacy_mood_id}; "
                f"allowed={case.get('allowed_primary_goals', [])}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
