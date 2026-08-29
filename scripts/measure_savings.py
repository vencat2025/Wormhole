"""Report what the gateway actually did, from logged requests.

Every number here comes from rows in the inference log: which model served
each request and how many tokens it used. Nothing is asserted or assumed.

One caveat is stated wherever the output is used: the baseline is a
counterfactual. WormHole cannot know what a request would have cost had it
gone elsewhere, so it prices the *same measured token counts* against
GPT-4o rates. Token counts are reported by the provider and rates come from
litellm's maintained pricing map, so both sides are measured or sourced; what
is constructed is the comparison against a model you did not actually run.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select
from db.database import engine
from db.models import InferenceLog


def main() -> int:
    with Session(engine) as session:
        rows = session.exec(select(InferenceLog)).all()

    if not rows:
        print("No logged requests. Run some traffic through the gateway first.")
        return 1

    actual = sum(r.actual_cost or 0.0 for r in rows)
    baseline = sum(r.baseline_cost or 0.0 for r in rows)
    in_tok = sum(r.prompt_tokens or 0 for r in rows)
    out_tok = sum(r.completion_tokens or 0 for r in rows)
    # 8.5 was written as a placeholder whenever the judge call failed, so it
    # cannot be distinguished from a real 8.5 after the fact. Both counts are
    # reported rather than silently averaging fabricated values together with
    # measured ones.
    all_scores = [r.judge_score for r in rows if r.judge_score is not None]
    placeholder = [v for v in all_scores if v == 8.5]
    scored = [v for v in all_scores if v != 8.5]

    print(f"Sample size: {len(rows)} logged requests")
    print(f"Tokens: {in_tok:,} in / {out_tok:,} out")
    print()
    print(f"Measured spend on models actually used: ${actual:.4f}")
    print(f"Same tokens priced at the GPT-4o baseline: ${baseline:.4f}")
    if baseline > 0:
        print(f"Difference: ${baseline - actual:.4f} ({(baseline - actual) / baseline * 100:.1f}% lower)")
    print("  (token counts are provider-reported; rates come from litellm's")
    print("   pricing map. The baseline is a counterfactual, not an observed bill)")
    print()

    dist = Counter(r.selected_model for r in rows)
    print("Requests by model actually used:")
    for model, n in dist.most_common():
        print(f"   {model:34s} {n:>5}  ({n / len(rows) * 100:.1f}%)")

    if all_scores:
        print()
        print(f"Rows carrying a judge score: {len(all_scores)}")
        print(f"  exactly 8.5, the old failure placeholder: {len(placeholder)}")
        if scored:
            print(f"  remaining: {sum(scored) / len(scored):.2f}/10 over {len(scored)} rows")
        print("  Treat this as unreliable. Historic rows were written by a judge")
        print("  pinned to a decommissioned model, and failures stored 8.5 rather")
        print("  than nothing. Judge failures now store no score, so a clean")
        print("  measurement needs traffic recorded after that fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
