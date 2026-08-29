"""Derive a measured difficulty score for real coding tasks.

Every other difficulty signal in this project is a guess: my keyword heuristic,
prompt length, a benchmark's published average applied uniformly to all its
items. This one is measured, and it is measured by other people.

SWE-bench publishes, for each of ~134 submitted systems, exactly which of the
500 Verified instances that system resolved. Those runs executed the projects'
real test suites, so "resolved" means the patch actually worked.

Pooling them gives a per-task difficulty: the fraction of systems that solved
it. An instance almost nobody resolved is genuinely hard; one most resolved is
not. That is an empirical property of the task, not an opinion about its
wording, and it is the label this router should learn from.

Data: https://github.com/SWE-bench/experiments (per-instance results)
      https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified (prompts)
Both are downloaded and cached; neither is committed to this repository.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "benchmark_cache")
OUT = os.path.join(CACHE, "swebench_difficulty.json")

GH_API = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
RAW = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
HF_ROWS = ("https://datasets-server.huggingface.co/rows"
           "?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test")


def fetch_resolution_counts(client: httpx.Client, limit: int = 200) -> Dict[str, Dict]:
    """How many submitted systems resolved each instance."""
    subs = [e["name"] for e in client.get(GH_API).json() if e["type"] == "dir"][:limit]
    print(f"  submissions found: {len(subs)}")

    solved = defaultdict(int)
    attempted = defaultdict(int)
    counted = 0
    for i, sub in enumerate(subs, 1):
        try:
            r = client.get(f"{RAW}/{sub}/results/results.json")
            if r.status_code != 200:
                continue
            d = r.json()
            resolved = set(d.get("resolved", []))
            # A submission counts as having attempted an instance if it appears
            # anywhere in that submission's results. Counting only "generated"
            # undercounts, because some submissions report a resolution without
            # listing the instance there, which drove solve rates above 100%.
            seen = resolved.copy()
            for key in ("generated", "with_logs", "applied", "no_apply",
                        "no_generation", "test_errored", "test_timeout"):
                seen.update(d.get(key, []))
            for iid in seen:
                attempted[iid] += 1
            for iid in resolved:
                solved[iid] += 1
            counted += 1
        except Exception:
            continue
        if i % 25 == 0:
            print(f"    read {i}/{len(subs)} submissions")
    print(f"  usable submissions: {counted}")
    return {"solved": dict(solved), "attempted": dict(attempted), "systems": counted}


def fetch_problem_statements(client: httpx.Client) -> Dict[str, str]:
    """instance_id -> problem statement, paged through the datasets server."""
    out = {}
    for offset in range(0, 500, 100):
        r = client.get(f"{HF_ROWS}&offset={offset}&length=100")
        if r.status_code != 200:
            break
        for row in r.json().get("rows", []):
            rec = row["row"]
            out[rec["instance_id"]] = " ".join((rec.get("problem_statement") or "").split())
    print(f"  problem statements: {len(out)}")
    return out


def build(force: bool = False) -> List[Dict]:
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(OUT) and not force:
        return json.load(open(OUT))

    with httpx.Client(timeout=60, follow_redirects=True,
                      headers={"User-Agent": "wormhole-router-trainer"}) as c:
        counts = fetch_resolution_counts(c)
        statements = fetch_problem_statements(c)

    rows = []
    for iid, text in statements.items():
        attempted = counts["attempted"].get(iid, 0)
        if attempted < 5:
            # Too few attempts to say anything about difficulty.
            continue
        solved = counts["solved"].get(iid, 0)
        rows.append({
            "instance_id": iid,
            "prompt": text[:1500],
            "solve_rate": round(solved / attempted, 4),
            "solved_by": solved,
            "attempted_by": attempted,
        })

    rows.sort(key=lambda r: r["solve_rate"])
    json.dump(rows, open(OUT, "w"))
    return rows


if __name__ == "__main__":
    rows = build(force="--force" in sys.argv)
    if not rows:
        sys.exit("No data; check network access to GitHub and Hugging Face.")
    print(f"\n  tasks with measured difficulty: {len(rows)}")
    print(f"  hardest  (solve rate {rows[0]['solve_rate']:.0%}): {rows[0]['prompt'][:70]}")
    print(f"  easiest  (solve rate {rows[-1]['solve_rate']:.0%}): {rows[-1]['prompt'][:70]}")
    buckets = {"hard <20%": 0, "medium 20-60%": 0, "easy >60%": 0}
    for r in rows:
        s = r["solve_rate"]
        buckets["hard <20%" if s < 0.2 else "medium 20-60%" if s < 0.6 else "easy >60%"] += 1
    for k, v in buckets.items():
        print(f"  {k:16s} {v}")
