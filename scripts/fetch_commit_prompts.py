"""Fetch real engineering instructions, in the register people actually type.

The bootstrap set has a register problem. Its prompts are SWE-bench bug reports
and HumanEval exercises, and neither reads like what someone types at a coding
harness: a short imperative sentence. Measured, the classifier trained on it
called "design a zero-downtime migration to shard the orders table" basic,
having never seen a short sentence that was hard.

An earlier attempt to fix this wrote the missing sentences by hand and filed
them under benchmark names. That was worse than the gap it filled.

Public commit subjects are the same register, written by engineers about work
they actually did, and they arrive with a difficulty signal attached: the diff
that shipped with them. "Fix typo in docstring" touches one file and two lines.
"Refactor the query compiler to support composite primary keys" touches thirty.
Scope is a proxy for difficulty rather than a measurement of it, and every row
this writes says so in label_source.

Usage:
  python scripts/fetch_commit_prompts.py            # ~800 commits, cached
  python scripts/fetch_commit_prompts.py --per 50   # fewer per repository
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "benchmark_cache", "commit_prompts.json")

# Mature Python projects with readable commit subjects and a genuine mix of
# trivial and structural work. Breadth matters more than any single repository:
# one project's conventions would teach the classifier that project's habits.
REPOS = [
    "django/django",
    "pallets/flask",
    "psf/requests",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "fastapi/fastapi",
    "encode/httpx",
    "pytest-dev/pytest",
]

# Commits that say nothing about engineering difficulty.
SKIP = re.compile(
    r"^(merge|revert|bump|release|v?\d+\.\d+|\[pre-commit\.ci\]|chore\(deps\)|"
    r"update changelog|back to development|post-release)",
    re.I,
)


def tier_for_scope(files: int, lines: int) -> str:
    """The tier a change's blast radius suggests, or "" when it suggests nothing.

    Only the unambiguous ends are labelled. Grading the whole range by size was
    tried first and made routing worse: "Include the invalid view type in the
    path() error" is a one-line idea that happened to touch three files, and
    labelling it high taught the classifier that ordinary engineering sentences
    are hard work. Held-out accuracy fell to 67% and routing to 2 of 7.

    A single tiny edit and a change across fifteen files are different kinds of
    task and the proxy is trustworthy at those ends. In between it is measuring
    how the diff happened to land, so those rows are dropped rather than
    guessed at.
    """
    if files == 1 and lines <= 8:
        return "basic"
    if files >= 15:
        return "frontier"
    return ""


def clean_subject(msg: str) -> str:
    """The commit subject as an instruction, without the bookkeeping."""
    subject = msg.split("\n")[0].strip()
    subject = re.sub(r"\s*\(#\d+\)\s*$", "", subject)      # trailing PR number
    subject = re.sub(r"^\[[^\]]+\]\s*", "", subject)        # [3.2.x] backport tags
    # Django-style "Fixed #37149 -- Switched to X" and "Refs #123 -- Y": the
    # ticket number is noise, and the clause after the dashes is the actual
    # description of the work.
    subject = re.sub(r"^(Fixed|Fixes|Fix|Refs|Closed|Closes)\s+#\d+\s*(--|—|–|:)?\s*",
                     "", subject, flags=re.I)
    subject = re.sub(r"^(feat|fix|docs|refactor|perf|test|build|ci)(\([^)]*\))?:\s*",
                     "", subject, flags=re.I)               # conventional-commit prefix
    subject = re.sub(r"^\s*[-–—]\s*", "", subject)           # leftover separator
    # Commit subjects are written in the past tense by convention in some
    # projects and the imperative in others. The register that matters is the
    # imperative, which is what someone types at a harness, so normalise the
    # common past-tense openings rather than teaching the classifier that
    # "Added" and "Add" are different kinds of work.
    PAST = {"added": "Add", "fixed": "Fix", "removed": "Remove", "updated": "Update",
            "changed": "Change", "made": "Make", "moved": "Move", "renamed": "Rename",
            "refactored": "Refactor", "improved": "Improve", "simplified": "Simplify",
            "corrected": "Correct", "replaced": "Replace", "switched": "Switch",
            "skipped": "Skip", "documented": "Document", "allowed": "Allow",
            "prevented": "Prevent", "avoided": "Avoid", "deprecated": "Deprecate"}
    first, _, rest = subject.partition(" ")
    if first.lower() in PAST:
        subject = f"{PAST[first.lower()]} {rest}".strip()
    return subject.strip()


def gh(path: str) -> Optional[Any]:
    try:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


def fetch_repo(repo: str, per: int) -> List[Dict[str, Any]]:
    listing = gh(f"repos/{repo}/commits?per_page={min(per * 2, 100)}")
    if not listing:
        print(f"  {repo}: could not list commits", file=sys.stderr)
        return []

    rows: List[Dict[str, Any]] = []
    for entry in listing:
        if len(rows) >= per:
            break
        sha = entry.get("sha")
        message = (entry.get("commit") or {}).get("message", "")
        subject = clean_subject(message)
        if not sha or SKIP.match(subject) or not (20 <= len(subject) <= 160):
            continue

        detail = gh(f"repos/{repo}/commits/{sha}")
        if not detail:
            continue
        stats = detail.get("stats") or {}
        files = len(detail.get("files") or [])
        lines = int(stats.get("additions", 0)) + int(stats.get("deletions", 0))
        if files == 0:
            continue

        tier = tier_for_scope(files, lines)
        if not tier:
            continue  # the middle of the range says nothing reliable
        rows.append({
            "prompt": subject,
            "benchmark": "commits",
            "repo": repo,
            "files_changed": files,
            "lines_changed": lines,
            "tier": tier,
            "label_source": "commit_scope_proxy",
        })
    print(f"  {repo}: {len(rows)} usable commits")
    return rows


def load_cached() -> List[Dict[str, Any]]:
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=100, help="commits per repository")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    if not args.refresh:
        cached = load_cached()
        if cached:
            print(f"{len(cached)} commit prompts already cached at {CACHE}")
            return 0

    if subprocess.run(["which", "gh"], capture_output=True).returncode != 0:
        sys.exit("The GitHub CLI (gh) is required, and must be authenticated: gh auth login")

    rows: List[Dict[str, Any]] = []
    for repo in REPOS:
        rows.extend(fetch_repo(repo, args.per))

    if not rows:
        sys.exit("No commits fetched. Check `gh auth status` and your rate limit.")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(rows, f, indent=2)

    from collections import Counter
    print(f"\nWrote {len(rows)} commit prompts -> {CACHE}")
    for tier, n in sorted(Counter(r["tier"] for r in rows).items()):
        print(f"   {tier:10s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
