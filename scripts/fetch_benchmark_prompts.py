"""Download real benchmark prompts for router training.

The router was previously trained on hand-written templates that borrowed
benchmark names but contained none of their items: 450 strings, generated from
a few dozen patterns. A classifier fitted to that matches phrasing like the
templates and little else, which is exactly the generalisation problem the
project kept running into.

These datasets are public and small. Fetching them gives the router real task
phrasing to learn from.

The data is downloaded at build time and cached, never committed. That keeps
this repository from redistributing someone else's dataset under its own
licence:

  HumanEval  MIT           https://github.com/openai/human-eval
  MBPP       CC-BY-4.0     https://github.com/google-research/google-research/tree/master/mbpp
  GSM8K      MIT           https://github.com/openai/grade-school-math
"""

import gzip
import json
import os
import sys
from typing import Dict, List

import httpx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark_cache")

SOURCES = {
    "HumanEval": {
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "field": "prompt",
        "gzip": True,
    },
    "MBPP": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
        "field": "text",
        "gzip": False,
    },
    "GSM8K": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "field": "question",
        "gzip": False,
    },
}


def fetch(name: str, spec: Dict, force: bool = False) -> List[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{name}.json")
    if os.path.exists(cache) and not force:
        return json.load(open(cache))

    with httpx.Client(timeout=60, follow_redirects=True) as c:
        r = c.get(spec["url"])
        r.raise_for_status()
        raw = gzip.decompress(r.content).decode() if spec["gzip"] else r.text

    prompts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        text = (json.loads(line).get(spec["field"]) or "").strip()
        # HumanEval prompts are function stubs; keep the docstring, which is
        # the part a router would actually see phrased as a request.
        if text:
            prompts.append(text)

    json.dump(prompts, open(cache, "w"))
    return prompts


def load_all(force: bool = False) -> Dict[str, List[str]]:
    out = {}
    for name, spec in SOURCES.items():
        try:
            out[name] = fetch(name, spec, force)
        except Exception as err:
            print(f"  {name}: unavailable ({type(err).__name__}); skipping", file=sys.stderr)
    return out


if __name__ == "__main__":
    data = load_all(force="--force" in sys.argv)
    total = sum(len(v) for v in data.values())
    for k, v in data.items():
        print(f"  {k:11s} {len(v):>5} prompts")
    print(f"  cached {total} real benchmark prompts in {CACHE_DIR}")
