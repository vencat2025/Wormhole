"""Build the training set for the local router SLM.

The classifier can only ever emit a label it saw in training, so this file
decides the routing vocabulary. Two rules keep it honest:

1. Every label must be an id in settings.CANDIDATE_MODELS. A label outside
   the fleet trains the router to pick something the gateway cannot price,
   reach, or fail over from.
2. Difficulty must be visible in the prompt text. The router sees only the
   prompt, so if an easy and a hard task are phrased identically it cannot
   separate them no matter how different their ideal models are.

Benchmark figures below are approximate published pass rates used as relative
capability ordering, not exact measurements.
"""

import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_PATH = os.path.join(DATA_DIR, "frontier_benchmark_dataset.json")

# Approximate pass rate per model per benchmark family.
BENCHMARK_PROFILES: Dict[str, Dict[str, float]] = {
    "HumanEval": {
        "ollama/qwen2.5-coder:7b": 0.72,
        "groq/openai/gpt-oss-20b": 0.81,
        "gemini/gemini-2.5-flash": 0.84,
        "groq/qwen/qwen3.6-27b": 0.85,
        "gpt-4o-mini": 0.86,
        "claude-3-haiku-20240307": 0.75,
        "groq/openai/gpt-oss-120b": 0.89,
        "gemini/gemini-2.5-pro": 0.91,
        "gpt-4o": 0.90,
    },
    "SWE-bench": {
        "ollama/qwen2.5-coder:7b": 0.12,
        "groq/openai/gpt-oss-20b": 0.19,
        "gemini/gemini-2.5-flash": 0.26,
        "groq/qwen/qwen3.6-27b": 0.24,
        "gpt-4o-mini": 0.22,
        "claude-3-haiku-20240307": 0.15,
        "groq/openai/gpt-oss-120b": 0.35,
        "gemini/gemini-2.5-pro": 0.42,
        "gpt-4o": 0.38,
    },
    "GSM8K": {
        "ollama/qwen2.5-coder:7b": 0.78,
        "groq/openai/gpt-oss-20b": 0.88,
        "gemini/gemini-2.5-flash": 0.90,
        "groq/qwen/qwen3.6-27b": 0.90,
        "gpt-4o-mini": 0.91,
        "claude-3-haiku-20240307": 0.82,
        "groq/openai/gpt-oss-120b": 0.94,
        "gemini/gemini-2.5-pro": 0.95,
        "gpt-4o": 0.96,
    },
}

# Quality bar per difficulty. A harder task demands a stronger model, which is
# what makes the router's decision depend on the prompt rather than only on
# which benchmark family it came from.
DIFFICULTY_THRESHOLDS = {"easy": 0.70, "medium": 0.82, "hard": 0.95}

# Phrasings must differ enough that the classifier can tell tiers apart from
# text alone.
# The TEMPLATES table that used to sit here manufactured prompts for MATH,
# GPQA, MMLU and IFEval, complete with per-model pass rates typed in by hand.
# Those rows were indistinguishable from fetched benchmark data once written to
# disk, and they were not benchmark data. Removed rather than relabelled: the
# dataset is smaller now and every row in it is real.



# Which fetched source each prompt is filed under. MBPP and HumanEval are both
# short Python function tasks, so they share a bucket.
REAL_PROMPT_BENCHMARKS = {"HumanEval": "HumanEval", "MBPP": "HumanEval", "GSM8K": "GSM8K"}


def load_real_prompts() -> Dict[str, List[str]]:
    """Real benchmark prompts keyed by the profile they should be scored under."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fetch_benchmark_prompts import load_all
    except Exception:
        return {}

    grouped: Dict[str, List[str]] = {}
    for source, prompts in load_all().items():
        profile = REAL_PROMPT_BENCHMARKS.get(source)
        if not profile:
            continue
        for p in prompts:
            text = " ".join(p.split())
            # Skip the degenerate ends: a stub with no description teaches
            # nothing, and a very long one is unrepresentative of a request.
            if 25 <= len(text) <= 900:
                grouped.setdefault(profile, []).append(text)
    return grouped


def difficulty_for(prompt: str) -> str:
    """Rough tier for a real benchmark prompt.

    Public coding and maths benchmarks do not label difficulty, so this reads
    the signals that actually change which model is needed: an explicit proof
    or complexity obligation, concurrency, or breadth across a system. It is a
    heuristic, and it is why the judged feedback loop matters more than this
    bootstrap.
    """
    low = prompt.lower()
    hard_signals = ("prove", "proof", "complexity", "o(n", "concurren", "race condition",
                    "distributed", "optimi", "architect", "migrat", "thread-safe")
    if any(w in low for w in hard_signals):
        return "hard"
    return "medium" if len(prompt) > 180 else "easy"




# Capability order, cheapest-capable first. Used to turn a measured solve rate
# directly into a tier.
TIER_ORDER = ["basic", "medium", "high", "frontier"]


def tier_from_solve_rate(rate: float) -> str:
    """The capability tier a measured solve rate implies.

    This is the honest label in the whole dataset: the rate is how much of the
    published SWE-bench field actually solved that instance, so a task almost
    nobody solved is frontier work by observation rather than by opinion.
    """
    if rate >= 0.75:
        return "basic"
    if rate >= 0.45:
        return "medium"
    if rate >= 0.20:
        return "high"
    return "frontier"


def label_from_solve_rate(rate: float, costs: Dict[str, float]) -> str:
    """Pick a tier straight from how often real systems solved the task.

    Routing a measured solve rate through a benchmark's published average
    throws the measurement away: SWE-bench pass rates sit in a narrow band, so
    its easy and medium tiers collapse onto the same model and every task looks
    equally hard. The observed rate is the better signal, so it selects the
    tier itself.
    """
    if rate >= 0.75:
        wanted = "basic"       # nearly everything solved it
    elif rate >= 0.45:
        wanted = "medium"
    elif rate >= 0.20:
        wanted = "high"
    else:
        wanted = "frontier"    # almost nothing solved it

    start = TIER_ORDER.index(wanted)
    for tier in TIER_ORDER[start:]:
        eligible = [m for m in settings.CANDIDATE_MODELS
                    if m.intelligence_tier == tier and m.id in costs]
        if eligible:
            return min(eligible, key=lambda m: costs[m.id]).id
    return min(costs, key=costs.get)


def load_measured_difficulty() -> List[Dict[str, Any]]:
    """Real coding tasks whose difficulty was measured, not guessed.

    Each row is a SWE-bench Verified instance plus the fraction of ~134
    published systems that actually resolved it, which those systems
    established by running the projects' own test suites. A task few systems
    resolved is hard; one most resolved is not.

    Everywhere else in this file difficulty is inferred from wording. Here it
    is an observed property of the task, which is the difference between the
    router learning what hard problems look like and learning which words tend
    to appear in them.
    """
    path = os.path.join(DATA_DIR, "benchmark_cache", "swebench_difficulty.json")
    if not os.path.exists(path):
        return []
    rows = json.load(open(path))
    out = []
    for r in rows:
        rate = r["solve_rate"]
        tier = "hard" if rate < 0.20 else "medium" if rate < 0.60 else "easy"
        text = r["prompt"].strip()
        if 40 <= len(text) <= 1500:
            out.append({"prompt": text, "difficulty": tier, "benchmark": "SWE-bench",
                        "solve_rate": rate})
    return out


def fleet_costs() -> Dict[str, float]:
    """Input cost per model id, from the live fleet definition.

    ROUTING_PROVIDERS is applied here as well as at request time. The
    classifier can only emit labels it saw in training, so restricting the
    fleet at runtime alone would leave it predicting models it is no longer
    allowed to pick and falling back to a single default every time. Training
    on the same restricted set is what produces real routing *within* a
    provider.
    """
    return {
        m.id: m.input_cost_per_1k
        for m in settings.CANDIDATE_MODELS
        if settings.provider_allowed(m.provider) and settings.model_allowed(m.id)
    }



def build_dataset(num_samples: int = 2000, seed: int = 42) -> List[Dict[str, Any]]:
    """Every row is a real benchmark prompt. Nothing here is written by us.

    An earlier version padded this out with hand-written templates filed under
    MATH, GPQA, MMLU and IFEval, with per-model pass rates typed into this
    file. Those rows looked like benchmark data and were not: "give a formal
    proof that the halting problem is undecidable" is a sentence someone wrote
    here, not a MATH item, and no published figure said what any model scores
    on it. They are gone.

    What remains is fetched: SWE-bench instances with the solve rate actually
    observed across the published field, and HumanEval, GSM8K and MBPP prompts
    as their authors wrote them. Each row records how its label was arrived at
    in label_source, because the two are not equally strong -- one is measured
    and the other is a keyword heuristic over the prompt.
    """
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []

    real = load_real_prompts()
    real_pool = [(bm, p) for bm, ps in real.items() for p in ps]
    rng.shuffle(real_pool)

    measured = load_measured_difficulty()
    rng.shuffle(measured)
    print(f"Real benchmark prompts available: {len(real_pool)}")
    print(f"Tasks with measured difficulty:   {len(measured)}")

    if not real_pool and not measured:
        raise SystemExit(
            "No real benchmark data available. Run scripts/fetch_benchmark_prompts.py "
            "and scripts/fetch_swebench_difficulty.py first; this builder no longer "
            "invents rows to fill the gap."
        )

    # Measured rows first, then as many real prompts as are needed to reach the
    # target. Both pools are finite, so the dataset is as large as the real
    # data allows and no larger.
    for m in measured:
        dataset.append({
            "benchmark": m["benchmark"],
            "prompt": m["prompt"],
            "difficulty": m["difficulty"],
            "expected_pass_rate": m["solve_rate"],
            "tier": tier_from_solve_rate(m["solve_rate"]),
            "label_source": "measured_solve_rate",
        })

    for benchmark, prompt in real_pool:
        if len(dataset) >= num_samples:
            break
        difficulty = difficulty_for(prompt)
        dataset.append({
            "benchmark": benchmark,
            "prompt": prompt,
            "difficulty": difficulty,
            "expected_pass_rate": None,
            "tier": {"easy": "basic", "medium": "medium", "hard": "high"}[difficulty],
            "label_source": "prompt_heuristic",
        })

    for i, row in enumerate(dataset, 1):
        row["id"] = i
    return dataset


if __name__ == "__main__":
    random.seed(42)
    os.makedirs(DATA_DIR, exist_ok=True)
    data = build_dataset(2000)

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    from collections import Counter
    print(f"Generated {len(data)} samples -> {OUT_PATH}")
    for src, n in Counter(d["label_source"] for d in data).most_common():
        print(f"   label source: {src:24s} {n}")
    for tier, n in sorted(Counter(d["tier"] for d in data).items()):
        print(f"   tier: {tier:30s} {n}")
    for bm, n in Counter(d["benchmark"] for d in data).most_common():
        print(f"   benchmark: {bm:25s} {n}")
