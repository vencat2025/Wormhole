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
    "MATH": {
        "ollama/qwen2.5-coder:7b": 0.45,
        "groq/openai/gpt-oss-20b": 0.62,
        "gemini/gemini-2.5-flash": 0.69,
        "groq/qwen/qwen3.6-27b": 0.68,
        "gpt-4o-mini": 0.70,
        "claude-3-haiku-20240307": 0.50,
        "groq/openai/gpt-oss-120b": 0.75,
        "gemini/gemini-2.5-pro": 0.82,
        "gpt-4o": 0.77,
    },
    "GPQA": {
        "ollama/qwen2.5-coder:7b": 0.25,
        "groq/openai/gpt-oss-20b": 0.35,
        "gemini/gemini-2.5-flash": 0.42,
        "groq/qwen/qwen3.6-27b": 0.41,
        "gpt-4o-mini": 0.40,
        "claude-3-haiku-20240307": 0.32,
        "groq/openai/gpt-oss-120b": 0.49,
        "gemini/gemini-2.5-pro": 0.58,
        "gpt-4o": 0.53,
    },
    "MMLU": {
        "ollama/qwen2.5-coder:7b": 0.68,
        "groq/openai/gpt-oss-20b": 0.78,
        "gemini/gemini-2.5-flash": 0.82,
        "groq/qwen/qwen3.6-27b": 0.81,
        "gpt-4o-mini": 0.82,
        "claude-3-haiku-20240307": 0.75,
        "groq/openai/gpt-oss-120b": 0.86,
        "gemini/gemini-2.5-pro": 0.89,
        "gpt-4o": 0.88,
    },
    "IFEval": {
        "ollama/qwen2.5-coder:7b": 0.65,
        "groq/openai/gpt-oss-20b": 0.75,
        "gemini/gemini-2.5-flash": 0.80,
        "groq/qwen/qwen3.6-27b": 0.79,
        "gpt-4o-mini": 0.80,
        "claude-3-haiku-20240307": 0.72,
        "groq/openai/gpt-oss-120b": 0.85,
        "gemini/gemini-2.5-pro": 0.87,
        "gpt-4o": 0.88,
    },
}

# Quality bar per difficulty. A harder task demands a stronger model, which is
# what makes the router's decision depend on the prompt rather than only on
# which benchmark family it came from.
DIFFICULTY_THRESHOLDS = {"easy": 0.70, "medium": 0.82, "hard": 0.95}

# Phrasings must differ enough that the classifier can tell tiers apart from
# text alone.
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "HumanEval": {
        "easy": [
            "Write a {lang} function to reverse a string.",
            "Write a {lang} helper that trims whitespace from a {struct}.",
            "Show me how to count items in a {struct} in {lang}.",
            "Write a function that returns the sum of a {struct} of numbers.",
            "Write a one-liner to sort a {struct} in {lang}.",
            "Write a helper that checks whether a number is even.",
        ],
        "medium": [
            "Write a {lang} function `{fn}` over a sorted {struct}, with type hints and a docstring.",
            "Implement `{fn}` in {lang} with input validation and unit tests.",
            "Refactor `{fn}` in {lang} to remove duplication and add error handling.",
            "Implement an LRU cache decorator with a configurable maximum size.",
            "Write a function that merges two sorted arrays into one sorted array in linear time.",
        ],
        "hard": [
            "Implement a thread-safe, lock-free concurrent hash map with resizing and a formal argument for correctness under contention.",
            "Write an incremental parser that recovers from syntax errors and reports precise source spans for each recovery point.",
            # Algorithmic prompts whose difficulty comes from a complexity
            # bound or proof obligation rather than from vocabulary. Without
            # these the classifier reads them as ordinary coding tasks,
            # because the surface language is identical.
            "Given an array of integers and a target, find all unique quadruplets summing to the target with O(n^3) complexity, and prove the bound is tight.",
            "Optimise this algorithm from O(n^2) to O(n log n) and give a formal proof of correctness for the new version.",
            "Find the maximum flow in this graph, justify the choice of algorithm, and prove the complexity bound you claim.",
            "Design an enterprise-scale distributed system architecture for {svc} with formal correctness guarantees under partition.",
            "Devise an autonomous scheduling algorithm with provable optimality and analyse its worst-case behaviour.",
        ],
    },
    "SWE-bench": {
        "easy": [
            "Fix the typo in this error message string.",
            "Update the deprecated import in this module to the new path.",
        ],
        "medium": [
            "Fix a failing unit test caused by an off-by-one error in the pagination offset of {svc}.",
            "Fix the bug in {svc} where {field} is dropped on retry.",
            "Debug why {svc} returns a stale {field2} after a config reload.",
            "Resolve the regression where the retry decorator swallows the original exception.",
        ],
        "hard": [
            "Fix a race condition in the {svc} connection pool retry mechanism under high load, across multiple files, without introducing deadlock.",
            "Redesign the {svc} architecture to support multi-region failover with exactly-once delivery, and migrate the existing data.",
            "Refactor the entire {svc} module into clean bounded contexts without breaking its public contract.",
            "Diagnose and repair a memory leak in the async stream handler when clients disconnect prematurely, and prove the fix holds under backpressure.",
            "Resolve a distributed transaction deadlock between order-service and inventory-service while preserving exactly-once semantics.",
        ],
    },
    "GSM8K": {
        "easy": [
            "If a pen costs $2 and I buy 5, how much do I spend?",
            "What is 15% of 200?",
        ],
        "medium": [
            "A store sells apples for $2 and oranges for $3. Sarah bought 12 fruits and spent $29. How many apples did she buy? Show your steps.",
            "A car travels 60 mph for 2 hours then 45 mph for 3 hours. What is the average speed for the journey? Show your steps.",
        ],
        "hard": [
            "Three pipes fill a tank at different rates and two drains empty it on staggered schedules. Derive the general fill-time formula, then evaluate it, showing every step.",
        ],
    },
    "MATH": {
        "easy": ["Simplify the fraction 12/18.", "What is the derivative of x^2?"],
        "medium": [
            "Compute the exact value of the integral of sin^3(x)cos(x) from 0 to pi/2 as a simplified fraction.",
            "Evaluate the limit of sin(3x)/x as x approaches 0, justifying each step.",
        ],
        "hard": [
            "Prove that the sum over n of 1/n^2 converges to pi^2/6, giving a rigorous derivation rather than a citation.",
            "Give a formal proof that the halting problem is undecidable, constructing the diagonalization explicitly.",
        ],
    },
    "GPQA": {
        "easy": ["What is the chemical symbol for gold?", "How many protons does a carbon atom have?"],
        "medium": [
            "Explain why entropy increases in an irreversible adiabatic expansion of an ideal gas.",
        ],
        "hard": [
            "In quantum mechanics, evaluate the transition probability amplitude for a particle in a 1D infinite square well under a time-dependent perturbation, to second order.",
            "Derive the energy spectrum of a 3D harmonic oscillator with spin-orbit coupling and identify the degeneracy of each level.",
        ],
    },
    "MMLU": {
        "easy": ["What is the capital of France?", "In what year did World War II end?"],
        "medium": [
            "Explain the historical significance of {topic} and its economic impact.",
            "Summarise {topic} in one paragraph for a non-specialist.",
            "Summarize the causes of the Industrial Revolution and its effect on urban labour.",
        ],
        "hard": [
            "Compare the monetary policy responses to the 1929 and 2008 crises, evaluating counterfactuals for each and the limits of the comparison.",
        ],
    },
    "IFEval": {
        "easy": ["Reply with only the word OK.", "List three colours as a comma-separated line."],
        "medium": [
            "Output a valid JSON object with keys `{field}` and `{field2}`. All values must be lowercase strings, no extra keys.",
            "Return exactly three bullet points about {topic}, no preamble.",
        ],
        "hard": [
            "Produce a JSON schema with nested objects, enum constraints and a conditional required-field rule, validating against draft 2020-12 with no prose outside the JSON.",
        ],
    },
}


# Real traffic never matches a template verbatim. Without surface variation
# the classifier memorises a few dozen strings and gives an arbitrary answer
# on anything else, so each template is expanded into many phrasings.
FILLERS: Dict[str, List[str]] = {
    "lang": ["Python", "JavaScript", "Go", "TypeScript", "Rust"],
    "struct": ["list", "array", "dictionary", "set", "queue"],
    "fn": ["find_duplicates", "calculate_median", "is_valid_email", "parse_timestamp", "normalize_path"],
    "svc": ["order-service", "billing-service", "inventory-service", "auth-service"],
    "field": ["user_id", "session_token", "transaction_id", "request_id"],
    "field2": ["status", "timestamp", "payload_hash", "retry_count"],
    "topic": ["the Industrial Revolution", "the Bretton Woods Conference", "the Treaty of Versailles"],
}

PREFIXES = {
    "easy": ["", "Quick one: ", "Simple task: ", "Can you ", "Please "],
    "medium": ["", "I need you to ", "Please ", "Task: "],
    "hard": ["", "This is involved: ", "Careful with this one: ", "Take your time: "],
}


def _fill(text: str, rng: random.Random) -> str:
    for key, values in FILLERS.items():
        token = "{" + key + "}"
        while token in text:
            text = text.replace(token, rng.choice(values), 1)
    return text


def render_prompt(benchmark: str, difficulty: str, rng: random.Random) -> str:
    body = _fill(rng.choice(TEMPLATES[benchmark][difficulty]), rng)
    prefix = rng.choice(PREFIXES[difficulty])
    if prefix and body:
        body = body[0].lower() + body[1:] if prefix.endswith(" ") and prefix[-2:] != ": " else body
    return f"{prefix}{body}"


def fleet_costs() -> Dict[str, float]:
    """Input cost per model id, from the live fleet definition."""
    return {m.id: m.input_cost_per_1k for m in settings.CANDIDATE_MODELS}


def select_optimal_model(benchmark: str, difficulty: str, costs: Dict[str, float]) -> Dict[str, Any]:
    """Cheapest model clearing the quality bar for this difficulty."""
    scores = {m: s for m, s in BENCHMARK_PROFILES[benchmark].items() if m in costs}
    if not scores:
        raise ValueError(f"No benchmark model for {benchmark} is present in CANDIDATE_MODELS.")

    threshold = DIFFICULTY_THRESHOLDS[difficulty] * max(scores.values())
    eligible = [m for m, s in scores.items() if s >= threshold] or [max(scores, key=scores.get)]
    chosen = min(eligible, key=lambda m: (costs[m], -scores[m]))
    return {"chosen_model": chosen, "pass_rate": scores[chosen], "input_cost": costs[chosen]}


def build_dataset(num_samples: int = 2000, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    costs = fleet_costs()
    pairs = [(b, d) for b in TEMPLATES for d in ("easy", "medium", "hard")]
    dataset = []

    for i in range(num_samples):
        benchmark, difficulty = pairs[i % len(pairs)]  # even coverage, not luck
        prompt = render_prompt(benchmark, difficulty, rng)
        opt = select_optimal_model(benchmark, difficulty, costs)
        dataset.append({
            "id": i + 1,
            "benchmark": benchmark,
            "difficulty": difficulty,
            "prompt": prompt,
            "selected_model": opt["chosen_model"],
            "expected_pass_rate": opt["pass_rate"],
            "input_cost_per_1k": opt["input_cost"],
        })
    return dataset


if __name__ == "__main__":
    random.seed(42)
    os.makedirs(DATA_DIR, exist_ok=True)
    data = build_dataset(2000)

    fleet = {m.id for m in settings.CANDIDATE_MODELS}
    labels = {d["selected_model"] for d in data}
    stray = labels - fleet
    if stray:
        raise SystemExit(f"Refusing to write: labels outside the fleet: {sorted(stray)}")

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    from collections import Counter
    print(f"Generated {len(data)} samples -> {OUT_PATH}")
    print(f"Distinct routable labels: {len(labels)}")
    for model, n in Counter(d["selected_model"] for d in data).most_common():
        print(f"   {model:32s} {n}")
