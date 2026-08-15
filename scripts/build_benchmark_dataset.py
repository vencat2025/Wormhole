import os
import json
import random
from typing import List, Dict, Any

# Benchmark Performance Matrix (Pass rates % on official public benchmarks)
BENCHMARK_PROFILES = {
    "HumanEval": {
        "gpt-4o-mini": 0.86,
        "gemini/gemini-1.5-flash": 0.79,
        "claude-3-haiku-20240307": 0.75,
        "gemini/gemini-1.5-pro": 0.88,
        "gpt-4o": 0.90,
        "claude-3-5-sonnet-20240620": 0.92
    },
    "MBPP": {
        "gpt-4o-mini": 0.87,
        "gemini/gemini-1.5-flash": 0.82,
        "claude-3-haiku-20240307": 0.78,
        "gemini/gemini-1.5-pro": 0.89,
        "gpt-4o": 0.91,
        "claude-3-5-sonnet-20240620": 0.93
    },
    "SWE-bench": {
        "gpt-4o-mini": 0.22,
        "gemini/gemini-1.5-flash": 0.18,
        "claude-3-haiku-20240307": 0.15,
        "gemini/gemini-1.5-pro": 0.32,
        "gpt-4o": 0.38,
        "claude-3-5-sonnet-20240620": 0.49
    },
    "LiveCodeBench": {
        "gpt-4o-mini": 0.42,
        "gemini/gemini-1.5-flash": 0.35,
        "claude-3-haiku-20240307": 0.30,
        "gemini/gemini-1.5-pro": 0.48,
        "gpt-4o": 0.54,
        "claude-3-5-sonnet-20240620": 0.58
    },
    "GSM8K": {
        "gpt-4o-mini": 0.91,
        "gemini/gemini-1.5-flash": 0.86,
        "claude-3-haiku-20240307": 0.82,
        "gemini/gemini-1.5-pro": 0.93,
        "gpt-4o": 0.96,
        "claude-3-5-sonnet-20240620": 0.96
    },
    "MATH": {
        "gpt-4o-mini": 0.70,
        "gemini/gemini-1.5-flash": 0.58,
        "claude-3-haiku-20240307": 0.50,
        "gemini/gemini-1.5-pro": 0.71,
        "gpt-4o": 0.77,
        "claude-3-5-sonnet-20240620": 0.78
    },
    "MMLU": {
        "gpt-4o-mini": 0.82,
        "gemini/gemini-1.5-flash": 0.79,
        "claude-3-haiku-20240307": 0.75,
        "gemini/gemini-1.5-pro": 0.85,
        "gpt-4o": 0.88,
        "claude-3-5-sonnet-20240620": 0.88
    },
    "GPQA": {
        "gpt-4o-mini": 0.40,
        "gemini/gemini-1.5-flash": 0.38,
        "claude-3-haiku-20240307": 0.32,
        "gemini/gemini-1.5-pro": 0.45,
        "gpt-4o": 0.53,
        "claude-3-5-sonnet-20240620": 0.59
    },
    "ARC-Challenge": {
        "gpt-4o-mini": 0.89,
        "gemini/gemini-1.5-flash": 0.86,
        "claude-3-haiku-20240307": 0.84,
        "gemini/gemini-1.5-pro": 0.91,
        "gpt-4o": 0.94,
        "claude-3-5-sonnet-20240620": 0.95
    },
    "IFEval": {
        "gpt-4o-mini": 0.80,
        "gemini/gemini-1.5-flash": 0.76,
        "claude-3-haiku-20240307": 0.72,
        "gemini/gemini-1.5-pro": 0.84,
        "gpt-4o": 0.88,
        "claude-3-5-sonnet-20240620": 0.89
    }
}

MODEL_INPUT_COSTS = {
    "gemini/gemini-1.5-flash": 0.000075,
    "gpt-4o-mini": 0.00015,
    "claude-3-haiku-20240307": 0.00025,
    "gemini/gemini-1.5-pro": 0.00125,
    "gpt-4o": 0.00250,
    "claude-3-5-sonnet-20240620": 0.00300
}

# Template prompt generators per benchmark category
BENCHMARK_TEMPLATES = [
    # HumanEval / MBPP (Simple-Medium Coding)
    ("HumanEval", "Write a Python function `{fn_name}` that takes `{args}` and returns `{return_desc}`. Add basic type hints."),
    ("MBPP", "Write a Python function to check whether {task_desc} and return the boolean result."),
    # SWE-bench / LiveCodeBench (Complex Software Architecture & Deep Refactoring)
    ("SWE-bench", "Fix an issue in a multi-file repository: {repo_issue}. Update the concurrent async loop without deadlocking."),
    ("LiveCodeBench", "Given an array of integers and target, find all unique quad tuples that sum to target using O(N^3) optimization and formal proof."),
    # GSM8K (Grade School Math)
    ("GSM8K", "Solve this word problem: {math_word_problem} Calculate step by step."),
    ("MATH", "Compute the exact value of {math_expr} and express the solution as a simplified fraction or radical."),
    # GPQA (Graduate Physics/Chemistry/Biology)
    ("GPQA", "In quantum mechanics, evaluate the transition probability amplitude for a particle in a {physics_context}."),
    # MMLU / ARC-Challenge (General Knowledge & Multi-hop Reasoning)
    ("MMLU", "Explain the historical significance of {topic} during the 19th century and its economic impact."),
    ("ARC-Challenge", "Identify which biological process occurs when {bio_condition} and justify the causal link."),
    # IFEval (Strict Constraint & Formatting)
    ("IFEval", "Output a valid JSON schema with keys `{key1}` and `{key2}`. Ensure all values are lowercased strings.")
]

SAMPLE_DATA_FILLERS = {
    "fn_name": ["find_duplicates", "calculate_median", "is_valid_email", "parse_timestamp", "binary_search"],
    "args": ["a list of integers", "a string", "a dictionary of key-values", "two sorted arrays"],
    "return_desc": ["the count of occurrences", "the sanitized string", "the merged dictionary", "the index of the element"],
    "task_desc": ["a given number is a prime", "all elements in a list are unique", "a string is a valid palindrome"],
    "repo_issue": [
        "Race condition in connection pool retry mechanism under high load",
        "Memory leak in async stream handler when client disconnects prematurely",
        "Distributed transaction deadlock between order-service and inventory-service"
    ],
    "math_word_problem": [
        "A store sells apples for $2 each and oranges for $3 each. If Sarah bought 12 fruits total and spent $29, how many apples did she buy?",
        "A car travels at 60 mph for 2 hours and then 45 mph for 3 hours. What is the average speed of the car for the entire journey?"
    ],
    "math_expr": ["\\int_0^{\\pi/2} \\sin^3(x) \\cos(x) dx", "\\sum_{n=1}^{\\infty} \\frac{1}{n^2}", "\\lim_{x \\to 0} \\frac{\\sin(3x)}{x}"],
    "physics_context": ["1D infinite square well under a time-dependent perturbation", "3D harmonic oscillator with spin-orbit coupling"],
    "topic": ["the Industrial Revolution", "the Bretton Woods Conference", "the Treaty of Versailles"],
    "bio_condition": ["ATP synthesis occurs during oxidative phosphorylation", "cellular respiration experiences anaerobic conditions"],
    "key1": ["user_id", "session_token", "transaction_id"],
    "key2": ["status", "timestamp", "payload_hash"]
}

def generate_prompt_for_benchmark(benchmark_name: str, template: str) -> str:
    filled = template
    for key, values in SAMPLE_DATA_FILLERS.items():
        placeholder = "{" + key + "}"
        if placeholder in filled:
            filled = filled.replace(placeholder, random.choice(values))
    return filled

def select_optimal_model(benchmark_name: str) -> Dict[str, Any]:
    scores = BENCHMARK_PROFILES[benchmark_name]
    
    # Target rule: Filter candidate models with pass rate >= 0.75 (or max pass rate if none >= 0.75),
    # then pick the model with the lowest input token cost.
    max_score = max(scores.values())
    threshold = 0.75 if max_score >= 0.75 else (max_score - 0.05)
    
    eligible_models = [m for m, score in scores.items() if score >= threshold]
    # Sort eligible models by input cost ascending
    eligible_models.sort(key=lambda m: MODEL_INPUT_COSTS[m])
    
    chosen_model = eligible_models[0]
    return {
        "chosen_model": chosen_model,
        "pass_rate": scores[chosen_model],
        "input_cost": MODEL_INPUT_COSTS[chosen_model]
    }

def build_dataset(num_samples: int = 1500) -> List[Dict[str, Any]]:
    dataset = []
    benchmarks = list(BENCHMARK_PROFILES.keys())
    
    for i in range(num_samples):
        b_name, template = random.choice(BENCHMARK_TEMPLATES)
        prompt = generate_prompt_for_benchmark(b_name, template)
        opt = select_optimal_model(b_name)
        
        dataset.append({
            "id": i + 1,
            "benchmark": b_name,
            "prompt": prompt,
            "selected_model": opt["chosen_model"],
            "expected_pass_rate": opt["pass_rate"],
            "input_cost_per_1k": opt["input_cost"]
        })
    return dataset

if __name__ == "__main__":
    random.seed(42)
    os.makedirs("/Users/venkat/Documents/AI/WormHole/data", exist_ok=True)
    out_path = "/Users/venkat/Documents/AI/WormHole/data/frontier_benchmark_dataset.json"
    
    data = build_dataset(2000)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"✅ Generated {len(data)} benchmark samples across HumanEval, SWE-bench, GSM8K, MATH, GPQA, MMLU, IFEval!")
    print(f"📁 Dataset saved to {out_path}")
