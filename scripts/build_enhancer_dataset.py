import os
import json
import random
from typing import List, Dict, Any

# Prompts across domains paired with expert quality-enhanced prompt structures
RAW_PROMPT_TEMPLATES = [
    ("Format a JSON list of top primary colors.", 
     "Objective: Format a JSON list of top primary colors.\nInstructions:\n- Return a valid JSON array containing primary color objects.\n- Include keys: `name`, `hex_code`, and `rgb`.\n- Ensure lowercase formatting and strict JSON syntax without markdown wrappers."),
    
    ("Write a function to check palindrome.", 
     "Objective: Implement a Python function to verify if a string is a palindrome.\nInstructions:\n- Case-insensitive comparison ignoring non-alphanumeric characters.\n- Provide type annotations and docstrings.\n- Include two unit tests checking standard and edge cases (empty string, single char)."),
    
    ("Summarize meeting notes.", 
     "Objective: Provide a executive summary of meeting notes.\nInstructions:\n- Extract Key Decisions, Action Items with assigned owners, and Open Discussion points.\n- Format using concise bullet points under markdown headers.\n- Maintain an objective, professional tone."),
    
    ("Refactor monolithic app to microservices.", 
     "Objective: Design a migration plan for monolithic codebase to microservices.\nInstructions:\n- Define service boundaries using Domain-Driven Design (DDD).\n- Outline data decomposition strategies and event-driven async communication.\n- Highlight fault tolerance, circuit breaking, and zero-downtime deployment patterns."),
    
    ("Fix SQL query performance issue.", 
     "Objective: Optimize slow SQL query execution.\nInstructions:\n- Analyze missing indexes, full table scans, and expensive JOIN operations.\n- Provide an updated SQL query utilizing appropriate composite indexes.\n- Explain execution plan optimizations (EXPLAIN ANALYZE).")
]

DOMAIN_VARIATIONS = [
    ("Write a Python function to {action}.", "Objective: Write a Python function to {action}.\nInstructions:\n- Implement clean Python 3.10+ code with type hints.\n- Include proper error handling for edge cases.\n- Add 2-3 unit tests."),
    ("Draft a technical documentation for {topic}.", "Objective: Draft technical documentation for {topic}.\nInstructions:\n- Structure with Overview, Prerequisites, Architecture, and Step-by-Step guide.\n- Include clear code snippets and environment configuration flags."),
    ("Solve this math problem: {expr}.", "Objective: Solve mathematical expression: {expr}.\nInstructions:\n- Show step-by-step algebraic derivations.\n- State final answer clearly in simplified fraction or exact radical form.")
]

DATA_VARIANTS = {
    "action": ["parse CSV files", "calculate Fibonacci numbers asynchronously", "validate JWT tokens", "sort a linked list"],
    "topic": ["Redis caching layer", "OAuth2 authentication flow", "Kafka message bus integration"],
    "expr": ["x^2 - 5x + 6 = 0", "limit of sin(x)/x as x approaches 0", "integral of e^(2x) dx"]
}

def build_enhancer_dataset(num_samples: int = 1000) -> List[Dict[str, Any]]:
    dataset = []
    
    # Add base pairs
    for orig, enh in RAW_PROMPT_TEMPLATES:
        dataset.append({
            "original_prompt": orig,
            "enhanced_prompt": enh
        })
        
    for i in range(num_samples - len(RAW_PROMPT_TEMPLATES)):
        tmpl_orig, tmpl_enh = random.choice(DOMAIN_VARIATIONS)
        filled_orig = tmpl_orig
        filled_enh = tmpl_enh
        
        for key, vals in DATA_VARIANTS.items():
            ph = "{" + key + "}"
            if ph in filled_orig:
                chosen_val = random.choice(vals)
                filled_orig = filled_orig.replace(ph, chosen_val)
                filled_enh = filled_enh.replace(ph, chosen_val)
                
        dataset.append({
            "id": i + 1,
            "original_prompt": filled_orig,
            "enhanced_prompt": filled_enh
        })
        
    return dataset

if __name__ == "__main__":
    random.seed(42)
    os.makedirs("/Users/venkat/Documents/AI/WormHole/data", exist_ok=True)
    out_path = "/Users/venkat/Documents/AI/WormHole/data/enhancer_dataset.json"
    
    data = build_enhancer_dataset(1200)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"✅ Generated {len(data)} prompt enhancement samples!")
    print(f"📁 Dataset saved to {out_path}")
