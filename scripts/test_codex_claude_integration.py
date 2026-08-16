#!/usr/bin/env python3
"""
Live Integration Tester for OpenAI Codex, Claude Code, Cursor, and Custom Developer Tools
Sends requests simulating various coding models & settings through WormHole proxy.
"""

import time
import requests
import json

WORMHOLE_BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "wh_live_demo123456789"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

TEST_WORKLOADS = [
    {
        "harness": "Claude Code CLI / Codex",
        "requested_model": "wormhole-auto",
        "prompt": "Write a Python script to validate email addresses using regular expressions and write 2 pytest cases.",
        "description": "Standard Utility Coding Workload"
    },
    {
        "harness": "Cursor IDE / Aider CLI",
        "requested_model": "gpt-4o-mini",
        "prompt": "Convert a CSV string into a JSON array of key-value objects in JavaScript.",
        "description": "Simple Formatting & Data Transformation"
    },
    {
        "harness": "Codex High-Reasoning / Claude 3.5 Sonnet Setting",
        "requested_model": "claude-3-5-sonnet-20240620",
        "prompt": "Design a high-concurrency distributed lock manager using Redis, Raft consensus, and Rust with memory safety guarantees.",
        "description": "High-Complexity System Architecture"
    }
]

def run_integration_demo():
    print("=" * 80)
    print(" 🚀 LIVE INTERACTIVE DEMO: CODEX & CLAUDE CODE HARNESS INTEGRATION")
    print(f" 🌐 WormHole Proxy Gateway: {WORMHOLE_BASE_URL}")
    print("=" * 80 + "\n")

    for idx, test in enumerate(TEST_WORKLOADS, 1):
        print(f"📌 TEST CASE {idx}: {test['harness'].upper()}")
        print(f"   Requested Model: '{test['requested_model']}' ({test['description']})")
        print(f"   Prompt: \"{test['prompt']}\"")
        print("   ⏳ Dispatching through WormHole Proxy Gateway...")

        start_t = time.time()
        payload = {
            "model": test["requested_model"],
            "messages": [
                {"role": "system", "content": "You are an expert AI software engineer."},
                {"role": "user", "content": test["prompt"]}
            ]
        }

        try:
            res = requests.post(f"{WORMHOLE_BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=10)
            latency = round((time.time() - start_t) * 1000, 2)
            
            if res.status_code == 200:
                data = res.json()
                meta = data.get("wormhole_metadata", {})
                print(f"   ✅ Response Received in {latency}ms!")
                print(f"   🎯 Model Selected by Router: '{data.get('model')}'")
                print(f"   💡 Rationale: {meta.get('router_reasoning')}")
                print(f"   💰 Actual Cost: ${meta.get('actual_cost_usd'):.6f} | Baseline Cost (GPT-4o): ${meta.get('baseline_cost_usd'):.6f}")
                print(f"   🎉 Net Savings: ${meta.get('cost_savings_usd'):.6f} ({meta.get('savings_percentage')} Saved!)")
                snippet = data["choices"][0]["message"]["content"].replace("\n", " ")[:120]
                print(f"   📝 Output Snippet: \"{snippet}...\"")
            else:
                print(f"   ❌ HTTP Error {res.status_code}: {res.text}")

        except Exception as e:
            print(f"   ❌ Connection Error: {e}")

        print("-" * 80 + "\n")

    print("=" * 80)
    print(" 📊 SUMMARY: Check your live Web Dashboard at http://127.0.0.1:8000/ to view live metrics!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_integration_demo()
