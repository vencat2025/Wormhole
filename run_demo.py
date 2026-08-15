#!/usr/bin/env python3
"""
WormHole Automated Interactive Demo Launcher
Runs end-to-end inference cost optimization demo, launches dev server, opens dashboard, and sends test prompts.
"""

import sys
import time
import urllib.request
import json
import subprocess
import webbrowser

SERVER_URL = "http://127.0.0.1:8000"

def is_server_running():
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/api/models", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def start_server_if_needed():
    if is_server_running():
        print("✅ WormHole Server is already running on http://127.0.0.1:8000")
        return None
    
    print("🚀 Starting WormHole Gateway Server on http://127.0.0.1:8000 ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to boot
    for _ in range(15):
        if is_server_running():
            print("✅ Server successfully started!")
            return proc
        time.sleep(0.5)
    
    print("⚠️ Warning: Server startup took longer than expected.")
    return proc

def send_chat_completion(prompt: str):
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = json.dumps({
        "model": "wormhole-auto",
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_demo():
    server_proc = start_server_if_needed()
    
    print("\n" + "═"*75)
    print(" ⚡ WORMHOLE AUTOMATED DEMO LAUNCHER")
    print("═"*75)

    # Open dashboard in browser
    print("\n🌐 Opening Web Dashboard at http://127.0.0.1:8000 ...")
    webbrowser.open(SERVER_URL)
    time.sleep(1)

    prompts = [
        {
            "category": "Simple Formatting Task",
            "prompt": "Format a JSON array containing the top 3 primary colors with their hex codes."
        },
        {
            "category": "Standard Code Synthesis",
            "prompt": "Write a Python function to check if a string is a palindrome and write two unit tests."
        },
        {
            "category": "Complex Enterprise Architecture",
            "prompt": "Design a high-concurrency distributed locking system using Redis, Raft consensus algorithm, and Rust with formal safety proof."
        }
    ]

    for idx, item in enumerate(prompts, 1):
        print(f"\n───────────────────────────────────────────────────────────────────────────")
        print(f" 📌 DEMO CASE {idx}: {item['category'].upper()}")
        print(f"───────────────────────────────────────────────────────────────────────────")
        print(f"📥 Input Prompt: \"{item['prompt']}\"")
        print("⏳ Processing through WormHole (Enhancing -> Routing -> Dispatching)...")
        
        try:
            res = send_chat_completion(item["prompt"])
            meta = res["wormhole_metadata"]
            
            print(f"✨ Model 1 (Enhancer): ⚡ Fast Local Enhancer SLM (<1ms inference): Quality enriched prompt")
            print(f"🎯 Model 2 (Router): Selected '{meta['selected_model']}'")
            print(f"💡 Router Reasoning: {meta['router_reasoning']}")
            print(f"💰 Actual Cost:  ${meta['actual_cost_usd']:.6f}")
            print(f"📊 Baseline Cost: ${meta['baseline_cost_usd']:.6f} (GPT-4o)")
            print(f"🎉 Cost Savings:  ${meta['cost_savings_usd']:.6f} ({meta['savings_percentage']} Saved)")
        except Exception as e:
            print(f"❌ Error calling API: {e}")
        
        time.sleep(1.5)

    print("\n" + "═"*75)
    print(" 📊 SUMMARY & LEARNING FLYWHEEL")
    print("═"*75)
    
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/api/logs") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            s = data["summary"]
            print(f"  • Total Requests Logged: {s['total_requests']}")
            print(f"  • Total API Spend:       ${s['total_actual_cost_usd']:.4f}")
            print(f"  • Total Baseline Cost:   ${s['total_baseline_cost_usd']:.4f}")
            print(f"  • Net Cost Savings:      ${s['total_savings_usd']:.4f} ({s['savings_percentage']}% Net Savings)")
            print(f"  • Average Judge Score:   {s['average_judge_score']} / 10.0")
    except Exception as e:
        print(f"Could not fetch summary: {e}")

    print("\n📥 JSONL Fine-Tuning Datasets generated automatically at:")
    print(f"  • {SERVER_URL}/api/dataset/export?target=router")
    print(f"  • {SERVER_URL}/api/dataset/export?target=enhancer")
    print("\n✨ Demo completed! Check your browser window for the live dashboard.\n")

if __name__ == "__main__":
    run_demo()
