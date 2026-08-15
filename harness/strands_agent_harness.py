#!/usr/bin/env python3
"""
Open-Source Agentic Framework Harness (Strands / Multi-Agent Pattern)
Demonstrates how multi-agent workflows (Planner, Coder, Reviewer) route through
WormHole to execute complex agentic loops at 90%+ cost reduction.
"""

import time
from openai import OpenAI

# Point standard OpenAI client to WormHole Proxy
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="wormhole-strands-agent-key"
)

class StrandsAgentNode:
    def __init__(self, role: str, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt

    def execute(self, user_input: str):
        print(f"🤖 [{self.role}] Processing task...")
        response = client.chat.completions.create(
            model="wormhole-auto",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        model_used = response.model
        output = response.choices[0].message.content
        print(f"   -> Model Selected by Router: '{model_used}'")
        print(f"   -> Output Snippet: {output[:100]}...\n")
        return output

def run_strands_agentic_workflow():
    print("="*75)
    print(" 🕸️ STRANDS OPEN-SOURCE AGENTIC MULTI-AGENT HARNESS DEMO")
    print("="*75)

    # Agent 1: Planner / Architect Agent
    planner = StrandsAgentNode(
        role="Planner Agent",
        system_prompt="You are a senior software architect breaking down requirements into actionable component tasks."
    )

    # Agent 2: Coder Agent
    coder = StrandsAgentNode(
        role="Coder Agent",
        system_prompt="You are a principal software engineer writing robust, idiomatic Python code."
    )

    # Agent 3: Code Auditor / QA Reviewer Agent
    auditor = StrandsAgentNode(
        role="QA Auditor Agent",
        system_prompt="You are a security and performance auditor checking code for vulnerabilities and edge cases."
    )

    user_goal = "Build a high-performance LRU Cache with thread-safety and O(1) time complexity."
    print(f"📥 Initial Goal: \"{user_goal}\"\n")

    start_t = time.time()
    # Step 1: Planner plans tasks
    plan = planner.execute(user_goal)

    # Step 2: Coder implements based on plan
    code = coder.execute(f"Implement code based on this architecture plan:\n{plan[:300]}")

    # Step 3: Auditor reviews code
    audit = auditor.execute(f"Audit this code implementation for security and thread-safety:\n{code[:300]}")

    total_t = round(time.time() - start_t, 2)
    print("="*75)
    print(f" 🎉 Multi-Agent Strands Loop Execution Completed in {total_t}s!")
    print(" Check your Web Dashboard at http://127.0.0.1:8000/ to view live metrics!")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_strands_agentic_workflow()
