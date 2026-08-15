#!/usr/bin/env python3
"""
Open-Source Enterprise Application Harness
Demonstrates how client applications (agents, RAG pipelines, data extractors)
use the standard OpenAI client SDK connected to WormHole proxy to get 90%+ cost savings.
"""

import sys
import time
from openai import OpenAI

WORMHOLE_BASE_URL = "http://127.0.0.1:8000/v1"

# Initialize standard OpenAI client pointing to WormHole proxy gateway
client = OpenAI(
    base_url=WORMHOLE_BASE_URL,
    api_key="wormhole-enterprise-key"
)

def run_code_repair_harness():
    print("\n🤖 [Agent Harness 1] Running Automated Code Repair Workload...")
    prompt = "Fix a memory leak in an async Python WebSocket handler when client disconnects unexpectedly."
    
    response = client.chat.completions.create(
        model="wormhole-auto",
        messages=[
            {"role": "system", "content": "You are an automated code repair agent."},
            {"role": "user", "content": prompt}
        ]
    )
    
    meta = getattr(response, "wormhole_metadata", {})
    print(f"  • Selected Model:  {response.model}")
    print(f"  • Completion:      {response.choices[0].message.content[:90]}...")
    return response

def run_rag_support_harness():
    print("\n📚 [Agent Harness 2] Running Enterprise Support RAG Workload...")
    prompt = "Summarize employee travel reimbursement policy guidelines for international business trips."
    
    response = client.chat.completions.create(
        model="wormhole-auto",
        messages=[
            {"role": "system", "content": "You are an enterprise HR assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    print(f"  • Selected Model:  {response.model}")
    print(f"  • Completion:      {response.choices[0].message.content[:90]}...")
    return response

def run_data_extraction_harness():
    print("\n📊 [Agent Harness 3] Running Structured Data Extraction Workload...")
    prompt = "Extract user_name, total_amount ($450.00), and date (2026-08-14) from invoice #94021 into JSON format."
    
    response = client.chat.completions.create(
        model="wormhole-auto",
        messages=[
            {"role": "system", "content": "You are a structured data extraction pipeline."},
            {"role": "user", "content": prompt}
        ]
    )
    
    print(f"  • Selected Model:  {response.model}")
    print(f"  • Completion:      {response.choices[0].message.content[:90]}...")
    return response

def main():
    print("="*75)
    print(" 🚀 OPEN-SOURCE ENTERPRISE CLIENT HARNESS DEMO")
    print(" Connecting standard OpenAI Client SDK -> WormHole Gateway")
    print("="*75)
    
    start_time = time.time()
    run_code_repair_harness()
    time.sleep(1)
    run_rag_support_harness()
    time.sleep(1)
    run_data_extraction_harness()
    
    total_time = round(time.time() - start_time, 2)
    print("\n" + "="*75)
    print(f" ✅ All Enterprise Harness Tasks Completed in {total_time}s!")
    print(" Check your Web Dashboard at http://127.0.0.1:8000/ to view live cost savings!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
