# WormHole — Enterprise AI Inference Cost Reducer

**WormHole** is an enterprise AI inference middleware layer designed to drastically reduce LLM API spend while preserving or elevating completion quality. 

![WormHole Automated Demo Recording](docs/wormhole_demo.gif)

It introduces a dual-model intermediate layer between your client application harness and downstream LLM providers:
1. **Prompt Enhancer (Model 1)**: Quality-enriches and structures the user prompt so smaller, cheaper models achieve frontier-quality output.
2. **Router Model (Model 2)**: Evaluates the enhanced prompt against registered enterprise models and dynamically routes it to the lowest-cost model capable of handling the task.

---

## 🏗️ Architecture & System Flow

```mermaid
graph TD
    %% Client & Gateway
    Client["Client Application / Harness"] -->|"1. Standard OpenAI Chat Spec"| GW["WormHole FastAPI Gateway"]
    
    %% Intermediate Layer
    subgraph Intermediate ["WormHole Intermediate Layer"]
        GW -->|"2. Raw Prompt"| Enhancer["Model 1: Prompt Enhancer LLM - Quality Optimization"]
        Enhancer -->|"3. Enhanced Prompt"| Router["Model 2: Router LLM - Cost & Capability Analysis"]
        Router -->|"4. Selected Model & Reasoning"| Dispatcher["Execution Dispatcher & Cost Engine"]
    end
    
    %% Target LLMs Fleet
    subgraph EnterpriseFleet ["Enterprise Candidate Models Fleet"]
        Dispatcher -->|"5. Dispatches Enhanced Prompt"| Downstream{"Chosen Model"}
        Downstream -->|"Option A"| GPT4oMini["GPT-4o Mini - $0.00015 per 1k in"]
        Downstream -->|"Option B"| Flash["Gemini 1.5 Flash - $0.000075 per 1k in"]
        Downstream -->|"Option C"| Haiku["Claude 3 Haiku - $0.00025 per 1k in"]
        Downstream -->|"Option D - Frontier"| GPT4o["GPT-4o / Sonnet 3.5 - $0.0025 per 1k in"]
    end

    %% Response Delivery
    GPT4oMini -->|"Completion"| Dispatcher
    Flash -->|"Completion"| Dispatcher
    Haiku -->|"Completion"| Dispatcher
    GPT4o -->|"Completion"| Dispatcher
    
    Dispatcher -->|"6. Completion + Cost Metadata"| GW
    GW -->|"7. Response Payload"| Client

    %% Async Feedback & Training Loop
    subgraph FeedbackLoop ["Learning & Auto-Evaluation Loop"]
        Dispatcher -.->|"8. Async Background Task"| Judge["LLM-as-a-Judge Auto-Evaluator"]
        Judge -->|"9. Quality Score 1.0 - 10.0"| DB[("SQLite Database - InferenceLogs")]
        DB -->|"10. Export Dataset JSONL"| FineTuning["Model Fine-Tuning Pipeline"]
    end
```

---

## 🔄 Step-by-Step Processing Pipeline

### 1. Request Ingestion (`/v1/chat/completions`)
Client applications send standard OpenAI-formatted completion payloads. WormHole acts as a drop-in replacement proxy.

### 2. Prompt Enhancer (LLM Model 1)
- **File**: [`services/enhancer.py`](file:///Users/venkat/Documents/AI/WormHole/services/enhancer.py)
- Takes terse or unformatted user prompts and enriches them with clear instructions, expected output structure, and edge-case handling guidelines.
- **Goal**: Quality maximization—ensuring that lower-cost downstream models receive sufficient context to output top-tier code/text.

### 3. Router Decision (LLM Model 2)
- **File**: [`services/router.py`](file:///Users/venkat/Documents/AI/WormHole/services/router.py)
- Inspects the enhanced prompt and matches its complexity against the registered fleet of enterprise models and their token pricing.
- Outputs structured JSON specifying `selected_model` and `reasoning`.

### 4. Dispatch & Cost Accounting
- **File**: [`services/dispatcher.py`](file:///Users/venkat/Documents/AI/WormHole/services/dispatcher.py)
- Executes the API request via **LiteLLM**.
- Calculates actual input/output token cost vs. baseline cost (cost if routed to `gpt-4o`) and tracks exact dollar and percentage savings.

### 5. Asynchronous Auto-Evaluation (LLM-as-a-Judge)
- **File**: [`services/judge.py`](file:///Users/venkat/Documents/AI/WormHole/services/judge.py)
- Asynchronously grades completion quality on a scale of **1.0 to 10.0**.
- Persists judge score, feedback, latency, and costs to SQLite.

### 6. Learning & Fine-Tuning Dataset Generator
- **File**: [`services/dataset.py`](file:///Users/venkat/Documents/AI/WormHole/services/dataset.py)
- Exports high-scoring historical inferences into JSONL format for fine-tuning custom student models for Model 1 (Enhancer) and Model 2 (Router).

---

## 💰 Candidate Models & Cost Accounting

| Model ID | Provider | Input Cost / 1k | Output Cost / 1k | Intelligence Tier | Typical Speed |
|---|---|---|---|---|---|
| `gpt-4o-mini` | OpenAI | $0.00015 | $0.00060 | Medium | Fast |
| `gemini/gemini-1.5-flash` | Google | $0.000075 | $0.00030 | Medium | Ultra Fast |
| `claude-3-haiku-20240307` | Anthropic | $0.00025 | $0.00125 | Medium | Fast |
| `gemini/gemini-1.5-pro` | Google | $0.00125 | $0.00500 | High | Medium |
| `gpt-4o` | OpenAI | $0.00250 | $0.01000 | Frontier | Medium |
| `claude-3-5-sonnet-20240620` | Anthropic | $0.00300 | $0.01500 | Frontier | Medium |

---

## 🔌 API Endpoints & Interfaces

### 1. OpenAI-Compatible Chat Proxy
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "wormhole-auto",
  "messages": [
    { "role": "user", "content": "Write a Python function to check if a string is a palindrome." }
  ]
}
```

### 2. Analytics & Historical Logs
```http
GET /api/logs
```
Returns summary metrics (total requests, total cost spent, total baseline cost, net savings $, net savings %, average judge score) and recent inference logs.

### 3. Registered Candidate Models
```http
GET /api/models
```

### 4. Fine-Tuning Dataset Export
```http
GET /api/dataset/export?target=router
GET /api/dataset/export?target=enhancer
```

### 5. Enterprise Web Dashboard
Access **`http://127.0.0.1:8000/`** in your browser for a live graphical analytics dashboard.

---

## ⚡ Quickstart & Local Execution

```bash
# Clone & Navigate to Repository
cd /Users/venkat/Documents/AI/WormHole

# Activate Virtual Environment
source .venv/bin/activate

# Install Dependencies
pip install -r requirements.txt

# Run Automated Test Suite
PYTHONPATH=. pytest

# Start Gateway Server
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## 📁 Repository Structure

```
WormHole/
├── config.py             # System configuration, API keys, & candidate model definitions
├── db/
│   ├── database.py       # SQLModel engine & session setup
│   └── models.py         # InferenceLog schema (prompts, costs, judge scores)
├── services/
│   ├── enhancer.py       # Model 1: Quality Prompt Enhancer Service
│   ├── router.py         # Model 2: Router LLM Decision Service
│   ├── judge.py          # LLM-as-a-Judge Auto-Evaluation Service
│   ├── dispatcher.py     # Execution Dispatcher & Real-time Cost Calculation Engine
│   └── dataset.py        # Fine-Tuning JSONL Dataset Exporter
├── tests/
│   ├── test_api.py       # API proxy & analytics endpoint unit tests
│   └── test_components.py # Unit tests for enhancer, router, dispatcher, judge, & exporter
├── main.py               # FastAPI application host, proxy, & Web Dashboard UI
└── requirements.txt      # Python dependencies
```
