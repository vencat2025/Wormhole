# WormHole — Enterprise AI Inference Cost Reducer

**WormHole** is an enterprise AI inference middleware layer designed to drastically reduce LLM API spend while preserving or elevating completion quality. 

![WormHole Automated Demo Recording](docs/wormhole_demo.gif)

It introduces a dual-model intermediate layer between your client application harness and downstream LLM providers:
1. **Prompt Enhancer (Model 1)**: Quality-enriches and structures the user prompt so smaller, cheaper models achieve frontier-quality output.
2. **Router Model (Model 2)**: Evaluates the enhanced prompt against registered enterprise models and dynamically routes it to the lowest-cost model capable of handling the task.

---

## 🏗️ Architecture & System Flow

## ⚡ 60-Second Explanation (How to Explain to a Friend)

> **Think of WormHole like a Smart Gateway for AI Coding Agents.**
> When you ask Codex CLI to *"Create an app"* or *"Fix a bug"*, instead of sending that request directly to expensive $0.03/req models (like GPT-4o), WormHole intercepts it:
> 1. **<2ms Router**: Picks the absolute fastest, cheapest open model (like Groq GPT-OSS-120B).
> 2. **<1ms Enhancer**: Enriches your prompt with explicit context so cheap models behave like frontier models.
> 3. **0ms Reasoning Suppressor**: Bypasses slow `<think>` tags so completions stream instantly in <1s.
> 4. **Stream Tool Engine**: Automatically turns code output into native terminal commands (`exec`, `mkdir`, `cat << 'EOF' > file`) so Codex CLI creates files live in your workspace.

```mermaid
flowchart LR
    User["👤 User in Codex CLI\n'Create an image app'"] --> GW["⚡ WormHole Gateway"]
    
    subgraph WormHoleCore ["WormHole Sub-2ms Intelligence Core"]
        GW --> Router["1. Router SLM (<2ms)\nPicks lowest cost model"]
        Router --> Enhancer["2. Enhancer SLM (<1ms)\nQuality-enriches prompt"]
        Enhancer --> Suppressor["3. Reasoning Suppressor\nHides <think> tags (0ms delay)"]
    end
    
    Suppressor --> LPUs["🚀 Groq LPU Cloud\n(GPT-OSS-120B / Qwen 3.6)"]
    LPUs --> StreamEngine["4. Stream Regex Engine\nConverts code blocks -> tool_calls"]
    StreamEngine --> Terminal["💻 Codex CLI Terminal\nExecutes mkdir & writes files live!"]
```

---

## 🏗️ Architecture & System Flow

```mermaid
graph TD
    %% Client & Gateway
    Client["Client Application / Codex CLI"] -->|"1. Standard OpenAI & Responses API Spec"| GW["WormHole FastAPI Gateway"]
    
    %% Intermediate Layer
    subgraph Intermediate ["WormHole Intermediate Layer"]
        GW -->|"2. Raw Prompt"| Router["Model 2: Router SLM (<2ms)"]
        Router -->|"3. Selected Model & Reasoning"| Decision{"Is Budget/Mid-Tier Model?"}
        Decision -->|"Yes (Budget/Mid-Tier)"| Enhancer["Model 1: Prompt Enhancer SLM (<1ms Quality Boost)"]
        Decision -->|"No (Frontier Model)"| Dispatcher["Execution Dispatcher & Cost Engine"]
        Enhancer -->|"Enhanced Prompt + reasoning_format='hidden'"| Dispatcher
    end
    
    %% Target LLMs Fleet
    subgraph EnterpriseFleet ["Enterprise Candidate Models Fleet (Groq LPUs + Cloud)"]
        Dispatcher -->|"4. Dispatches Prompt"| Downstream{"Chosen Model"}
        Downstream -->|"Option A (Ultra-Fast 120B)"| GPTOSS120B["GPT OSS 120B (Groq) - $0.00015 per 1k in"]
        Downstream -->|"Option B (Fast 20B)"| GPTOSS20B["GPT OSS 20B (Groq) - $0.000075 per 1k in"]
        Downstream -->|"Option C (Coding 27B)"| QWEN["Qwen 3.6 27B (Groq) - $0.00010 per 1k in"]
        Downstream -->|"Option D - Frontier"| GPT4o["GPT-4o / Sonnet 3.5 - $0.0025 per 1k in"]
    end

    %% Response Delivery
    GPTOSS120B -->|"Completion Stream"| StreamEngine["Stream Tool Conversion Engine"]
    GPTOSS20B -->|"Completion Stream"| StreamEngine
    QWEN -->|"Completion Stream"| StreamEngine
    GPT4o -->|"Completion Stream"| StreamEngine
    
    StreamEngine -->|"5. Native function_call Events (mkdir/write)"| GW
    GW -->|"6. Response SSE Events"| Client

    %% Async Feedback & Training Loop
    subgraph FeedbackLoop ["Learning & Auto-Evaluation Loop"]
        Dispatcher -.->|"7. Async Background Task"| Judge["LLM-as-a-Judge Auto-Evaluator"]
        Judge -->|"8. Quality Score 1.0 - 10.0"| DB[("SQLite Database - InferenceLogs")]
        DB -->|"9. Export Dataset JSONL"| FineTuning["Model Fine-Tuning Pipeline"]
    end
```

---

## 🔄 Step-by-Step Processing Pipeline

### 1. Request Ingestion (`/v1/responses` & `/v1/chat/completions`)
Client applications send standard OpenAI-formatted completion payloads. WormHole acts as a drop-in replacement proxy for Codex CLI and web apps.

### 2. Router Decision (Model 2 Local SLM)
- **File**: [`services/router.py`](file:///Users/venkat/Documents/AI/WormHole/services/router.py)
- Evaluates raw prompt complexity in **< 2 milliseconds** against registered enterprise models and benchmark capability profiles.

### 3. Selective Prompt Enhancement (Model 1 Local SLM)
- **File**: [`services/enhancer.py`](file:///Users/venkat/Documents/AI/WormHole/services/enhancer.py)
- **If a Frontier Model is selected** (`gpt-4o`, `claude-3-5-sonnet`): Prompt enhancement is **bypassed** to save unnecessary latency and token overhead.
- **If a Budget/Mid-Tier Model is selected** (`groq/openai/gpt-oss-120b`, `groq/qwen/qwen3.6-27b`): Model 1 quality-enriches the prompt in **< 1 millisecond** so the budget model outputs frontier-level completions.

### 4. Reasoning Suppression (`reasoning_format="hidden"`)
- **File**: [`services/dispatcher.py`](file:///Users/venkat/Documents/AI/WormHole/services/dispatcher.py)
- Bypasses internal reasoning/thinking tags (`<think> ... </think>`) on Groq LPUs, eliminating 12s+ token delays and streaming output code blocks immediately from Token #1.

### 5. Stream Tool Conversion Engine
- **File**: [`services/dispatcher.py`](file:///Users/venkat/Documents/AI/WormHole/services/dispatcher.py)
- Intercepts streaming code blocks (`# app.py`, `<!-- templates/index.html -->`, `<exec>`) and converts them in real time into native OpenAI Responses API `function_call` events (`exec`, `mkdir`, `cat << 'EOF' > file`).
- Codex CLI receives native execution events and creates files live in your workspace.

### 6. Asynchronous Auto-Evaluation (LLM-as-a-Judge)
- **File**: [`services/judge.py`](file:///Users/venkat/Documents/AI/WormHole/services/judge.py)
- Asynchronously grades completion quality on a scale of **1.0 to 10.0**.
- Persists judge score, feedback, latency, and costs to SQLite database.

### 7. Continuous SLM Self-Improvement & Dataset Exporter
- **File**: [`services/dataset.py`](file:///Users/venkat/Documents/AI/WormHole/services/dataset.py)
- Filters high-scoring historical completions (`judge_score >= 7.0`) and exports formatted JSONL fine-tuning datasets for:
  - **Model 1 (Prompt Enhancer SLM)**: `(Original Prompt → Quality Enhanced Prompt)`
  - **Model 2 (Router SLM)**: `(Enhanced Prompt → Selected Model ID & Reasoning)`
- Used to continuously fine-tune local student SLMs (`models/router_slm.joblib`), making routing completely local with **<2ms latency** and **$0 API cost**.

```mermaid
flowchart TD
    Inference["1. Live Request Executed & Delivered"] --> Judge["2. LLM-as-a-Judge Auto-Evaluator\nAsync Background Task (services/judge.py)"]
    Judge --> Score["3. Grades Quality (1.0 - 10.0)\n& Stores in SQLite DB"]
    
    subgraph DataPipeline ["Dataset Exporter Engine (services/dataset.py)"]
        Score --> Filter{"Filter Quality\nScore >= 7.0?"}
        Filter -->|"Yes (High Quality)"| JSONL["4. Export JSONL Fine-Tuning Datasets\n• Enhancer Dataset\n• Router Dataset"]
        Filter -->|"No"| Discard["Discard Bad Log"]
    end

    JSONL --> Training["5. Model Fine-Tuning Pipeline\nTrains Student SLMs"]
    Training --> LocalSLM["6. Update Local SLM (<2ms)\nmodels/router_slm.joblib"]
    LocalSLM --> Influx["7. Powers Sub-2ms Local Routing\n$0 Cost & Zero Latency!"]
```

---

## 💰 Candidate Models & Cost Accounting

| Model ID | Provider | Input Cost / 1k | Output Cost / 1k | Intelligence Tier | Typical Speed |
|---|---|---|---|---|---|
| `groq/openai/gpt-oss-120b` | Groq LPU | $0.00015 | $0.00060 | Frontier | Ultra Fast (<0.8s) |
| `groq/openai/gpt-oss-20b` | Groq LPU | $0.00075 | $0.00030 | High | Ultra Fast (<0.5s) |
| `groq/qwen/qwen3.6-27b` | Groq LPU | $0.00010 | $0.00040 | High | Fast (<1.0s) |
| `gpt-4o-mini` | OpenAI | $0.00015 | $0.00060 | Medium | Fast |
| `gpt-4o` | OpenAI | $0.00250 | $0.01000 | Frontier | Medium |

---

## ⚔️ Competitive Analysis & Differentiation

For detailed feature comparison matrices, technological moats, and enterprise TCO breakdowns versus OpenRouter, Portkey, RouteLLM, Martian, and LiteLLM Enterprise, see **[COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md)**.

### Summary Comparison Matrix:

| Feature | ⚡ **WormHole** | 🔌 **OpenRouter** | 🔑 **Portkey** | 🔬 **RouteLLM** |
|---|---|---|---|---|
| **Model 1: Prompt Enhancer** | **Yes (Local SLM <1ms)** | ❌ No | ❌ No | ❌ No |
| **Model 2: Router Engine** | **Yes (Local SLM <2ms, $0 Cost)** | Static user lists | Rule-based configs | Matrix router |
| **Auto-Judge Feedback** | **Yes (1.0-10.0 scale)** | ❌ No | ❌ No | ❌ No |
| **Fine-Tuning Flywheel** | **Yes (JSONL Export & `/api/router/retrain`)** | ❌ No | ❌ No | ❌ No |
| **Deployment Model** | **100% Private VPC / Self-Hosted** | Third-party Cloud SaaS | Cloud / Self-Hosted | Open Source |

---

## 🛠️ Custom AI Harness Integration (Claude Code, Cursor, Codex, Aider)

WormHole acts as a seamless drop-in replacement proxy for all custom developer coding harnesses and AI tools. For complete step-by-step setup guides, see **[HARNESS_INTEGRATION_GUIDE.md](HARNESS_INTEGRATION_GUIDE.md)**.

- **Claude Code CLI**: `export ANTHROPIC_BASE_URL="http://127.0.0.1:8000/v1"`
- **Cursor IDE / VS Code**: Set `OpenAI Base URL` to `http://127.0.0.1:8000/v1`
- **Aider CLI / Continue.dev**: Point API base to `http://127.0.0.1:8000/v1`

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
