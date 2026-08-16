import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Security, Response
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import Session, select, func

from config import settings
from db.database import init_db, engine
from db.models import InferenceLog
from services.enhancer import enhance_prompt
from services.router import route_prompt
from services.dispatcher import (
    dispatch_inference,
    dispatch_streaming_inference,
    dispatch_responses_streaming_inference
)
from services.judge import evaluate_completion
from services.dataset import export_dataset_jsonl
from services.auth import verify_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wormhole.main")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("WormHole DB Initialized successfully.")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise AI Inference Middleware Gateway for Cost Optimization & Quality Enhancement.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models for OpenAI API Spec ---
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any], Dict[str, Any]]] = ""

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "wormhole-auto"  # Default routing keyword
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

ChatMessage.model_rebuild()
ChatCompletionRequest.model_rebuild()

def extract_clean_text(content_obj: Any) -> str:
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, list):
        texts = []
        for item in content_obj:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    texts.append(str(item["text"]))
                elif "content" in item:
                    texts.append(extract_clean_text(item["content"]))
        if texts:
            return " ".join(texts)
    if isinstance(content_obj, dict):
        if "text" in content_obj:
            return str(content_obj["text"])
        if "content" in content_obj:
            return extract_clean_text(content_obj["content"])
    return str(content_obj)

# --- Core Gateway Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    auth_token: str = Depends(verify_api_key)
):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")
    
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raw_prompt = request.messages[-1].content
    else:
        raw_prompt = user_messages[-1].content

    original_prompt = extract_clean_text(raw_prompt)

    # Step 1: Model 2 - Local Router SLM Decision (<2ms)
    selected_model, router_reasoning = await route_prompt(original_prompt)

    # Step 2: Selective Prompt Enhancement (Model 1)
    is_frontier = any(f in selected_model.lower() for f in ["gpt-4o", "sonnet", "gemini-1.5-pro"]) and not "mini" in selected_model.lower()
    
    if is_frontier:
        enhanced_prompt = original_prompt
        router_reasoning += " | Prompt Enhancement Bypassed (Frontier model selected)"
    else:
        enhanced_prompt = await enhance_prompt(original_prompt)
        router_reasoning += " | Selective Prompt Enhancement Applied (Quality boost for budget model)"

    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Handle SSE token streaming if stream=True
    if request.stream:
        return StreamingResponse(
            dispatch_streaming_inference(
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt,
                enhancer_model=settings.ENHANCER_MODEL if not is_frontier else "bypassed",
                router_model=settings.ROUTER_MODEL,
                selected_model=selected_model,
                router_reasoning=router_reasoning,
                original_messages=raw_messages
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Step 3: Execution Dispatcher & Cost Metric Computation (Synchronous JSON)
    result = await dispatch_inference(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=settings.ENHANCER_MODEL if not is_frontier else "bypassed",
        router_model=settings.ROUTER_MODEL,
        selected_model=selected_model,
        router_reasoning=router_reasoning,
        original_messages=raw_messages
    )

    # Step 4: Asynchronous LLM-as-a-Judge Auto-Evaluation Task
    background_tasks.add_task(
        evaluate_completion,
        request_id=result["request_id"],
        enhanced_prompt=enhanced_prompt,
        completion=result["completion"]
    )

    # Format OpenAI-compatible completion response
    response_payload = {
        "id": f"chatcmpl-{result['request_id']}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": selected_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["completion"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": result["metrics"]["prompt_tokens"],
            "completion_tokens": result["metrics"]["completion_tokens"],
            "total_tokens": result["metrics"]["total_tokens"]
        },
        "wormhole_metadata": {
            "request_id": result["request_id"],
            "enhancer_model": settings.ENHANCER_MODEL,
            "router_model": settings.ROUTER_MODEL,
            "selected_model": selected_model,
            "router_reasoning": router_reasoning,
            "actual_cost_usd": result["metrics"]["actual_cost_usd"],
            "baseline_cost_usd": result["metrics"]["baseline_cost_usd"],
            "cost_savings_usd": result["metrics"]["cost_savings_usd"],
            "savings_percentage": f"{result['metrics']['savings_percentage']}%"
        }
    }
    
    return response_payload

# --- OpenAI Responses API Endpoint (For Codex CLI & Agentic Tools) ---
@app.post("/v1/responses")
async def openai_responses_endpoint(
    raw_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    auth_token: str = Depends(verify_api_key)
):
    original_prompt = "Hello"
    if "input" in raw_request:
        inp = raw_request["input"]
        original_prompt = extract_clean_text(inp)
    elif "messages" in raw_request:
        msgs = raw_request["messages"]
        if msgs and len(msgs) > 0:
            original_prompt = extract_clean_text(msgs[-1].get("content", "Hello"))

    # Step 1: Model 2 - Local Router SLM Decision (<2ms)
    selected_model, router_reasoning = await route_prompt(original_prompt)

    # Step 2: Selective Prompt Enhancement (Model 1)
    is_frontier = any(f in selected_model.lower() for f in ["gpt-4o", "sonnet", "gemini-1.5-pro"]) and not "mini" in selected_model.lower()
    if is_frontier:
        enhanced_prompt = original_prompt
        router_reasoning += " | Prompt Enhancement Bypassed (Frontier model selected)"
    else:
        enhanced_prompt = await enhance_prompt(original_prompt)
        router_reasoning += " | Selective Prompt Enhancement Applied (Quality boost for budget model)"

    raw_messages = [{"role": "user", "content": enhanced_prompt}]

    # Handle streaming for OpenAI Codex CLI (v0.142+)
    if raw_request.get("stream", True):
        return StreamingResponse(
            dispatch_responses_streaming_inference(
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt,
                enhancer_model=settings.ENHANCER_MODEL if not is_frontier else "bypassed",
                router_model=settings.ROUTER_MODEL,
                selected_model=selected_model,
                router_reasoning=router_reasoning,
                original_messages=raw_messages
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Step 3: Execution Dispatcher (Synchronous JSON)
    result = await dispatch_inference(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=settings.ENHANCER_MODEL if not is_frontier else "bypassed",
        router_model=settings.ROUTER_MODEL,
        selected_model=selected_model,
        router_reasoning=router_reasoning,
        original_messages=raw_messages
    )

    background_tasks.add_task(
        evaluate_completion,
        request_id=result["request_id"],
        enhanced_prompt=enhanced_prompt,
        completion=result["completion"]
    )

    completion_text = result["completion"]
    
    response_payload = {
        "id": f"resp-{result['request_id']}",
        "object": "response",
        "created_at": int(time.time()),
        "created": int(time.time()),
        "status": "completed",
        "model": selected_model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": completion_text
                    }
                ]
            }
        ],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "input_tokens": result["metrics"]["prompt_tokens"],
            "output_tokens": result["metrics"]["completion_tokens"],
            "total_tokens": result["metrics"]["total_tokens"]
        },
        "wormhole_metadata": result["metrics"]
    }
    return response_payload

# --- OpenAI-Compatible Models Endpoint ---
@app.get("/v1/models")
def list_v1_models():
    reasoning_presets = [
        {"effort": "none", "description": "No reasoning"},
        {"effort": "low", "description": "Low reasoning"},
        {"effort": "medium", "description": "Medium reasoning"},
        {"effort": "high", "description": "High reasoning"},
        {"effort": "xhigh", "description": "Extra High reasoning"}
    ]
    models_list = [
        {
            "id": "wormhole-auto",
            "name": "WormHole Auto Router",
            "slug": "wormhole-auto",
            "display_name": "WormHole Auto Router",
            "description": "Sub-2ms SLM Auto Router & Selective Enhancer",
            "base_instructions": "You are an expert AI software engineer.",
            "priority": 0,
            "truncation_policy": {"mode": "tokens", "limit": 4096},
            "support_verbosity": False,
            "supported_in_api": True,
            "visibility": "list",
            "shell_type": "default",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "wormhole",
            "supports_parallel_tool_calls": True,
            "supports_reasoning_summaries": True,
            "supports_multiline": False,
            "supports_tools": True,
            "supports_images": False,
            "supported_reasoning_levels": reasoning_presets
        },
        {
            "id": "gpt-5.5",
            "name": "GPT-5.5",
            "slug": "gpt-5.5",
            "display_name": "GPT-5.5",
            "description": "OpenAI GPT-5.5 Agentic Coding Model",
            "base_instructions": "You are an expert AI software engineer.",
            "priority": 0,
            "truncation_policy": {"mode": "tokens", "limit": 4096},
            "support_verbosity": False,
            "supported_in_api": True,
            "visibility": "list",
            "shell_type": "default",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "wormhole",
            "supports_parallel_tool_calls": True,
            "supports_reasoning_summaries": True,
            "supports_multiline": False,
            "supports_tools": True,
            "supports_images": False,
            "supported_reasoning_levels": reasoning_presets
        },
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "slug": "gpt-4o",
            "display_name": "GPT-4o",
            "description": "OpenAI GPT-4o Frontier Model",
            "base_instructions": "You are an expert AI software engineer.",
            "priority": 0,
            "truncation_policy": {"mode": "tokens", "limit": 4096},
            "support_verbosity": False,
            "supported_in_api": True,
            "visibility": "list",
            "shell_type": "default",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "wormhole",
            "supports_parallel_tool_calls": True,
            "supports_reasoning_summaries": False,
            "supports_tools": True,
            "supports_images": False,
            "supported_reasoning_levels": reasoning_presets
        },
        {
            "id": "gpt-4o-mini",
            "name": "GPT-4o Mini",
            "slug": "gpt-4o-mini",
            "display_name": "GPT-4o Mini",
            "description": "OpenAI GPT-4o-mini Budget Model",
            "base_instructions": "You are an expert AI software engineer.",
            "priority": 0,
            "truncation_policy": {"mode": "tokens", "limit": 4096},
            "support_verbosity": False,
            "supported_in_api": True,
            "visibility": "list",
            "shell_type": "default",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "wormhole",
            "supports_reasoning_summaries": False,
            "supports_tools": True,
            "supports_images": False,
            "supported_reasoning_levels": reasoning_presets
        },
        {
            "id": "claude-3-5-sonnet-20240620",
            "name": "Claude 3.5 Sonnet",
            "slug": "claude-3-5-sonnet-20240620",
            "display_name": "Claude 3.5 Sonnet",
            "description": "Anthropic Claude 3.5 Sonnet",
            "visibility": "public",
            "shell_type": "default",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "wormhole",
            "supports_reasoning_summaries": False,
            "supports_tools": True,
            "supports_images": False,
            "supported_reasoning_levels": reasoning_presets
        }
    ]
    return {
        "object": "list",
        "data": models_list,
        "models": models_list
    }

# --- Enterprise Admin & Analytics APIs ---
@app.get("/api/models")
def list_candidate_models():
    return {"models": settings.CANDIDATE_MODELS}

@app.get("/api/logs")
def list_logs(limit: int = 50, offset: int = 0):
    with Session(engine) as session:
        statement = select(InferenceLog).order_by(InferenceLog.id.desc()).offset(offset).limit(limit)
        logs = session.exec(statement).all()
        
        # Summary Analytics
        total_requests = session.exec(select(func.count(InferenceLog.id))).one()
        total_actual_cost = session.exec(select(func.sum(InferenceLog.actual_cost))).one() or 0.0
        total_baseline_cost = session.exec(select(func.sum(InferenceLog.baseline_cost))).one() or 0.0
        total_savings = session.exec(select(func.sum(InferenceLog.cost_savings))).one() or 0.0
        avg_score = session.exec(select(func.avg(InferenceLog.judge_score))).one() or 0.0

        return {
            "summary": {
                "total_requests": total_requests,
                "total_actual_cost_usd": round(total_actual_cost, 4),
                "total_baseline_cost_usd": round(total_baseline_cost, 4),
                "total_savings_usd": round(total_savings, 4),
                "savings_percentage": round((total_savings / max(total_baseline_cost, 0.0001)) * 100, 1),
                "average_judge_score": round(avg_score, 2)
            },
            "logs": logs
        }

@app.get("/api/dataset/export")
def export_dataset(target: str = "router", min_score: float = 7.0):
    jsonl_content = export_dataset_jsonl(target_type=target, min_score=min_score)
    return PlainTextResponse(
        content=jsonl_content,
        media_type="application/x-jsonlines",
        headers={"Content-Disposition": f'attachment; filename="wormhole_{target}_dataset.jsonl"'}
    )

@app.post("/api/router/retrain")
def retrain_models_from_feedback():
    from models.train_router import train_router_slm
    from models.train_quality_evaluator import train_quality_evaluator_slm
    
    try:
        train_router_slm()
        train_quality_evaluator_slm()
        return {"status": "success", "message": "Local Router SLM and Quality Evaluator SLM successfully retrained from latest database completions and judge feedback."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# --- Admin Dashboard UI ---
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WormHole | AI Inference Cost Reducer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: #111827;
            --border: #1f2937;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.2);
            --green: #10b981;
            --green-glow: rgba(16, 185, 129, 0.15);
            --text: #f9fafb;
            --text-sub: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 20px; font-weight: 700; color: #fff; }
        .badge { background: linear-gradient(135deg, var(--accent), #8b5cf6); padding: 4px 10px; borderRadius: 12px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }
        .card-title { font-size: 13px; font-weight: 500; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .card-value { font-size: 28px; font-weight: 700; color: #fff; }
        .card-sub { font-size: 13px; color: var(--green); margin-top: 4px; font-weight: 500; }

        .section-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
        .export-btn { background: var(--panel); border: 1px solid var(--accent); color: #fff; padding: 8px 16px; border-radius: 8px; font-size: 13px; cursor: pointer; text-decoration: none; transition: background 0.2s; }
        .export-btn:hover { background: var(--accent); }

        table { width: 100%; border-collapse: collapse; background: var(--panel); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
        th, td { padding: 14px 16px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--border); }
        th { background: #1f2937; font-weight: 600; color: var(--text-sub); text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
        tr:hover { background: rgba(255,255,255,0.02); }
        
        .tag { display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
        .tag-model { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
        .tag-score { background: var(--green-glow); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
        .prompt-preview { max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-sub); }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            ⚡ WormHole <span class="badge">Enterprise Gateway</span>
        </div>
        <div>
            <button onclick="triggerRetrain()" class="export-btn" style="background: rgba(16, 185, 129, 0.2); border-color: var(--green); margin-right: 8px;">🔄 Retrain Local SLMs</button>
            <a href="/api/dataset/export?target=router" class="export-btn">📥 Export Router Dataset (JSONL)</a>
            <a href="/api/dataset/export?target=enhancer" class="export-btn" style="margin-left:8px;">📥 Export Enhancer Dataset (JSONL)</a>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="card">
            <div class="card-title">Total Requests</div>
            <div class="card-value" id="total-requests">0</div>
            <div class="card-sub">Middleware Active</div>
        </div>
        <div class="card">
            <div class="card-title">Total Cost Savings</div>
            <div class="card-value" id="total-savings" style="color: var(--green);">$0.0000</div>
            <div class="card-sub" id="savings-pct">0.0% Saved vs GPT-4o Baseline</div>
        </div>
        <div class="card">
            <div class="card-title">Actual API Spend</div>
            <div class="card-value" id="actual-spend">$0.0000</div>
            <div class="card-sub" id="baseline-spend">Baseline: $0.0000</div>
        </div>
        <div class="card">
            <div class="card-title">Avg Judge Score</div>
            <div class="card-value" id="avg-score">0.0 / 10</div>
            <div class="card-sub">Auto LLM-as-a-Judge Score</div>
        </div>
    </div>

    <div class="section-title">
        <span>Inference Traffic & Routing Decisions</span>
    </div>

    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>📥 Original Prompt</th>
                <th>✨ Enhanced Prompt (Model 1 SLM)</th>
                <th>🎯 Target Model & Reasoning</th>
                <th>Cost / Savings</th>
                <th>Judge Score</th>
            </tr>
        </thead>
        <tbody id="logs-body">
            <tr><td colspan="6" style="text-align: center; color: var(--text-sub);">Loading inference logs...</td></tr>
        </tbody>
    </table>

    <script>
        function escapeHtml(str) {
            if (!str) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        async function fetchAnalytics() {
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                const s = data.summary;
                
                document.getElementById('total-requests').innerText = s.total_requests;
                document.getElementById('total-savings').innerText = `$${s.total_savings_usd.toFixed(4)}`;
                document.getElementById('savings-pct').innerText = `${s.savings_percentage}% Saved vs Baseline`;
                document.getElementById('actual-spend').innerText = `$${s.total_actual_cost_usd.toFixed(4)}`;
                document.getElementById('baseline-spend').innerText = `Baseline (GPT-4o): $${s.total_baseline_cost_usd.toFixed(4)}`;
                document.getElementById('avg-score').innerText = `${s.average_judge_score} / 10`;

                const tbody = document.getElementById('logs-body');
                if (data.logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-sub);">No requests logged yet. Send chat completion calls to <code>/v1/chat/completions</code>.</td></tr>';
                    return;
                }

                tbody.innerHTML = data.logs.map(log => {
                    const origPrompt = escapeHtml(log.original_prompt || '');
                    const enhPrompt = escapeHtml(log.enhanced_prompt || '');
                    const selModel = escapeHtml(log.selected_model || '');
                    const reason = escapeHtml(log.router_reasoning || 'N/A');
                    const reqId = escapeHtml(log.request_id || '');
                    return `
                    <tr>
                        <td style="font-family: monospace; font-size: 11px; color: var(--text-sub);">${reqId}</td>
                        <td>
                            <div class="prompt-preview" title="${origPrompt}">${origPrompt}</div>
                        </td>
                        <td>
                            <details style="font-size: 12px; color: #a5b4fc; cursor: pointer;">
                                <summary style="font-weight: 500;">View Enhanced Prompt</summary>
                                <div style="margin-top: 6px; padding: 8px; background: rgba(0,0,0,0.4); border-radius: 6px; white-space: pre-wrap; font-family: monospace; font-size: 11px; color: #e0e7ff; max-width: 320px; max-height: 150px; overflow-y: auto;">${enhPrompt}</div>
                            </details>
                        </td>
                        <td>
                            <span class="tag tag-model">${selModel}</span>
                            <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px; max-width: 220px;">${reason}</div>
                        </td>
                        <td>
                            <div style="font-weight: 600;">$${log.actual_cost.toFixed(6)}</div>
                            <div style="font-size: 11px; color: var(--green);">Saved $${log.cost_savings.toFixed(6)}</div>
                        </td>
                        <td>
                            ${log.judge_score !== null ? `<span class="tag tag-score">★ ${log.judge_score.toFixed(1)}/10</span>` : '<span style="color:var(--text-sub); font-size:11px;">Pending...</span>'}
                        </td>
                    </tr>
                `;
                }).join('');
            } catch (err) {
                console.error("Error loading analytics", err);
            }
        }

        async function triggerRetrain() {
            try {
                alert("⚡ Retraining local SLMs from latest DB completions and judge feedback...");
                const res = await fetch('/api/router/retrain', { method: 'POST' });
                const data = await res.json();
                alert(`✅ Retraining Complete!\n${data.message}`);
                fetchAnalytics();
            } catch (err) {
                alert("❌ Retraining failed: " + err);
            }
        }

        fetchAnalytics();
        setInterval(fetchAnalytics, 5000);
    </script>
</body>
</html>
"""
