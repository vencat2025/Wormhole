import asyncio
import os
import time
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
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
from services.codex_models import build_models_response

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

CAPTURE_PATH = os.getenv("WORMHOLE_CAPTURE_FILE")

if CAPTURE_PATH:
    @app.middleware("http")
    async def capture_raw_requests(request, call_next):
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            body = await request.body()
            with open(CAPTURE_PATH, "a") as f:
                f.write(json.dumps({"path": request.url.path, "body": body.decode("utf-8", "replace")}) + "\n")
        return await call_next(request)

# --- Models for OpenAI API Spec ---
class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any], Dict[str, Any]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "wormhole-auto"  # Default routing keyword
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

ChatMessage.model_rebuild()
ChatCompletionRequest.model_rebuild()

def strip_codex_system_context(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return text

def extract_clean_text(content_obj: Any) -> str:
    res = ""
    if isinstance(content_obj, str):
        res = content_obj
    elif isinstance(content_obj, list):
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
            res = " ".join(texts)
    elif isinstance(content_obj, dict):
        if "text" in content_obj:
            res = str(content_obj["text"])
        elif "content" in content_obj:
            res = extract_clean_text(content_obj["content"])
    else:
        res = str(content_obj)
    return strip_codex_system_context(res)

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
    selected_model, router_reasoning = await route_prompt(original_prompt, has_tools=bool(request.tools))

    # Step 2: Selective Prompt Enhancement (Model 1)
    is_frontier = any(f in selected_model.lower() for f in ["gpt-4o", "sonnet", "gemini-1.5-pro"]) and not "mini" in selected_model.lower()
    
    if is_frontier:
        enhanced_prompt = original_prompt
        router_reasoning += " | Prompt Enhancement Bypassed (Frontier model selected)"
    else:
        enhanced_prompt = await enhance_prompt(original_prompt)
        router_reasoning += " | Selective Prompt Enhancement Applied (Quality boost for budget model)"

    raw_messages = [m.model_dump(exclude_none=True) for m in request.messages]

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
                original_messages=raw_messages,
                tools=request.tools,
                tool_choice=request.tool_choice
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
        original_messages=raw_messages,
        tools=request.tools,
        tool_choice=request.tool_choice
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

def extract_user_prompt(inp: Any) -> str:
    res = ""
    if isinstance(inp, list) and len(inp) > 0:
        user_msgs = [m for m in inp if isinstance(m, dict) and m.get("role") == "user"]
        if user_msgs:
            target = user_msgs[-1].get("content")
            res = extract_clean_text(target)
        else:
            last_item = inp[-1]
            if isinstance(last_item, dict):
                if "content" in last_item:
                    res = extract_clean_text(last_item["content"])
                elif "text" in last_item:
                    res = str(last_item["text"])
                else:
                    res = str(last_item)
            else:
                res = str(last_item)
    else:
        res = extract_clean_text(inp)
    return strip_codex_system_context(res)

def _flatten_codex_tools(tools: List[Any]) -> List[Dict[str, Any]]:
    """Flatten Codex's nested tool tree into a flat OpenAI function-tool list."""
    flat = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type")
        if t_type == "namespace":
            flat.extend(_flatten_codex_tools(t.get("tools") or []))
        elif t_type == "custom":
            # Freeform tools take raw text; expose them as a single string arg
            # so open-weight models can still target them.
            flat.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            })
        elif t_type == "function":
            flat.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                },
            })
    return flat


def convert_responses_input_to_messages(raw_request: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    messages = []
    collected_tools: List[Dict[str, Any]] = []

    # 1. System instructions
    instructions = raw_request.get("instructions")
    if instructions:
        clean_inst = extract_clean_text(instructions)
        if clean_inst:
            messages.append({"role": "system", "content": clean_inst})

    # 2. Input items
    inp = raw_request.get("input", [])
    if isinstance(inp, str):
        messages.append({"role": "user", "content": extract_clean_text(inp)})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": extract_clean_text(item)})
            elif isinstance(item, dict):
                item_type = item.get("type", "message")
                
                if item_type == "additional_tools":
                    # Codex 0.147 nests its tool definitions inside an input
                    # item rather than the top-level `tools` array.
                    collected_tools.extend(_flatten_codex_tools(item.get("tools") or []))

                elif item_type == "message":
                    role = item.get("role", "user")
                    content_raw = item.get("content", "")
                    clean_content = extract_clean_text(content_raw)
                    messages.append({"role": role, "content": clean_content})

                elif item_type == "function_call":
                    call_id = item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:8]}"))
                    name = item.get("name", "")
                    args = item.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args}
                            }
                        ]
                    })
                
                elif item_type == "function_call_output":
                    call_id = item.get("call_id", item.get("id", ""))
                    output = item.get("output", "")
                    clean_output = extract_clean_text(output)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": clean_output or "Tool executed successfully."
                    })
                else:
                    role = item.get("role", "user")
                    content = extract_clean_text(item.get("content", item.get("text", "")))
                    messages.append({"role": role, "content": content})

    elif "messages" in raw_request and isinstance(raw_request["messages"], list):
        for m in raw_request["messages"]:
            if isinstance(m, dict):
                messages.append({"role": m.get("role", "user"), "content": extract_clean_text(m.get("content", ""))})

    # Sanitize messages for provider compatibility (Groq, OpenAI, Anthropic).
    # Codex sends its agent preamble as role "developer", which no provider
    # below accepts; it becomes a system message rather than being dropped.
    sanitized_messages = []
    for m in messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        content = m.get("content")
        tool_calls = m.get("tool_calls")

        if role in ["user", "system", "tool"]:
            msg_dict = {"role": role, "content": str(content) if content else "..."}
            if "tool_call_id" in m:
                msg_dict["tool_call_id"] = m["tool_call_id"]
            sanitized_messages.append(msg_dict)
        elif role == "assistant":
            msg_dict = {"role": "assistant", "content": str(content) if content is not None else ""}
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            sanitized_messages.append(msg_dict)

    # 3. Tools normalization. Codex may supply them at the top level or, since
    # 0.147, nested inside an `additional_tools` input item.
    raw_tools = raw_request.get("tools")
    if raw_tools and isinstance(raw_tools, list):
        collected_tools.extend(_flatten_codex_tools(raw_tools))

    seen_names = set()
    normalized_tools = []
    for t in collected_tools:
        name = (t.get("function") or {}).get("name")
        if name and name not in seen_names:
            seen_names.add(name)
            normalized_tools.append(t)
    normalized_tools = normalized_tools or None

    # 4. Reinforce tool usage. The model catalog already frames the agent role
    # via instructions_template; this names the concrete tools of this turn.
    if normalized_tools:
        exec_tool = next(
            (n for n in ("shell", "exec_command", "local_shell", "bash", "container.exec", "exec")
             if n in seen_names),
            None
        )
        if exec_tool:
            agent_directive = (
                f"\n\nTOOL USE FOR THIS TURN: you have `{exec_tool}` available and it runs in the "
                f"user's real workspace. Any request to create, edit, run, or fix something must be "
                f"carried out by calling `{exec_tool}` now. Code shown in a reply is not a file on "
                f"disk; only the tool call creates one. Available tools: {', '.join(sorted(seen_names))}."
            )
            for m in sanitized_messages:
                if m.get("role") == "system":
                    m["content"] = str(m.get("content", "")) + agent_directive
                    break
            else:
                sanitized_messages.insert(0, {"role": "system", "content": agent_directive.strip()})

    return sanitized_messages, normalized_tools

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
        original_prompt = extract_user_prompt(inp)
    elif "messages" in raw_request:
        msgs = raw_request["messages"]
        if msgs and len(msgs) > 0:
            original_prompt = extract_user_prompt(msgs)

    raw_messages, tools = convert_responses_input_to_messages(raw_request)

    # Step 1: Model 2 - Local Router SLM Decision (<2ms)
    selected_model, router_reasoning = await route_prompt(original_prompt, has_tools=bool(tools))

    # Step 2: Selective Prompt Enhancement (Model 1)
    is_frontier = any(f in selected_model.lower() for f in ["gpt-4o", "sonnet", "gemini-1.5-pro"]) and not "mini" in selected_model.lower()
    if is_frontier:
        enhanced_prompt = original_prompt
        router_reasoning += " | Prompt Enhancement Bypassed (Frontier model selected)"
    else:
        enhanced_prompt = await enhance_prompt(original_prompt)
        router_reasoning += " | Selective Prompt Enhancement Applied (Quality boost for budget model)"

    if not raw_messages:
        raw_messages = [{"role": "user", "content": enhanced_prompt}]

    tool_choice = raw_request.get("tool_choice")

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
                original_messages=raw_messages,
                tools=tools,
                tool_choice=tool_choice
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
        original_messages=raw_messages,
        tools=tools,
        tool_choice=tool_choice
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

# --- Anthropic Messages API Endpoint (For Claude Code CLI) ---
@app.post("/v1/messages")
async def anthropic_messages_endpoint(
    raw_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    auth_token: str = Depends(verify_api_key)
):
    system_prompt = raw_request.get("system", "")
    msgs = raw_request.get("messages", [])
    
    openai_messages = []
    if system_prompt:
        openai_messages.append({"role": "system", "content": extract_clean_text(system_prompt)})
    for m in msgs:
        openai_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    original_prompt = extract_user_prompt(msgs)
    selected_model, router_reasoning = await route_prompt(original_prompt)

    is_frontier = any(f in selected_model.lower() for f in ["gpt-4o", "sonnet", "gemini-1.5-pro"]) and not "mini" in selected_model.lower()
    enhanced_prompt = original_prompt if is_frontier else await enhance_prompt(original_prompt)

    result = await dispatch_inference(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=settings.ENHANCER_MODEL if not is_frontier else "bypassed",
        router_model=settings.ROUTER_MODEL,
        selected_model=selected_model,
        router_reasoning=router_reasoning,
        original_messages=openai_messages,
        tools=raw_request.get("tools"),
        tool_choice=raw_request.get("tool_choice")
    )

    background_tasks.add_task(
        evaluate_completion,
        request_id=result["request_id"],
        enhanced_prompt=enhanced_prompt,
        completion=result["completion"]
    )

    return {
        "id": f"msg_{result['request_id']}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": result["completion"]}],
        "model": selected_model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": result["metrics"]["prompt_tokens"],
            "output_tokens": result["metrics"]["completion_tokens"]
        }
    }

# --- OpenAI-Compatible Models Endpoint ---
@app.get("/v1/models")
def list_v1_models():
    return build_models_response()

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
                <th>⏰ Timestamp</th>
                <th>📥 Original Prompt</th>
                <th>✨ Enhanced Prompt (Model 1 SLM)</th>
                <th>🎯 Target Model & Reasoning</th>
                <th>Cost / Savings</th>
                <th>Judge Score</th>
            </tr>
        </thead>
        <tbody id="logs-body">
            <tr><td colspan="7" style="text-align: center; color: var(--text-sub);">Loading inference logs...</td></tr>
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

        function formatLocalDate(isoStr) {
            if (!isoStr) return 'N/A';
            let str = String(isoStr).trim();
            const hasTimezone = str.endsWith('Z') || /[+-]\d{2}:?\d{2}$/.test(str);
            if (!hasTimezone) {
                str += 'Z';
            }
            const d = new Date(str);
            if (isNaN(d.getTime())) return 'N/A';
            return d.toLocaleString(undefined, {
                year: 'numeric', month: '2-digit', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
                hour12: true
            });
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
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-sub);">No requests logged yet. Send chat completion calls to <code>/v1/chat/completions</code>.</td></tr>';
                    return;
                }

                tbody.innerHTML = data.logs.map(log => {
                    const origPrompt = escapeHtml(log.original_prompt || '');
                    const enhPrompt = escapeHtml(log.enhanced_prompt || '');
                    const selModel = escapeHtml(log.selected_model || '');
                    const reason = escapeHtml(log.router_reasoning || 'N/A');
                    const reqId = escapeHtml(log.request_id || '');
                    const dateStr = formatLocalDate(log.created_at);
                    return `
                    <tr>
                        <td style="font-family: monospace; font-size: 11px; color: var(--text-sub);">${reqId}</td>
                        <td style="font-size: 11px; color: #cbd5e1; white-space: nowrap;">${dateStr}</td>
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
