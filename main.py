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

from config import settings, TIER_ORDER
from db.database import init_db, engine
from db.models import InferenceLog, RoutingDecision
from services.enhancer import enhance_prompt
from services.router import route_prompt, route_among
from services.dispatcher import (
    dispatch_inference,
    dispatch_streaming_inference,
    dispatch_responses_streaming_inference,
    dispatch_anthropic_streaming_inference
)
from services.judge import evaluate_completion
from services.dataset import export_dataset_jsonl
from services.auth import verify_api_key
from services.codex_models import build_models_response
from services.anthropic_api import convert_anthropic_messages, build_anthropic_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wormhole.main")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("WormHole DB Initialized successfully.")
    from services.dispatcher import _load_exhausted_providers, is_model_routable
    _load_exhausted_providers()

    # Say plainly, at startup, what this instance can actually serve. Without
    # this a first-time user who set up only a local model sees a coding agent
    # fail on its first turn with a routing error, and has no way to tell that
    # the cause is configuration rather than a bug.
    chat = [m.id for m in settings.CANDIDATE_MODELS if is_model_routable(m.id)]
    agentic = [m.id for m in settings.CANDIDATE_MODELS if is_model_routable(m.id, need_tools=True)]
    if not chat:
        logger.error(
            "No model is reachable. Add at least one provider key to .env "
            "(GROQ_API_KEY is free to obtain), or start Ollama for local models."
        )
    elif not agentic:
        logger.warning(
            "Chat works (%s), but no tool-capable model is reachable, so Codex, "
            "Claude Code and OpenCode will fail on their first turn. Local models "
            "cannot drive an agentic harness; add a cloud provider key to .env.",
            ", ".join(chat),
        )
    else:
        logger.info("Routing over %d models (%d can drive coding agents).",
                    len(chat), len(agentic))
        if settings.MIN_ROUTING_TIER:
            logger.info("Quality floor: nothing below '%s' will be chosen.",
                        settings.MIN_ROUTING_TIER)

        # A classifier that predicts tiers is fleet-independent, so there is
        # nothing to warn about. One that predicts model ids is not: restrict
        # the fleet to models outside its training set and every prediction
        # misses. Worth saying out loud, because the symptom is bad routing
        # rather than an error.
        if settings.ROUTER_MODE != "llm":
            from services.router import _get_local_router_slm
            slm = _get_local_router_slm()
            known = {str(c) for c in getattr(slm, "classes_", [])} if slm is not None else set()
            if known and not known <= set(TIER_ORDER) and not (known & set(chat)):
                logger.warning(
                    "The local router predicts model ids and was trained on a fleet that is "
                    "no longer reachable (%s), so every prediction falls back to a tier "
                    "substitution. Retrain to get a fleet-independent router: "
                    "python models/train_router.py",
                    ", ".join(sorted(known)[:3]) + ("..." if len(known) > 3 else ""),
                )

        # Routing down only saves money if something cheaper exists. When the
        # cheapest model that can drive a harness already sits at a strong
        # tier, every decision correctly lands on it and the gateway is doing
        # nothing for cost -- which looks like broken routing unless said.
        cheap = [settings.model_config_for(m) for m in agentic]
        cheap = [c for c in cheap if c]
        if cheap:
            floor_model = min(cheap, key=lambda c: c.input_cost_per_1k)
            if floor_model.intelligence_tier.lower() in ("high", "frontier"):
                logger.info(
                    "Cheapest tool-capable model (%s) is already '%s' tier, so most requests "
                    "will route there and cost savings will be small. Add a cheaper capable "
                    "model to widen the spread.",
                    floor_model.id, floor_model.intelligence_tier,
                )
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise AI Inference Middleware Gateway for Cost Optimization & Quality Enhancement.",
    version="1.0.0",
    lifespan=lifespan
)

# Cross-origin access is off by default, and that is a security boundary
# rather than a default nobody thought about.
#
# This gateway listens on loopback and holds the plaintext prompt and
# completion history in a local database. The attacker that matters is not a
# remote host -- it cannot reach 127.0.0.1 -- it is any page the operator
# happens to have open in a browser, which can issue requests to localhost.
# With allow_origins=["*"] and allow_credentials=True, Starlette reflects the
# requesting origin back and the browser hands that page the response. Measured
# before this change: a request carrying "Origin: https://evil.example.com" got
# back "access-control-allow-origin: https://evil.example.com" and a readable
# body containing the operator's prompts. For a gateway whose entire pitch is
# that prompts stay on the machine, that was the worst possible bug.
#
# The dashboard is served from this same app and uses relative URLs, so it is
# same-origin and needs no CORS grant at all. Anyone genuinely fronting this
# from another origin can name it explicitly; a wildcard cannot be reinstated
# by accident.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    if "*" in CORS_ORIGINS:
        logger.warning(
            "CORS_ORIGINS contains '*', which lets any website you visit read this "
            "gateway's prompt history from your browser. Name the origins instead."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        # Never with a wildcard, and not needed here: this gateway authenticates
        # with a bearer token, not a cookie.
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
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


def apply_enhanced_prompt(messages: List[Dict[str, Any]], enhanced: str) -> List[Dict[str, Any]]:
    """Substitute the enhanced text into the last user message.

    Enhancing a prompt and then sending the original wastes the work entirely,
    which is what happened before: `enhanced_prompt` was computed on every
    request and only ever used when the message list was empty, so no harness
    request ever benefited from it.
    """
    if not enhanced or not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            replaced = dict(out[i])
            replaced["content"] = enhanced
            out[i] = replaced
            return out
    return out


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
    raw_messages = [m.model_dump(exclude_none=True) for m in request.messages]

    if settings.ENABLE_ENHANCEMENT and settings.should_enhance_for(selected_model):
        enhanced_prompt = await enhance_prompt(original_prompt, for_tools=bool(request.tools))
        raw_messages = apply_enhanced_prompt(raw_messages, enhanced_prompt)
        router_reasoning += f" | Prompt enhanced for {selected_model}"
    else:
        enhanced_prompt = original_prompt
        if not settings.ENABLE_ENHANCEMENT:
            router_reasoning += " | Enhancement disabled"
        else:
            router_reasoning += " | Enhancement skipped (model already in a strong tier)"

    # Handle SSE token streaming if stream=True
    if request.stream:
        return StreamingResponse(
            dispatch_streaming_inference(
                requested_model=request.model,
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt,
                enhancer_model=settings.ENHANCER_MODEL if settings.should_enhance_for(selected_model) else "bypassed",
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
        enhancer_model=settings.ENHANCER_MODEL if settings.should_enhance_for(selected_model) else "bypassed",
        router_model=settings.ROUTER_MODEL,
        selected_model=selected_model,
        router_reasoning=router_reasoning,
        original_messages=raw_messages,
        tools=request.tools,
        tool_choice=request.tool_choice
    )

    # Step 4: Asynchronous LLM-as-a-Judge Auto-Evaluation Task
    # Judge scores completions and feeds back into router training.
    # Can be disabled to reduce costs, at the trade-off of losing the learning loop.
    if settings.ENABLE_JUDGING:
        import random
        if random.random() < settings.JUDGE_SAMPLE_RATE:
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
        # The model that actually served the request, which differs from the
        # routed choice whenever failover ran. Reporting the routed model
        # would misattribute both the completion and its cost.
        "model": result["selected_model"],
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
            "selected_model": result["selected_model"],
            "router_reasoning": router_reasoning,
            "actual_cost_usd": result["metrics"]["actual_cost_usd"],
            "prompt_tokens": result["metrics"]["prompt_tokens"],
            "completion_tokens": result["metrics"]["completion_tokens"]
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
    """Flatten Codex's tool tree into a flat OpenAI function-tool list.

    Namespaced tools (the MCP and multi-agent groups) are skipped. They are
    the overwhelming bulk of the payload -- the Slack group alone is ~17k
    tokens, enough on its own to blow a small provider's per-minute token
    limit -- and a namespaced call would have to be routed back through its
    namespace to be dispatched, which this gateway does not do. Dropping them
    leaves the coding tools the harness actually needs for a turn.
    """
    flat = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type")
        if t_type == "namespace":
            continue
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

    # Step 2: Selective Prompt Enhancement (Model 1). Applied whenever the
    # chosen model sits in a tier that benefits from it, including agentic
    # turns -- lifting a weaker model's output is the whole point, and the
    # local enhancer costs under a millisecond.
    if settings.should_enhance_for(selected_model):
        enhanced_prompt = await enhance_prompt(original_prompt, for_tools=bool(tools))
        raw_messages = apply_enhanced_prompt(raw_messages, enhanced_prompt)
        router_reasoning += f" | Prompt enhanced for {selected_model}"
    else:
        enhanced_prompt = original_prompt
        router_reasoning += " | Enhancement skipped (model already in a strong tier)"

    if not raw_messages:
        raw_messages = [{"role": "user", "content": enhanced_prompt}]

    tool_choice = raw_request.get("tool_choice")

    # Handle streaming for OpenAI Codex CLI (v0.142+)
    if raw_request.get("stream", True):
        return StreamingResponse(
            dispatch_responses_streaming_inference(
                requested_model=raw_request.get("model"),
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt,
                enhancer_model=settings.ENHANCER_MODEL if settings.should_enhance_for(selected_model) else "bypassed",
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
        enhancer_model=settings.ENHANCER_MODEL if settings.should_enhance_for(selected_model) else "bypassed",
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
        "model": result["selected_model"],
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
    messages, tools, tool_choice = convert_anthropic_messages(raw_request)
    # Anthropic clients always send max_tokens and expect it honoured;
    # dropping it leaves the provider default in charge of output length.
    max_tokens = raw_request.get("max_tokens")
    original_prompt = extract_user_prompt(raw_request.get("messages", []))

    selected_model, router_reasoning = await route_prompt(original_prompt, has_tools=bool(tools))

    if settings.should_enhance_for(selected_model):
        enhanced_prompt = await enhance_prompt(original_prompt, for_tools=bool(tools))
        messages = apply_enhanced_prompt(messages, enhanced_prompt)
        router_reasoning += f" | Prompt enhanced for {selected_model}"
    else:
        enhanced_prompt = original_prompt

    if not messages:
        messages = [{"role": "user", "content": enhanced_prompt or "Hello"}]

    if raw_request.get("stream"):
        return StreamingResponse(
            dispatch_anthropic_streaming_inference(
                requested_model=raw_request.get("model"),
                original_prompt=original_prompt,
                enhanced_prompt=enhanced_prompt,
                enhancer_model=settings.ENHANCER_MODEL if settings.should_enhance_for(selected_model) else "bypassed",
                router_model=settings.ROUTER_MODEL,
                selected_model=selected_model,
                router_reasoning=router_reasoning,
                original_messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    result = await dispatch_inference(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=settings.ENHANCER_MODEL if settings.should_enhance_for(selected_model) else "bypassed",
        router_model=settings.ROUTER_MODEL,
        selected_model=selected_model,
        router_reasoning=router_reasoning,
        original_messages=messages,
        tools=tools,
        tool_choice=tool_choice
    )

    background_tasks.add_task(
        evaluate_completion,
        request_id=result["request_id"],
        enhanced_prompt=enhanced_prompt,
        completion=result["completion"]
    )

    return build_anthropic_message(
        message_id=f"msg_{result['request_id']}",
        model=result["selected_model"],
        text=result["completion"],
        tool_calls=result.get("tool_calls"),
        input_tokens=result["metrics"]["prompt_tokens"],
        output_tokens=result["metrics"]["completion_tokens"],
    )

# --- OpenAI-Compatible Models Endpoint ---
@app.get("/v1/models")
def list_v1_models():
    return build_models_response()

# --- Advisory Routing (client executes the call itself) ---
@app.post("/api/route")
async def advise_route(
    raw_request: Dict[str, Any],
    auth_token: str = Depends(verify_api_key)
):
    """Recommend a model without performing the inference.

    For clients that call their provider directly -- Codex on a ChatGPT
    subscription, for instance -- proxying the request through this gateway
    would move the spend onto pay-per-token API billing. Here the gateway only
    decides, and the client keeps using the entitlement it already pays for.

    `models` is ordered lightest first and supplied by the caller, so this
    works for models the gateway itself cannot reach.
    """
    prompt = raw_request.get("prompt") or ""
    if not prompt:
        raise HTTPException(status_code=400, detail="`prompt` is required.")

    ladder = raw_request.get("models") or []
    normalised = [
        {"id": m, "description": ""} if isinstance(m, str)
        else {"id": m.get("id", ""), "description": m.get("description", "")}
        for m in ladder
    ]
    normalised = [m for m in normalised if m["id"]]
    if not normalised:
        raise HTTPException(status_code=400, detail="`models` must be a non-empty ordered list.")

    selected, reasoning = await route_among(prompt, normalised)

    ids = [m["id"] for m in normalised]
    try:
        with Session(engine) as session:
            session.add(RoutingDecision(
                prompt=prompt[:4000],
                selected_model=selected,
                reasoning=reasoning,
                router_model=settings.ROUTER_MODEL,
                tier_index=ids.index(selected) if selected in ids else 0,
                ladder_size=len(ids),
                client=raw_request.get("client"),
            ))
            session.commit()
    except Exception as db_err:
        # Never fail the caller's routing because logging broke.
        logger.error(f"Failed to record routing decision: {db_err}")

    return {"selected_model": selected, "reasoning": reasoning, "candidates": ids}

# --- Enterprise Admin & Analytics APIs ---
@app.get("/api/models")
def list_candidate_models():
    return {"models": settings.CANDIDATE_MODELS}

@app.get("/api/logs")
def list_logs(limit: int = 50, offset: int = 0, auth_token: str = Depends(verify_api_key)):
    with Session(engine) as session:
        statement = select(InferenceLog).order_by(InferenceLog.id.desc()).offset(offset).limit(limit)
        logs = session.exec(statement).all()
        
        # Summary Analytics
        total_requests = session.exec(select(func.count(InferenceLog.id))).one()
        total_actual_cost = session.exec(select(func.sum(InferenceLog.actual_cost))).one() or 0.0
        total_in = session.exec(select(func.sum(InferenceLog.prompt_tokens))).one() or 0
        total_out = session.exec(select(func.sum(InferenceLog.completion_tokens))).one() or 0

        # Share of traffic that avoided the most expensive tier. Every part of
        # this is observed: which model ran, and how many there were.
        by_model = session.exec(
            select(InferenceLog.selected_model, func.count(InferenceLog.id))
            .group_by(InferenceLog.selected_model)
        ).all()
        heavy = {m.id for m in settings.CANDIDATE_MODELS if m.intelligence_tier == "frontier"}
        off_heavy = sum(n for m, n in by_model if m not in heavy)
        avg_score = session.exec(select(func.avg(InferenceLog.judge_score))).one() or 0.0

        return {
            "summary": {
                "total_requests": total_requests,
                "total_actual_cost_usd": round(total_actual_cost, 4),
                "total_input_tokens": int(total_in),
                "total_output_tokens": int(total_out),
                "avg_cost_per_request": round(total_actual_cost / total_requests, 6) if total_requests else 0.0,
                "requests_off_top_tier": off_heavy,
                "off_top_tier_percentage": round(off_heavy / total_requests * 100, 1) if total_requests else 0.0,
                "by_model": {m: n for m, n in by_model},
                "average_judge_score": round(avg_score, 2)
            },
            "logs": logs
        }

@app.get("/api/routing/decisions")
def routing_decisions(limit: int = 50, auth_token: str = Depends(verify_api_key)):
    """Advisory routing history.

    Reports tier usage rather than spend. Under a subscription the bill is
    fixed, so the meaningful question is how often the heavyweight tier was
    reached for, not what it cost.
    """
    with Session(engine) as session:
        rows = session.exec(
            select(RoutingDecision).order_by(RoutingDecision.id.desc()).limit(limit)
        ).all()
        total = session.exec(select(func.count(RoutingDecision.id))).one()

        by_model = {}
        for r in session.exec(select(RoutingDecision)).all():
            by_model[r.selected_model] = by_model.get(r.selected_model, 0) + 1

        # "Top tier" means the last rung of the ladder the caller offered.
        top = session.exec(select(func.count(RoutingDecision.id)).where(
            RoutingDecision.tier_index == RoutingDecision.ladder_size - 1
        )).one() or 0

    return {
        "summary": {
            "total_decisions": total,
            "top_tier_decisions": top,
            "top_tier_percentage": round((top / total * 100) if total else 0.0, 1),
            "by_model": by_model,
        },
        "decisions": rows,
    }

@app.get("/api/dataset/export")
def export_dataset(target: str = "router", min_score: float = 7.0,
                   auth_token: str = Depends(verify_api_key)):
    jsonl_content = export_dataset_jsonl(target_type=target, min_score=min_score)
    return PlainTextResponse(
        content=jsonl_content,
        media_type="application/x-jsonlines",
        headers={"Content-Disposition": f'attachment; filename="wormhole_{target}_dataset.jsonl"'}
    )

@app.post("/api/router/retrain")
def retrain_models_from_feedback(auth_token: str = Depends(verify_api_key)):
    from models.train_router import train_router_slm
    from models.train_quality_evaluator import train_quality_evaluator_slm
    
    from services.feedback import collect_feedback_examples

    try:
        # Report what the retrain actually learned from. Previously this
        # claimed to train "from latest database completions" while reading
        # only the static benchmark file, so pressing it changed nothing.
        examples = collect_feedback_examples()
        train_router_slm()
        train_quality_evaluator_slm()
        return {
            "status": "success",
            "feedback_examples_used": len(examples),
            "message": (
                f"Retrained on the benchmark bootstrap plus {len(examples)} judged real prompts. "
                "Only requests the judge actually scored are used; unscored ones are ignored."
                if examples else
                "Retrained on the benchmark bootstrap only: no judged real traffic yet. "
                "Send traffic through the gateway so the judge can score it, then retrain again."
            ),
        }
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
            <div class="card-title">Kept off the top tier</div>
            <div class="card-value" id="off-top" style="color: var(--green);">0%</div>
            <div class="card-sub" id="off-top-sub">of requests</div>
        </div>
        <div class="card">
            <div class="card-title">Actual API Spend</div>
            <div class="card-value" id="actual-spend">$0.0000</div>
            <div class="card-sub" id="avg-cost">Avg per request: $0.000000</div>
        </div>
        <div class="card">
            <div class="card-title">Avg Judge Score</div>
            <div class="card-value" id="avg-score">0.0 / 10</div>
            <div class="card-sub">Auto LLM-as-a-Judge Score</div>
        </div>
    </div>

    <p style="color:var(--text-sub); font-size:13px; margin:-8px 0 20px 0;">
        Every figure above is observed: which model ran, what it reported using, and what that cost.
        For a like-for-like comparison against a single strong model, run
        <code>scripts/evaluate_routing_quality.py</code> &mdash; it executes both arms for real.
    </p>

    <div class="section-title">
        <span>Inference Traffic &amp; Routing Decisions</span>
    </div>

    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>⏰ Timestamp</th>
                <th>📥 Original Prompt</th>
                <th>✨ Enhanced Prompt (Model 1 SLM)</th>
                <th>🙋 Model Asked For</th>
                <th>🎯 Model That Ran &amp; Reasoning</th>
                <th>Tokens / Cost</th>
                <th>Judge Score</th>
            </tr>
        </thead>
        <tbody id="logs-body">
            <tr><td colspan="8" style="text-align: center; color: var(--text-sub);">Loading inference logs...</td></tr>
        </tbody>
    </table>

    <div class="section-title" style="margin-top:32px;">
        <span>Advisory Routing Decisions (client executed the call itself)</span>
        <span id="toptier-badge" class="tag tag-model">&mdash;</span>
    </div>
    <p style="color:var(--text-sub); font-size:13px; margin-bottom:12px;">
        These runs never passed through the gateway, so there are no tokens or cost to report.
        Under a subscription the bill is fixed &mdash; what matters is how often the heaviest tier was used.
    </p>
    <table>
        <thead>
            <tr>
                <th>⏰ Timestamp</th>
                <th>📥 Task</th>
                <th>🎯 Tier Chosen</th>
                <th>Reasoning</th>
            </tr>
        </thead>
        <tbody id="routing-body">
            <tr><td colspan="4" style="text-align:center;color:var(--text-sub);">No advisory routing yet. Try <code>scripts/codex-routed</code>.</td></tr>
        </tbody>
    </table>

    <script>
        // The endpoints holding prompt history sit behind verify_api_key, so
        // when ENABLE_AUTH is on this same-origin dashboard needs the key too.
        // Ask once, keep it for the tab only, and never write it to disk.
        async function api(path, options) {
            const opts = Object.assign({}, options || {});
            const key = sessionStorage.getItem('wormhole_key');
            if (key) {
                opts.headers = Object.assign({}, opts.headers, {'Authorization': 'Bearer ' + key});
            }
            let res = await fetch(path, opts);
            if (res.status === 401) {
                const entered = window.prompt(
                    'This gateway has ENABLE_AUTH on. Enter one of your ' +
                    'WORMHOLE_API_KEYS to view the dashboard:');
                if (!entered) return res;
                sessionStorage.setItem('wormhole_key', entered);
                opts.headers = Object.assign({}, opts.headers, {'Authorization': 'Bearer ' + entered});
                res = await fetch(path, opts);
                if (res.status === 401) sessionStorage.removeItem('wormhole_key');
            }
            return res;
        }

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
                const res = await api('/api/logs');
                const data = await res.json();
                const s = data.summary;
                
                document.getElementById('total-requests').innerText = s.total_requests;
                document.getElementById('off-top').innerText = `${s.off_top_tier_percentage}%`;
                document.getElementById('off-top-sub').innerText =
                    `${s.requests_off_top_tier} of ${s.total_requests} requests`;
                document.getElementById('actual-spend').innerText = `$${s.total_actual_cost_usd.toFixed(4)}`;
                document.getElementById('avg-cost').innerText =
                    `Avg per request: $${s.avg_cost_per_request.toFixed(6)}  ·  ${s.total_input_tokens.toLocaleString()} in / ${s.total_output_tokens.toLocaleString()} out`;
                document.getElementById('avg-score').innerText = `${s.average_judge_score} / 10`;

                const tbody = document.getElementById('logs-body');
                if (data.logs.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-sub);">No requests logged yet. Send chat completion calls to <code>/v1/chat/completions</code>.</td></tr>';
                    return;
                }

                tbody.innerHTML = data.logs.map(log => {
                    const origPrompt = escapeHtml(log.original_prompt || '');
                    const enhPrompt = escapeHtml(log.enhanced_prompt || '');
                    const selModel = escapeHtml(log.selected_model || '');
                    // What the harness pinned in its own config, kept in its
                    // own column beside what actually ran. Side by side is the
                    // whole point: the developer chose one model and a cheaper
                    // one did the work, which is invisible if you only log the
                    // second.
                    const reqModel = escapeHtml(log.requested_model || '');
                    const routedDown = reqModel && log.selected_model &&
                                       !log.selected_model.endsWith(reqModel);
                    const askedCell = !reqModel
                        ? `<span style="color:var(--text-sub);font-size:11px;">&mdash;</span>`
                        : routedDown
                            ? `<span class="tag" style="background:rgba(251,191,36,0.12);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);">${reqModel}</span>
                               <div style="font-size:10px;color:var(--text-sub);margin-top:4px;">routed to something cheaper</div>`
                            : `<span class="tag tag-model">${reqModel}</span>
                               <div style="font-size:10px;color:var(--text-sub);margin-top:4px;">kept</div>`;
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
                        <td>${askedCell}</td>
                        <td>
                            <span class="tag tag-model">${selModel}</span>
                            <div style="font-size: 11px; color: var(--text-sub); margin-top: 4px; max-width: 220px;">${reason}</div>
                        </td>
                        <td>
                            <div style="font-weight: 600;">$${log.actual_cost.toFixed(6)}</div>
                            <div style="font-size: 11px; color: var(--text-sub);">${log.prompt_tokens} in / ${log.completion_tokens} out</div>
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
                const res = await api('/api/router/retrain', { method: 'POST' });
                const data = await res.json();
                alert(`✅ Retraining Complete!\n${data.message}`);
                fetchAnalytics();
            } catch (err) {
                alert("❌ Retraining failed: " + err);
            }
        }

        async function fetchRouting() {
            try {
                const res = await api('/api/routing/decisions?limit=25');
                const data = await res.json();
                const s = data.summary;
                const badge = document.getElementById('toptier-badge');
                badge.innerText = `Top tier used in ${s.top_tier_decisions} of ${s.total_decisions} (${s.top_tier_percentage}%)`;

                const tbody = document.getElementById('routing-body');
                if (!data.decisions.length) return;
                tbody.innerHTML = data.decisions.map(d => `
                    <tr>
                        <td style="font-size:11px;color:#cbd5e1;white-space:nowrap;">${formatLocalDate(d.created_at)}</td>
                        <td><div class="prompt-preview" title="${escapeHtml(d.prompt)}">${escapeHtml(d.prompt)}</div></td>
                        <td><span class="tag tag-model">${escapeHtml(d.selected_model)}</span>
                            <div style="font-size:11px;color:var(--text-sub);margin-top:4px;">tier ${d.tier_index + 1} of ${d.ladder_size}</div></td>
                        <td style="font-size:12px;color:var(--text-sub);max-width:320px;">${escapeHtml(d.reasoning || '')}</td>
                    </tr>`).join('');
            } catch (err) {
                console.error('Error loading routing decisions', err);
            }
        }

        fetchAnalytics();
        fetchRouting();
        setInterval(fetchAnalytics, 5000);
        setInterval(fetchRouting, 5000);
    </script>
</body>
</html>
"""
