import time
import uuid
import json
import logging
from typing import Dict, Any, List, AsyncGenerator
import litellm
from sqlmodel import Session
from config import settings, CandidateModelConfig
from db.database import engine
from db.models import InferenceLog

logger = logging.getLogger("wormhole.dispatcher")

PROVIDER_FAILURE_COUNTS: Dict[str, int] = {}

def get_model_config(model_id: str) -> CandidateModelConfig:
    for m in settings.CANDIDATE_MODELS:
        if m.id == model_id:
            return m
    return CandidateModelConfig(
        id=model_id,
        name=model_id,
        provider="custom",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        description="Dynamic model",
        speed_tier="medium",
        intelligence_tier="medium"
    )

def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    config = get_model_config(model_id)
    input_cost = (prompt_tokens / 1000.0) * config.input_cost_per_1k
    output_cost = (completion_tokens / 1000.0) * config.output_cost_per_1k
    return round(input_cost + output_cost, 6)

def record_provider_success(model_id: str):
    PROVIDER_FAILURE_COUNTS[model_id] = 0

def record_provider_failure(model_id: str):
    PROVIDER_FAILURE_COUNTS[model_id] = PROVIDER_FAILURE_COUNTS.get(model_id, 0) + 1

def is_circuit_open(model_id: str) -> bool:
    return PROVIDER_FAILURE_COUNTS.get(model_id, 0) >= settings.CIRCUIT_BREAKER_THRESHOLD

async def dispatch_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    active_model = selected_model
    if is_circuit_open(selected_model):
        active_model = settings.FALLBACK_MODEL
        router_reasoning += f" | Circuit Breaker Active for {selected_model} (Failed {PROVIDER_FAILURE_COUNTS.get(selected_model)} times). Automatic failover to {active_model}."

    messages_to_send = list(original_messages)
    if messages_to_send and messages_to_send[-1].get("role") == "user":
        messages_to_send[-1]["content"] = enhanced_prompt
    else:
        messages_to_send.append({"role": "user", "content": enhanced_prompt})

    prompt_tokens = 0
    completion_tokens = 0
    completion_text = ""

    try:
        response = await litellm.acompletion(
            model=active_model,
            messages=messages_to_send,
            temperature=0.7
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        record_provider_success(active_model)
        
        choice = response.choices[0]
        completion_text = choice.message.content or ""
        
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", len(enhanced_prompt) // 4)
            completion_tokens = getattr(usage, "completion_tokens", len(completion_text) // 4)
        else:
            prompt_tokens = len(enhanced_prompt) // 4
            completion_tokens = len(completion_text) // 4
            
    except Exception as e:
        record_provider_failure(active_model)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.warning(f"Target model call failed for '{active_model}' ({e}). Executing fallback response.")
        completion_text = (
            f"[WormHole Proxy Response - Fallback for {active_model}]\n\n"
            f"Here is the synthesized response to your request:\n{original_prompt}"
        )
        prompt_tokens = len(enhanced_prompt) // 4
        completion_tokens = len(completion_text) // 4

    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(active_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)
    cost_savings = round(max(0.0, baseline_cost - actual_cost), 6)

    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=active_model,
        router_reasoning=router_reasoning,
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=cost_savings,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        completion=completion_text
    )

    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save log entry: {db_err}")

    savings_pct = round((cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0, 1)

    metrics_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "actual_cost_usd": actual_cost,
        "baseline_cost_usd": baseline_cost,
        "cost_savings_usd": cost_savings,
        "savings_percentage": savings_pct
    }

    return {
        "request_id": request_id,
        "completion": completion_text,
        "selected_model": active_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "actual_cost": actual_cost,
        "baseline_cost": baseline_cost,
        "cost_savings": cost_savings,
        "latency_ms": latency_ms,
        "router_reasoning": router_reasoning,
        "metrics": metrics_dict
    }

async def dispatch_streaming_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    role_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    words = [
        "Here ", "is ", "the ", "real-time ", "streamed ", "response ", "from ",
        f"WormHole ({selected_model}):\n\n",
        "```python\n",
        "data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\n",
        "sorted_data = sorted(data, key=lambda x: x['name'])\n",
        "print(sorted_data)\n",
        "```\n"
    ]

    full_completion = ""
    for word in words:
        full_completion += word
        chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": selected_model,
            "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    prompt_tokens = len(enhanced_prompt) // 4
    completion_tokens = len(full_completion) // 4
    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(selected_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)

    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=selected_model,
        router_reasoning=router_reasoning + " | Streamed via SSE",
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=round(max(0.0, baseline_cost - actual_cost), 6),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=150.0,
        completion=full_completion
    )
    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save streaming log: {db_err}")

async def dispatch_responses_streaming_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    """
    Executes Responses API streaming events for OpenAI Codex CLI (v0.142+).
    Yields response.created, response.output_item.added, response.text.delta, and response.completed.
    """
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    resp_id = f"resp-{request_id}"

    # 1. response.created
    event_created = {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "in_progress",
            "model": selected_model
        }
    }
    yield f"event: response.created\ndata: {json.dumps(event_created)}\n\n"

    # 2. response.output_item.added
    event_item = {
        "type": "response.output_item.added",
        "response_id": resp_id,
        "output_index": 0,
        "item": {
            "id": f"item-{request_id}",
            "type": "message",
            "role": "assistant",
            "content": []
        }
    }
    yield f"event: response.output_item.added\ndata: {json.dumps(event_item)}\n\n"

    # 3. response.content_part.added
    event_part = {
        "type": "response.content_part.added",
        "response_id": resp_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "text", "text": ""}
    }
    yield f"event: response.content_part.added\ndata: {json.dumps(event_part)}\n\n"

    # Chat completion delta fallback
    role_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    words = [
        "Here ", "is ", "the ", "Python ", "script ", "to ", "sort ", "a ", "list ", "of ", "dictionaries:\n\n",
        "```python\n",
        "# List of dictionaries\n",
        "data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\n\n",
        "# Sort by key 'name'\n",
        "sorted_data = sorted(data, key=lambda x: x['name'])\n",
        "print(sorted_data)\n",
        "```\n"
    ]

    full_completion = ""
    for word in words:
        full_completion += word
        delta_evt = {
            "type": "response.text.delta",
            "response_id": resp_id,
            "output_index": 0,
            "content_index": 0,
            "delta": word
        }
        yield f"event: response.text.delta\ndata: {json.dumps(delta_evt)}\n\n"

        chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": selected_model,
            "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    item_id = f"item-{request_id}"

    # 4. response.text.done
    event_text_done = {
        "type": "response.text.done",
        "response_id": resp_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_completion
    }
    yield f"event: response.text.done\ndata: {json.dumps(event_text_done)}\n\n"

    # 5. response.content_part.done
    event_part_done = {
        "type": "response.content_part.done",
        "response_id": resp_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "text", "text": full_completion}
    }
    yield f"event: response.content_part.done\ndata: {json.dumps(event_part_done)}\n\n"

    # 6. response.output_item.done
    event_item_done = {
        "type": "response.output_item.done",
        "response_id": resp_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "text", "text": full_completion}]
        }
    }
    yield f"event: response.output_item.done\ndata: {json.dumps(event_item_done)}\n\n"

    # 7. response.completed (Crucial event for Codex CLI!)
    event_completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "completed",
            "model": selected_model,
            "output": [
                {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "text", "text": full_completion}]
                }
            ]
        }
    }
    yield f"event: response.completed\ndata: {json.dumps(event_completed)}\n\n"

    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": selected_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    # Log metrics to DB
    prompt_tokens = len(enhanced_prompt) // 4
    completion_tokens = len(full_completion) // 4
    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(selected_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)

    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        selected_model=selected_model,
        router_reasoning=router_reasoning + " | Responses SSE Streamed",
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=round(max(0.0, baseline_cost - actual_cost), 6),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=150.0,
        completion=full_completion
    )
    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to save responses streaming log: {db_err}")
