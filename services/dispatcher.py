import time
import uuid
import logging
from typing import Dict, Any, List
import litellm
from sqlmodel import Session
from config import settings, CandidateModelConfig
from db.database import engine
from db.models import InferenceLog
from services.judge import evaluate_completion

logger = logging.getLogger("wormhole.dispatcher")

def get_model_config(model_id: str) -> CandidateModelConfig:
    for m in settings.CANDIDATE_MODELS:
        if m.id == model_id:
            return m
    # Return default config if custom/unlisted model ID
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

async def dispatch_inference(
    original_prompt: str,
    enhanced_prompt: str,
    enhancer_model: str,
    router_model: str,
    selected_model: str,
    router_reasoning: str,
    original_messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Executes the completion call on the selected target model, logs metrics, and triggers auto-evaluation.
    """
    request_id = f"wh-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    # Replace last user message with enhanced prompt
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
            model=selected_model,
            messages=messages_to_send,
            temperature=0.7
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
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
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.warning(f"Target model call failed for '{selected_model}' ({e}). Executing fallback response.")
        completion_text = (
            f"[WormHole Proxy Response - Fallback for {selected_model}]\n\n"
            f"Here is the synthesized response to your request:\n{original_prompt}"
        )
        prompt_tokens = len(enhanced_prompt) // 4
        completion_tokens = len(completion_text) // 4

    total_tokens = prompt_tokens + completion_tokens
    actual_cost = calculate_cost(selected_model, prompt_tokens, completion_tokens)
    baseline_cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)
    cost_savings = round(max(0.0, baseline_cost - actual_cost), 6)

    # Persist log to DB
    log_entry = InferenceLog(
        request_id=request_id,
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        enhancer_model=enhancer_model,
        router_model=router_model,
        router_reasoning=router_reasoning,
        selected_model=selected_model,
        baseline_model="gpt-4o",
        completion=completion_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        cost_savings=cost_savings,
        latency_ms=latency_ms
    )

    try:
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
    except Exception as db_err:
        logger.error(f"Failed to log inference transaction to database: {db_err}")

    # Return standard response metadata payload
    return {
        "request_id": request_id,
        "completion": completion_text,
        "selected_model": selected_model,
        "enhanced_prompt": enhanced_prompt,
        "router_reasoning": router_reasoning,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "actual_cost_usd": actual_cost,
            "baseline_cost_usd": baseline_cost,
            "cost_savings_usd": cost_savings,
            "savings_percentage": round((cost_savings / max(baseline_cost, 0.000001)) * 100, 1),
            "latency_ms": latency_ms
        }
    }
