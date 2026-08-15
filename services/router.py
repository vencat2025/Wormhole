import json
import logging
from typing import Dict, Any, Tuple
import litellm
from config import settings

logger = logging.getLogger("wormhole.router")

ROUTER_SYSTEM_PROMPT = """You are the AI Routing Model (Model 2) for an enterprise LLM cost optimization platform.
Your objective is to evaluate an ENHANCED PROMPT and choose the CHEAPEST candidate model capable of executing it with high quality.

Available Candidate Models in Enterprise Fleet:
{candidate_models_info}

Instructions:
1. Carefully analyze the task complexity, required reasoning depth, domain (e.g. simple formatting, basic summarization, standard code, vs advanced math/complex architecture synthesis).
2. Pick the model with the LOWEST input/output cost that still achieves high quality for this specific task.
3. Respond ONLY with valid JSON matching this schema:
{{
  "selected_model": "<model_id>",
  "reasoning": "<brief 1-2 sentence justification for why this model was chosen over more expensive alternatives>"
}}
"""

def _format_candidate_models() -> str:
    lines = []
    for model in settings.CANDIDATE_MODELS:
        lines.append(
            f"- Model ID: `{model.id}` | Name: {model.name} | Input: ${model.input_cost_per_1k}/1k, Output: ${model.output_cost_per_1k}/1k | Intel Tier: {model.intelligence_tier} | Description: {model.description}"
        )
    return "\n".join(lines)

async def route_prompt(enhanced_prompt: str, model_name: str = None) -> Tuple[str, str]:
    """
    Evaluates the enhanced prompt and returns (selected_model_id, reasoning).
    """
    model = model_name or settings.ROUTER_MODEL
    candidate_info = _format_candidate_models()
    system_prompt = ROUTER_SYSTEM_PROMPT.format(candidate_models_info=candidate_info)
    
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhanced Prompt to Route:\n{enhanced_prompt}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        selected_model = data.get("selected_model", settings.FALLBACK_MODEL)
        reasoning = data.get("reasoning", "Selected optimal model based on prompt complexity.")
        
        # Verify selected model is in candidates
        valid_ids = [m.id for m in settings.CANDIDATE_MODELS]
        if selected_model not in valid_ids:
            logger.warning(f"Router output invalid model '{selected_model}', defaulting to {settings.FALLBACK_MODEL}")
            selected_model = settings.FALLBACK_MODEL
            
        return selected_model, reasoning

    except Exception as e:
        logger.warning(f"Router API call failed or unconfigured ({e}). Utilizing fallback heuristic routing.")
        # Heuristic routing for offline/dev fallback
        prompt_len = len(enhanced_prompt)
        has_complex_keywords = any(w in enhanced_prompt.lower() for w in ["architecture", "refactor whole system", "formal proof", "quantum"])
        
        if has_complex_keywords:
            selected_model = "gpt-4o"
            reasoning = "Routed to gpt-4o due to detected high-complexity reasoning keywords."
        elif prompt_len > 3000:
            selected_model = "gemini/gemini-1.5-flash"
            reasoning = "Routed to Gemini 1.5 Flash due to long context and ultra-low token cost."
        else:
            selected_model = "gpt-4o-mini"
            reasoning = "Routed to GPT-4o Mini as low-cost model suitable for standard task complexity."
            
        return selected_model, reasoning
