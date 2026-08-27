import os
import json
import logging
from typing import Tuple
import joblib
import litellm
from config import settings

logger = logging.getLogger("wormhole.router")

# Resolve paths relative to the repository so the project runs from any
# checkout location, not only the machine it was written on.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, "models", "router_slm.joblib")
_LOCAL_ROUTER_SLM = None

def _get_local_router_slm():
    global _LOCAL_ROUTER_SLM
    if _LOCAL_ROUTER_SLM is None and os.path.exists(MODEL_FILE_PATH):
        try:
            _LOCAL_ROUTER_SLM = joblib.load(MODEL_FILE_PATH)
            logger.info("⚡ Local Router SLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load local SLM ({e}).")
    return _LOCAL_ROUTER_SLM

ROUTER_SYSTEM_PROMPT = """You are the routing model for an LLM gateway. Choose which model should answer a user's prompt.

Available models, cheapest first:
{candidate_models_info}

Routing is a trade-off, not a discount. Under-routing a hard task produces a
wrong answer that costs far more than the tokens saved; over-routing an easy
task wastes money for no quality gain. Judge what the task actually demands.

Choose the CHEAPEST model for tasks that are:
- factual recall, definitions, translation, summarisation, formatting
- single-function code, boilerplate, renames, simple edits and regexes
- anything a competent junior engineer would finish without deliberation

Escalate to a HIGHER tier when the task involves any of:
- correctness proofs, complexity bounds, or formal reasoning
- concurrency, race conditions, distributed systems, or data consistency
- changes spanning multiple files, modules or services
- system or schema design, migrations, or architectural trade-offs
- debugging from a symptom rather than a known cause
- security-sensitive logic, money, auth, or anything hard to reverse
- long or ambiguous requirements needing decomposition

Judge the task, not its vocabulary. "Optimise this to O(n log n) and prove the
bound" is hard despite being one sentence. "Refactor our payment engine to be
idempotent across partial failures" is hard because of blast radius.

Respond ONLY with valid JSON:
{{
  "selected_model": "<model_id>",
  "reasoning": "<one sentence: what the task demands and why this tier meets it>"
}}
"""

def _format_candidate_models() -> str:
    lines = []
    for model in settings.CANDIDATE_MODELS:
        if not settings.provider_allowed(model.provider) or not settings.model_allowed(model.id):
            continue
        lines.append(
            f"- Model ID: `{model.id}` | Name: {model.name} | Input: ${model.input_cost_per_1k}/1k, Output: ${model.output_cost_per_1k}/1k | Intel Tier: {model.intelligence_tier} | Description: {model.description}"
        )
    return "\n".join(lines)

def is_model_routable_lazy(model_id: str, need_tools: bool) -> bool:
    from services.dispatcher import is_model_routable
    return is_model_routable(model_id, need_tools=need_tools)


def _first_routable(model_ids, need_tools: bool) -> str:
    """First model in preference order that traffic can actually reach."""
    from services.dispatcher import is_model_routable
    for model_id in model_ids:
        if is_model_routable(model_id, need_tools=need_tools):
            return model_id
    for m in sorted(settings.CANDIDATE_MODELS, key=lambda c: c.input_cost_per_1k):
        if is_model_routable(m.id, need_tools=need_tools):
            return m.id
    return ""


async def route_prompt(enhanced_prompt: str, model_name: str = None, has_tools: bool = False) -> Tuple[str, str]:
    """
    Evaluates the enhanced prompt and returns (selected_model_id, reasoning).
    Prioritizes fast local SLM inference (<2ms) trained on public benchmark performance profiles.
    """
    from services.dispatcher import is_model_routable

    local_slm = _get_local_router_slm() if settings.ROUTER_MODE != "llm" else None
    if local_slm is not None:
        try:
            predicted_model = local_slm.predict([enhanced_prompt])[0]

            # The classifier's pick stands unless it cannot actually be
            # reached: no credentials, a rejected key, no tool support on an
            # agentic turn, or an open circuit. Substituting only in those
            # cases keeps routing meaningful, where an unconditional override
            # would retire the classifier entirely.
            if is_model_routable(predicted_model, need_tools=has_tools):
                return predicted_model, (
                    f"⚡ Local Router SLM (<2ms): selected '{predicted_model}' on benchmark capability matching."
                )

            substitute = _first_routable(
                [settings.AGENTIC_MODEL, settings.FALLBACK_MODEL] if has_tools else [settings.FALLBACK_MODEL],
                has_tools
            )
            if substitute:
                return substitute, (
                    f"⚡ Local Router SLM (<2ms) selected '{predicted_model}', which is not currently routable "
                    f"(missing/rejected credentials, no tool support, or open circuit); using '{substitute}'."
                )
            logger.warning(
                f"SLM picked '{predicted_model}' and no configured substitute is routable; falling through to API router."
            )
        except Exception as slm_err:
            logger.warning(f"Local Router SLM inference failed ({slm_err}), falling back to API/heuristic router.")

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
        
        if not is_model_routable_lazy(selected_model, has_tools):
            substitute = _first_routable([settings.FALLBACK_MODEL], has_tools)
            logger.warning(f"Router output '{selected_model}' is not routable; using '{substitute}'.")
            selected_model = substitute

        return selected_model, reasoning

    except Exception as e:
        # The router itself consumes provider quota, so under rapid traffic it
        # can fail while the fleet is healthy. The old behaviour fell through
        # to the cheapest allowed model, which silently under-routes exactly
        # when routing stopped working. Degrade toward capability instead:
        # a task wrongly sent to a stronger model costs tokens, one wrongly
        # sent to a weaker model costs a wrong answer.
        logger.warning(f"Router call failed ({e}); using degraded tier heuristics.")

        prompt_lower = enhanced_prompt.lower()
        complex_signals = (
            "architecture", "migration", "concurren", "race condition", "deadlock",
            "distributed", "prove", "proof", "complexity", "optimis", "optimiz",
            "refactor", "security", "idempoten", "multi-file", "across services",
        )
        looks_hard = any(w in prompt_lower for w in complex_signals) or len(enhanced_prompt) > 3000

        def by_tier(*tiers):
            for tier in tiers:
                for m in sorted(settings.CANDIDATE_MODELS, key=lambda c: -c.input_cost_per_1k):
                    if m.intelligence_tier == tier and is_model_routable(m.id, need_tools=has_tools):
                        return m.id
            return ""

        if looks_hard:
            selected_model = by_tier("frontier", "high", "medium")
            why = "degraded heuristics: complexity signals present, holding at a capable tier"
        else:
            # Not "cheapest available" but a middle tier, since this path runs
            # precisely when the gateway cannot assess the task properly.
            selected_model = by_tier("medium", "high", "basic")
            why = "degraded heuristics: no complexity signals, mid tier chosen"

        selected_model = selected_model or settings.FALLBACK_MODEL
        return selected_model, f"⚠️ Router unavailable ({type(e).__name__}); {why}. Selected '{selected_model}'."
