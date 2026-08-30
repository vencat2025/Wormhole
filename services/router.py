import os
import json
import logging
from typing import Tuple, List, Dict
import joblib
import litellm
from config import settings, TIER_ORDER

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


def _substitute_at_tier(predicted_model: str, need_tools: bool) -> str:
    """Cheapest reachable model no weaker than the one the classifier wanted.

    The classifier only knows the fleet it was trained on. Point the gateway at
    a different one -- ROUTING_MODELS naming an all-5.6 ladder, say -- and
    every prediction names a model that is not on it. Falling back to the
    cheapest reachable model at that point throws the whole decision away:
    measured with a 5.6-only ladder, a rename and a zero-downtime sharding
    migration both landed on the cheapest tier, because "not routable" was
    being treated as "no opinion".

    The prediction still carries the part that matters, which is *how hard the
    task looked*. That survives as the predicted model's tier, so translate it:
    keep the tier, spend the least money that buys it on the fleet actually
    available. Only if nothing reaches that bar does this settle for the
    strongest thing left.
    """
    from services.dispatcher import is_model_routable

    cfg = settings.model_config_for(predicted_model)
    if not cfg:
        return ""
    try:
        want = TIER_ORDER.index(cfg.intelligence_tier.lower())
    except ValueError:
        return ""

    reachable = [
        m for m in settings.CANDIDATE_MODELS
        if is_model_routable(m.id, need_tools=need_tools)
    ]
    if not reachable:
        return ""

    def tier_of(m) -> int:
        try:
            return TIER_ORDER.index(m.intelligence_tier.lower())
        except ValueError:
            return -1

    at_or_above = [m for m in reachable if tier_of(m) >= want]
    if at_or_above:
        return min(at_or_above, key=lambda m: (m.input_cost_per_1k, m.output_cost_per_1k)).id
    # Nothing that strong is reachable; the best available is the honest answer.
    return max(reachable, key=lambda m: (tier_of(m), -m.input_cost_per_1k)).id


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

            # Keep the difficulty the classifier judged, on the fleet that is
            # actually reachable, before falling back to a fixed model.
            substitute = _substitute_at_tier(predicted_model, has_tools)
            if substitute:
                cfg = settings.model_config_for(predicted_model)
                tier = cfg.intelligence_tier if cfg else "unknown"
                return substitute, (
                    f"⚡ Local Router SLM (<2ms) selected '{predicted_model}', which is not currently routable "
                    f"(missing/rejected credentials, no tool support, or open circuit); "
                    f"substituted '{substitute}', the cheapest reachable model at the '{tier}' tier or above."
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
        router_kwargs = {"api_base": settings.OLLAMA_BASE_URL} if model.startswith("ollama/") else {}
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhanced Prompt to Route:\n{enhanced_prompt}"}
            ],
            **router_kwargs,
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


LADDER_ROUTER_PROMPT = """You choose which tier of coding model should handle a task.

The tiers below are ordered from lightest to strongest. Reply with the index of
the lightest tier that can do the job properly.

{ladder}

Routing is a trade-off, not a discount. Sending a hard task to a light tier
produces a wrong answer that costs far more than the tokens saved; sending a
trivial one to the strongest tier wastes capacity for no gain.

Use the LIGHTEST tier for: factual questions, renames, formatting, single
functions, boilerplate, simple edits, and anything a competent junior engineer
would finish without deliberation.

Escalate for: correctness proofs or complexity bounds; concurrency, races or
distributed consistency; changes spanning multiple files or services; system,
schema or migration design; debugging from a symptom rather than a known cause;
security, money, auth, or anything hard to reverse; long or ambiguous
requirements needing decomposition.

Judge the task, not its vocabulary.

Respond ONLY with valid JSON:
{{"index": <integer>, "reasoning": "<one sentence>"}}
"""


async def route_among(prompt: str, ladder: List[Dict[str, str]]) -> Tuple[str, str]:
    """Pick a rung from an explicitly ordered ladder of models.

    Used when the gateway advises but does not execute -- for a client that
    calls its provider itself, such as Codex on a ChatGPT subscription. The
    ladder is supplied by the caller and ordered lightest first, so no pricing
    table is needed and nothing is assumed about models this gateway cannot
    reach.
    """
    if not ladder:
        return "", "No candidate models supplied."

    rendered = "\n".join(
        f"{i}. {m.get('id')} - {m.get('description', 'no description')}"
        for i, m in enumerate(ladder)
    )
    router_kwargs = {}
    if settings.ROUTER_MODEL.startswith("ollama/"):
        router_kwargs["api_base"] = settings.OLLAMA_BASE_URL

    try:
        response = await litellm.acompletion(
            model=settings.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": LADDER_ROUTER_PROMPT.format(ladder=rendered)},
                {"role": "user", "content": f"Task:\n{prompt}"},
            ],
            **router_kwargs,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=256,
        )
        data = json.loads(response.choices[0].message.content.strip())
        idx = int(data.get("index", 0))
        idx = max(0, min(idx, len(ladder) - 1))
        return ladder[idx]["id"], data.get("reasoning", "")
    except Exception as e:
        # Degrade toward capability, matching route_prompt: an advisory router
        # that fails should not quietly recommend the weakest tier.
        logger.warning(f"Ladder router failed ({e}); defaulting to a middle rung.")
        idx = len(ladder) // 2
        return ladder[idx]["id"], f"Router unavailable ({type(e).__name__}); defaulted to a middle tier."
