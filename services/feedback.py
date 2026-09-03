"""Turn judged completions into router training data.

This is the loop that makes the router improve instead of being replaced. The
benchmark dataset only cold-starts it: those prompts are synthetic templates,
which is why the classifier matches training phrasing well and generalises
poorly. Real traffic, labelled by how the chosen model actually performed, is
what fixes that.

The learning signal already exists in the log, but it is one-sided and reading
it symmetrically is what poisoned an earlier version of this file.

A judge score bounds the difficulty; it does not measure it. A failure at tier
T proves the task needs more than T, which is real evidence at any tier. A
success at tier T proves only that the task needs at most T -- informative when
T was cheap, worthless when T was expensive, because a frontier model
succeeding at a trivial task is exactly what you would expect either way.

So: difficulty is learned from failures, ease from cheap successes, and a
success on a strong tier is dropped rather than believed.

Labels are tiers, not model ids, for the reason set out in
models/train_router.py: a tier is a property of the prompt and survives any
change to the fleet, where a model id does not.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from config import settings, TIER_ORDER
from db.database import engine
from db.models import InferenceLog

logger = logging.getLogger("wormhole.feedback")

# Above this the served model is treated as having been sufficient.
GOOD_SCORE = 7.0

# Judge failures used to be stored as this exact value, indistinguishable from
# a real score after the fact. Rows carrying it are dropped rather than trusted.
LEGACY_PLACEHOLDER_SCORE = 8.5

# Tiers where a good score actually carries information. See the reasoning at
# the point of use: a success only ever bounds the difficulty from above, so it
# is worth learning from when the model was cheap and worth nothing when it was
# not.
INFORMATIVE_SUCCESS_TIERS = {"basic", "medium"}

# Written by the dispatcher when no candidate model could serve the request.
# Kept in sync with services/dispatcher.py by this constant rather than by
# hoping two strings stay identical.
GATEWAY_FAILURE_MARKER = "This text is not an answer."


def _tier_of(model_id: str) -> Optional[str]:
    """The capability tier the given model sits in."""
    cfg = settings.model_config_for(model_id)
    if cfg is None:
        return None
    tier = (cfg.intelligence_tier or "").lower()
    return tier if tier in TIER_ORDER else None


def _next_tier_up(tier: str) -> Optional[str]:
    idx = TIER_ORDER.index(tier)
    return TIER_ORDER[idx + 1] if idx + 1 < len(TIER_ORDER) else None


def collect_feedback_examples(min_prompt_chars: int = 12) -> List[Dict[str, Any]]:
    """Build (prompt, tier) examples from judged real traffic."""
    examples: List[Dict[str, Any]] = []
    skipped_unscored = skipped_placeholder = skipped_empty = 0
    skipped_gateway_failure = 0
    skipped_uninformative = escalated = 0

    with Session(engine) as session:
        rows = session.exec(select(InferenceLog)).all()

    for row in rows:
        prompt = (row.original_prompt or "").strip()
        if len(prompt) < min_prompt_chars:
            continue

        if row.judge_score is None:
            skipped_unscored += 1
            continue
        if row.judge_score == LEGACY_PLACEHOLDER_SCORE:
            skipped_placeholder += 1
            continue

        # A judge with nothing in front of it scored the plumbing, not the
        # task. Tool-only turns on the chat-completions path used to reach it
        # with an empty string, collect 1.0 for "no completion was provided",
        # and get escalated a tier -- which is how "rename userCnt to
        # userCount" ended up labelled frontier. The path is fixed, but the
        # rows it already wrote are still in the log, and a low score on an
        # empty completion is never evidence about difficulty.
        if not (row.completion or "").strip():
            skipped_empty += 1
            continue

        # The gateway writes a placeholder when every candidate model failed.
        # It is not a completion, the judge scores it 1.0 for containing no
        # work, and the escalation rule then reads that as "this prompt needs a
        # stronger tier" -- teaching the router from an outage. Same shape as
        # the empty-completion case above.
        if GATEWAY_FAILURE_MARKER in (row.completion or ""):
            skipped_gateway_failure += 1
            continue

        served_tier = _tier_of(row.selected_model)
        if served_tier is None:
            # A model no longer in the fleet. Its tier is what the label needed
            # and that is gone, so the row cannot be placed on the scale.
            continue

        if row.judge_score >= GOOD_SCORE:
            # A success is an upper bound, not a measurement: it proves the
            # task needed *at most* this tier. From a cheap tier that is real
            # evidence the task is easy, and it is the only way the router
            # ever learns to route down. From a strong tier it proves nothing
            # -- of course the expensive model managed it -- and taking it as
            # a label is how "How do I format a JSON string in Python?" came
            # to be labelled frontier, purely because a 120B model answered
            # it. Difficulty is learned from failures; ease is learned from
            # cheap successes.
            if served_tier not in INFORMATIVE_SUCCESS_TIERS:
                skipped_uninformative += 1
                continue
            label = served_tier
        else:
            # The served model underperformed, so this prompt belongs a tier up.
            label = _next_tier_up(served_tier)
            if label is None:
                # Already the strongest tier there is; a low score there is a
                # model-quality problem, not a routing one, so it teaches
                # nothing about where to send the prompt.
                continue
            escalated += 1

        # No credential or allow-list filtering here, unlike the previous
        # model-id labels. A tier is not something the gateway can fail to
        # reach: it describes the prompt, and the router resolves it against
        # whatever fleet is available at request time.
        examples.append({
            "prompt": prompt,
            "tier": label,
            "source": "feedback",
            "judge_score": row.judge_score,
        })

    logger.info(
        f"Feedback examples: {len(examples)} usable "
        f"({escalated} relabelled upward); skipped {skipped_unscored} unscored, "
        f"{skipped_placeholder} carrying the legacy placeholder score, "
        f"{skipped_empty} judged with an empty completion, "
        f"{skipped_gateway_failure} where every model failed, "
        f"{skipped_uninformative} passing on a tier too strong to prove anything."
    )
    return examples
