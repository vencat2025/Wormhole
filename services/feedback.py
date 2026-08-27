"""Turn judged completions into router training data.

This is the loop that makes the router improve instead of being replaced. The
benchmark dataset only cold-starts it: those prompts are synthetic templates,
which is why the classifier matches training phrasing well and generalises
poorly. Real traffic, labelled by how the chosen model actually performed, is
what fixes that.

The learning signal already exists in the log. A good judge score means the
model that ran was adequate for that prompt, so it is a correct label. A poor
one means the task needed more than it got, so the prompt is relabelled to the
next tier up.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from config import settings
from db.database import engine
from db.models import InferenceLog

logger = logging.getLogger("wormhole.feedback")

# Above this the served model is treated as having been sufficient.
GOOD_SCORE = 7.0

# Judge failures used to be stored as this exact value, indistinguishable from
# a real score after the fact. Rows carrying it are dropped rather than trusted.
LEGACY_PLACEHOLDER_SCORE = 8.5


def _tier_ladder() -> List[str]:
    """Fleet model ids ordered cheapest first, used to escalate a bad label."""
    return [m.id for m in sorted(settings.CANDIDATE_MODELS, key=lambda c: c.input_cost_per_1k)]


def _next_tier_up(model_id: str) -> Optional[str]:
    ladder = _tier_ladder()
    if model_id not in ladder:
        return None
    idx = ladder.index(model_id)
    return ladder[idx + 1] if idx + 1 < len(ladder) else None


def collect_feedback_examples(min_prompt_chars: int = 12) -> List[Dict[str, Any]]:
    """Build (prompt, selected_model) examples from judged real traffic."""
    examples: List[Dict[str, Any]] = []
    skipped_unscored = skipped_placeholder = escalated = 0

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

        if row.judge_score >= GOOD_SCORE:
            label = row.selected_model
        else:
            # The served model underperformed, so this prompt belongs a tier up.
            label = _next_tier_up(row.selected_model)
            if label is None:
                # Already the strongest tier available; a low score there is a
                # model-quality problem, not a routing one, so it teaches
                # nothing about where to send the prompt.
                continue
            escalated += 1

        cfg = settings.model_config_for(label)
        if cfg is None:
            continue
        # Training on a model the gateway cannot reach spends classifier
        # capacity on a label whose every prediction gets substituted at
        # request time. Historic traffic contains several of these.
        if not settings.provider_has_credentials(cfg.provider):
            continue
        if not settings.provider_allowed(cfg.provider) or not settings.model_allowed(cfg.id):
            continue

        examples.append({
            "prompt": prompt,
            "selected_model": label,
            "source": "feedback",
            "judge_score": row.judge_score,
        })

    logger.info(
        f"Feedback examples: {len(examples)} usable "
        f"({escalated} relabelled upward); skipped {skipped_unscored} unscored, "
        f"{skipped_placeholder} carrying the legacy placeholder score."
    )
    return examples
