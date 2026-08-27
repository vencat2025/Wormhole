import json
import logging
from typing import Tuple, Optional
import litellm
from sqlmodel import Session, select
from config import settings
from db.database import engine
from db.models import InferenceLog

logger = logging.getLogger("wormhole.judge")

JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous LLM Judge evaluating enterprise AI completions.
Evaluate the AI Completion against the User's Enhanced Prompt.

Scoring Criteria (1.0 to 10.0):
- 1.0 - 4.0: Severe hallucination, missing core requirements, broken code, or irrelevant output.
- 5.0 - 7.0: Partially correct, minor formatting issues or minor logical omissions.
- 8.0 - 10.0: High quality, accurate, complete, highly adhering to instructions and edge cases.

Respond ONLY with valid JSON matching this schema:
{
  "score": <float between 1.0 and 10.0>,
  "feedback": "<brief 1-2 sentence explanation of the score>"
}
"""

async def evaluate_completion(
    request_id: str,
    enhanced_prompt: str,
    completion: str,
    judge_model_name: Optional[str] = None
) -> Tuple[float, str]:
    """
    Evaluates the quality of the completion using LLM-as-a-Judge and saves results to DB.
    """
    model = judge_model_name or settings.JUDGE_MODEL
    
    # No score until the judge actually returns one. Writing a placeholder
    # would put a fabricated number in the same column as real evaluations,
    # where nothing downstream can tell them apart.
    score = None
    feedback = ""
    
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"ENHANCED PROMPT:\n{enhanced_prompt}\n\nAI COMPLETION:\n{completion}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            # 256 truncated the JSON mid-object, which the provider then
            # rejected as json_validate_failed, so nothing was ever scored.
            max_tokens=800
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        raw_score = data.get("score")
        score = float(raw_score) if raw_score is not None else None
        feedback = data.get("feedback", feedback)
    except Exception as e:
        logger.warning(
            f"LLM Judge call failed ({e}). Leaving this request unscored; "
            f"completion length is not a measure of quality."
        )
        score = None
        feedback = f"Not evaluated: judge call failed ({type(e).__name__})."

    # Update database record with judge score & feedback
    try:
        with Session(engine) as session:
            statement = select(InferenceLog).where(InferenceLog.request_id == request_id)
            log_record = session.exec(statement).first()
            if log_record:
                log_record.judge_score = score
                log_record.judge_feedback = feedback
                log_record.judge_model = model
                session.add(log_record)
                session.commit()
    except Exception as db_err:
        logger.error(f"Failed to persist judge evaluation to DB: {db_err}")

    return score, feedback
