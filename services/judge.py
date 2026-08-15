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
    
    score = 8.5
    feedback = "Auto-evaluated: Satisfactory completion matching enhanced prompt."
    
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
            max_tokens=256
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        score = float(data.get("score", 8.5))
        feedback = data.get("feedback", feedback)
    except Exception as e:
        logger.warning(f"LLM Judge call failed or unconfigured ({e}). Utilizing default evaluation score.")
        # Fallback scoring heuristic
        if len(completion) > 20:
            score = 8.5
            feedback = "Fallback heuristic judge score: Completion length adequate."
        else:
            score = 5.0
            feedback = "Fallback heuristic judge score: Short output."

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
