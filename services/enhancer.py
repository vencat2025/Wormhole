import os
import logging
import joblib
import litellm
from config import settings
from models.slm_structures import LocalEnhancerSLM

logger = logging.getLogger("wormhole.enhancer")

MODEL_FILE_PATH = "/Users/venkat/Documents/AI/WormHole/models/enhancer_slm.joblib"
_LOCAL_ENHANCER_SLM = None

def _get_local_enhancer_slm():
    global _LOCAL_ENHANCER_SLM
    if _LOCAL_ENHANCER_SLM is None and os.path.exists(MODEL_FILE_PATH):
        try:
            _LOCAL_ENHANCER_SLM = joblib.load(MODEL_FILE_PATH)
            logger.info("⚡ Local Enhancer SLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load local Enhancer SLM ({e}).")
    return _LOCAL_ENHANCER_SLM

SYSTEM_ENHANCER_PROMPT = """You are an expert Prompt Engineering AI. Your task is Quality Enhancement of enterprise prompts.
Take the user's input prompt and optimize it into a clear, highly structured, unambiguous, and comprehensive prompt.
Guidelines:
1. Expand terse or implicit requirements into explicit criteria (edge cases, formatting, expected outputs).
2. Maintain the original core intent of the user.
3. Structure the prompt with clear headings, constraints, and instructions so that downstream models produce frontier-quality outputs.
4. Output ONLY the enhanced prompt content. Do not include meta-conversational text like "Here is your enhanced prompt:".
5. IMPORTANT: If the prompt requests creating an application, writing code, or generating workspace files, explicitly instruct the model to execute shell commands (<exec>...</exec>) to create all necessary files directly, rather than writing conversational tutorial guides or instructions.
"""

async def enhance_prompt(original_prompt: str, model_name: str = None) -> str:
    """
    Enhance the input prompt to improve downstream completion quality.
    Prioritizes fast local SLM inference (<1ms) trained on prompt structuring templates.
    """
    local_slm = _get_local_enhancer_slm()
    if local_slm is not None:
        try:
            enhanced_text = local_slm.enhance(original_prompt)
            return enhanced_text
        except Exception as slm_err:
            logger.warning(f"Local Enhancer SLM execution failed ({slm_err}), falling back to API/heuristic enhancer.")

    model = model_name or settings.ENHANCER_MODEL
    
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ENHANCER_PROMPT},
                {"role": "user", "content": f"Original Prompt:\n{original_prompt}"}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        enhanced_text = response.choices[0].message.content.strip()
        return enhanced_text
    except Exception as e:
        logger.warning(f"Enhancer API call failed or unconfigured ({e}). Falling back to heuristic enhancement.")
        enhanced_text = (
            f"[ENHANCED FOR QUALITY & PRECISION]\n\n"
            f"Objective: {original_prompt}\n\n"
            f"Instructions:\n"
            f"- Provide a complete, correct, and well-structured response.\n"
            f"- Pay strict attention to accuracy, clarity, and edge cases.\n"
            f"- Format code blocks cleanly with proper syntax highlighting if applicable."
        )
        return enhanced_text
