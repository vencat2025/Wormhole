import logging
import litellm
from config import settings

logger = logging.getLogger("wormhole.enhancer")

SYSTEM_ENHANCER_PROMPT = """You are an expert Prompt Engineering AI. Your task is Quality Enhancement of enterprise prompts.
Take the user's input prompt and optimize it into a clear, highly structured, unambiguous, and comprehensive prompt.
Guidelines:
1. Expand terse or implicit requirements into explicit criteria (edge cases, formatting, expected outputs).
2. Maintain the original core intent of the user.
3. Structure the prompt with clear headings, constraints, and instructions so that even smaller or cheaper LLMs can follow it seamlessly and produce frontier-quality outputs.
4. Output ONLY the enhanced prompt content. Do not include meta-conversational text like "Here is your enhanced prompt:".
"""

async def enhance_prompt(original_prompt: str, model_name: str = None) -> str:
    """
    Enhance the input prompt to improve downstream completion quality.
    """
    model = model_name or settings.ENHANCER_MODEL
    
    # Check if API keys are configured, otherwise provide structured fallback enhancement
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
        # Rule-based fallback enhancement for offline/development mode
        enhanced_text = (
            f"[ENHANCED FOR QUALITY & PRECISION]\n\n"
            f"Objective: {original_prompt}\n\n"
            f"Instructions:\n"
            f"- Provide a complete, correct, and well-structured response.\n"
            f"- Pay strict attention to accuracy, clarity, and edge cases.\n"
            f"- Format code blocks cleanly with proper syntax highlighting if applicable."
        )
        return enhanced_text
