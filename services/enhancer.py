import os
import logging
import joblib
import litellm
from config import settings
from models.slm_structures import LocalEnhancerSLM

logger = logging.getLogger("wormhole.enhancer")

# Resolve paths relative to the repository so the project runs from any
# checkout location, not only the machine it was written on.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, "models", "enhancer_slm.joblib")
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


# The chat-oriented enhancement templates ask for structured prose and
# "code blocks with clean syntax highlighting". On an agentic turn that is
# actively harmful: it steers a weak model into printing a tutorial instead of
# calling the tools, which is the failure this gateway exists to prevent. Tool
# turns get criteria pointed the other way.
AGENTIC_ENHANCEMENT = """{original}

Execution requirements:
- Carry out this task now by calling the available tools. Do not describe the steps.
- Code shown in a reply is not a file. Only a tool call creates one.
- Create any parent directories before writing into them.
- After the tools have run, verify the result, then summarise what now exists in one or two sentences."""


def enhance_for_tools(original_prompt: str) -> str:
    """Deterministic, local enhancement suited to a tool-using turn."""
    return AGENTIC_ENHANCEMENT.format(original=original_prompt.strip())


async def enhance_prompt(original_prompt: str, model_name: str = None, for_tools: bool = False) -> str:
    """
    Enhance the input prompt to improve downstream completion quality.
    Prioritizes fast local SLM inference (<1ms) trained on prompt structuring templates.
    """
    if for_tools:
        return enhance_for_tools(original_prompt)

    local_slm = _get_local_enhancer_slm()
    if local_slm is not None:
        try:
            enhanced_text = local_slm.enhance(original_prompt)
            return enhanced_text
        except Exception as slm_err:
            logger.warning(f"Local Enhancer SLM execution failed ({slm_err}), falling back to API/heuristic enhancer.")

    model = model_name or settings.ENHANCER_MODEL
    
    try:
        # A local model needs its base URL supplied explicitly, so that
        # JUDGE_MODEL/ENHANCER_MODEL can point at ollama/ and keep the
        # whole pipeline on the machine.
        extra = {"api_base": settings.OLLAMA_BASE_URL} if model.startswith("ollama/") else {}
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ENHANCER_PROMPT},
                {"role": "user", "content": f"Original Prompt:\n{original_prompt}"}
            ],
            temperature=0.3,
            **extra,
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
