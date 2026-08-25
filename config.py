import os
from typing import List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class CandidateModelConfig(BaseModel):
    id: str
    name: str
    provider: str
    input_cost_per_1k: float  # USD per 1,000 input tokens
    output_cost_per_1k: float # USD per 1,000 output tokens
    description: str
    max_tokens: int = 4096
    speed_tier: str = "fast" # "fast", "medium", "slow"
    intelligence_tier: str = "high" # "basic", "medium", "high", "frontier"

class Settings:
    PROJECT_NAME: str = "WormHole AI Cost Reducer"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./wormhole.db")

    # Default LLM configuration for the intermediate layer
    ENHANCER_MODEL: str = os.getenv("ENHANCER_MODEL", "groq/openai/gpt-oss-120b")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "groq/openai/gpt-oss-120b")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "groq/openai/gpt-oss-120b")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "groq/openai/gpt-oss-120b")

    # Default API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Enterprise Security & Auth Settings
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() in ("true", "1", "yes")
    VALID_API_KEYS: List[str] = [
        os.getenv("WORMHOLE_API_KEY", "wh_live_demo123456789"),
        "wh_live_enterprise_default_key"
    ]

    # Circuit Breaker & Failover Settings
    CIRCUIT_BREAKER_THRESHOLD: int = 3  # Max consecutive failures before bypassing provider

    # Registered Candidate Target Models in Enterprise Fleet (Cloud + Open-Source Local)
    CANDIDATE_MODELS: List[CandidateModelConfig] = [
        CandidateModelConfig(
            id="groq/openai/gpt-oss-120b",
            name="GPT OSS 120B (Groq Cloud)",
            provider="groq",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            description="Ultra fast 120B frontier model hosted on Groq LPU hardware.",
            speed_tier="ultra-fast",
            intelligence_tier="frontier"
        ),
        CandidateModelConfig(
            id="groq/openai/gpt-oss-20b",
            name="GPT OSS 20B (Groq Cloud)",
            provider="groq",
            input_cost_per_1k=0.000075,
            output_cost_per_1k=0.0003,
            description="High-speed 20B reasoning model hosted on Groq LPU hardware.",
            speed_tier="ultra-fast",
            intelligence_tier="high"
        ),
        CandidateModelConfig(
            id="groq/qwen/qwen3.6-27b",
            name="Qwen 3.6 27B (Groq Cloud)",
            provider="groq",
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0004,
            description="Qwen 3.6 27B model hosted on Groq LPU hardware.",
            speed_tier="fast",
            intelligence_tier="high"
        ),
        CandidateModelConfig(
            id="ollama/qwen2.5-coder:7b",
            name="Qwen 2.5 Coder 7B (Local)",
            provider="ollama",
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            description="Open-source Qwen 2.5 Coder 7B model running locally on Ollama with 100% free $0 API cost.",
            speed_tier="fast",
            intelligence_tier="medium"
        ),
        CandidateModelConfig(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider="openai",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
            description="Fast, low-cost model ideal for simple to medium reasoning, code generation, summarization, and formatting.",
            speed_tier="fast",
            intelligence_tier="medium"
        ),
        CandidateModelConfig(
            id="gemini/gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider="google",
            input_cost_per_1k=0.000075,
            output_cost_per_1k=0.0003,
            description="Ultra fast and cheap Google model with huge context window and solid overall reasoning.",
            speed_tier="fast",
            intelligence_tier="medium"
        ),
        CandidateModelConfig(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider="anthropic",
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.00125,
            description="Lightweight Anthropic model for rapid responses, text processing, and structured extraction.",
            speed_tier="fast",
            intelligence_tier="medium"
        ),
        CandidateModelConfig(
            id="gemini/gemini-2.5-pro",
            name="Gemini 2.5 Pro",
            provider="google",
            input_cost_per_1k=0.00125,
            output_cost_per_1k=0.0050,
            description="High-capacity Google model with 2M token context, suited for deep context search and complex logic.",
            speed_tier="fast",
            intelligence_tier="high"
        ),
        CandidateModelConfig(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            input_cost_per_1k=0.0025,
            output_cost_per_1k=0.0100,
            description="Frontier multimodal model for highly complex reasoning, advanced math, and deep architectural code synthesis.",
            speed_tier="medium",
            intelligence_tier="frontier"
        ),
        CandidateModelConfig(
            id="gpt-4.5",
            name="GPT-4.5",
            provider="openai",
            input_cost_per_1k=0.0050,
            output_cost_per_1k=0.0200,
            description="Next-generation OpenAI frontier model for deep reasoning, mathematical synthesis, and software engineering.",
            speed_tier="medium",
            intelligence_tier="frontier"
        ),
        CandidateModelConfig(
            id="gpt-5.4",
            name="GPT-5.4",
            provider="openai",
            input_cost_per_1k=0.0060,
            output_cost_per_1k=0.0240,
            description="Advanced OpenAI 5-series model optimized for enterprise system architecture, multi-step planning, and agentic workflows.",
            speed_tier="medium",
            intelligence_tier="frontier"
        ),
        CandidateModelConfig(
            id="gpt-5.6",
            name="GPT-5.6",
            provider="openai",
            input_cost_per_1k=0.0075,
            output_cost_per_1k=0.0300,
            description="Flagship OpenAI 5-series frontier coding and reasoning model for high-complexity autonomous software engineering.",
            speed_tier="medium",
            intelligence_tier="frontier"
        ),
    ]

settings = Settings()
