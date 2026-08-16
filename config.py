import os
from typing import List, Dict, Any
from pydantic import BaseModel

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
    ENHANCER_MODEL: str = os.getenv("ENHANCER_MODEL", "groq/llama-3.3-70b-versatile")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "groq/llama-3.3-70b-versatile")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "groq/llama-3.3-70b-versatile")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "groq/llama-3.3-70b-versatile")

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
            id="groq/llama-3.3-70b-versatile",
            name="Llama 3.3 70B (Groq Cloud)",
            provider="groq",
            input_cost_per_1k=0.00059,
            output_cost_per_1k=0.00079,
            description="Ultra fast 70B cloud model hosted on Groq LPU hardware with high intelligence and zero latency.",
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
            id="claude-3-5-sonnet-20240620",
            name="Claude 3.5 Sonnet",
            provider="anthropic",
            input_cost_per_1k=0.0030,
            output_cost_per_1k=0.0150,
            description="High-tier reasoning and coding model with exceptional code generation capabilities.",
            speed_tier="medium",
            intelligence_tier="frontier"
        ),
    ]

settings = Settings()
