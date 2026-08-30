import logging
import os
from typing import List, Dict, Any, Optional
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
    # Whether the model emits native OpenAI-style function calls. A model that
    # cannot do this is unusable for agentic turns no matter how capable it is.
    supports_tools: bool = True
    # False means the rates below are placeholders chosen to preserve tier
    # ordering, not published prices. Routing is unaffected (it depends on
    # ordering), but every dollar figure derived from them is provisional.
    pricing_verified: bool = True
    # True means an agentic turn must go out over /v1/responses rather than
    # /v1/chat/completions. The reasoning tiers reject function tools on chat
    # completions unless reasoning_effort is 'none', which turns off the very
    # thing you picked the tier for. See services/openai_responses.py.
    requires_responses_api: bool = False


# Capability tiers, weakest first. Order is the whole meaning of the values.
TIER_ORDER = ["basic", "medium", "high", "frontier"]

# Credential each provider needs before any of its models can be reached.
# Ollama is local and needs none.
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

class Settings:
    PROJECT_NAME: str = "WormHole AI Cost Reducer"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./wormhole.db")

    # Default LLM configuration for the intermediate layer
    ENHANCER_MODEL: str = os.getenv("ENHANCER_MODEL", "groq/openai/gpt-oss-120b")
    ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "groq/openai/gpt-oss-120b")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "groq/openai/gpt-oss-120b")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "groq/openai/gpt-oss-120b")

    # Model used whenever the client supplies tools. Agentic turns are only
    # useful if the model reliably emits native function calls, so these are
    # routed to a known-good caller rather than through cost-based routing.
    AGENTIC_MODEL: str = os.getenv("AGENTIC_MODEL", "groq/openai/gpt-oss-120b")

    # Default API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Enterprise Security & Auth Settings.
    # Keys come only from the environment. Shipping a default key in source
    # would mean every deployment that turns auth on shares one publicly
    # known credential, which is worse than running with auth off, because
    # it looks protected.
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() in ("true", "1", "yes")
    VALID_API_KEYS: List[str] = [
        k.strip() for k in os.getenv("WORMHOLE_API_KEYS", os.getenv("WORMHOLE_API_KEY", "")).split(",")
        if k.strip()
    ]

    # Judging & Learning Loop Configuration
    # Judging runs async after every completion, scoring quality on 1-10 scale.
    # Scores feed back into router training for continuous improvement.
    # Trade-off: adds 10-30% overhead (extra API calls), but enables learning.
    # Disable if you want cheapest-possible routing without adaptation.
    ENABLE_JUDGING: bool = os.getenv("ENABLE_JUDGING", "true").lower() in ("true", "1", "yes")

    # Sample rate for judging (0.0 to 1.0). Only applicable if ENABLE_JUDGING=true.
    # 1.0 = judge every request (default, full training signal)
    # 0.1 = judge 10% of requests (90% cost savings, sparser training data)
    # Use this to trade off between cost and learning speed.
    JUDGE_SAMPLE_RATE: float = float(os.getenv("JUDGE_SAMPLE_RATE", "1.0"))

    # Base URL for locally hosted models. Everything routed through an
    # ollama/ id stays on this machine.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    # Enable or disable prompt enhancement. Enhancement rewrites prompts for weak
    # models, intending to lift output quality. Trade-off: adds ~500ms latency and
    # an extra API call per request, but no measured proof it helps enough to justify
    # the cost. Users should decide: cheaper routing (disable) or smarter routing (enable).
    ENABLE_ENHANCEMENT: bool = os.getenv("ENABLE_ENHANCEMENT", "true").lower() in ("true", "1", "yes")

    # Capability tiers that get prompt enhancement when ENABLE_ENHANCEMENT=true.
    # The enhancer exists to lift a weaker model's output toward what a frontier
    # model would have produced, so it is wasted on the strong tiers and skipped there.
    ENHANCE_TIERS: List[str] = [
        t.strip().lower() for t in os.getenv("ENHANCE_TIERS", "basic,medium").split(",") if t.strip()
    ]

    # Which router decides the model.
    #   slm  - local classifier, sub-millisecond, but it is bag-of-words over
    #          synthetic templates and does not generalise to unseen phrasing
    #   llm  - one cheap model call, ~200-400ms, semantic and far more robust
    #   auto - slm first, falling back to llm only if the classifier is absent
    ROUTER_MODE: str = os.getenv("ROUTER_MODE", "auto").strip().lower()

    # Lowest capability tier the router may choose. Empty means no floor.
    #
    # ROUTING_MODELS can already express this, but only by naming every model
    # you will accept, which has to be rewritten each time the fleet changes.
    # A floor survives that: it says "nothing weaker than this", and new models
    # are admitted or excluded on their own tier.
    #
    # This exists because nominally-capable is not the same as capable enough.
    # A model can advertise tool calling, pass a short-answer benchmark, and
    # still be unable to drive an agentic harness: routed to gpt-5-nano, a
    # one-line file-creation task in Claude Code spent 19 requests and never
    # created the file. Set MIN_ROUTING_TIER=medium before pointing a coding
    # agent at this gateway.
    MIN_ROUTING_TIER: str = os.getenv("MIN_ROUTING_TIER", "").strip().lower()

    # Restrict routing to an explicit set of model ids. Narrower than
    # ROUTING_PROVIDERS and usually what a real policy looks like: a short
    # ladder of approved tiers rather than "whatever this vendor sells".
    # Overlapping tiers make the choice arbitrary, so keep the ladder clean.
    ROUTING_MODELS: List[str] = [
        m.strip() for m in os.getenv("ROUTING_MODELS", "").split(",") if m.strip()
    ]

    # Restrict routing to specific providers. Empty means the whole fleet.
    # Set e.g. ROUTING_PROVIDERS=openai to keep every request inside one
    # vendor while still choosing the cheapest capable tier within it, which
    # is the common enterprise case: the policy is "not always the flagship",
    # not "switch vendors".
    ROUTING_PROVIDERS: List[str] = [
        p.strip().lower() for p in os.getenv("ROUTING_PROVIDERS", "").split(",") if p.strip()
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
            description="Open-source Qwen 2.5 Coder 7B running locally on Ollama at zero API cost. Chat only: see supports_tools.",
            speed_tier="fast",
            intelligence_tier="medium",
            # It emits native tool calls for a short prompt and a single tool,
            # but degrades to printing JSON as text under a real harness
            # preamble -- measured with both Codex and Claude Code. Routing an
            # agentic turn here produces an agent that explains instead of
            # acting, which is the failure this gateway exists to prevent.
            supports_tools=False
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
        # OpenAI 5-series. These ids were confirmed against /v1/models on a
        # live account; an earlier revision removed them believing they were
        # invented, which was wrong -- what was invented were their benchmark
        # figures. Rates below are litellm's published figures, checked against
        # its pricing map rather than estimated: an earlier revision guessed
        # them and had luna 7.5x too expensive and gpt-5.4 2x too cheap, which
        # is exactly the kind of error that silently distorts routing order.
        CandidateModelConfig(
            id="gpt-5-nano",
            name="GPT-5 Nano",
            provider="openai",
            input_cost_per_1k=0.00005,
            output_cost_per_1k=0.0004,
            description="Smallest 5-series tier for trivial recall, formatting and one-line edits.",
            speed_tier="ultra-fast",
            intelligence_tier="basic",
            pricing_verified=True
        ),
        CandidateModelConfig(
            id="gpt-5-mini",
            name="GPT-5 Mini",
            provider="openai",
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.0020,
            description="Everyday 5-series tier for routine coding, refactors and summarisation.",
            speed_tier="fast",
            intelligence_tier="medium",
            pricing_verified=True
        ),
        CandidateModelConfig(
            id="gpt-5.4",
            name="GPT-5.4",
            provider="openai",
            input_cost_per_1k=0.00250,
            output_cost_per_1k=0.0150,
            description="Strong 5-series reasoning tier for multi-file work and non-trivial debugging.",
            speed_tier="medium",
            intelligence_tier="high",
            pricing_verified=True
        ),
        CandidateModelConfig(
            id="gpt-5.6-luna",
            name="GPT-5.6 Luna",
            provider="openai",
            input_cost_per_1k=0.00020,
            output_cost_per_1k=0.0012,
            description="Fastest 5.6 reasoning tier. Everyday agentic coding and refactors.",
            speed_tier="fast",
            intelligence_tier="high",
            pricing_verified=True,
            requires_responses_api=True
        ),
        CandidateModelConfig(
            id="gpt-5.6-terra",
            name="GPT-5.6 Terra",
            provider="openai",
            input_cost_per_1k=0.00200,
            output_cost_per_1k=0.0120,
            description="Middle 5.6 reasoning tier. Multi-file work and non-trivial debugging.",
            speed_tier="medium",
            intelligence_tier="high",
            pricing_verified=True,
            requires_responses_api=True
        ),
        CandidateModelConfig(
            id="gpt-5.6-sol",
            name="GPT-5.6 Sol",
            provider="openai",
            input_cost_per_1k=0.00400,
            output_cost_per_1k=0.0200,
            description="Top 5.6 tier. Architecture, proofs, and the hardest agentic work.",
            speed_tier="medium",
            intelligence_tier="frontier",
            pricing_verified=True,
            # OpenAI rejects function tools for all three 5.6 tiers on
            # /v1/chat/completions unless reasoning_effort is 'none' -- which
            # would buy tool support by switching off the reasoning this tier
            # is paid for. The gateway sends them over /v1/responses instead,
            # where tools and full reasoning coexist, so supports_tools stays
            # True. See requires_responses_api above.
            requires_responses_api=True
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
    ]

    def should_enhance_for(self, model_id: str) -> bool:
        cfg = self.model_config_for(model_id)
        return bool(cfg) and cfg.intelligence_tier.lower() in self.ENHANCE_TIERS

    def tier_allowed(self, tier: str) -> bool:
        """Whether a capability tier clears the configured floor."""
        if not self.MIN_ROUTING_TIER:
            return True
        try:
            floor = TIER_ORDER.index(self.MIN_ROUTING_TIER)
        except ValueError:
            # An unrecognised floor must not silently disable the fleet, and
            # must not silently disable itself either. Ignore it loudly.
            logging.getLogger("wormhole.config").warning(
                "MIN_ROUTING_TIER=%r is not one of %s; ignoring the floor.",
                self.MIN_ROUTING_TIER, ", ".join(TIER_ORDER),
            )
            return True
        try:
            return TIER_ORDER.index((tier or "").lower()) >= floor
        except ValueError:
            return False  # unknown tier cannot be shown to clear a floor

    def provider_allowed(self, provider: str) -> bool:
        return not self.ROUTING_PROVIDERS or provider.lower() in self.ROUTING_PROVIDERS

    def model_allowed(self, model_id: str) -> bool:
        return not self.ROUTING_MODELS or model_id in self.ROUTING_MODELS

    def provider_has_credentials(self, provider: str) -> bool:
        env_var = PROVIDER_KEY_ENV.get(provider)
        if env_var is None:
            return True  # local or keyless provider
        return bool(getattr(self, env_var, "") or os.getenv(env_var, ""))

    def model_config_for(self, model_id: str) -> Optional[CandidateModelConfig]:
        for m in self.CANDIDATE_MODELS:
            if m.id == model_id:
                return m
        return None


settings = Settings()
