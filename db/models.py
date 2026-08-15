from typing import Optional
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

class InferenceLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Prompt Details
    original_prompt: str
    enhanced_prompt: str
    
    # Routing Details
    enhancer_model: str
    router_model: str
    router_reasoning: Optional[str] = None
    selected_model: str
    baseline_model: str = "gpt-4o"
    
    # Execution & Cost Metrics
    completion: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    actual_cost: float = 0.0
    baseline_cost: float = 0.0
    cost_savings: float = 0.0
    latency_ms: float = 0.0
    
    # Auto-Judge Feedback & Learning Score
    judge_score: Optional[float] = None  # Scale 1.0 - 10.0
    judge_feedback: Optional[str] = None
    judge_model: Optional[str] = None
