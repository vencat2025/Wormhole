import json
from typing import List, Dict, Any, Optional
from sqlmodel import Session, select
from db.database import engine
from db.models import InferenceLog

def export_router_dataset(min_score: float = 7.0) -> List[Dict[str, Any]]:
    """
    Exports training dataset for training/fine-tuning Model 2 (Router Model).
    Filters logs with judge_score >= min_score.
    """
    dataset = []
    with Session(engine) as session:
        statement = select(InferenceLog).where(
            (InferenceLog.judge_score != None) & (InferenceLog.judge_score >= min_score)
        )
        logs = session.exec(statement).all()
        for log in logs:
            dataset.append({
                "input": {
                    "enhanced_prompt": log.enhanced_prompt,
                },
                "output": {
                    "selected_model": log.selected_model,
                    "reasoning": log.router_reasoning
                },
                "metadata": {
                    "judge_score": log.judge_score,
                    "actual_cost": log.actual_cost,
                    "cost_savings": log.cost_savings
                }
            })
    return dataset

def export_enhancer_dataset(min_score: float = 7.0) -> List[Dict[str, Any]]:
    """
    Exports training dataset for training/fine-tuning Model 1 (Prompt Enhancer).
    """
    dataset = []
    with Session(engine) as session:
        statement = select(InferenceLog).where(
            (InferenceLog.judge_score != None) & (InferenceLog.judge_score >= min_score)
        )
        logs = session.exec(statement).all()
        for log in logs:
            dataset.append({
                "messages": [
                    {"role": "system", "content": "You are an expert Prompt Engineering AI. Enhance prompt for quality."},
                    {"role": "user", "content": f"Original Prompt:\n{log.original_prompt}"},
                    {"role": "assistant", "content": log.enhanced_prompt}
                ],
                "judge_score": log.judge_score
            })
    return dataset

def export_dataset_jsonl(target_type: str = "router", min_score: float = 7.0) -> str:
    """
    Returns string JSONL formatted lines ready for model fine-tuning API ingestion.
    """
    if target_type == "enhancer":
        records = export_enhancer_dataset(min_score)
    else:
        records = export_router_dataset(min_score)
        
    lines = [json.dumps(r) for r in records]
    return "\n".join(lines)
