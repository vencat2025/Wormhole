import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sqlmodel import Session, select
from db.database import engine
from db.models import InferenceLog

# Resolve paths relative to the repository so the project runs from any
# checkout location, not only the machine it was written on.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
EVALUATOR_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "quality_evaluator_slm.joblib")

class QualityEvaluatorSLM:
    """
    Adaptive Quality Evaluator SLM.
    Predicts completion quality score (1.0 - 10.0) for a given (enhanced_prompt, candidate_model) pair.
    Learns continuously from historical LLM-as-a-Judge ratings.
    """
    def __init__(self, vectorizer, regressor, model_q_table=None):
        self.vectorizer = vectorizer
        self.regressor = regressor
        self.model_q_table = model_q_table or {}

    def predict_quality(self, enhanced_prompt: str, candidate_model_id: str) -> float:
        prompt_vec = self.vectorizer.transform([enhanced_prompt])
        base_predicted_score = float(self.regressor.predict(prompt_vec)[0])
        
        # Apply model-specific learned Q-value adjustment
        q_bias = self.model_q_table.get(candidate_model_id, 0.0)
        final_score = min(10.0, max(1.0, base_predicted_score + q_bias))
        return round(final_score, 2)

def train_quality_evaluator_slm():
    print("⚡ Training Quality Evaluator & Adaptive Learning SLM...")
    
    # 1. Fetch historical DB logs with judge scores
    db_samples = []
    try:
        with Session(engine) as session:
            logs = session.exec(select(InferenceLog).where(InferenceLog.judge_score != None)).all()
            for log in logs:
                db_samples.append({
                    "prompt": log.enhanced_prompt,
                    "model": log.selected_model,
                    "score": log.judge_score
                })
    except Exception as e:
        print(f"DB fetch note: {e}")

    # Fallback synthetic training pairs if DB has few entries
    if len(db_samples) < 20:
        print("📥 Augmenting DB logs with benchmark feedback samples for training...")
        with open(os.path.join(PROJECT_ROOT, "data", "frontier_benchmark_dataset.json"), "r") as f:
            bm_data = json.load(f)
        for item in bm_data:
            db_samples.append({
                "prompt": item["prompt"],
                "model": item["selected_model"],
                "score": float(item["expected_pass_rate"]) * 10.0
            })

    X_prompts = [item["prompt"] for item in db_samples]
    y_scores = [item["score"] for item in db_samples]

    # Compute Q-table offsets per candidate model
    model_q_table = {}
    model_scores = {}
    for item in db_samples:
        m = item["model"]
        if m not in model_scores:
            model_scores[m] = []
        model_scores[m].append(item["score"])
        
    global_avg = float(np.mean(y_scores)) if y_scores else 8.5
    for m, scores in model_scores.items():
        model_avg = float(np.mean(scores))
        model_q_table[m] = round(model_avg - global_avg, 2)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words="english")
    X_vecs = vectorizer.fit_transform(X_prompts)

    regressor = Ridge(alpha=1.0)
    regressor.fit(X_vecs, y_scores)

    evaluator_slm = QualityEvaluatorSLM(vectorizer, regressor, model_q_table)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(evaluator_slm, EVALUATOR_MODEL_PATH)
    print(f"💾 Trained Quality Evaluator SLM saved to {EVALUATOR_MODEL_PATH}")
    print(f"🎯 Learned Model Q-Value Offsets: {model_q_table}")

if __name__ == "__main__":
    train_quality_evaluator_slm()
