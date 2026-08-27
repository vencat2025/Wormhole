import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Resolve paths relative to the repository so the project runs from any
# checkout location, not only the machine it was written on.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "frontier_benchmark_dataset.json")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE_PATH = os.path.join(MODEL_OUTPUT_DIR, "router_slm.joblib")

def train_router_slm():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run scripts/build_benchmark_dataset.py first.")

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    # Close the loop: judged real traffic is what teaches the router to
    # generalise. The benchmark file only cold-starts it, and retraining on
    # that alone can never change the model no matter how often it is run.
    feedback = []
    try:
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from services.feedback import collect_feedback_examples
        feedback = collect_feedback_examples()
    except Exception as fb_err:
        print(f"⚠️  Could not load feedback examples ({fb_err}); training on benchmarks only.")

    # Real prompts are worth more than synthetic templates, so they are
    # repeated to carry weight against the much larger bootstrap set.
    FEEDBACK_WEIGHT = 3
    combined = [{"prompt": d["prompt"], "selected_model": d["selected_model"]} for d in dataset]
    for ex in feedback:
        combined.extend([{"prompt": ex["prompt"], "selected_model": ex["selected_model"]}] * FEEDBACK_WEIGHT)

    # A class the split cannot stratify on breaks training outright.
    from collections import Counter
    counts = Counter(c["selected_model"] for c in combined)
    combined = [c for c in combined if counts[c["selected_model"]] >= 2]

    X = [item["prompt"] for item in combined]
    y = [item["selected_model"] for item in combined]

    print(f"📊 Dataset size: {len(X)} samples "
          f"({len(dataset)} benchmark + {len(feedback)} judged real prompts x{FEEDBACK_WEIGHT})")
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("⚡ Training Local Router SLM (TF-IDF + Gradient Boosting Classifier)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2500, stop_words="english")),
        ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE_PATH)
    print(f"💾 Trained Local Router SLM saved to {MODEL_FILE_PATH}")

if __name__ == "__main__":
    train_router_slm()
