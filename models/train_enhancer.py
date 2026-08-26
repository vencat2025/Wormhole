import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from models.slm_structures import LocalEnhancerSLM

# Resolve paths relative to the repository so the project runs from any
# checkout location, not only the machine it was written on.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "enhancer_dataset.json")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE_PATH = os.path.join(MODEL_OUTPUT_DIR, "enhancer_slm.joblib")

def train_enhancer_slm():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run scripts/build_enhancer_dataset.py first.")

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    X_texts = [item["original_prompt"] for item in dataset]

    print(f"📊 Training Local Enhancer SLM on {len(X_texts)} prompt samples...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, stop_words="english")
    X_vecs = vectorizer.fit_transform(X_texts)

    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(X_vecs)

    slm_model = LocalEnhancerSLM(vectorizer, knn, dataset)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(slm_model, MODEL_FILE_PATH)
    print(f"💾 Trained Local Enhancer SLM saved to {MODEL_FILE_PATH}")

if __name__ == "__main__":
    train_enhancer_slm()
