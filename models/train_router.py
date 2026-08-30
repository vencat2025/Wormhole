"""Train the local routing classifier.

The classifier predicts a **capability tier** -- basic / medium / high /
frontier -- and not a model id. That distinction is the whole design.

A tier is a property of the prompt: "this task needs real reasoning" stays true
no matter which models you have keys for. A model id is a property of your
fleet, and the moment the fleet changes, every label the classifier learned is
wrong. Measured on the previous model-id classifier: pointed at an all-5.6
ladder, none of its labels existed any more, so every prediction missed and a
rename and a zero-downtime sharding migration both landed on the cheapest tier.

Training on tiers instead means adding a provider key, retiring a model, or
switching vendors entirely never invalidates the classifier. The router asks it
how hard the task is, then spends the least money that buys that capability
from whatever the user actually has credentials for.

The bootstrap labels come from measured benchmark difficulty rather than
opinion. SWE-bench carries a real per-instance pass rate across 134 systems, so
the instances almost nothing solved are the ones marked frontier.
"""

import os
import re
import json
import joblib
from collections import Counter

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

# Almost no system in the SWE-bench field solved an instance below this rate,
# which is the strongest evidence available that a task needs the top tier.
FRONTIER_PASS_RATE = 0.2


def tier_for_record(record: dict) -> str:
    """Map one benchmark record onto the capability tier it demands.

    `difficulty` carries most of the signal, but it only has three values while
    the fleet has four tiers, and collapsing hard-and-frontier loses exactly the
    distinction that matters most: it is the difference between a cheap
    reasoning tier and the expensive one. Where a measured pass rate exists,
    use it to recover that split.
    """
    difficulty = (record.get("difficulty") or "").lower()
    if difficulty == "easy":
        return "basic"
    if difficulty == "medium":
        return "medium"

    benchmark = record.get("benchmark")
    pass_rate = record.get("expected_pass_rate")

    # Graduate-level science questions, and the SWE-bench instances the field
    # as a whole could not solve.
    if benchmark == "GPQA":
        return "frontier"
    if benchmark == "SWE-bench" and pass_rate is not None and pass_rate < FRONTIER_PASS_RATE:
        return "frontier"
    return "high"


def short_form_of(record: dict) -> str:
    """The opening line of a SWE-bench issue, which is its title.

    The bootstrap otherwise teaches the classifier one register only: long
    prose. SWE-bench items are multi-paragraph bug reports, HumanEval items are
    function specifications, and nothing in it resembles what people actually
    type at a coding harness -- a single short imperative line. Measured, the
    classifier called "design a zero-downtime migration to shard the orders
    table" basic, having never seen a short sentence that was hard.

    An issue title is the same real text with the same measured label, just in
    the short form, so adding it teaches that brevity is not evidence of ease
    without inventing a single prompt.
    """
    title = re.split(r"\s*(?:###|\*\*|\n)", record.get("prompt", ""), maxsplit=1)[0].strip()
    # Too short to carry meaning, or so long it is not really a title.
    return title if 20 <= len(title) <= 160 else ""


def train_router_slm():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run scripts/build_benchmark_dataset.py first."
        )

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

    # Real prompts are worth more than benchmark items, so they are repeated to
    # carry weight against the much larger bootstrap set.
    FEEDBACK_WEIGHT = 3
    combined = [{"prompt": d["prompt"], "tier": tier_for_record(d)} for d in dataset]

    # Same items, short form, same measured labels. See short_form_of.
    short_added = 0
    for d in dataset:
        if d.get("benchmark") != "SWE-bench":
            continue
        title = short_form_of(d)
        if title:
            combined.append({"prompt": title, "tier": tier_for_record(d)})
            short_added += 1
    for ex in feedback:
        combined.extend([{"prompt": ex["prompt"], "tier": ex["tier"]}] * FEEDBACK_WEIGHT)

    # A class the split cannot stratify on breaks training outright.
    counts = Counter(c["tier"] for c in combined)
    combined = [c for c in combined if counts[c["tier"]] >= 2]

    X = [item["prompt"] for item in combined]
    y = [item["tier"] for item in combined]

    print(f"📊 Dataset size: {len(X)} samples "
          f"({len(dataset)} benchmark + {short_added} issue titles "
          f"+ {len(feedback)} judged real prompts x{FEEDBACK_WEIGHT})")
    print("   tier distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(y).items())))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("⚡ Training Local Router SLM (TF-IDF + Gradient Boosting Classifier)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2500, stop_words="english")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE_PATH)
    print(f"💾 Trained Local Router SLM saved to {MODEL_FILE_PATH}")


if __name__ == "__main__":
    train_router_slm()
