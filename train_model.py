from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODEL_PATH = BASE_DIR / "spam_classifier.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

def main():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["label"].isin(["spam", "ham"])]

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"],
        test_size=0.20, random_state=42, stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=5000
        )),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)), 4),
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test))
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nModel training completed!")
    print(f"Dataset rows: {len(data)}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1-score:  {metrics['f1_score']:.2%}")
    print("\nConfusion matrix (rows = actual [ham, spam], columns = predicted [ham, spam]):")
    print(metrics["confusion_matrix"])
    print(f"\nSaved model to: {MODEL_PATH.name}")

if __name__ == "__main__":
    main()
