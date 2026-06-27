"""
train_classification.py
-------------------------
Trains and evaluates multiple classification models on the customer
churn dataset, selects the best performer (by ROC-AUC), and saves it
(along with the fitted preprocessor) to the models/ directory.

Run:
    python src/train_classification.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from preprocessing import (
    add_churn_features,
    build_preprocessor,
    CHURN_NUMERIC_FEATURES,
    CHURN_CATEGORICAL_FEATURES,
    CHURN_TARGET,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, pos_label=1)
    rec = recall_score(y_test, preds, pos_label=1)
    f1 = f1_score(y_test, preds, pos_label=1)
    auc = roc_auc_score(y_test, probs)

    print(f"\n📊 {name}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")

    return {
        "model_name": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC_AUC": auc
    }


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = add_churn_features(df)

    X = df[CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES]
    y = df[CHURN_TARGET].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor(CHURN_NUMERIC_FEATURES, CHURN_CATEGORICAL_FEATURES)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_SEED
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test, name)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    print("\n🏆 Model comparison (sorted by ROC-AUC):")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model_name"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n✅ Best model: {best_name}")

    # Print full classification report for the winning model
    final_preds = best_pipeline.predict(X_test)
    print("\nClassification report (best model):")
    print(classification_report(y_test, final_preds, target_names=["No Churn", "Churn"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, final_preds))

    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"\n💾 Saved best pipeline to {model_path}")

    results_df.to_csv(os.path.join(MODEL_DIR, "churn_model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
