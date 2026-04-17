"""
train.py
Trains LogisticRegression, RandomForest, and LightGBM on crash severity.
Evaluates all three, saves the best (by macro F1), and writes reports.
"""

import sys
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier

# Make local imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import fetch_all_crashes
from preprocess import build_features

warnings.filterwarnings("ignore")

REPO_ROOT   = Path(__file__).resolve().parents[2]
MODELS_DIR  = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def evaluate(name, model, X_test, y_test, classes):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="macro")

    print_section(f"{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=classes))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())

    return {"model": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def plot_feature_importance(model, feature_names, model_name):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            print("Model has no feature_importances_, skipping plot.")
            return

        indices = np.argsort(importances)[::-1][:20]
        top_features = [feature_names[i] for i in indices]
        top_values   = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 20 Feature Importances — {model_name}")
        plt.tight_layout()
        out = REPORTS_DIR / "feature_importance.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nFeature importance plot saved to {out}")

        print("\nTop 10 most important features:")
        for i, (feat, val) in enumerate(zip(top_features[:10], top_values[:10]), 1):
            print(f"  {i:2d}. {feat:<45s}  {val:.4f}")

    except Exception as e:
        print(f"Could not plot feature importances: {e}")


def main():
    # ── Load & preprocess ────────────────────────────────────────────────────
    print_section("Loading Data")
    df = fetch_all_crashes()

    print_section("Preprocessing")
    X, y = build_features(df)

    classes = sorted(y.unique())
    print(f"\nClasses: {classes}")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,}   Test size: {len(X_test):,}")

    feature_names = list(X.columns)

    # ── Models ───────────────────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1
        ),
    }

    results = []
    trained = {}

    for name, model in models.items():
        print_section(f"Training {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
        result = evaluate(name, model, X_test, y_test, classes)
        results.append(result)

    # ── Comparison table ─────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print_section("Model Comparison")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"\nComparison table saved to {REPORTS_DIR / 'model_comparison.csv'}")

    # ── Best model ───────────────────────────────────────────────────────────
    best_row   = comparison_df.iloc[0]
    best_name  = best_row["model"]
    best_f1    = best_row["macro_f1"]
    best_model = trained[best_name]

    print_section(f"Winner: {best_name}  (Macro F1 = {best_f1})")

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump({"model": best_model, "features": feature_names, "classes": classes}, model_path)
    print(f"Best model saved to {model_path}")

    # ── Feature importance ───────────────────────────────────────────────────
    plot_feature_importance(best_model, feature_names, best_name)


if __name__ == "__main__":
    main()
