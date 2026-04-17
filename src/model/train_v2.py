"""
train_v2.py
Trains LogisticRegression, RandomForest, and LightGBM on crash severity
using ONLY forward-knowable (pre-event) features (see preprocess_v2.py).

Saves:
    models/best_model_v2.pkl          — best model dict {model, features, classes}
    models/feature_list_v2.txt        — one feature name per line
    reports/model_comparison_v2.csv   — accuracy + macro F1 for all 3 models
    reports/feature_importance_v2.png — top-20 feature importances for the winner
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
from preprocess_v2 import build_features_v2, CACHE_PATH

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

    print_section(name)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=classes))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())

    return {"model": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    try:
        if not hasattr(model, "feature_importances_"):
            print("Model has no feature_importances_, skipping plot.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values   = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances — {model_name} (v2)")
        plt.tight_layout()
        out = REPORTS_DIR / "feature_importance_v2.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nFeature importance plot saved to {out}")

        print("\nTop 10 most important features:")
        for i, (feat, val) in enumerate(zip(top_features[:10], top_values[:10]), 1):
            print(f"  {i:2d}. {feat:<50s}  {val:.4f}")

    except Exception as e:
        print(f"Could not plot feature importances: {e}")


def main():
    # ── Load & preprocess ────────────────────────────────────────────────────
    print_section("Loading Data (from cache)")
    df = pd.read_parquet(CACHE_PATH)
    print(f"Loaded {len(df):,} rows from {CACHE_PATH}")

    print_section("Preprocessing (v2 — forward-knowable features only)")
    X, y, feature_names = build_features_v2(df)

    classes = sorted(y.unique())
    print(f"\nClasses: {classes}")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size : {len(X_train):,}   Test size : {len(X_test):,}")

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
    print_section("Model Comparison (v2)")
    print(comparison_df.to_string(index=False))

    out_csv = REPORTS_DIR / "model_comparison_v2.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"\nComparison table saved to {out_csv}")

    # ── Best model ───────────────────────────────────────────────────────────
    best_row   = comparison_df.iloc[0]
    best_name  = best_row["model"]
    best_f1    = best_row["macro_f1"]
    best_model = trained[best_name]

    print_section(f"Winner: {best_name}  (Macro F1 = {best_f1})")

    # Save model dict (same format as v1)
    model_path = MODELS_DIR / "best_model_v2.pkl"
    joblib.dump(
        {"model": best_model, "features": feature_names, "classes": classes},
        model_path
    )
    print(f"Best model saved to {model_path}")

    # Save feature list
    feat_list_path = MODELS_DIR / "feature_list_v2.txt"
    with open(feat_list_path, "w") as f:
        for feat in feature_names:
            f.write(feat + "\n")
    print(f"Feature list saved to {feat_list_path}")

    # ── Feature importance ───────────────────────────────────────────────────
    plot_feature_importance(best_model, feature_names, best_name)


if __name__ == "__main__":
    main()
