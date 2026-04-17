"""
train_v3.py
Trains LogisticRegression, RandomForest, and LightGBM on crash severity with
SMOTE oversampling on the training set to improve High-class recall.

Key differences from v2:
  - SMOTE(random_state=42, k_neighbors=5) applied ONLY to X_train / y_train.
  - Test set is NEVER touched by SMOTE (no label-leakage).
  - Same 80-feature schema as v2 (imported from preprocess_v2).
  - Same class_weight='balanced' on all three models.

Saves:
    models/best_model_v3.pkl          — best model dict {model, features, classes}
    models/feature_list_v3.txt        — one feature name per line (same as v2)
    reports/model_comparison_v3.csv   — accuracy + macro F1 for all 3 models
    reports/feature_importance_v3.png — top-20 feature importances for the winner
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
from imblearn.over_sampling import SMOTE

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
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values   = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features[::-1], top_values[::-1], color="teal")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances — {model_name} (v3)")
        plt.tight_layout()
        out = REPORTS_DIR / "feature_importance_v3.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nFeature importance plot saved to {out}")

        print("\nTop 10 most important features (v3):")
        for i, (feat, val) in enumerate(zip(top_features[:10], top_values[:10]), 1):
            print(f"  {i:2d}. {feat:<50s}  {val:.4f}")

        return list(zip(top_features[:10], [round(float(v), 4) for v in top_values[:10]]))

    except Exception as e:
        print(f"Could not plot feature importances: {e}")
        return None


def main():
    # ── Load & preprocess ────────────────────────────────────────────────────
    print_section("Loading Data (from cache)")
    df = pd.read_parquet(CACHE_PATH)
    print(f"Loaded {len(df):,} rows from {CACHE_PATH}")

    print_section("Preprocessing (v2 features — same schema, new training strategy)")
    X, y, feature_names = build_features_v2(df)

    classes = sorted(y.unique())
    print(f"\nClasses: {classes}")
    print("\nClass distribution (original):")
    for cls in classes:
        count = (y == cls).sum()
        print(f"  {cls:<10s}: {count:>6,}  ({100*count/len(y):.2f}%)")

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size : {len(X_train):,}   Test size : {len(X_test):,}")

    # ── SMOTE oversampling — ONLY on training set ────────────────────────────
    print_section("Applying SMOTE to training set (k_neighbors=5)")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — train size: {len(X_train_res):,}")
    print("Class distribution after SMOTE:")
    for cls in classes:
        count = (y_train_res == cls).sum()
        print(f"  {cls:<10s}: {count:>6,}  ({100*count/len(y_train_res):.2f}%)")

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
        print_section(f"Training {name} (on SMOTE-resampled data) ...")
        model.fit(X_train_res, y_train_res)
        trained[name] = model
        result = evaluate(name, model, X_test, y_test, classes)
        results.append(result)

    # ── Comparison table ─────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print_section("Model Comparison (v3 — SMOTE + balanced weights)")
    print(comparison_df.to_string(index=False))

    out_csv = REPORTS_DIR / "model_comparison_v3.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"\nComparison table saved to {out_csv}")

    # ── v1 vs v2 vs v3 comparison ─────────────────────────────────────────────
    print_section("v1 vs v2 vs v3 Macro F1 Comparison")
    v1_f1 = 0.4616  # LightGBM v1 (includes leakage — for reference only)
    v2_f1 = 0.4025  # LightGBM v2 (forward-knowable features, no SMOTE)
    v3_f1 = comparison_df.iloc[0]["macro_f1"]
    print(f"  v1 LightGBM (leakage): {v1_f1:.4f}")
    print(f"  v2 LightGBM (clean)  : {v2_f1:.4f}")
    print(f"  v3 {comparison_df.iloc[0]['model']:<20s} : {v3_f1:.4f}  ← SMOTE")
    if v3_f1 > v2_f1:
        print(f"  >> Improvement over v2: +{v3_f1 - v2_f1:.4f}")
    else:
        print(f"  >> Change vs v2: {v3_f1 - v2_f1:+.4f}")

    # ── Best model ───────────────────────────────────────────────────────────
    best_row   = comparison_df.iloc[0]
    best_name  = best_row["model"]
    best_f1    = best_row["macro_f1"]
    best_model = trained[best_name]

    print_section(f"Winner: {best_name}  (Macro F1 = {best_f1})")

    # Save model dict (same format as v1/v2)
    model_path = MODELS_DIR / "best_model_v3.pkl"
    joblib.dump(
        {"model": best_model, "features": feature_names, "classes": classes},
        model_path
    )
    print(f"Best model saved to {model_path}")

    # Save feature list (same features as v2, kept separate for traceability)
    feat_list_path = MODELS_DIR / "feature_list_v3.txt"
    with open(feat_list_path, "w") as f:
        for feat in feature_names:
            f.write(feat + "\n")
    print(f"Feature list saved to {feat_list_path}")

    # ── Feature importance ───────────────────────────────────────────────────
    plot_feature_importance(best_model, feature_names, best_name)


if __name__ == "__main__":
    main()
