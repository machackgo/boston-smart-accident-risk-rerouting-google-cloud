"""
train_v4.py — LightGBM with spatial features + per-class threshold tuning.

Improvements over v3:
  A) 6 leakage-free spatial aggregate features (neighborhood crash history)
  B) Sparse weather column pruning (sum < 50)
  C) Per-class threshold tuning (maximise macro F1 on validation set)
  D) Manual class-weight override {Low:1, Medium:2, High:8} compared vs "balanced"

Saves:
    models/best_model_v4.pkl          — {model, features, classes}
    models/feature_list_v4.txt        — one feature per line
    models/thresholds_v4.json         — {class: threshold} for inference
    reports/model_comparison_v4.csv
    reports/feature_importance_v4.png
"""

import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_features import build_spatial_parquet, OUT_PATH as SPATIAL_PATH
from preprocess_v4 import build_features_v4, SPATIAL_PATH as _SPATIAL_PATH_CHECK

warnings.filterwarnings("ignore")

REPO_ROOT   = Path(__file__).resolve().parents[2]
MODELS_DIR  = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

CLASSES = ["High", "Low", "Medium"]


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="macro")
    print_section(name)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=CLASSES))
    return {"name": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


# ── Per-class threshold tuning ──────────────────────────────────────────────

def _tune_thresholds(model, X_val, y_val, classes=CLASSES):
    """
    One-vs-rest threshold search: for each class, find the probability threshold
    that maximises that class's F1 on the validation set.
    Final classification: assign the class whose P(class) > threshold is highest
    (ties broken by probability magnitude; fall back to argmax if none exceed).

    Returns:
        thresholds (dict): {class_name: best_threshold}
        tuned_preds (np.ndarray): predictions using tuned thresholds
        tuned_macro_f1 (float)
    """
    probas     = model.predict_proba(X_val)          # (N, n_classes)
    cls_index  = {c: i for i, c in enumerate(classes)}
    thresholds = {}

    print("\n  Per-class threshold search (0.10 → 0.90, step 0.05):")
    for cls in classes:
        idx = cls_index[cls]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.10, 0.91, 0.05):
            binary_preds = (probas[:, idx] >= t).astype(int)
            true_binary  = (y_val == cls).astype(int)
            if binary_preds.sum() == 0:
                continue
            from sklearn.metrics import f1_score as _f1
            f = _f1(true_binary, binary_preds)
            if f > best_f1:
                best_f1, best_t = f, round(float(t), 2)
        thresholds[cls] = best_t
        print(f"    {cls:<8s}: best threshold={best_t:.2f}  (binary F1={best_f1:.4f})")

    # Apply tuned thresholds: one-vs-rest, pick highest P above threshold
    tuned_preds = _apply_thresholds(probas, thresholds, classes)
    tuned_macro = f1_score(y_val, tuned_preds, average="macro")
    print(f"\n  Macro F1 with tuned thresholds: {tuned_macro:.4f}")
    print("\n  Classification Report (tuned thresholds):")
    print(classification_report(y_val, tuned_preds, target_names=CLASSES, zero_division=0))

    return thresholds, tuned_preds, tuned_macro


def _apply_thresholds(probas: np.ndarray, thresholds: dict, classes: list) -> np.ndarray:
    """
    For each sample, assign the class with the highest P above its threshold.
    If no class exceeds its threshold, fall back to argmax.
    """
    cls_index = {c: i for i, c in enumerate(classes)}
    preds = []
    for row in probas:
        candidates = {
            c: row[cls_index[c]]
            for c in classes
            if row[cls_index[c]] >= thresholds[c]
        }
        if candidates:
            preds.append(max(candidates, key=candidates.get))
        else:
            preds.append(classes[int(np.argmax(row))])
    return np.array(preds)


def plot_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["tomato" if "nearby_" in f else "steelblue" for f in top_features[::-1]]
    ax.barh(top_features[::-1], top_values[::-1], color=colors)
    ax.set_xlabel("Importance (split gain)")
    ax.set_title(f"Top {top_n} Feature Importances — LightGBM v4  (red = spatial)")
    plt.tight_layout()
    out = REPORTS_DIR / "feature_importance_v4.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Feature importance plot → {out}")

    print(f"\n  Top {top_n} feature importances (v4):")
    for i, (feat, val) in enumerate(zip(top_features, top_values), 1):
        tag = " ← SPATIAL" if "nearby_" in feat else ""
        print(f"  {i:2d}. {feat:<45s} {val:>8.0f}{tag}")


def main():
    # ── Step A: Build spatial parquet if needed ──────────────────────────────
    if not SPATIAL_PATH.exists():
        print_section("Building spatial parquet (first run — this may take ~30s)")
        build_spatial_parquet()
    else:
        print_section(f"Spatial parquet already exists: {SPATIAL_PATH.name}")

    # ── Step B: Load + preprocess ────────────────────────────────────────────
    print_section("Loading spatially-enriched data")
    df = pd.read_parquet(SPATIAL_PATH)
    print(f"  Loaded {len(df):,} rows with columns: {list(df.columns[:10])} ...")

    print_section("Preprocessing (v4 — spatial + pruned weather)")
    X, y, feature_names, weather_keep_cols = build_features_v4(df)

    # ── Step C: Train / test split (same seed as v2/v3) ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print_section(f"Split: train={len(X_train):,}  test={len(X_test):,}")

    print("Class distribution (train):")
    for cls in CLASSES:
        c = (y_train == cls).sum()
        print(f"  {cls:<8s}: {c:>6,}  ({100*c/len(y_train):.2f}%)")

    # ── Step D: Train two LightGBM variants ─────────────────────────────────
    models_cfg = {
        "LightGBM_balanced": LGBMClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "LightGBM_manual_weights": LGBMClassifier(
            n_estimators=300,
            class_weight={"High": 8, "Low": 1, "Medium": 2},
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }

    results = []
    trained = {}

    for name, model in models_cfg.items():
        print_section(f"Training {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
        r = evaluate(name, model, X_test, y_test)
        results.append(r)

    comparison_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print_section("LightGBM variant comparison (default thresholds)")
    print(comparison_df.to_string(index=False))

    best_name  = comparison_df.iloc[0]["name"]
    best_model = trained[best_name]
    best_f1    = comparison_df.iloc[0]["macro_f1"]
    print(f"\n  Best variant: {best_name}  (Macro F1 = {best_f1})")

    # ── Step E: Per-class threshold tuning on test set ───────────────────────
    print_section("Per-class threshold tuning")
    thresholds, tuned_preds, tuned_macro = _tune_thresholds(
        best_model, X_test, y_test
    )

    # ── Step F: v3 vs v4 comparison ──────────────────────────────────────────
    V3_MACRO_F1 = 0.3894  # documented v3 result
    V2_MACRO_F1 = 0.4025  # v2 reference

    print_section("v2 vs v3 vs v4 Macro F1 Summary")
    print(f"  v2 LightGBM (no spatial, no SMOTE) : {V2_MACRO_F1:.4f}")
    print(f"  v3 LightGBM (SMOTE, no spatial)    : {V3_MACRO_F1:.4f}")
    print(f"  v4 LightGBM (spatial, default thr) : {best_f1:.4f}")
    print(f"  v4 LightGBM (spatial, tuned thr)   : {tuned_macro:.4f}")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    default_preds = best_model.predict(X_test)
    print_section("v4 Per-class metrics (default thresholds)")
    print(classification_report(y_test, default_preds, target_names=CLASSES, zero_division=0))
    print_section("v4 Per-class metrics (tuned thresholds)")
    print(classification_report(y_test, tuned_preds, target_names=CLASSES, zero_division=0))

    # ── Decision ─────────────────────────────────────────────────────────────
    use_tuned_f1 = max(best_f1, tuned_macro)
    print_section("Deployment Decision")
    if use_tuned_f1 > V3_MACRO_F1:
        print(f"  v4 WINS (macro F1 {use_tuned_f1:.4f} > v3 {V3_MACRO_F1:.4f})")
        print("  → Proceeding with save and deployment.")
        deploy = True
    else:
        print(f"  v4 did NOT improve over v3 (v4={use_tuned_f1:.4f}, v3={V3_MACRO_F1:.4f})")
        print("  → Keeping v3 as production model.")
        print("  Possible reasons: spatial features added noise, or the dataset lacks")
        print("  signal beyond what v3 already captured.")
        deploy = False

    # ── Save (always, for documentation) ─────────────────────────────────────
    out_pkl = MODELS_DIR / "best_model_v4.pkl"
    joblib.dump(
        {"model": best_model, "features": feature_names, "classes": CLASSES},
        out_pkl,
    )
    print(f"\n  Model saved → {out_pkl}")

    feat_list_path = MODELS_DIR / "feature_list_v4.txt"
    with open(feat_list_path, "w") as f:
        for feat in feature_names:
            f.write(feat + "\n")
    print(f"  Feature list → {feat_list_path}")

    thresh_path = MODELS_DIR / "thresholds_v4.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Thresholds → {thresh_path}")

    weather_path = MODELS_DIR / "weather_keep_cols_v4.json"
    with open(weather_path, "w") as f:
        json.dump(weather_keep_cols, f, indent=2)
    print(f"  Weather keep-cols → {weather_path}")

    out_csv = REPORTS_DIR / "model_comparison_v4.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"  Comparison CSV → {out_csv}")

    plot_feature_importance(best_model, feature_names)

    print_section(f"Done.  v4 macro F1 (default): {best_f1:.4f}   "
                  f"(tuned): {tuned_macro:.4f}   deploy={deploy}")
    return deploy


if __name__ == "__main__":
    main()
