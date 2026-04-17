"""
train_v4_binary.py — Binary reframe: Safe (Low) vs Elevated (Medium + High).

Uses the same v4 spatial features but collapses the three-class problem into
a binary one.  This is a parallel analysis model — it does NOT replace v4.

Saves:
    models/best_model_v4_binary.pkl  — {model, features, classes}
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, accuracy_score,
)
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_features import OUT_PATH as SPATIAL_PATH
from preprocess_v4 import build_features_v4

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("Loading spatial parquet ...")
    df = pd.read_parquet(SPATIAL_PATH)

    print("Preprocessing (v4 features) ...")
    X, y_multi, feature_names, _ = build_features_v4(df)

    # Map to binary: Low → Safe(0), Medium/High → Elevated(1)
    y = (y_multi != "Low").astype(int)
    print(f"Class distribution: Safe(0)={( y==0).sum():,}  Elevated(1)={(y==1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc    = accuracy_score(y_test, preds)
    f1_bin = f1_score(y_test, preds, average="binary")
    f1_mac = f1_score(y_test, preds, average="macro")
    auc    = roc_auc_score(y_test, probas)

    print("\n=== Binary Model (Safe vs Elevated) ===")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Binary F1      : {f1_bin:.4f}")
    print(f"  Macro F1       : {f1_mac:.4f}")
    print(f"  ROC AUC        : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Safe", "Elevated"]))

    out = MODELS_DIR / "best_model_v4_binary.pkl"
    joblib.dump(
        {"model": model, "features": feature_names, "classes": ["Safe", "Elevated"]},
        out,
    )
    print(f"\n  Binary model saved → {out}")
    return {"binary_f1": round(f1_bin, 4), "macro_f1": round(f1_mac, 4), "roc_auc": round(auc, 4)}


if __name__ == "__main__":
    main()
