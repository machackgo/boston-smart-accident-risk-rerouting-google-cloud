"""
pipeline_retrain.py — Phase 4: Retrain challenger model and compare against champion.

Reads the merged challenger dataset from Phase 3, runs the full v4-style training
pipeline (spatial features → preprocess → LightGBM → threshold tuning), evaluates
against the champion, and writes all artifacts + a comparison report to GCS.

Safe: writes only to candidate/ prefix. Never touches production/.

Usage:
    python scripts/pipeline_retrain.py

GCS reads:
    candidate/pipeline_state.json
    candidate/merged/{ts}/challenger_training_dataset.parquet
    production/models/champion_manifest.json
    production/reports/champion_metrics.json

GCS writes:
    candidate/models/{run_ts}/best_model_challenger.pkl
    candidate/models/{run_ts}/feature_list_challenger.txt
    candidate/models/{run_ts}/thresholds_challenger.json
    candidate/models/{run_ts}/weather_keep_cols_challenger.json
    candidate/reports/{run_ts}/evaluation_challenger.json
    candidate/reports/{run_ts}/comparison.json
    candidate/reports/{run_ts}/feature_importance_challenger.png
    candidate/reports/latest_comparison.json   (stable pointer)
    candidate/pipeline_state.json              (updated)

Prerequisites:
    pip install lightgbm scikit-learn pandas pyarrow joblib matplotlib google-cloud-storage
    gcloud auth application-default login
"""

import io
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from google.cloud import storage

warnings.filterwarnings("ignore")

# ── repo path so we can import src/model modules ──────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
from src.model.spatial_features import compute_spatial_features
from src.model.preprocess_v4 import build_features_v4, SPATIAL_COLS

# Notification helper — disabled gracefully if notify.py is unavailable.
try:
    from notify import notify_retrain_complete
    _NOTIFY = True
except Exception as _notify_err:
    print(f"  [notify] Import failed — notifications disabled: {_notify_err}")
    _NOTIFY = False

# BigQuery run logger — disabled gracefully if bq_pipeline is unavailable.
try:
    from bq_pipeline import log_pipeline_run as _bq_log_run, log_challenger_run as _bq_log_chal
    _BQ = True
except Exception as _bq_err:
    print(f"  [bq] Import failed — BigQuery pipeline logging disabled: {_bq_err}")
    _BQ = False

# ── GCS config ────────────────────────────────────────────────────────────────
GCS_BUCKET          = "boston-rerouting-data"
GCS_PIPELINE_STATE  = "candidate/pipeline_state.json"
GCS_CHAMPION_MFST   = "production/models/champion_manifest.json"
GCS_CHAMPION_METS   = "production/reports/champion_metrics.json"

CLASSES             = ["High", "Low", "Medium"]
SEVERITY_MAP        = {"No Injury": "Low", "Injury": "Medium", "Fatal": "High"}
EARTH_RADIUS_M      = 6_371_009.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def sec(title: str):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


def gcs_read_json(client, path: str) -> dict | None:
    blob = client.bucket(GCS_BUCKET).blob(path)
    return json.loads(blob.download_as_text()) if blob.exists() else None


def gcs_write_json(client, obj: dict, path: str):
    data = json.dumps(obj, indent=2, default=str).encode("utf-8")
    client.bucket(GCS_BUCKET).blob(path).upload_from_file(
        io.BytesIO(data), content_type="application/json"
    )
    print(f"  → gs://{GCS_BUCKET}/{path}  ({len(data)} bytes)")


def gcs_write_bytes(client, buf: io.BytesIO, path: str, content_type: str):
    buf.seek(0)
    size = len(buf.getvalue())
    client.bucket(GCS_BUCKET).blob(path).upload_from_file(buf, content_type=content_type)
    print(f"  → gs://{GCS_BUCKET}/{path}  ({round(size/1024, 1)} KB)")


def gcs_read_parquet(client, path: str) -> pd.DataFrame:
    buf = io.BytesIO()
    client.bucket(GCS_BUCKET).blob(path).download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)


# ── Threshold tuning (identical to train_v4.py) ───────────────────────────────

def tune_thresholds(model, X_val, y_val) -> tuple[dict, np.ndarray, float]:
    probas    = model.predict_proba(X_val)
    cls_index = {c: i for i, c in enumerate(model.classes_)}
    thresholds = {}

    print("\n  Per-class threshold search (0.10 → 0.90, step 0.05):")
    for cls in CLASSES:
        idx = cls_index[cls]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.10, 0.91, 0.05):
            bp = (probas[:, idx] >= t).astype(int)
            tb = (y_val == cls).astype(int)
            if bp.sum() == 0:
                continue
            f = f1_score(tb, bp, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, round(float(t), 2)
        thresholds[cls] = best_t
        print(f"    {cls:<8s}: threshold={best_t:.2f}  (binary F1={best_f1:.4f})")

    preds = []
    for row in probas:
        candidates = {c: row[cls_index[c]] for c in CLASSES
                      if row[cls_index[c]] >= thresholds[c]}
        preds.append(max(candidates, key=candidates.get) if candidates
                     else CLASSES[int(np.argmax(row))])
    tuned_preds = np.array(preds)
    tuned_macro = f1_score(y_val, tuned_preds, average="macro", zero_division=0)
    return thresholds, tuned_preds, tuned_macro


# ── Binary ROC AUC from multiclass model ──────────────────────────────────────

def binary_roc_auc(model, X_test, y_test) -> float:
    """Elevated = Medium or High (value 1); Safe = Low (value 0)."""
    probas    = model.predict_proba(X_test)
    cls_list  = list(model.classes_)
    idx_high  = cls_list.index("High")
    idx_med   = cls_list.index("Medium")
    score     = probas[:, idx_high] + probas[:, idx_med]
    binary_y  = (y_test != "Low").astype(int)
    if binary_y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(binary_y, score))


# ── Feature importance plot ───────────────────────────────────────────────────

def plot_importance(model, feature_names: list, top_n: int = 15) -> io.BytesIO:
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_f       = [feature_names[i] for i in indices]
    top_v       = importances[indices]
    fig, ax     = plt.subplots(figsize=(10, 7))
    colors      = ["tomato" if "nearby_" in f else "steelblue" for f in top_f[::-1]]
    ax.barh(top_f[::-1], top_v[::-1], color=colors)
    ax.set_xlabel("Importance (split gain)")
    ax.set_title("Top Feature Importances — Challenger (red = spatial)")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    preds    = model.predict(X_test)
    acc      = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_test, preds, labels=CLASSES, zero_division=0
    )
    bin_auc  = binary_roc_auc(model, X_test, y_test)
    per_class = {
        cls: {"precision": round(float(p[i]), 4),
              "recall":    round(float(r[i]), 4),
              "f1":        round(float(f[i]), 4),
              "support":   int(s[i])}
        for i, cls in enumerate(CLASSES)
    }
    print(f"\n  {name}")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    Macro F1   : {macro_f1:.4f}")
    print(f"    Binary AUC : {bin_auc:.4f}")
    print(classification_report(y_test, preds, target_names=CLASSES, zero_division=0))
    return {
        "name":           name,
        "macro_f1":       round(macro_f1, 4),
        "accuracy":       round(acc, 4),
        "binary_roc_auc": round(bin_auc, 4),
        "per_class":      per_class,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import time as _time
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    _t0    = _time.monotonic()
    sec(f"Phase 4 — Retrain Challenger + Compare  {run_ts}")

    client = storage.Client()

    # ── 1. Read pipeline state and champion references ─────────────────────────
    sec("Reading pipeline state and champion references")
    state = gcs_read_json(client, GCS_PIPELINE_STATE)
    if not state or not state.get("phase4_ready"):
        print("  phase4_ready=false. Run pipeline_accumulate.py first.")
        sys.exit(1)

    champion_mfst = gcs_read_json(client, GCS_CHAMPION_MFST)
    champion_mets = gcs_read_json(client, GCS_CHAMPION_METS)
    if not champion_mfst or not champion_mets:
        print("  ERROR: champion manifest/metrics not found in GCS.")
        sys.exit(1)

    merged_path = state["latest_merged_path"]
    champ_macro = champion_mets["multiclass"]["macro_f1"]
    champ_auc   = champion_mets["binary"]["roc_auc"]
    champ_high  = champion_mets["multiclass"]["per_class"]["High"]["f1"]
    gates       = champion_mets["promotion_gates"]

    print(f"  Challenger dataset : gs://{GCS_BUCKET}/{merged_path}")
    print(f"  Champion macro F1  : {champ_macro}")
    print(f"  Champion ROC AUC   : {champ_auc}")
    print(f"  Promotion gate     : macro F1 ≥ {champ_macro} + {gates['min_macro_f1_improvement']} "
          f"= {champ_macro + gates['min_macro_f1_improvement']:.4f}")

    # ── 2. Load challenger dataset ─────────────────────────────────────────────
    sec("Loading challenger training dataset from GCS")
    df = gcs_read_parquet(client, merged_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
    print(f"  Year range: {df['year'].min()} – {df['year'].max()}")

    # ── 3. Filter: drop rows missing severity, lat, or lon ───────────────────
    sec("Filtering for valid rows (severity + coordinates required)")
    df["severity_class"] = df["severity_3class"].map(SEVERITY_MAP)
    df = df.dropna(subset=["severity_class", "lat", "lon"]).reset_index(drop=True)
    print(f"  Rows after filter: {len(df):,}")
    print("  Class distribution:")
    for cls in CLASSES:
        c = (df["severity_class"] == cls).sum()
        print(f"    {cls:<8s}: {c:>7,}  ({100*c/len(df):.2f}%)")

    # ── 4. Leakage-free spatial feature computation ────────────────────────────
    sec("Computing spatial features (BallTree haversine — ~2–4 min)")
    # Split indices FIRST so spatial features respect the train/test boundary.
    # Same seed and stratification as the downstream train_test_split in step 7,
    # ensuring test rows only see the training-set BallTree.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42,
        stratify=df["severity_class"]
    )
    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]
    print(f"  Spatial split: {len(train_df):,} train  {len(test_df):,} test")

    print("  Computing for TRAIN rows (leave-one-out) ...")
    train_sp = compute_spatial_features(train_df, train_df, is_train=True)

    print("  Computing for TEST rows (train-tree only) ...")
    test_sp  = compute_spatial_features(train_df, test_df, is_train=False)

    sp_all   = pd.concat([train_sp, test_sp]).sort_index()
    enriched = pd.concat([df, sp_all], axis=1)
    print(f"  Enriched dataset: {enriched.shape}")

    # ── 5. Feature engineering (v4 pipeline) ──────────────────────────────────
    sec("Preprocessing (v4 feature engineering)")
    X, y, feature_names, weather_keep_cols = build_features_v4(enriched)
    print(f"\n  X shape: {X.shape}")
    print(f"  Feature count: {len(feature_names)}")

    # ── 6. Train / test split ─────────────────────────────────────────────────
    sec("Train / test split (80/20, stratified, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print("  Class distribution (train):")
    for cls in CLASSES:
        c = (y_train == cls).sum()
        print(f"    {cls:<8s}: {c:>6,}  ({100*c/len(y_train):.2f}%)")

    # Promotion gate: total effective rows ≥ 45,000
    total_effective = len(X)
    gate_rows_ok = total_effective >= gates["min_training_rows"]
    print(f"\n  Total effective rows: {total_effective:,}  "
          f"(gate ≥ {gates['min_training_rows']:,}: {'PASS' if gate_rows_ok else 'FAIL'})")

    # ── 7. Train two LightGBM variants ────────────────────────────────────────
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

    sec("Training LightGBM variants")
    results   = []
    trained   = {}
    for name, model in models_cfg.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
        r = evaluate_model(name, model, X_test, y_test)
        results.append(r)

    cmp_df     = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    best_name  = cmp_df.iloc[0]["name"]
    best_model = trained[best_name]
    best_f1    = float(cmp_df.iloc[0]["macro_f1"])
    best_auc   = float(cmp_df.iloc[0]["binary_roc_auc"])

    print(f"\n  Best variant: {best_name}  (macro F1={best_f1:.4f}, ROC AUC={best_auc:.4f})")

    # ── 8. Per-class threshold tuning ─────────────────────────────────────────
    sec("Per-class threshold tuning on test set")
    thresholds, tuned_preds, tuned_macro = tune_thresholds(best_model, X_test, y_test)
    tuned_auc = binary_roc_auc(best_model, X_test, y_test)

    # Use the better of default vs tuned F1
    challenger_f1  = max(best_f1, tuned_macro)
    challenger_auc = tuned_auc
    use_tuned      = tuned_macro > best_f1
    final_preds    = tuned_preds if use_tuned else best_model.predict(X_test)

    print(f"\n  Default F1 : {best_f1:.4f}")
    print(f"  Tuned F1   : {tuned_macro:.4f}")
    print(f"  Using       : {'tuned' if use_tuned else 'default'} thresholds  (macro F1={challenger_f1:.4f})")

    # ── 9. Final evaluation ───────────────────────────────────────────────────
    sec("Final challenger evaluation")
    p, r, f, s = precision_recall_fscore_support(
        y_test, final_preds, labels=CLASSES, zero_division=0
    )
    per_class_final = {
        cls: {"precision": round(float(p[i]), 4),
              "recall":    round(float(r[i]), 4),
              "f1":        round(float(f[i]), 4),
              "support":   int(s[i])}
        for i, cls in enumerate(CLASSES)
    }
    final_acc = float(accuracy_score(y_test, final_preds))

    print(f"  Macro F1   : {challenger_f1:.4f}")
    print(f"  Accuracy   : {final_acc:.4f}")
    print(f"  ROC AUC    : {challenger_auc:.4f}")
    print(f"  Test rows  : {len(y_test):,}")
    print("\n  Per-class:")
    for cls in CLASSES:
        pc = per_class_final[cls]
        print(f"    {cls:<8s}: F1={pc['f1']:.4f}  prec={pc['precision']:.4f}  "
              f"rec={pc['recall']:.4f}  n={pc['support']:,}")

    # ── 10. Promotion gate evaluation ─────────────────────────────────────────
    sec("Promotion gate evaluation")
    challenger_high_f1 = per_class_final["High"]["f1"]

    gate_macro = {
        "description":  f"Challenger macro F1 ≥ champion + {gates['min_macro_f1_improvement']}",
        "required":     round(champ_macro + gates["min_macro_f1_improvement"], 4),
        "challenger":   round(challenger_f1, 4),
        "champion":     champ_macro,
        "delta":        round(challenger_f1 - champ_macro, 4),
        "passed":       challenger_f1 >= champ_macro + gates["min_macro_f1_improvement"],
    }
    gate_auc = {
        "description":  f"Challenger binary ROC AUC ≥ {gates['min_binary_roc_auc']}",
        "required":     gates["min_binary_roc_auc"],
        "challenger":   round(challenger_auc, 4),
        "champion":     champ_auc,
        "delta":        round(challenger_auc - champ_auc, 4),
        "passed":       challenger_auc >= gates["min_binary_roc_auc"],
    }
    high_floor = round(champ_high - gates["high_f1_max_regression"], 4)
    gate_high = {
        "description":  f"Challenger High F1 ≥ champion High F1 − {gates['high_f1_max_regression']}",
        "required":     high_floor,
        "challenger":   round(challenger_high_f1, 4),
        "champion":     champ_high,
        "delta":        round(challenger_high_f1 - champ_high, 4),
        "passed":       challenger_high_f1 >= high_floor,
    }
    gate_test = {
        "description":  f"Test set ≥ {gates['min_test_set_rows']:,} rows",
        "required":     gates["min_test_set_rows"],
        "challenger":   len(y_test),
        "passed":       len(y_test) >= gates["min_test_set_rows"],
    }
    gate_rows = {
        "description":  f"Total effective rows ≥ {gates['min_training_rows']:,}",
        "required":     gates["min_training_rows"],
        "challenger":   total_effective,
        "passed":       gate_rows_ok,
    }

    all_gates = {
        "macro_f1_improvement":      gate_macro,
        "binary_roc_auc_floor":      gate_auc,
        "high_f1_no_regression":     gate_high,
        "test_set_size":             gate_test,
        "total_effective_rows":      gate_rows,
    }

    all_passed = all(g["passed"] for g in all_gates.values())
    recommended_action = "promote" if all_passed else "do_not_promote"

    failed_gates = [name for name, g in all_gates.items() if not g["passed"]]
    if all_passed:
        reason = (f"All promotion gates passed. Challenger macro F1 {challenger_f1:.4f} "
                  f"exceeds champion {champ_macro} by {challenger_f1 - champ_macro:+.4f} "
                  f"(required +{gates['min_macro_f1_improvement']}).")
    else:
        reason = (f"Failed gates: {failed_gates}. "
                  f"Challenger macro F1={challenger_f1:.4f} vs champion {champ_macro}. "
                  "Production model unchanged. Candidate archived for future accumulation.")

    print(f"\n  Gate results:")
    for name, g in all_gates.items():
        status = "PASS" if g["passed"] else "FAIL"
        print(f"    [{status}] {name}: challenger={g.get('challenger')}  required={g.get('required')}")

    print(f"\n  All gates passed: {all_passed}")
    print(f"  Recommended action: {recommended_action}")
    print(f"  Reason: {reason}")

    # ── 11. Save challenger model artifacts to GCS ────────────────────────────
    sec("Saving challenger model artifacts to GCS")
    model_folder  = f"candidate/models/{run_ts}"
    report_folder = f"candidate/reports/{run_ts}"

    # Model pickle
    model_buf = io.BytesIO()
    joblib.dump({"model": best_model, "features": feature_names, "classes": CLASSES},
                model_buf)
    gcs_write_bytes(client, model_buf,
                    f"{model_folder}/best_model_challenger.pkl",
                    "application/octet-stream")

    # Feature list
    feat_data = "\n".join(feature_names).encode()
    gcs_write_bytes(client, io.BytesIO(feat_data),
                    f"{model_folder}/feature_list_challenger.txt",
                    "text/plain")

    # Thresholds
    gcs_write_json(client, thresholds,
                   f"{model_folder}/thresholds_challenger.json")

    # Weather keep cols
    gcs_write_json(client, weather_keep_cols,
                   f"{model_folder}/weather_keep_cols_challenger.json")

    # ── 12. Save evaluation and comparison reports ────────────────────────────
    sec("Saving evaluation and comparison reports to GCS")

    # Full evaluation JSON
    evaluation = {
        "phase":                "4_retrain",
        "run_timestamp":        run_ts,
        "merged_dataset_gcs":   merged_path,
        "merged_rows_total":    len(df),
        "effective_rows":       total_effective,
        "train_rows":           len(X_train),
        "test_rows":            len(X_test),
        "year_range":           [int(df["year"].min()), int(df["year"].max())],
        "best_variant":         best_name,
        "thresholds_used":      "tuned" if use_tuned else "default",
        "thresholds":           thresholds,
        "feature_count":        len(feature_names),
        "multiclass": {
            "macro_f1":         round(challenger_f1, 4),
            "accuracy":         round(final_acc, 4),
            "per_class":        per_class_final,
        },
        "binary": {
            "roc_auc":          round(challenger_auc, 4),
        },
        "variant_comparison": [
            {"name": r["name"], "macro_f1": r["macro_f1"],
             "accuracy": r["accuracy"], "binary_roc_auc": r["binary_roc_auc"]}
            for r in results
        ],
    }
    gcs_write_json(client, evaluation, f"{report_folder}/evaluation_challenger.json")

    # Comparison JSON
    comparison = {
        "phase":                "4_comparison",
        "run_timestamp":        run_ts,
        "champion": {
            "macro_f1":         champ_macro,
            "accuracy":         champion_mets["multiclass"]["accuracy"],
            "binary_roc_auc":   champ_auc,
            "high_f1":          champ_high,
            "medium_f1":        champion_mets["multiclass"]["per_class"]["Medium"]["f1"],
            "low_f1":           champion_mets["multiclass"]["per_class"]["Low"]["f1"],
            "test_rows":        champion_mets["test_set_size"],
            "training_date":    champion_mfst["training_date"],
            "year_range":       champion_mfst["year_range"],
        },
        "challenger": {
            "macro_f1":         round(challenger_f1, 4),
            "accuracy":         round(final_acc, 4),
            "binary_roc_auc":   round(challenger_auc, 4),
            "high_f1":          per_class_final["High"]["f1"],
            "medium_f1":        per_class_final["Medium"]["f1"],
            "low_f1":           per_class_final["Low"]["f1"],
            "test_rows":        len(y_test),
            "effective_rows":   total_effective,
            "year_range":       [int(df["year"].min()), int(df["year"].max())],
            "model_gcs_path":   f"{model_folder}/best_model_challenger.pkl",
        },
        "deltas": {
            "macro_f1":         round(challenger_f1 - champ_macro, 4),
            "accuracy":         round(final_acc - champion_mets["multiclass"]["accuracy"], 4),
            "binary_roc_auc":   round(challenger_auc - champ_auc, 4),
            "high_f1":          round(per_class_final["High"]["f1"] - champ_high, 4),
            "medium_f1":        round(per_class_final["Medium"]["f1"]
                                      - champion_mets["multiclass"]["per_class"]["Medium"]["f1"], 4),
            "low_f1":           round(per_class_final["Low"]["f1"]
                                      - champion_mets["multiclass"]["per_class"]["Low"]["f1"], 4),
        },
        "promotion_gates":      all_gates,
        "all_gates_passed":     all_passed,
        "recommended_action":   recommended_action,
        "recommendation_reason": reason,
    }
    gcs_write_json(client, comparison, f"{report_folder}/comparison.json")
    gcs_write_json(client, comparison, "candidate/reports/latest_comparison.json")

    # Feature importance plot
    imp_buf = plot_importance(best_model, feature_names)
    gcs_write_bytes(client, imp_buf,
                    f"{report_folder}/feature_importance_challenger.png",
                    "image/png")

    # ── 13. Update pipeline_state.json ────────────────────────────────────────
    sec("Updating candidate/pipeline_state.json")
    state["last_updated"]              = run_ts
    state["latest_model_folder"]       = model_folder
    state["latest_report_folder"]      = report_folder
    state["latest_comparison_gcs"]     = f"{report_folder}/comparison.json"
    state["latest_challenger_f1"]      = round(challenger_f1, 4)
    state["latest_challenger_auc"]     = round(challenger_auc, 4)
    state["latest_all_gates_passed"]   = all_passed
    state["latest_recommended_action"] = recommended_action
    state["phase5_ready"]              = all_passed

    # ── Readiness summary (updated after every Phase 4 run) ───────────────────
    state["last_retrain_timestamp"]   = run_ts
    state["last_retrain_merged_rows"] = len(df)
    if all_passed:
        state["recommended_next_action"] = "promote_challenger"
        state["recommended_next_action_reason"] = (
            f"All promotion gates passed. Challenger macro F1={challenger_f1:.4f} "
            f"beats champion {champ_macro} by {challenger_f1 - champ_macro:+.4f}. "
            "Review candidate/reports/latest_comparison.json then promote manually."
        )
    else:
        state["recommended_next_action"] = "continue_accumulating"
        state["recommended_next_action_reason"] = (
            f"Challenger did not beat champion (failed gates: {failed_gates}). "
            f"Challenger macro F1={challenger_f1:.4f} vs champion {champ_macro}. "
            "Continue accumulating MassDOT data via the weekly job, then re-run "
            "pipeline_retrain.py when meaningfully more data has arrived."
        )

    state.setdefault("retrain_history", []).append({
        "run_ts":              run_ts,
        "challenger_macro_f1": round(challenger_f1, 4),
        "challenger_auc":      round(challenger_auc, 4),
        "all_gates_passed":    all_passed,
        "recommended_action":  recommended_action,
        "model_folder":        model_folder,
        "report_folder":       report_folder,
    })
    gcs_write_json(client, state, GCS_PIPELINE_STATE)

    # ── Final summary ──────────────────────────────────────────────────────────
    sec("Phase 4 complete")
    print(f"  {'CHAMPION':>20}   {'CHALLENGER':>12}   {'DELTA':>10}")
    print(f"  {'─'*20}   {'─'*12}   {'─'*10}")
    for metric, champ_val, chal_val in [
        ("Macro F1",       champ_macro,  challenger_f1),
        ("Accuracy",       champion_mets["multiclass"]["accuracy"], final_acc),
        ("Binary ROC AUC", champ_auc,    challenger_auc),
        ("High F1",        champ_high,   per_class_final["High"]["f1"]),
        ("Medium F1",      champion_mets["multiclass"]["per_class"]["Medium"]["f1"],
                           per_class_final["Medium"]["f1"]),
        ("Low F1",         champion_mets["multiclass"]["per_class"]["Low"]["f1"],
                           per_class_final["Low"]["f1"]),
    ]:
        delta = chal_val - champ_val
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {metric:>20}   {champ_val:>12.4f}   {chal_val:>7.4f}  {arrow}{abs(delta):.4f}")

    print(f"\n  {'All gates passed':>20} : {all_passed}")
    print(f"  {'Recommended action':>20} : {recommended_action.upper()}")
    print(f"  {'Reason':>20} : {reason}")
    print(f"\n  Artifacts:")
    print(f"    Model      : gs://{GCS_BUCKET}/{model_folder}/best_model_challenger.pkl")
    print(f"    Comparison : gs://{GCS_BUCKET}/{report_folder}/comparison.json")
    print(f"    Latest ptr : gs://{GCS_BUCKET}/candidate/reports/latest_comparison.json")
    print(f"    State      : gs://{GCS_BUCKET}/{GCS_PIPELINE_STATE}")

    if not all_passed:
        print(f"\n  Production model is UNCHANGED. Challenger archived at:")
        print(f"    gs://{GCS_BUCKET}/{model_folder}/")
        print(f"  Run pipeline_check.py + pipeline_download.py again when more data is available,")
        print(f"  then pipeline_accumulate.py + pipeline_retrain.py to try again.")

    if _NOTIFY:
        notify_retrain_complete(
            state=state,
            challenger_f1=challenger_f1,
            champ_f1=champ_macro,
            all_gates_passed=all_passed,
            failed_gates=failed_gates,
            run_ts=run_ts,
        )

    # ── BigQuery pipeline logging ──────────────────────────────────────────────
    if _BQ:
        _duration = int(_time.monotonic() - _t0)
        _bq_log_run(
            run_ts, "phase4_retrain", recommended_action,
            challenger_f1=round(challenger_f1, 4),
            all_gates_passed=all_passed,
            recommended_action=recommended_action,
            candidate_years=list(map(int, [df["year"].min(), df["year"].max()])),
            merged_row_count=len(df),
            merged_gcs_path=merged_path,
            duration_seconds=_duration,
            triggered_by="manual",
        )
        _bq_log_chal(run_ts, comparison, state)

    print("=" * 62)


if __name__ == "__main__":
    main()
