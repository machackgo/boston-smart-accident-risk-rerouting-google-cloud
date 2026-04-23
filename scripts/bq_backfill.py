"""
bq_backfill.py — One-time backfill of BigQuery tables from existing GCS artifacts.

Populates three tables from GCS files already produced by the pipeline:

  model_versions
    Source: production/models/champion_manifest.json
            production/reports/champion_metrics.json
    One row — the champion (v4).

  challenger_runs
    Source: candidate/pipeline_state.json  (retrain_history list)
            candidate/reports/{run_ts}/comparison.json  (full gate details)
    One row per past Phase 4 run found in retrain_history.

  pipeline_runs
    Source: pipeline/logs/*.json  (one file per weekly run, written by pipeline_weekly.py)
    One row per existing weekly run log.

Run once after bq_setup.py:
    python scripts/bq_backfill.py

Add --dry-run to print what would be inserted without writing to BigQuery:
    python scripts/bq_backfill.py --dry-run

Safe: read-only on GCS, append-only on BigQuery.
Re-running will insert duplicate rows — intended to be run exactly once.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from bq_pipeline import (
    log_pipeline_run,
    log_challenger_run,
    register_model_version,
)

GCS_BUCKET   = "boston-rerouting-data"
DRY_RUN      = "--dry-run" in sys.argv


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _gcs_read_json(client: storage.Client, path: str) -> dict | None:
    blob = client.bucket(GCS_BUCKET).blob(path)
    if not blob.exists():
        print(f"  [backfill] NOT FOUND: gs://{GCS_BUCKET}/{path}")
        return None
    return json.loads(blob.download_as_text())


def _list_blobs(client: storage.Client, prefix: str) -> list[str]:
    return [b.name for b in client.bucket(GCS_BUCKET).list_blobs(prefix=prefix)]


def sec(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


# ── Backfill: model_versions — champion v4 ───────────────────────────────────

def backfill_champion(client: storage.Client) -> None:
    sec("model_versions — Champion v4")

    manifest = _gcs_read_json(client, "production/models/champion_manifest.json")
    metrics  = _gcs_read_json(client, "production/reports/champion_metrics.json")
    if not manifest or not metrics:
        print("  ERROR: Champion manifest or metrics missing. Skipping.")
        return

    year_range   = manifest.get("year_range", [])
    training_date = manifest.get("training_date")
    artifact_gcs  = manifest.get("model_gcs_path", "production/models/best_model.pkl")

    mc       = metrics.get("multiclass", {})
    pc       = mc.get("per_class", {})
    binary   = metrics.get("binary", {})

    model_metrics = {
        "training_date":        training_date,
        "training_dataset_gcs": manifest.get("training_dataset_gcs",
                                             "production/data/training_dataset.parquet"),
        "training_row_count":   manifest.get("training_row_count"),
        "year_range_min":       year_range[0] if len(year_range) > 0 else None,
        "year_range_max":       year_range[-1] if len(year_range) > 0 else None,
        "macro_f1":             mc.get("macro_f1"),
        "accuracy":             mc.get("accuracy"),
        "binary_roc_auc":       binary.get("roc_auc"),
        "high_f1":              pc.get("High",   {}).get("f1"),
        "medium_f1":            pc.get("Medium", {}).get("f1"),
        "low_f1":               pc.get("Low",    {}).get("f1"),
        "test_set_size":        metrics.get("test_set_size"),
        "champion_metrics_gcs": "production/reports/champion_metrics.json",
    }

    print(f"  model_id       : v4")
    print(f"  display_name   : Champion v4")
    print(f"  training_date  : {training_date}")
    print(f"  year_range     : {year_range}")
    print(f"  macro_f1       : {model_metrics['macro_f1']}")
    print(f"  binary_roc_auc : {model_metrics['binary_roc_auc']}")
    print(f"  artifact_gcs   : gs://{GCS_BUCKET}/{artifact_gcs}")

    if DRY_RUN:
        print("  [DRY RUN] Would call register_model_version(v4, is_champion=True)")
        return

    register_model_version(
        model_id="v4",
        display_name="Champion v4",
        metrics=model_metrics,
        artifact_gcs=artifact_gcs,
        is_champion=True,
    )


# ── Backfill: challenger_runs — from retrain_history ─────────────────────────

def backfill_challenger_runs(client: storage.Client) -> None:
    sec("challenger_runs — from retrain_history")

    pipeline_state = _gcs_read_json(client, "candidate/pipeline_state.json")
    if not pipeline_state:
        print("  pipeline_state.json not found. Skipping.")
        return

    history = pipeline_state.get("retrain_history", [])
    if not history:
        print("  retrain_history is empty — no challenger runs to backfill.")
        return

    print(f"  Found {len(history)} retrain run(s) in retrain_history.")

    for i, entry in enumerate(history):
        run_ts = entry.get("run_ts")
        if not run_ts:
            print(f"  Entry {i}: missing run_ts — skipping.")
            continue

        print(f"\n  [{i+1}/{len(history)}] run_ts={run_ts}")

        # Try to read the full comparison.json for rich gate details
        comparison_path = f"candidate/reports/{run_ts}/comparison.json"
        comparison = _gcs_read_json(client, comparison_path)

        if comparison is None:
            # Fall back to the summary in retrain_history
            print(f"    comparison.json not found — building minimal row from history entry.")
            comparison = {
                "merged_rows_total":   pipeline_state.get("last_retrain_merged_rows"),
                "all_gates_passed":    entry.get("all_gates_passed"),
                "recommended_action":  entry.get("recommended_action"),
                "recommendation_reason": "",
                "champion": {},
                "challenger": {
                    "macro_f1":         entry.get("challenger_macro_f1"),
                    "binary_roc_auc":   entry.get("challenger_auc"),
                    "model_gcs_path":   f"{entry.get('model_folder', '')}/best_model_challenger.pkl",
                },
                "promotion_gates": {},
            }
        else:
            print(f"    Read comparison.json  ({len(json.dumps(comparison))} bytes)")
            print(f"    all_gates_passed : {comparison.get('all_gates_passed')}")
            print(f"    recommended      : {comparison.get('recommended_action')}")

        if DRY_RUN:
            print(f"    [DRY RUN] Would call log_challenger_run({run_ts}, ...)")
            continue

        log_challenger_run(run_ts, comparison, pipeline_state)

    print()


# ── Backfill: pipeline_runs — from pipeline/logs/*.json ──────────────────────

def backfill_pipeline_runs(client: storage.Client) -> None:
    sec("pipeline_runs — from pipeline/logs/*.json")

    log_paths = sorted(_list_blobs(client, prefix="pipeline/logs/"))
    if not log_paths:
        print("  No log files found at pipeline/logs/. Skipping.")
        return

    print(f"  Found {len(log_paths)} log file(s).")

    for path in log_paths:
        log = _gcs_read_json(client, path)
        if not log:
            print(f"  Could not read {path} — skipping.")
            continue

        run_ts  = log.get("run_timestamp")
        outcome = log.get("outcome")
        if not run_ts or not outcome:
            print(f"  {path}: missing run_timestamp or outcome — skipping.")
            continue

        print(f"  {path}  →  outcome={outcome}")

        kwargs: dict = {
            "is_dry_run":   log.get("dry_run", False),
            "triggered_by": "cloud_scheduler",
        }

        # Populate optional fields based on outcome
        new_years = log.get("new_years")
        if new_years:
            kwargs["years_with_new_data"] = new_years if isinstance(new_years, list) else [new_years]

        if outcome == "success":
            merged_rows = log.get("merged_rows")
            if merged_rows:
                try:
                    kwargs["merged_row_count"] = int(merged_rows)
                except (TypeError, ValueError):
                    pass
            kwargs["merged_gcs_path"] = log.get("merged_gcs_path")

        if DRY_RUN:
            print(f"    [DRY RUN] Would call log_pipeline_run({run_ts}, weekly_phases_1_3, {outcome})")
            continue

        log_pipeline_run(run_ts, "weekly_phases_1_3", outcome, **kwargs)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'=' * 62}")
    print(f"  BigQuery Backfill — boston-rerouting.boston_rerouting")
    print(f"{'=' * 62}")
    if DRY_RUN:
        print("  Mode: DRY RUN (no BigQuery writes)")
    else:
        print("  Mode: LIVE (will write to BigQuery)")
        print("  WARNING: Re-running will insert duplicate rows.")
        print("           This script is intended to be run exactly once.")

    client = storage.Client()

    backfill_champion(client)
    backfill_challenger_runs(client)
    backfill_pipeline_runs(client)

    sec("Backfill complete")
    if DRY_RUN:
        print("  Dry run — no data written. Re-run without --dry-run to commit.")
    else:
        print("  Verify in BigQuery console:")
        print("    SELECT * FROM `boston-rerouting.boston_rerouting.model_versions`")
        print("    SELECT * FROM `boston-rerouting.boston_rerouting.challenger_runs` ORDER BY run_timestamp")
        print("    SELECT * FROM `boston-rerouting.boston_rerouting.pipeline_runs`   ORDER BY run_timestamp")
    print()


if __name__ == "__main__":
    main()
