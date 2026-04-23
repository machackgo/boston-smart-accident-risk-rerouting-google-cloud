"""
pipeline_accumulate.py — Phase 3: Build the accumulated challenger training dataset.

Reads the current production base dataset from GCS and all processed candidate
parquets from every prior Phase 2 run, merges them into a single challenger
training dataset, deduplicates on crash_numb, and writes the result to GCS.

Safe: writes only to candidate/merged/ prefix. Never touches production/.

Usage:
    python scripts/pipeline_accumulate.py

    # Dry-run: show what would be merged without writing anything
    python scripts/pipeline_accumulate.py --dry-run

GCS reads:
    production/datasets/crashes_v4_2015_2024.parquet  (production base)
    candidate/pipeline_state.json                      (Phase 2 run history)
    candidate/processed/{run_ts}/{year}_boston_processed.parquet  (all Phase 2 outputs)

GCS outputs:
    candidate/merged/{run_ts}/challenger_training_dataset.parquet
    candidate/merged/{run_ts}/manifest.json
    candidate/pipeline_state.json  (updated with latest merged folder)

Prerequisites:
    pip install pandas pyarrow google-cloud-storage
    gcloud auth application-default login
"""

import io
import json
import sys
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET          = "boston-rerouting-data"
GCS_PROD_DATASET    = "production/datasets/crashes_v4_2015_2024.parquet"
GCS_PIPELINE_STATE  = "candidate/pipeline_state.json"

# Columns the v4 training pipeline actually uses — anything outside this list
# is either metadata (id, EMS) or informational. All are preserved in the merge
# but this documents what matters for training.
V4_FEATURE_SOURCE_COLUMNS = [
    "crash_numb", "crash_datetime_clean", "year", "crash_hour",
    "lat", "lon", "speed_limit", "weath_cond_descr",
    "severity_3class",  # target
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def sec(title: str):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def gcs_read_json(client: storage.Client, path: str) -> dict | None:
    blob = client.bucket(GCS_BUCKET).blob(path)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def gcs_write_json(client: storage.Client, obj: dict, path: str):
    data = json.dumps(obj, indent=2).encode("utf-8")
    client.bucket(GCS_BUCKET).blob(path).upload_from_file(
        io.BytesIO(data), content_type="application/json"
    )
    print(f"  → gs://{GCS_BUCKET}/{path}  ({len(data)} bytes)")


def gcs_read_parquet(client: storage.Client, path: str) -> pd.DataFrame:
    buf = io.BytesIO()
    client.bucket(GCS_BUCKET).blob(path).download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def gcs_write_parquet(client: storage.Client, df: pd.DataFrame, path: str) -> int:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    size = buf.tell()
    buf.seek(0)
    client.bucket(GCS_BUCKET).blob(path).upload_from_file(
        buf, content_type="application/octet-stream"
    )
    print(f"  → gs://{GCS_BUCKET}/{path}  ({len(df):,} rows, {round(size/1024,1)} KB)")
    return size


# ── Collect all candidate processed parquet paths ─────────────────────────────

def collect_candidate_paths(client: storage.Client, state: dict) -> list[dict]:
    """
    Walk run_history in pipeline_state.json, read each Phase 2 processed manifest,
    and return a flat list of {year, gcs_path} for every processed parquet.
    This ensures all Phase 2 runs (past and present) are included in the merge.
    """
    collected = []
    seen_paths = set()

    for run in state.get("run_history", []):
        manifest_path = run.get("processed_manifest")
        if not manifest_path:
            continue

        manifest = gcs_read_json(client, manifest_path)
        if not manifest:
            print(f"  WARNING: could not read manifest {manifest_path} — skipping")
            continue

        for result in manifest.get("results", []):
            if result.get("status") != "ok":
                continue
            gcs_path = result.get("gcs_path")
            if not gcs_path or gcs_path in seen_paths:
                continue   # skip duplicates (same file referenced in multiple manifests)
            seen_paths.add(gcs_path)
            collected.append({
                "year":     result["year"],
                "gcs_path": gcs_path,
                "run_ts":   run["run_ts"],
            })

    return sorted(collected, key=lambda x: x["year"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    sec(f"Phase 3 — Accumulate Challenger Dataset  {run_ts}"
        + ("  [DRY RUN]" if dry_run else ""))

    client = storage.Client()

    # ── 1. Read pipeline_state.json ───────────────────────────────────────────
    sec("Reading pipeline_state.json")
    state = gcs_read_json(client, GCS_PIPELINE_STATE)
    if not state:
        print(f"  ERROR: {GCS_PIPELINE_STATE} not found. Run pipeline_download.py first.")
        sys.exit(1)

    if not state.get("phase3_ready"):
        print("  pipeline_state.json reports phase3_ready=false.")
        print("  Run pipeline_download.py first to produce processed candidate files.")
        sys.exit(1)

    print(f"  Last updated      : {state['last_updated']}")
    print(f"  Downloaded years  : {state['all_downloaded_years']}")
    print(f"  Phase 2 run count : {len(state.get('run_history', []))}")

    # ── 2. Collect all candidate processed file paths ─────────────────────────
    sec("Collecting all candidate processed parquets (all Phase 2 runs)")
    candidate_entries = collect_candidate_paths(client, state)

    if not candidate_entries:
        print("  No processed candidate files found. Nothing to merge. Exiting.")
        sys.exit(0)

    for entry in candidate_entries:
        print(f"  Year {entry['year']}  run={entry['run_ts']}  path={entry['gcs_path']}")

    if dry_run:
        print(f"\n  [DRY RUN] Would merge production + {[e['year'] for e in candidate_entries]}")
        sec("Phase 3 complete (dry-run — nothing written)")
        return

    # ── 3. Load production base dataset ──────────────────────────────────────
    sec("Loading production base dataset")
    prod_df = gcs_read_parquet(client, GCS_PROD_DATASET)
    print(f"  Loaded {len(prod_df):,} rows × {len(prod_df.columns)} cols  "
          f"(years {prod_df['year'].min()}–{prod_df['year'].max()})")

    # Drop 'id' — will be reassigned after merge
    if "id" in prod_df.columns:
        prod_df = prod_df.drop(columns=["id"])
    prod_df["_source"] = "production"

    # ── 4. Load all candidate processed parquets ──────────────────────────────
    sec("Loading candidate processed parquets")
    cand_frames = []
    cand_year_rows = {}

    for entry in candidate_entries:
        df = gcs_read_parquet(client, entry["gcs_path"])
        df["_source"] = f"candidate_{entry['year']}"
        cand_frames.append(df)
        cand_year_rows[entry["year"]] = len(df)
        print(f"  Year {entry['year']}: {len(df):,} rows from {entry['gcs_path']}")

    total_cand_rows = sum(cand_year_rows.values())
    print(f"\n  Total candidate rows: {total_cand_rows:,}")

    # ── 5. Merge: production + all candidate ─────────────────────────────────
    sec("Merging production + candidate data")

    # Concatenate — pandas fills missing columns (EMS cols) with NaN for
    # candidate rows automatically. This is correct: EMS cols are not used
    # by the v4 training pipeline.
    combined = pd.concat([prod_df] + cand_frames, ignore_index=True, sort=False)
    print(f"  Combined (before dedup): {len(combined):,} rows")

    # Deduplication on crash_numb.
    # Strategy: candidate rows override production rows for the same crash_numb.
    # Sort so candidate rows come last, then keep='last' when deduplicating.
    # 'production' sorts before 'candidate_*' alphabetically.
    combined = combined.sort_values("_source", ascending=True)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["crash_numb"], keep="last")
    dupes_removed = before_dedup - len(combined)

    if dupes_removed:
        print(f"  Deduplication removed {dupes_removed:,} rows "
              f"(candidate version kept for duplicate crash_numbs)")
    else:
        print(f"  Deduplication: 0 duplicates found (crash_numb is fully disjoint)")

    # Drop the helper column
    combined = combined.drop(columns=["_source"])

    # ── 6. Sort and reassign id ───────────────────────────────────────────────
    sort_cols = [c for c in ["year", "crash_numb"] if c in combined.columns]
    combined = combined.sort_values(sort_cols, ignore_index=True)
    combined.insert(0, "id", range(1, len(combined) + 1))

    print(f"\n  Final merged rows: {len(combined):,}")
    print(f"  Columns          : {len(combined.columns)}")
    print(f"  Year range       : {combined['year'].min()} – {combined['year'].max()}")

    # Row breakdown by year and source
    print("\n  Rows per year:")
    for yr, cnt in combined.groupby("year").size().sort_index().items():
        print(f"    {yr}: {cnt:,}")

    print("\n  Severity breakdown:")
    for sev, cnt in combined["severity_3class"].value_counts().sort_index().items():
        print(f"    {sev:<15} {cnt:>7,}")

    # ── 7. Null-rate summary for training-critical columns ────────────────────
    print("\n  Null rates (training-critical columns):")
    for col in ["lat", "lon", "crash_datetime_clean", "severity_3class", "weath_cond_descr"]:
        if col in combined.columns:
            null_pct = combined[col].isna().mean() * 100
            print(f"    {col:<30} {null_pct:.2f}% null")

    # ── 8. Write merged dataset to GCS ───────────────────────────────────────
    sec("Writing challenger training dataset to GCS")
    merged_folder = f"candidate/merged/{run_ts}"
    merged_path   = f"{merged_folder}/challenger_training_dataset.parquet"
    size_bytes = gcs_write_parquet(client, combined, merged_path)

    # ── 9. Write manifest ─────────────────────────────────────────────────────
    manifest = {
        "phase":                    "3_accumulate",
        "run_timestamp":            run_ts,
        "production_base_gcs":      GCS_PROD_DATASET,
        "production_base_rows":     len(prod_df),
        "candidate_years":          [e["year"] for e in candidate_entries],
        "candidate_year_rows":      cand_year_rows,
        "total_candidate_rows":     total_cand_rows,
        "duplicates_removed":       dupes_removed,
        "merged_row_count":         len(combined),
        "merged_column_count":      len(combined.columns),
        "merged_year_range":        [int(combined["year"].min()), int(combined["year"].max())],
        "merged_gcs_path":          merged_path,
        "merged_size_bytes":        size_bytes,
        "dedup_key":                "crash_numb",
        "dedup_strategy":           "candidate_row_wins_on_conflict",
        "columns_ems_null_for_candidate": [
            "ems_hotspot_flag", "ems_ped_hotspot_flag", "ems_peak_hour"
        ],
        "ems_impact": (
            "EMS columns are null for all candidate rows. "
            "These columns are absent from v4 feature_list_v4.txt and are not used in training."
        ),
        "severity_breakdown": {str(k): int(v) for k, v in combined["severity_3class"].value_counts().items()},
        "rows_per_year": {int(k): int(v) for k, v in combined.groupby("year").size().items()},
    }

    manifest_path = f"{merged_folder}/manifest.json"
    gcs_write_json(client, manifest, manifest_path)

    # ── 10. Update pipeline_state.json ───────────────────────────────────────
    sec("Updating candidate/pipeline_state.json")
    state["latest_merged_folder"]   = merged_folder
    state["latest_merged_path"]     = merged_path
    state["latest_merged_manifest"] = manifest_path
    state["latest_merged_rows"]     = len(combined)
    state["latest_merged_years"]    = [int(combined["year"].min()), int(combined["year"].max())]
    state["phase4_ready"]           = True
    state["last_updated"]           = run_ts

    # ── Readiness summary (human-readable Phase 4 trigger signal) ─────────────
    state["ready_for_phase4"]            = True
    state["latest_merged_timestamp"]     = run_ts
    state["latest_merged_row_count"]     = len(combined)
    state["years_in_candidate_dataset"]  = sorted([e["year"] for e in candidate_entries])
    state["recommended_next_action"]     = "run_phase4_retrain"
    state["recommended_next_action_reason"] = (
        f"Phase 3 built a challenger dataset with {len(combined):,} total rows "
        f"({total_cand_rows:,} new candidate rows from year(s) "
        f"{sorted([e['year'] for e in candidate_entries])}). "
        "Run pipeline_retrain.py when ready to evaluate the challenger vs. the champion."
    )

    # Append to merge history
    state.setdefault("merge_history", []).append({
        "run_ts":          run_ts,
        "merged_rows":     len(combined),
        "candidate_years": [e["year"] for e in candidate_entries],
        "merged_path":     merged_path,
        "manifest_path":   manifest_path,
    })

    gcs_write_json(client, state, GCS_PIPELINE_STATE)

    # ── Summary ───────────────────────────────────────────────────────────────
    sec("Phase 3 complete")
    print(f"  Production base rows : {len(prod_df):,}  (years 2015–2024)")
    print(f"  Candidate rows added : {total_cand_rows:,}  (years {[e['year'] for e in candidate_entries]})")
    print(f"  Duplicates removed   : {dupes_removed:,}")
    print(f"  Final merged rows    : {len(combined):,}")
    print(f"  Year range           : {combined['year'].min()} – {combined['year'].max()}")
    print(f"\n  Challenger dataset   : gs://{GCS_BUCKET}/{merged_path}")
    print(f"  Manifest             : gs://{GCS_BUCKET}/{manifest_path}")
    print(f"  Pipeline state       : gs://{GCS_BUCKET}/{GCS_PIPELINE_STATE}")
    print("=" * 62)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
