"""
pipeline_download.py — Phase 2: Download + transform newly detected MassDOT data.

Reads Phase 1's check manifest from GCS, downloads only the years flagged as
new or changed, archives the raw files, and writes a processed candidate parquet
that is compatible with the v4 training pipeline.

Safe: writes only to candidate/ prefix in GCS. Never touches production/.

Usage:
    python scripts/pipeline_download.py

    # Dry-run: print what would be downloaded without writing anything
    python scripts/pipeline_download.py --dry-run

GCS outputs:
    candidate/raw/{run_ts}/{year}_boston_raw.parquet
    candidate/raw/{run_ts}/manifest.json
    candidate/processed/{run_ts}/{year}_boston_processed.parquet
    candidate/processed/{run_ts}/manifest.json
    candidate/pipeline_state.json          (updated after each successful run)

Prerequisites:
    pip install requests pandas pyarrow google-cloud-storage
    gcloud auth application-default login
"""

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from google.cloud import storage

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET         = "boston-rerouting-data"
GCS_CHECK_LATEST   = "candidate/checks/latest.json"
GCS_PIPELINE_STATE = "candidate/pipeline_state.json"

PAGE_SIZE = 2000   # max records per ArcGIS paginated request

# Columns to keep in the processed output — must match the project schema
WANTED_COLUMNS = [
    "crash_numb",
    "crash_datetime_clean",
    "year",
    "crash_hour",
    "city_town_name",
    "lat",
    "lon",
    "crash_severity_descr",
    "severity_3class",
    "numb_fatal_injr",
    "numb_nonfatal_injr",
    "numb_vehc",
    "weath_cond_descr",
    "road_surf_cond_descr",
    "ambnt_light_descr",
    "manr_coll_descr",
    "rdwy_jnct_type_descr",
    "rdwy",
    "speed_limit",
    "hit_run_descr",
    "district_num",
    "cnty_name",
]

# severity_3class mapping from raw CRASH_SEVERITY_DESCR values
SEVERITY_MAP = {
    "Fatal injury":                        "Fatal",
    "Non-fatal injury":                    "Injury",
    "Property damage only (none injured)": "No Injury",
    "Not Reported":                        "Unknown",
    "Unknown":                             "Unknown",
}


# ── Logging helper ────────────────────────────────────────────────────────────

def sec(title: str):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


# ── GCS helpers ───────────────────────────────────────────────────────────────

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


def gcs_write_parquet(client: storage.Client, df: pd.DataFrame, path: str):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    size = buf.tell()
    buf.seek(0)
    client.bucket(GCS_BUCKET).blob(path).upload_from_file(
        buf, content_type="application/octet-stream"
    )
    print(f"  → gs://{GCS_BUCKET}/{path}  ({len(df):,} rows, {round(size/1024,1)} KB)")
    return size


# ── ArcGIS paginated download ─────────────────────────────────────────────────

def download_year(year: int, url: str, where: str) -> pd.DataFrame | None:
    """
    Paginate through the ArcGIS REST endpoint and return a DataFrame of all
    Boston crash records for the given year. Returns None on failure.
    """
    all_records = []
    offset = 0

    while True:
        params = {
            "where":             where,
            "outFields":         "*",
            "resultRecordCount": PAGE_SIZE,
            "resultOffset":      offset,
            "f":                 "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [year {year}] Request failed at offset {offset}: {e}")
            return None

        if "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            print(f"  [year {year}] API error: {msg}")
            return None

        features = data.get("features", [])
        records  = [f["attributes"] for f in features]
        all_records.extend(records)

        exceeded = data.get("exceededTransferLimit", False)
        if not exceeded or len(features) == 0:
            break

        offset += PAGE_SIZE
        time.sleep(0.3)   # polite pacing

    if not all_records:
        print(f"  [year {year}] No records returned")
        return None

    df = pd.DataFrame(all_records)
    print(f"  [year {year}] {len(df):,} rows fetched ({len(df.columns)} fields)")
    return df


# ── Transform / clean ─────────────────────────────────────────────────────────

def transform(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Clean and standardise one year's raw DataFrame into the project schema.
    Mirrors the logic in scripts/fetch_massdot_boston.py clean().
    """
    # Lowercase all column names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Parse crash_datetime_clean from epoch milliseconds
    if "crash_datetime" in df.columns:
        dt_parsed = pd.to_datetime(
            df["crash_datetime"], unit="ms", utc=True, errors="coerce"
        )
        df["crash_datetime_clean"] = dt_parsed.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        if "year" not in df.columns:
            df["year"] = dt_parsed.dt.year.astype("Int64")
    elif "crash_date_text" in df.columns and "year" not in df.columns:
        df["year"] = pd.to_datetime(
            df["crash_date_text"], errors="coerce"
        ).dt.year.astype("Int64")

    # Ensure year is Int64
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Derive severity_3class
    if "crash_severity_descr" in df.columns:
        df["severity_3class"] = (
            df["crash_severity_descr"].map(SEVERITY_MAP).fillna("Unknown")
        )

    # Select and order columns to match project schema
    present = [c for c in WANTED_COLUMNS if c in df.columns]
    missing = [c for c in WANTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [year {year}] WARNING: columns absent from source: {missing}")

    df = df[present].copy()

    # Coerce numeric types
    for col in ["numb_fatal_injr", "numb_nonfatal_injr", "numb_vehc", "district_num"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["lat", "lon", "speed_limit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure city_town_name is uppercase and trimmed
    if "city_town_name" in df.columns:
        df["city_town_name"] = df["city_town_name"].str.upper().str.strip()

    # Safety filter: Boston only
    if "city_town_name" in df.columns:
        before = len(df)
        df = df[df["city_town_name"] == "BOSTON"].copy()
        removed = before - len(df)
        if removed:
            print(f"  [year {year}] Safety filter removed {removed} non-Boston rows")

    return df.reset_index(drop=True)


# ── Validation summary ────────────────────────────────────────────────────────

def validation_summary(df: pd.DataFrame, year: int) -> dict:
    """Return a dict of key quality stats for the manifest."""
    sev_counts = df["severity_3class"].value_counts().to_dict() if "severity_3class" in df.columns else {}
    return {
        "year":               year,
        "row_count":          len(df),
        "columns_present":    list(df.columns),
        "lat_null_pct":       round(df["lat"].isna().mean() * 100, 2) if "lat" in df.columns else None,
        "lon_null_pct":       round(df["lon"].isna().mean() * 100, 2) if "lon" in df.columns else None,
        "datetime_null_pct":  round(df["crash_datetime_clean"].isna().mean() * 100, 2) if "crash_datetime_clean" in df.columns else None,
        "severity_null_pct":  round((df["severity_3class"] == "Unknown").mean() * 100, 2) if "severity_3class" in df.columns else None,
        "severity_breakdown": sev_counts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    sec(f"Phase 2 — Download + Transform  {run_ts}"
        + ("  [DRY RUN]" if dry_run else ""))

    client = storage.Client()

    # ── 1. Read Phase 1 check manifest ────────────────────────────────────────
    sec("Reading Phase 1 check manifest")
    check_manifest = gcs_read_json(client, GCS_CHECK_LATEST)
    if not check_manifest:
        print(f"  ERROR: {GCS_CHECK_LATEST} not found in GCS.")
        print("  Run pipeline_check.py first.")
        sys.exit(1)

    action   = check_manifest["summary"]["recommended_action"]
    new_years = check_manifest["summary"]["years_with_new_data"]

    print(f"  Check timestamp      : {check_manifest['run_timestamp']}")
    print(f"  Recommended action   : {action}")
    print(f"  Years with new data  : {new_years}")

    if action != "download_recommended":
        print(f"\n  Action is '{action}' — nothing to download. Exiting.")
        sys.exit(0)

    if not new_years:
        print("\n  No new years flagged — nothing to download. Exiting.")
        sys.exit(0)

    # Build a lookup: year → {url, where_clause} from the check manifest
    endpoint_lookup = {
        ep["year"]: {"url": ep["url"], "where": ep["where_clause"]}
        for ep in check_manifest["endpoints"]
        if ep["status"] == "ok"
    }

    # ── 2. Download raw data year by year ─────────────────────────────────────
    sec(f"Downloading {len(new_years)} year(s): {new_years}")

    raw_results   = []
    proc_results  = []
    raw_frames    = {}

    for year in sorted(new_years):
        if year not in endpoint_lookup:
            print(f"  [year {year}] Not in endpoint lookup — skipping")
            continue

        ep    = endpoint_lookup[year]
        url   = ep["url"]
        where = ep["where"]

        print(f"\n  [{year}] Downloading from MassDOT ArcGIS ...")
        print(f"    URL  : {url}")
        print(f"    WHERE: {where}")

        if dry_run:
            print(f"    [DRY RUN] Skipping actual download")
            continue

        raw_df = download_year(year, url, where)
        if raw_df is None:
            print(f"  [{year}] Download failed — skipping this year")
            raw_results.append({
                "year": year, "status": "failed",
                "url": url, "error": "download_failed"
            })
            continue

        raw_frames[year] = raw_df
        raw_gcs_path = f"candidate/raw/{run_ts}/{year}_boston_raw.parquet"
        size = gcs_write_parquet(client, raw_df, raw_gcs_path)

        raw_results.append({
            "year":       year,
            "status":     "ok",
            "url":        url,
            "where":      where,
            "row_count":  len(raw_df),
            "field_count": len(raw_df.columns),
            "gcs_path":   raw_gcs_path,
            "size_bytes": size,
        })

    # ── 3. Write raw manifest ─────────────────────────────────────────────────
    if not dry_run and raw_results:
        raw_manifest = {
            "phase":              "2a_raw_download",
            "run_timestamp":      run_ts,
            "check_manifest_gcs": GCS_CHECK_LATEST,
            "years_attempted":    new_years,
            "results":            raw_results,
        }
        raw_manifest_path = f"candidate/raw/{run_ts}/manifest.json"
        sec("Writing raw download manifest")
        gcs_write_json(client, raw_manifest, raw_manifest_path)

    # ── 4. Transform each downloaded year ────────────────────────────────────
    if not dry_run and raw_frames:
        sec("Transforming raw data into processed candidate format")

        for year, raw_df in raw_frames.items():
            print(f"\n  [{year}] Applying transform ...")
            proc_df = transform(raw_df, year)

            print(f"  [{year}] After transform: {len(proc_df):,} rows, {len(proc_df.columns)} columns")
            if "crash_datetime_clean" in proc_df.columns:
                null_dt = proc_df["crash_datetime_clean"].isna().sum()
                print(f"  [{year}] crash_datetime_clean nulls: {null_dt}")
            if "severity_3class" in proc_df.columns:
                print(f"  [{year}] severity_3class distribution:")
                for sev, cnt in proc_df["severity_3class"].value_counts().items():
                    print(f"    {sev:<15} {cnt:>5,}")

            proc_gcs_path = f"candidate/processed/{run_ts}/{year}_boston_processed.parquet"
            size = gcs_write_parquet(client, proc_df, proc_gcs_path)

            proc_results.append({
                "year":               year,
                "status":             "ok",
                "gcs_path":           proc_gcs_path,
                "size_bytes":         size,
                "validation":         validation_summary(proc_df, year),
            })

        # ── 5. Write processed manifest ───────────────────────────────────────
        proc_manifest = {
            "phase":                "2b_processed",
            "run_timestamp":        run_ts,
            "raw_manifest_gcs":     f"candidate/raw/{run_ts}/manifest.json",
            "schema_columns":       WANTED_COLUMNS,
            "missing_vs_production": [
                "ems_hotspot_flag",
                "ems_ped_hotspot_flag",
                "ems_peak_hour",
            ],
            "missing_impact": (
                "None — these 3 columns are absent from the v4 feature list "
                "and are never used in model training."
            ),
            "results": proc_results,
        }
        proc_manifest_path = f"candidate/processed/{run_ts}/manifest.json"
        sec("Writing processed candidate manifest")
        gcs_write_json(client, proc_manifest, proc_manifest_path)

        # ── 6. Update pipeline_state.json ────────────────────────────────────
        sec("Updating candidate/pipeline_state.json")
        prev_state = gcs_read_json(client, GCS_PIPELINE_STATE) or {}

        # Accumulate all years downloaded across all Phase 2 runs
        all_downloaded_years = set(prev_state.get("all_downloaded_years", []))
        newly_downloaded = [r["year"] for r in proc_results if r["status"] == "ok"]
        all_downloaded_years.update(newly_downloaded)

        new_state = {
            "last_updated":               run_ts,
            "phase1_check_gcs":           GCS_CHECK_LATEST,
            "all_downloaded_years":       sorted(all_downloaded_years),
            "latest_raw_folder":          f"candidate/raw/{run_ts}/",
            "latest_processed_folder":    f"candidate/processed/{run_ts}/",
            "latest_raw_manifest":        f"candidate/raw/{run_ts}/manifest.json",
            "latest_processed_manifest":  f"candidate/processed/{run_ts}/manifest.json",
            "latest_processed_timestamp": run_ts,
            "phase3_ready":               len(proc_results) > 0,
            "recommended_next_action":    "run_phase3_accumulate",
            "recommended_next_action_reason": (
                f"Phase 2 downloaded and processed year(s) {newly_downloaded}. "
                "Run pipeline_accumulate.py to merge into the challenger training dataset."
            ),
            "run_history": prev_state.get("run_history", []) + [{
                "run_ts":                 run_ts,
                "years_downloaded":       newly_downloaded,
                "raw_manifest":           f"candidate/raw/{run_ts}/manifest.json",
                "processed_manifest":     f"candidate/processed/{run_ts}/manifest.json",
            }],
        }
        gcs_write_json(client, new_state, GCS_PIPELINE_STATE)

    # ── Summary ───────────────────────────────────────────────────────────────
    sec("Phase 2 summary")
    if dry_run:
        print("  [DRY RUN] No files were written.")
        print(f"  Would have downloaded years: {new_years}")
    else:
        ok_raw  = [r for r in raw_results  if r.get("status") == "ok"]
        ok_proc = [r for r in proc_results if r.get("status") == "ok"]
        print(f"  Years downloaded (raw)    : {[r['year'] for r in ok_raw]}")
        print(f"  Years processed           : {[r['year'] for r in ok_proc]}")
        print()
        for r in proc_results:
            v = r.get("validation", {})
            print(f"  Year {r['year']}:")
            print(f"    Rows           : {v.get('row_count', '?'):,}")
            print(f"    Lat/lon nulls  : {v.get('lat_null_pct', '?')}%")
            print(f"    Datetime nulls : {v.get('datetime_null_pct', '?')}%")
            print(f"    Severity unk.  : {v.get('severity_null_pct', '?')}%")
            for sev, cnt in (v.get("severity_breakdown") or {}).items():
                print(f"    severity {sev:<12}: {cnt:>4,}")
        print()
        print(f"  Raw files      : gs://{GCS_BUCKET}/candidate/raw/{run_ts}/")
        print(f"  Processed files: gs://{GCS_BUCKET}/candidate/processed/{run_ts}/")
        print(f"  Pipeline state : gs://{GCS_BUCKET}/{GCS_PIPELINE_STATE}")
    print("=" * 62)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
