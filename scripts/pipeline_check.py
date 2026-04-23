"""
pipeline_check.py — Phase 1: Weekly MassDOT data availability check.

Queries each known MassDOT ArcGIS endpoint using returnCountOnly=true (no data
downloaded), compares against the previous weekly manifest, and writes a
machine-readable check record to GCS.

Safe to run at any time — the only side effect is writing JSON to GCS.
Does NOT download crash data, retrain models, or touch production.

Usage:
    python scripts/pipeline_check.py

Outputs (GCS):
    candidate/checks/YYYY-MM-DD_HH-MM-SS.json   — timestamped check record
    candidate/checks/latest.json                 — stable pointer to most recent check

Prerequisites:
    pip install requests google-cloud-storage
    gcloud auth application-default login
"""

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from google.cloud import storage

# ── GCS config ────────────────────────────────────────────────────────────────
GCS_BUCKET          = "boston-rerouting-data"
GCS_CHECK_DIR       = "candidate/checks"
GCS_LATEST_KEY      = f"{GCS_CHECK_DIR}/latest.json"
GCS_PIPELINE_STATE  = "candidate/pipeline_state.json"

# ── Production baseline coverage ──────────────────────────────────────────────
# Years already incorporated into the production champion dataset.
# Any year outside this range is "new" by definition.
PRODUCTION_YEAR_MIN = 2015
PRODUCTION_YEAR_MAX = 2024

# ── MassDOT endpoint registry ─────────────────────────────────────────────────
# Each entry: year → {url, where, notes}
# Special cases are documented inline.
#
# Domain 1 (2015–2020): gis.massdot.state.ma.us  — older per-year services
# Domain 2 (2021+):     gis.crashdata.dot.mass.gov — newer open-data platform
#
# 2020 lives in a combined multi-year service; YEAR must be filtered explicitly.
# 2023 has a 'v' version suffix in the service name (MASSDOT_ODP_OPEN_2023v).

def _old_domain_url(year: int) -> str:
    return (
        f"https://gis.massdot.state.ma.us/arcgis/rest/services/"
        f"CrashClosedYear/CrashClosedYear{year}/FeatureServer/0/query"
    )

def _new_domain_url(year: int, suffix: str = "") -> str:
    return (
        f"https://gis.crashdata.dot.mass.gov/arcgis/rest/services/"
        f"MassDOT/MASSDOT_ODP_OPEN_{year}{suffix}/FeatureServer/0/query"
    )

KNOWN_ENDPOINTS: dict[int, dict] = {
    2015: {"url": _old_domain_url(2015), "where": "CITY_TOWN_NAME='BOSTON'"},
    2016: {"url": _old_domain_url(2016), "where": "CITY_TOWN_NAME='BOSTON'"},
    2017: {"url": _old_domain_url(2017), "where": "CITY_TOWN_NAME='BOSTON'"},
    2018: {"url": _old_domain_url(2018), "where": "CITY_TOWN_NAME='BOSTON'"},
    2019: {"url": _old_domain_url(2019), "where": "CITY_TOWN_NAME='BOSTON'"},
    2020: {
        "url": (
            "https://gis.massdot.state.ma.us/arcgis/rest/services/"
            "Roads/CrashClosedYears/FeatureServer/0/query"
        ),
        "where": "CITY_TOWN_NAME='BOSTON' AND YEAR='2020'",
        "note": "2020 data lives in a combined multi-year service",
    },
    2021: {"url": _new_domain_url(2021), "where": "CITY_TOWN_NAME='BOSTON'"},
    2022: {"url": _new_domain_url(2022), "where": "CITY_TOWN_NAME='BOSTON'"},
    2023: {
        "url": _new_domain_url(2023, suffix="v"),
        "where": "CITY_TOWN_NAME='BOSTON'",
        "note": "2023 service has a 'v' suffix: MASSDOT_ODP_OPEN_2023v",
    },
    2024: {"url": _new_domain_url(2024), "where": "CITY_TOWN_NAME='BOSTON'"},
    2025: {"url": _new_domain_url(2025), "where": "CITY_TOWN_NAME='BOSTON'"},
    2026: {"url": _new_domain_url(2026), "where": "CITY_TOWN_NAME='BOSTON'"},
}

# How many future years beyond the registry to probe blindly each run.
# E.g. if the registry goes to 2026, also try 2027 and 2028.
FUTURE_PROBE_COUNT = 2


# ── Endpoint query ────────────────────────────────────────────────────────────

def query_count(url: str, where: str, timeout: int = 30) -> tuple[int | None, str | None]:
    """
    Returns (count, error_message).
    count is None if the endpoint is unreachable or returns an API error.
    """
    params = {
        "where":           where,
        "returnCountOnly": "true",
        "f":               "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        return None, f"http_error: {e}"
    except ValueError:
        return None, "invalid_json"

    if "error" in data:
        msg = data["error"].get("message", str(data["error"]))
        return None, f"api_error: {msg}"
    if "count" in data:
        return int(data["count"]), None

    return None, f"unexpected_response: {str(data)[:120]}"


# ── GCS helpers ───────────────────────────────────────────────────────────────

def load_latest_manifest(client: storage.Client) -> dict | None:
    """Read the previous latest.json from GCS. Returns None if it doesn't exist."""
    blob = client.bucket(GCS_BUCKET).blob(GCS_LATEST_KEY)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def upload_json(client: storage.Client, obj: dict, gcs_path: str) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    blob = client.bucket(GCS_BUCKET).blob(gcs_path)
    blob.upload_from_file(io.BytesIO(data), content_type="application/json")


# ── Main check logic ──────────────────────────────────────────────────────────

def run_check() -> dict:
    run_ts = datetime.now(timezone.utc)
    run_ts_str = run_ts.strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 62)
    print(f"  Phase 1 — MassDOT weekly check  {run_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 62)

    gcs_client = storage.Client()

    # ── Load previous manifest ────────────────────────────────────────────────
    previous = load_latest_manifest(gcs_client)
    if previous:
        prev_counts = {
            int(e["year"]): e.get("count")
            for e in previous.get("endpoints", [])
            if e.get("status") == "ok"
        }
        print(f"\n  Previous check: {previous.get('run_timestamp', 'unknown')}")
        print(f"  Previous check path: gs://{GCS_BUCKET}/{previous.get('gcs_path', GCS_LATEST_KEY)}")
    else:
        prev_counts = {}
        print("\n  No previous check found — this is the first run (baseline).")

    # ── Build full endpoint list: known + future probes ───────────────────────
    max_known_year = max(KNOWN_ENDPOINTS.keys())
    probe_endpoints = {}
    for offset in range(1, FUTURE_PROBE_COUNT + 1):
        probe_year = max_known_year + offset
        probe_endpoints[probe_year] = {
            "url":   _new_domain_url(probe_year),
            "where": "CITY_TOWN_NAME='BOSTON'",
            "note":  "future probe — may not exist yet",
        }

    all_endpoints = {**KNOWN_ENDPOINTS, **probe_endpoints}

    # ── Query every endpoint ──────────────────────────────────────────────────
    print(f"\n  Querying {len(all_endpoints)} endpoints (count-only, no data downloaded) ...\n")

    endpoint_results = []
    years_with_new_data     = []
    years_with_count_change = []
    years_with_errors       = []
    new_endpoints_found     = []

    for year in sorted(all_endpoints.keys()):
        cfg   = all_endpoints[year]
        url   = cfg["url"]
        where = cfg["where"]
        note  = cfg.get("note", "")

        count, error = query_count(url, where)
        time.sleep(0.2)   # polite rate limiting

        prev_count     = prev_counts.get(year)
        in_production  = PRODUCTION_YEAR_MIN <= year <= PRODUCTION_YEAR_MAX
        is_first_seen  = year not in prev_counts and count is not None

        # Determine change type
        if error:
            status      = "error"
            change_type = "none"
            years_with_errors.append(year)
            flag = "ERROR"
        elif count is not None:
            status = "ok"
            if is_first_seen and not in_production:
                # Year we've never recorded AND it's beyond production coverage
                change_type = "new_endpoint_beyond_production"
                new_endpoints_found.append(year)
                years_with_new_data.append(year)
                flag = "NEW (beyond production)"
            elif is_first_seen:
                # Year in production range but never recorded in a check
                change_type = "first_recorded"
                flag = f"FIRST RECORD (count={count})"
            elif prev_count is not None and count != prev_count:
                # Count changed since last check
                change_type = "count_changed"
                years_with_count_change.append(year)
                years_with_new_data.append(year)
                flag = f"COUNT CHANGED ({prev_count} → {count})"
            else:
                change_type = "unchanged"
                flag = "unchanged"
        else:
            # count is None and no error string means endpoint likely doesn't exist
            status      = "not_found"
            change_type = "none"
            flag = "not found"

        result = {
            "year":           year,
            "url":            url,
            "where_clause":   where,
            "status":         status,
            "count":          count,
            "previous_count": prev_count,
            "change_type":    change_type,
            "in_production":  in_production,
            "is_probe":       year in probe_endpoints,
        }
        if note:
            result["note"] = note
        if error:
            result["error"] = error

        endpoint_results.append(result)

        status_display = f"count={count}" if count is not None else f"error: {error}"
        print(f"  {year}  {status_display:<25}  {flag}")

    # ── Determine recommended action ──────────────────────────────────────────
    if years_with_new_data:
        recommended_action = "download_recommended"
        action_reason = (
            f"New or changed data detected in years: {years_with_new_data}. "
            "Run pipeline_download.py next."
        )
    elif years_with_errors and not years_with_new_data:
        recommended_action = "retry_check"
        action_reason = (
            f"Errors on years {years_with_errors} but no confirmed new data. "
            "Retry next run."
        )
    elif not previous:
        recommended_action = "baseline_recorded"
        action_reason = (
            "First run — baseline counts recorded. "
            "No download triggered (no delta to compare against)."
        )
    else:
        recommended_action = "no_action"
        action_reason = "All known endpoints unchanged since previous check."

    # ── Build manifest ────────────────────────────────────────────────────────
    gcs_path = f"{GCS_CHECK_DIR}/{run_ts_str}.json"

    manifest = {
        "schema_version":             "1.0",
        "run_timestamp":              run_ts.isoformat(),
        "gcs_path":                   gcs_path,
        "previous_check_gcs_path":    previous.get("gcs_path") if previous else None,
        "production_year_range":      [PRODUCTION_YEAR_MIN, PRODUCTION_YEAR_MAX],
        "endpoints_queried":          len(all_endpoints),
        "endpoints":                  endpoint_results,
        "summary": {
            "years_with_new_data":      years_with_new_data,
            "years_with_count_change":  years_with_count_change,
            "new_endpoints_found":      new_endpoints_found,
            "years_with_errors":        years_with_errors,
            "recommended_action":       recommended_action,
            "action_reason":            action_reason,
        },
    }

    # ── Upload to GCS ─────────────────────────────────────────────────────────
    print(f"\n  Uploading manifest → gs://{GCS_BUCKET}/{gcs_path}")
    upload_json(gcs_client, manifest, gcs_path)

    print(f"  Updating latest pointer → gs://{GCS_BUCKET}/{GCS_LATEST_KEY}")
    upload_json(gcs_client, manifest, GCS_LATEST_KEY)

    # ── Update pipeline_state.json with check result ──────────────────────────
    state_blob     = gcs_client.bucket(GCS_BUCKET).blob(GCS_PIPELINE_STATE)
    pipeline_state = (json.loads(state_blob.download_as_text())
                      if state_blob.exists() else {})

    pipeline_state["latest_check_timestamp"]             = run_ts_str
    pipeline_state["latest_check_action"]                = recommended_action
    pipeline_state["latest_check_years_with_new_data"]   = years_with_new_data

    # Only set recommended_next_action when Phase 1 has something actionable to say.
    # If no new data was found, don't overwrite a more advanced phase's recommendation
    # (e.g. "run_phase4_retrain" set by Phase 3 should not be reset to "check_for_new_data").
    if recommended_action == "download_recommended":
        pipeline_state["recommended_next_action"] = "download_new_data"
        pipeline_state["recommended_next_action_reason"] = (
            f"Phase 1 detected new or changed MassDOT data for year(s) {years_with_new_data}. "
            "Run pipeline_download.py to fetch and transform it."
        )
    elif not pipeline_state.get("recommended_next_action"):
        pipeline_state["recommended_next_action"] = "check_for_new_data"
        pipeline_state["recommended_next_action_reason"] = (
            "No new MassDOT data detected this week. "
            "The weekly job will check again next Monday."
        )

    upload_json(gcs_client, pipeline_state, GCS_PIPELINE_STATE)
    print(f"  Pipeline state updated → gs://{GCS_BUCKET}/{GCS_PIPELINE_STATE}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Endpoints queried  : {len(all_endpoints)}")
    print(f"  New data detected  : {years_with_new_data or 'none'}")
    print(f"  Count changes      : {years_with_count_change or 'none'}")
    print(f"  New endpoints      : {new_endpoints_found or 'none'}")
    print(f"  Errors             : {years_with_errors or 'none'}")
    print(f"  Recommended action : {recommended_action}")
    print(f"  Reason             : {action_reason}")
    print(f"\n  Manifest written to:")
    print(f"    gs://{GCS_BUCKET}/{gcs_path}")
    print(f"    gs://{GCS_BUCKET}/{GCS_LATEST_KEY}  (latest pointer)")
    print("=" * 62)

    return manifest


if __name__ == "__main__":
    manifest = run_check()
    # Exit code 0 always — errors within individual endpoints are logged in the
    # manifest but don't fail the overall check job.
    sys.exit(0)
