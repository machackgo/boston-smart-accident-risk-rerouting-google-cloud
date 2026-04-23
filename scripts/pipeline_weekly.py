"""
pipeline_weekly.py — Weekly orchestrator for Phases 1–3 of the champion–challenger pipeline.

Runs the three safe automated phases in sequence:
  Phase 1  pipeline_check.py      — query MassDOT for new data (count-only, no download)
  Phase 2  pipeline_download.py   — download + archive raw data, transform to processed format
  Phase 3  pipeline_accumulate.py — merge processed data into challenger training dataset

Phase 4 (pipeline_retrain.py) is intentionally NOT run here — it is manual.
Run it separately when you decide there is enough accumulated data to justify a retrain:
    python scripts/pipeline_retrain.py

Usage:
    python scripts/pipeline_weekly.py          # normal run
    python scripts/pipeline_weekly.py --dry-run # pass --dry-run to each phase (no writes)

This script is safe to run at any time:
  - It writes only to candidate/ and pipeline/ prefixes in GCS
  - It never touches production/ data, models, or the running API
  - If Phase 1 finds no new data it exits 0 immediately (nothing else runs)

Cloud Run Job invocation (see infrastructure setup below):
    Entry point is just this script — Cloud Scheduler triggers it weekly.

Exit codes:
    0 = success (either "no new data" clean exit, or all three phases completed)
    1 = one of the phases failed; check Cloud Run Job logs
"""

import io
import json
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

GCS_BUCKET    = "boston-rerouting-data"
SCRIPTS_DIR   = Path(__file__).resolve().parent

# Notification helper — imported lazily so a broken notify.py never kills the pipeline.
try:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from notify import (notify_no_new_data, notify_new_data_detected,
                        notify_phase3_complete, notify_pipeline_failure)
    _NOTIFY = True
except Exception as _notify_err:
    print(f"  [notify] Import failed — notifications disabled: {_notify_err}")
    _NOTIFY = False

# BigQuery run logger — imported lazily so a BQ failure never kills the pipeline.
try:
    from bq_pipeline import log_pipeline_run as _bq_log_run
    _BQ = True
except Exception as _bq_err:
    print(f"  [bq] Import failed — BigQuery pipeline logging disabled: {_bq_err}")
    _BQ = False


# ── helpers ───────────────────────────────────────────────────────────────────

def sec(title: str):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


def run_phase(script: str, label: str, extra_args: list[str] | None = None) -> bool:
    """
    Run a pipeline script as a subprocess.
    stdout/stderr flow directly to the parent process (and therefore to Cloud Run logs).
    Returns True on success (exit code 0), False on failure.
    """
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + (extra_args or [])
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n  [{label}] completed successfully.")
        return True
    else:
        print(f"\n  [{label}] FAILED (exit code {result.returncode}).")
        return False


def gcs_read_json(client: storage.Client, path: str) -> dict | None:
    blob = client.bucket(GCS_BUCKET).blob(path)
    return json.loads(blob.download_as_text()) if blob.exists() else None


def gcs_write_json(client: storage.Client, obj: dict, path: str):
    data = json.dumps(obj, indent=2, default=str).encode("utf-8")
    client.bucket(GCS_BUCKET).blob(path).upload_from_file(
        io.BytesIO(data), content_type="application/json"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    run_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    dry_run   = "--dry-run" in sys.argv
    extra     = ["--dry-run"] if dry_run else []
    t0        = time.monotonic()

    sec(f"Weekly Pipeline Run  {run_ts}" + ("  [DRY RUN]" if dry_run else ""))
    print("  Automated scope : Phase 1 (check) → Phase 2 (download) → Phase 3 (accumulate)")
    print("  Phase 4 (retrain) : MANUAL — run pipeline_retrain.py when ready")

    client = storage.Client()

    def _elapsed() -> int:
        return int(time.monotonic() - t0)

    def _bq(outcome: str, **kw) -> None:
        if _BQ and not dry_run:
            _bq_log_run(run_ts, "weekly_phases_1_3", outcome,
                        is_dry_run=dry_run, triggered_by="cloud_scheduler",
                        duration_seconds=_elapsed(), **kw)

    # ── Phase 1: check for new MassDOT data ───────────────────────────────────
    sec("Phase 1 — MassDOT availability check")
    ok = run_phase("pipeline_check.py", "Phase 1", extra)
    if not ok:
        _write_log(client, run_ts, "phase1_failed", {}, dry_run)
        print("\n  Phase 1 failed. Aborting weekly run.")
        if _NOTIFY and not dry_run:
            notify_pipeline_failure("Phase 1 (MassDOT availability check)", run_ts)
        _bq("phase1_failed")
        sys.exit(1)

    # Read Phase 1 result to decide whether to proceed
    check_result = gcs_read_json(client, "candidate/checks/latest.json")
    if not check_result:
        print("\n  ERROR: Could not read Phase 1 output from GCS.")
        sys.exit(1)

    action       = check_result["summary"]["recommended_action"]
    new_years    = check_result["summary"]["years_with_new_data"]
    years_checked = check_result.get("years_checked")

    print(f"\n  Phase 1 result    : {action}")
    print(f"  Years with new data: {new_years or 'none'}")

    if action != "download_recommended" or not new_years:
        msg = "No new MassDOT data detected. Weekly run complete — nothing to download."
        print(f"\n  {msg}")
        _write_log(client, run_ts, "no_new_data", {"action": action}, dry_run)
        notif_sent = False
        if _NOTIFY and not dry_run:
            pipeline_state = gcs_read_json(client, "candidate/pipeline_state.json") or {}
            notif_sent = notify_no_new_data(run_ts, action, pipeline_state)
        _bq("no_new_data", years_checked=years_checked, notification_sent=notif_sent)
        sys.exit(0)

    # New data confirmed — notify before starting Phase 2
    notif_new = False
    if _NOTIFY and not dry_run:
        notif_new = notify_new_data_detected(new_years, run_ts)

    # ── Phase 2: download + transform ─────────────────────────────────────────
    sec(f"Phase 2 — Download + transform  (years: {new_years})")
    ok = run_phase("pipeline_download.py", "Phase 2", extra)
    if not ok:
        _write_log(client, run_ts, "phase2_failed", {"new_years": new_years}, dry_run)
        print("\n  Phase 2 failed. Raw data may be partially written. Check GCS.")
        if _NOTIFY and not dry_run:
            notify_pipeline_failure(
                "Phase 2 (download + transform)", run_ts,
                f"New years that failed: {new_years}",
            )
        _bq("phase2_failed", years_checked=years_checked,
            years_with_new_data=new_years, notification_sent=notif_new)
        sys.exit(1)

    # ── Phase 3: accumulate into challenger dataset ───────────────────────────
    sec("Phase 3 — Merge into challenger training dataset")
    ok = run_phase("pipeline_accumulate.py", "Phase 3", extra)
    if not ok:
        _write_log(client, run_ts, "phase3_failed", {"new_years": new_years}, dry_run)
        print("\n  Phase 3 failed. Processed data is safely archived. Re-run Phase 3 to retry.")
        if _NOTIFY and not dry_run:
            notify_pipeline_failure(
                "Phase 3 (accumulate / merge)", run_ts,
                "Processed parquets are safe in candidate/processed/. Re-run Phase 3 to retry.",
            )
        _bq("phase3_failed", years_checked=years_checked,
            years_with_new_data=new_years, notification_sent=notif_new)
        sys.exit(1)

    # ── All phases complete ────────────────────────────────────────────────────
    state = gcs_read_json(client, "candidate/pipeline_state.json") or {}
    merged_rows = state.get("latest_merged_rows", "unknown")
    merged_path = state.get("latest_merged_path", "unknown")

    notif_p3 = False
    if _NOTIFY and not dry_run:
        notif_p3 = notify_phase3_complete(state, run_ts)

    _write_log(client, run_ts, "success", {
        "new_years":       new_years,
        "merged_rows":     merged_rows,
        "merged_gcs_path": merged_path,
    }, dry_run)

    _bq("success",
        years_checked=years_checked,
        years_with_new_data=new_years,
        merged_row_count=merged_rows if isinstance(merged_rows, int) else None,
        merged_gcs_path=merged_path if merged_path != "unknown" else None,
        candidate_years=state.get("years_in_candidate_dataset"),
        notification_sent=notif_p3)

    sec("Weekly run complete")
    print(f"  Phases 1–3 completed successfully.")
    print(f"  New data years   : {new_years}")
    print(f"  Merged rows      : {merged_rows:,}" if isinstance(merged_rows, int) else
          f"  Merged rows      : {merged_rows}")
    print(f"  Challenger dataset: gs://{GCS_BUCKET}/{merged_path}")
    print()
    print("  Phase 4 (retrain + compare) is MANUAL.")
    print("  When you are ready to evaluate whether the challenger beats the champion:")
    print("    python scripts/pipeline_retrain.py")
    print("=" * 62)


def _write_log(client: storage.Client, run_ts: str, outcome: str,
               details: dict, dry_run: bool) -> None:
    log = {
        "run_timestamp": run_ts,
        "dry_run":       dry_run,
        "outcome":       outcome,
        **details,
    }
    path = f"pipeline/logs/{run_ts}.json"
    if not dry_run:
        gcs_write_json(client, log, path)
        print(f"\n  Run log → gs://{GCS_BUCKET}/{path}")
    else:
        print(f"\n  [DRY RUN] Would write log → gs://{GCS_BUCKET}/{path}")


if __name__ == "__main__":
    main()
