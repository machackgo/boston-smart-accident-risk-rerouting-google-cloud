"""
notify.py — Email notification helper for the Boston accident risk pipeline.

All public functions are safe to call from any pipeline script:
  - They never raise exceptions under any circumstances
  - A notification failure prints a warning but never aborts the pipeline
  - Gmail App Password is fetched from Google Cloud Secret Manager at call time
  - Deduplication state is stored in GCS so duplicate emails are never sent

Deduplication rules (simple summary):
  - "No new data"     → at most one email per calendar week
  - "New data found"  → always send (each run_ts is unique)
  - "Phase 3 done"    → send only when the merged GCS path is new (dataset changed)
  - "Phase 4 done"    → send only once per retrain run_ts
  - "Failure"         → always send (never deduplicate failures)

GCS dedup state file:
  candidate/notifications_state.json

Setup required (one-time — see bottom of this file for gcloud commands):
  1. Create a Gmail App Password for mohammedmubashir149@gmail.com
  2. Store it as Secret Manager secret "pipeline-gmail-app-password"
  3. Grant the Cloud Run service account secretAccessor on that secret

Public API:
  notify_no_new_data(run_ts, action, pipeline_state)
  notify_new_data_detected(new_years, run_ts)
  notify_phase3_complete(state, run_ts)
  notify_retrain_complete(state, challenger_f1, champ_f1, all_gates_passed, failed_gates, run_ts)
  notify_pipeline_failure(phase, run_ts, extra_info="")
  send_test_email()
"""

import io
import json
import smtplib
import traceback
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google.cloud import secretmanager, storage

# ── Config ────────────────────────────────────────────────────────────────────
RECIPIENT_EMAIL   = "mohammedmubashir149@gmail.com"
SENDER_EMAIL      = "mohammedmubashir149@gmail.com"
GCP_PROJECT       = "boston-rerouting"
SECRET_NAME       = "pipeline-gmail-app-password"
SMTP_HOST         = "smtp.gmail.com"
SMTP_PORT         = 587
SUBJECT_PREFIX    = "[Boston Pipeline]"

GCS_BUCKET        = "boston-rerouting-data"
GCS_NOTIF_STATE   = "candidate/notifications_state.json"
GCS_PIPELINE_STATE_URL = "gs://boston-rerouting-data/candidate/pipeline_state.json"


# ── GCS deduplication state ───────────────────────────────────────────────────

def _gcs_client():
    """Return a GCS client, or None on failure. Never raises."""
    try:
        return storage.Client()
    except Exception as e:
        print(f"  [notify] Could not create GCS client for dedup: {e}")
        return None


def _read_notif_state(client) -> dict:
    """Read candidate/notifications_state.json. Returns {} on any failure."""
    try:
        blob = client.bucket(GCS_BUCKET).blob(GCS_NOTIF_STATE)
        return json.loads(blob.download_as_text()) if blob.exists() else {}
    except Exception as e:
        print(f"  [notify] Could not read notification state: {e}")
        return {}


def _write_notif_state(client, state: dict) -> None:
    """Write candidate/notifications_state.json. Silently swallows failures."""
    try:
        data = json.dumps(state, indent=2).encode("utf-8")
        client.bucket(GCS_BUCKET).blob(GCS_NOTIF_STATE).upload_from_file(
            io.BytesIO(data), content_type="application/json"
        )
    except Exception as e:
        print(f"  [notify] Could not write notification state (dedup will retry next run): {e}")


def _iso_week(run_ts: str) -> str:
    """Return 'YYYY-WNN' for deduplicating weekly 'no new data' emails."""
    try:
        d = datetime.strptime(run_ts[:10], "%Y-%m-%d")
        return f"{d.year}-W{d.isocalendar()[1]:02d}"
    except Exception:
        return run_ts[:10]


# ── Core: credential fetch + SMTP send ───────────────────────────────────────

def _get_gmail_password() -> str | None:
    """Fetch Gmail App Password from Secret Manager. Returns None on failure."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name   = f"projects/{GCP_PROJECT}/secrets/{SECRET_NAME}/versions/latest"
        resp   = client.access_secret_version(request={"name": name})
        return resp.payload.data.decode("utf-8").strip()
    except Exception as e:
        print(f"  [notify] Secret Manager lookup failed: {e}")
        return None


def _send(subject: str, body: str) -> bool:
    """
    Send a plain-text email via Gmail SMTP.
    Returns True on success, False on any failure. Never raises.
    """
    try:
        password = _get_gmail_password()
        if not password:
            print("  [notify] Skipping email — credentials unavailable.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{SUBJECT_PREFIX} {subject}"
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECIPIENT_EMAIL
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, password)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())

        print(f"  [notify] Email sent → {RECIPIENT_EMAIL} | {subject[:70]}")
        return True

    except Exception:
        print(f"  [notify] Email failed (pipeline continues):\n"
              f"{traceback.format_exc(limit=3)}")
        return False


# ── Shared formatting ─────────────────────────────────────────────────────────

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_rows(val) -> str:
    return f"{val:,}" if isinstance(val, int) else str(val or "—")


def _state_block(state: dict) -> str:
    """Format the key readiness fields from pipeline_state.json for email bodies."""
    lines = [
        "── Pipeline state ──────────────────────────────────────────────",
        f"  recommended_next_action         : {state.get('recommended_next_action', '—')}",
        f"  recommended_next_action_reason  :",
        f"    {state.get('recommended_next_action_reason', '—')}",
        f"  latest_check_timestamp          : {state.get('latest_check_timestamp', '—')}",
        f"  latest_check_years_with_new_data: {state.get('latest_check_years_with_new_data', '—')}",
        f"  latest_merged_row_count         : {_fmt_rows(state.get('latest_merged_row_count'))}",
        f"  years_in_candidate_dataset      : {state.get('years_in_candidate_dataset', '—')}",
        f"  latest_challenger_f1            : {state.get('latest_challenger_f1', '—')}",
        f"  latest_recommended_action       : {state.get('latest_recommended_action', '—')}",
        "────────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


# ── Event: no new data (deduplicated per calendar week) ──────────────────────

def notify_no_new_data(run_ts: str, action: str,
                       pipeline_state: dict) -> bool:
    """
    Sent when Phase 1 finds no new MassDOT data.

    Deduplication: at most one email per ISO calendar week.
    If the weekly job runs every Monday and consistently finds no data,
    you receive one email per week — not silence, not spam.
    """
    this_week = _iso_week(run_ts)

    client = _gcs_client()
    if client:
        notif_state = _read_notif_state(client)
        if notif_state.get("last_no_new_data_week") == this_week:
            print(f"  [notify] Skipping 'no new data' — already sent for week {this_week}.")
            return False

    merged_rows = pipeline_state.get("latest_merged_row_count")
    cand_years  = pipeline_state.get("years_in_candidate_dataset", "—")
    merged_ts   = pipeline_state.get("latest_merged_timestamp", "—")
    next_action = pipeline_state.get("recommended_next_action", "—")

    subject = f"Weekly check — no new data  [{run_ts[:10]}]"
    body = f"""\
Boston Accident Risk Pipeline — {_now_str()}

Phase 1 (weekly MassDOT check) ran and found no new crash data.
Nothing was downloaded or changed. The pipeline exited cleanly.

  Check result       : {action}
  Run timestamp      : {run_ts}

Current candidate dataset status:
  Merged row count   : {_fmt_rows(merged_rows)} rows
  Candidate years    : {cand_years}
  Last merged        : {merged_ts}
  Recommended action : {next_action}

What to do:
  Nothing right now. The weekly job will check again next Monday.
  When MassDOT publishes new data, you will receive a separate email.

{_state_block(pipeline_state)}

Pipeline state: {GCS_PIPELINE_STATE_URL}
"""
    ok = _send(subject, body)
    if ok and client:
        notif_state["last_no_new_data_week"] = this_week
        _write_notif_state(client, notif_state)
    return ok


# ── Event: new data detected (always send — each run_ts is unique) ───────────

def notify_new_data_detected(new_years: list, run_ts: str) -> bool:
    """
    Sent when Phase 1 detects new or changed MassDOT data.
    The weekly job continues into Phase 2 and Phase 3 automatically.
    A follow-up email arrives when Phase 3 finishes.

    No deduplication: each run_ts is unique, so this is never a duplicate.
    """
    subject = f"New MassDOT data detected — pipeline running  [{run_ts[:10]}]"
    body = f"""\
Boston Accident Risk Pipeline — {_now_str()}

Phase 1 (weekly availability check) detected new or changed MassDOT
crash data. The weekly job is now running Phase 2 (download) and
Phase 3 (accumulate) automatically.

  Years with new data : {new_years}
  Run timestamp       : {run_ts}

No action needed from you right now.
You will receive a follow-up email when the candidate dataset has been
updated and Phase 4 retraining can be manually triggered.

Pipeline state: {GCS_PIPELINE_STATE_URL}
"""
    return _send(subject, body)


# ── Event: Phase 3 complete / dataset updated (dedup on merged path) ─────────

def notify_phase3_complete(state: dict, run_ts: str) -> bool:
    """
    Sent when Phases 1–3 all succeed and the challenger training dataset
    has been updated. This is the primary signal to consider running Phase 4.

    Deduplication: if we already sent a notification for this exact merged
    dataset (same GCS path), we skip. This prevents re-sending if pipeline_weekly
    is re-run manually for any reason.
    """
    merged_path = state.get("latest_merged_path", "")
    row_count   = state.get("latest_merged_row_count")
    cand_years  = state.get("years_in_candidate_dataset", [])
    last_retrain_rows = state.get("last_retrain_merged_rows")

    client = _gcs_client()
    if client and merged_path:
        notif_state = _read_notif_state(client)
        if notif_state.get("last_notified_merged_path") == merged_path:
            print(f"  [notify] Skipping Phase 3 complete — already notified for this dataset.")
            return False

    # Compute how many new rows since the last retrain (if there was one)
    if isinstance(row_count, int) and isinstance(last_retrain_rows, int):
        new_since_retrain = row_count - last_retrain_rows
        delta_line = (f"  Rows since last retrain : +{new_since_retrain:,} rows "
                      f"(last retrain had {last_retrain_rows:,} rows)")
    elif last_retrain_rows is None:
        delta_line = "  Rows since last retrain : N/A (Phase 4 has not been run yet)"
    else:
        delta_line = ""

    subject = f"Challenger dataset updated — Phase 4 ready  [{run_ts[:10]}]"
    body = f"""\
Boston Accident Risk Pipeline — {_now_str()}

Phases 1–3 completed successfully. The challenger training dataset
has been updated and is ready for manual Phase 4 retraining.

  Merged row count    : {_fmt_rows(row_count)} rows
  Candidate years     : {cand_years}
  Run timestamp       : {run_ts}
{delta_line}

What to do next:
  When you are ready to evaluate the challenger model:
    python scripts/pipeline_retrain.py

  You will receive another email with the full comparison results.
  The production model is NOT changed until you promote manually.

{_state_block(state)}

Candidate dataset : gs://boston-rerouting-data/{merged_path}
Pipeline state    : {GCS_PIPELINE_STATE_URL}
"""
    ok = _send(subject, body)
    if ok and client and merged_path:
        notif_state["last_notified_merged_path"] = merged_path
        _write_notif_state(client, notif_state)
    return ok


# ── Event: Phase 4 retrain complete (dedup on retrain run_ts) ────────────────

def notify_retrain_complete(state: dict, challenger_f1: float,
                             champ_f1: float, all_gates_passed: bool,
                             failed_gates: list, run_ts: str) -> bool:
    """
    Sent at the end of every Phase 4 run, win or lose.

    Deduplication: skip if we already sent a notification for this exact
    retrain run_ts (guards against accidental double-calling).
    """
    client = _gcs_client()
    if client:
        notif_state = _read_notif_state(client)
        if notif_state.get("last_notified_retrain_ts") == run_ts:
            print(f"  [notify] Skipping retrain complete — already notified for {run_ts}.")
            return False

    delta = challenger_f1 - champ_f1
    arrow = "▲" if delta > 0 else "▼"

    if all_gates_passed:
        subject = f"CHALLENGER WINS — manual promotion required  [{run_ts[:10]}]"
        verdict = "ALL PROMOTION GATES PASSED."
        action_section = """\
What to do next — promote the challenger manually:
  1. Review the full comparison report (GCS link below)
  2. If satisfied, replace the production model artifacts in GCS:
       production/models/best_model_v4.pkl
       production/models/feature_list_v4.txt
       production/models/thresholds_v4.json
       production/models/weather_keep_cols_v4.json
  3. Update production/models/champion_manifest.json
  4. Update production/reports/champion_metrics.json
  5. Redeploy the Cloud Run service

  The production model is currently UNCHANGED."""
    else:
        subject = f"Challenger did not beat champion — keep accumulating  [{run_ts[:10]}]"
        verdict = f"Promotion gates FAILED: {failed_gates}"
        action_section = """\
What to do next:
  Continue accumulating MassDOT data via the weekly automated pipeline.
  Re-run pipeline_retrain.py when meaningfully more data has arrived
  (typically 3,000–5,000+ new rows makes a meaningful difference).

  The production model is UNCHANGED."""

    body = f"""\
Boston Accident Risk Pipeline — {_now_str()}

Phase 4 (retrain + compare) completed.

  Verdict             : {verdict}
  Challenger macro F1 : {challenger_f1:.4f}
  Champion macro F1   : {champ_f1:.4f}
  Delta               : {arrow}{abs(delta):.4f}
  Run timestamp       : {run_ts}

{action_section}

{_state_block(state)}

Latest comparison : gs://boston-rerouting-data/candidate/reports/latest_comparison.json
Pipeline state    : {GCS_PIPELINE_STATE_URL}
"""
    ok = _send(subject, body)
    if ok and client:
        notif_state["last_notified_retrain_ts"] = run_ts
        _write_notif_state(client, notif_state)
    return ok


# ── Event: pipeline failure (always send — never deduplicated) ───────────────

def notify_pipeline_failure(phase: str, run_ts: str,
                             extra_info: str = "") -> bool:
    """
    Sent when any pipeline phase exits with a non-zero return code.
    Failures are never deduplicated — every failure is worth knowing about.
    """
    subject = f"PIPELINE FAILURE — {phase} failed  [{run_ts[:10]}]"
    extra_block = f"\n  Details             : {extra_info}" if extra_info else ""
    body = f"""\
Boston Accident Risk Pipeline — {_now_str()}

A pipeline phase failed and the run was aborted.

  Failed phase        : {phase}
  Run timestamp       : {run_ts}{extra_block}

What to check:
  1. Cloud Run Job logs (full Python traceback):
       gcloud logging read \\
         'resource.type="cloud_run_job" AND resource.labels.job_name="boston-pipeline-weekly"' \\
         --limit=100 --project=boston-rerouting

  2. GCS candidate/ prefix for any partial writes:
       gsutil ls gs://boston-rerouting-data/candidate/

  3. Re-run manually to reproduce:
       python scripts/pipeline_weekly.py

Pipeline state: {GCS_PIPELINE_STATE_URL}
"""
    return _send(subject, body)


# ── Manual test helper ────────────────────────────────────────────────────────

def send_test_email() -> bool:
    """
    Send a test email to verify Secret Manager credentials and SMTP.
    Run directly:  python scripts/notify.py
    """
    subject = f"Test notification — credentials working  [{_now_str()}]"
    body = f"""\
Boston Accident Risk Pipeline — {_now_str()}

This is a test email confirming that:
  - The Gmail App Password was retrieved from Secret Manager successfully
  - Gmail SMTP authentication succeeded
  - Emails are delivered to {RECIPIENT_EMAIL}

No pipeline action is required.
"""
    return _send(subject, body)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"

    if cmd == "test":
        print("Sending test email ...")
        ok = send_test_email()
        print("Done." if ok else "Failed — check the error above.")

    elif cmd == "no-new-data":
        print("Sending 'no new data' test email (ignoring dedup) ...")
        ok = _send(
            f"TEST — no new data  [{_now_str()}]",
            f"Boston Accident Risk Pipeline — {_now_str()}\n\n"
            "This is a manual test of the 'no new data' notification.\n"
            "Deduplication was bypassed for this test.",
        )
        print("Done." if ok else "Failed.")

    elif cmd == "phase3":
        print("Sending 'Phase 3 complete' test email (ignoring dedup) ...")
        fake_state = {
            "recommended_next_action": "run_phase4_retrain",
            "recommended_next_action_reason": "Test — Phase 3 merge complete.",
            "latest_check_timestamp": _now_str(),
            "latest_check_years_with_new_data": [2025, 2026],
            "latest_merged_row_count": 55000,
            "years_in_candidate_dataset": [2025, 2026],
            "latest_challenger_f1": None,
            "latest_recommended_action": None,
            "latest_merged_path": "candidate/merged/test/challenger_training_dataset.parquet",
        }
        ok = _send(
            f"TEST — Challenger dataset updated  [{_now_str()}]",
            f"Boston Accident Risk Pipeline — {_now_str()}\n\n"
            "This is a manual test of the 'Phase 3 complete' notification.\n\n"
            + _state_block(fake_state),
        )
        print("Done." if ok else "Failed.")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python scripts/notify.py [test|no-new-data|phase3]")


# ── Setup instructions ────────────────────────────────────────────────────────
#
# ONE-TIME SETUP (already done if test email works — skip if credentials exist):
#
# 1. Create a Gmail App Password:
#    - Go to myaccount.google.com → Security → 2-Step Verification
#    - Scroll down → App passwords → create one named "boston-pipeline"
#    - Copy the 16-character password (shown only once)
#
# 2. Store in Secret Manager:
#    gcloud secrets create pipeline-gmail-app-password \
#      --replication-policy=automatic --project=boston-rerouting
#
#    echo -n "your-16-char-app-password" | \
#      gcloud secrets versions add pipeline-gmail-app-password \
#        --data-file=- --project=boston-rerouting
#
# 3. Grant Cloud Run service account access:
#    gcloud secrets add-iam-policy-binding pipeline-gmail-app-password \
#      --member=serviceAccount:294613088058-compute@developer.gserviceaccount.com \
#      --role=roles/secretmanager.secretAccessor \
#      --project=boston-rerouting
#
# 4. Grant Cloud Run service account write access to GCS (for dedup state):
#    Already granted via roles/storage.objectAdmin on the bucket.
#
# 5. Test locally:
#    python scripts/notify.py
#    python scripts/notify.py no-new-data
#    python scripts/notify.py phase3
