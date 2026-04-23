"""
bq_pipeline.py — BigQuery logging helper for pipeline run events.

Imported by pipeline_weekly.py and pipeline_retrain.py.
Writes to boston-rerouting.boston_rerouting.pipeline_runs.

Safe contract:
  - Never raises under any circumstances
  - A BigQuery failure prints a warning and the pipeline continues
  - Creates its own BQ client per call (pipeline scripts are long-running
    jobs, not request handlers — no need for a persistent singleton)
"""

import json
from datetime import datetime, timezone

from google.cloud import bigquery

BQ_PROJECT         = "boston-rerouting"
BQ_DATASET         = "boston_rerouting"
_TABLE_PIPELINE    = f"{BQ_PROJECT}.{BQ_DATASET}.pipeline_runs"
_TABLE_CHALLENGERS = f"{BQ_PROJECT}.{BQ_DATASET}.challenger_runs"
_TABLE_MODELS      = f"{BQ_PROJECT}.{BQ_DATASET}.model_versions"


def _parse_run_ts(run_ts: str) -> str:
    """
    Convert a pipeline run_ts string (e.g. '2026-04-21_06-00-00')
    to a BigQuery-compatible ISO 8601 UTC timestamp string.
    Returns the input unchanged if parsing fails.
    """
    try:
        dt = datetime.strptime(run_ts, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return run_ts


def _to_timestamp(value: str | None) -> str | None:
    """
    Ensure a date or datetime string is a full ISO 8601 timestamp BigQuery accepts.
    Bare dates like '2026-04-17' are promoted to '2026-04-17T00:00:00+00:00'.
    Values that already contain a time component are returned unchanged.
    """
    if value is None:
        return None
    if "T" in value or " " in value:
        return value
    try:
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return value


def _json_or_none(value) -> str | None:
    """Serialize a list/dict to a JSON string, or return None."""
    if value is None:
        return None
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def log_pipeline_run(run_ts: str, phase: str, outcome: str, **kwargs) -> None:
    """
    Write one pipeline execution record to BigQuery pipeline_runs table.

    Args:
        run_ts:   The run timestamp string (e.g. '2026-04-21_06-00-00').
        phase:    'weekly_phases_1_3' or 'phase4_retrain'.
        outcome:  'success' | 'no_new_data' | 'phase1_failed' | 'phase2_failed' |
                  'phase3_failed'.
        **kwargs: Optional extra fields:
            years_checked          (list)
            years_with_new_data    (list)
            years_downloaded       (list)
            merged_row_count       (int)
            merged_gcs_path        (str)
            candidate_years        (list)
            challenger_f1          (float)
            all_gates_passed       (bool)
            recommended_action     (str)
            is_dry_run             (bool)
            notification_sent      (bool)
            duration_seconds       (int)
            triggered_by           (str)  default 'cloud_scheduler'
    """
    try:
        row = {
            "run_id":               run_ts,
            "phase":                phase,
            "run_timestamp":        _parse_run_ts(run_ts),
            "outcome":              outcome,
            "years_checked":        _json_or_none(kwargs.get("years_checked")),
            "years_with_new_data":  _json_or_none(kwargs.get("years_with_new_data")),
            "years_downloaded":     _json_or_none(kwargs.get("years_downloaded")),
            "merged_row_count":     kwargs.get("merged_row_count"),
            "merged_gcs_path":      kwargs.get("merged_gcs_path"),
            "candidate_years":      _json_or_none(kwargs.get("candidate_years")),
            "challenger_f1":        kwargs.get("challenger_f1"),
            "all_gates_passed":     kwargs.get("all_gates_passed"),
            "recommended_action":   kwargs.get("recommended_action"),
            "is_dry_run":           kwargs.get("is_dry_run", False),
            "notification_sent":    kwargs.get("notification_sent"),
            "duration_seconds":     kwargs.get("duration_seconds"),
            "triggered_by":         kwargs.get("triggered_by", "cloud_scheduler"),
        }
        client = bigquery.Client(project=BQ_PROJECT)
        errors = client.insert_rows_json(_TABLE_PIPELINE, [row])
        if errors:
            print(f"  [bq_pipeline] insert_rows_json errors: {errors}")
        else:
            print(f"  [bq_pipeline] Pipeline run logged → BigQuery {_TABLE_PIPELINE}")
    except Exception as exc:
        print(f"  [bq_pipeline] log_pipeline_run failed (pipeline continues): {exc}")


def log_challenger_run(run_ts: str, comparison: dict, pipeline_state: dict) -> None:
    """
    Write one Phase 4 challenger comparison result to BigQuery challenger_runs.

    Args:
        run_ts:         The retrain run_ts string.
        comparison:     The comparison dict written to comparison.json by pipeline_retrain.py.
        pipeline_state: The current pipeline_state dict (for last_retrain_merged_rows).
    """
    try:
        champ   = comparison.get("champion", {})
        chal    = comparison.get("challenger", {})
        gates   = comparison.get("promotion_gates", {})
        failed  = [n for n, g in gates.items() if not g.get("passed", True)]

        row = {
            "run_id":                 run_ts,
            "run_timestamp":          _parse_run_ts(run_ts),
            "merged_dataset_gcs":     chal.get("model_gcs_path", "").replace(
                "best_model_challenger.pkl", "").rstrip("/"),  # folder path
            "merged_row_count":       comparison.get("merged_rows_total"),
            "effective_row_count":    chal.get("effective_rows"),
            "candidate_years":        _json_or_none(chal.get("year_range")),
            "last_retrain_row_count": pipeline_state.get("last_retrain_merged_rows"),
            "challenger_model_id":    f"challenger_{run_ts[:10].replace('-', '')}",
            "challenger_macro_f1":    chal.get("macro_f1"),
            "challenger_roc_auc":     chal.get("binary_roc_auc"),
            "challenger_high_f1":     chal.get("high_f1"),
            "challenger_medium_f1":   chal.get("medium_f1"),
            "challenger_low_f1":      chal.get("low_f1"),
            "champion_model_id":      "v4",
            "champion_macro_f1":      champ.get("macro_f1"),
            "champion_roc_auc":       champ.get("binary_roc_auc"),
            "all_gates_passed":       comparison.get("all_gates_passed"),
            "failed_gates":           _json_or_none(failed),
            "recommended_action":     comparison.get("recommended_action"),
            "recommendation_reason":  comparison.get("recommendation_reason"),
            "report_gcs":             None,  # filled by caller if needed
            "model_gcs":              chal.get("model_gcs_path"),
            "comparison_gcs":         f"candidate/reports/{run_ts}/comparison.json",
        }

        client = bigquery.Client(project=BQ_PROJECT)
        errors = client.insert_rows_json(_TABLE_CHALLENGERS, [row])
        if errors:
            print(f"  [bq_pipeline] challenger_runs insert errors: {errors}")
        else:
            print(f"  [bq_pipeline] Challenger run logged → BigQuery {_TABLE_CHALLENGERS}")
    except Exception as exc:
        print(f"  [bq_pipeline] log_challenger_run failed (pipeline continues): {exc}")


def register_model_version(model_id: str, display_name: str,
                            metrics: dict, artifact_gcs: str,
                            is_champion: bool = False) -> None:
    """
    Write one model version record to BigQuery model_versions.
    Called by bq_backfill.py for the champion and by pipeline_retrain.py for challengers.
    """
    try:
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "model_id":             model_id,
            "display_name":         display_name,
            "training_date":        _to_timestamp(metrics.get("training_date")),
            "training_dataset_gcs": metrics.get("training_dataset_gcs"),
            "training_row_count":   metrics.get("training_row_count"),
            "year_range_min":       metrics.get("year_range_min"),
            "year_range_max":       metrics.get("year_range_max"),
            "macro_f1":             metrics.get("macro_f1"),
            "accuracy":             metrics.get("accuracy"),
            "binary_roc_auc":       metrics.get("binary_roc_auc"),
            "high_f1":              metrics.get("high_f1"),
            "medium_f1":            metrics.get("medium_f1"),
            "low_f1":               metrics.get("low_f1"),
            "test_set_size":        metrics.get("test_set_size"),
            "is_champion":          is_champion,
            "is_active":            True,
            "vertex_model_id":      None,
            "vertex_endpoint_id":   None,
            "traffic_pct":          100 if is_champion else 0,
            "registered_at":        now,
            "promoted_at":          _to_timestamp(metrics.get("training_date")) if is_champion else None,
            "deprecated_at":        None,
            "champion_metrics_gcs": metrics.get("champion_metrics_gcs"),
            "model_artifact_gcs":   artifact_gcs,
        }

        client = bigquery.Client(project=BQ_PROJECT)
        errors = client.insert_rows_json(_TABLE_MODELS, [row])
        if errors:
            print(f"  [bq_pipeline] model_versions insert errors: {errors}")
        else:
            print(f"  [bq_pipeline] Model version registered → BigQuery {_TABLE_MODELS}")
    except Exception as exc:
        print(f"  [bq_pipeline] register_model_version failed (pipeline continues): {exc}")
