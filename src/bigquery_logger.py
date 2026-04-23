"""
bigquery_logger.py — BigQuery analytics sink for prediction events.

Parallel to database.py (Cloud SQL). Called as a FastAPI background task
alongside log_prediction() so both sinks receive every event independently.

Design rules (same as database.py):
  - Never raises to the caller under any circumstances
  - A BigQuery failure does not affect Cloud SQL, the API response, or the user
  - Client is a lazy singleton — created once per container instance

What it writes:
  boston-rerouting.boston_rerouting.route_predictions
  One row per /predict or /predict/segmented call.
  Same fields as Cloud SQL + prediction_id (UUID) + timestamp (TIMESTAMP)
  + vertex_* fields (null until Phase B) + gemini_* (false until Phase E).
"""

import uuid
from datetime import datetime, timezone

from google.cloud import bigquery

BQ_PROJECT    = "boston-rerouting"
BQ_DATASET    = "boston_rerouting"
_TABLE_PREDS  = f"{BQ_PROJECT}.{BQ_DATASET}.route_predictions"

_bq: bigquery.Client | None = None


def _client() -> bigquery.Client:
    global _bq
    if _bq is None:
        _bq = bigquery.Client(project=BQ_PROJECT)
    return _bq


def log_prediction_bq(row: dict) -> None:
    """
    Write one prediction event to BigQuery route_predictions.

    Accepts the same row dict as log_prediction() in database.py.
    Errors are printed and swallowed — Cloud SQL is unaffected.
    Called as a FastAPI background task:
        background_tasks.add_task(log_prediction_bq, row_dict)
    """
    try:
        bq_row = {
            "prediction_id":               str(uuid.uuid4()),
            "timestamp":                   datetime.now(timezone.utc).isoformat(),
            # ── Fields shared with Cloud SQL log ──────────────────────────────
            "endpoint":                    row.get("endpoint"),
            "origin":                      row.get("origin"),
            "destination":                 row.get("destination"),
            "departure_time":              row.get("departure_time"),
            "duration_minutes":            row.get("duration_minutes"),
            "distance_miles":              row.get("distance_miles"),
            "num_alternatives":            row.get("num_alternatives"),
            "recommended_route_index":     row.get("recommended_route_index"),
            "risk_class":                  row.get("risk_class"),
            "confidence":                  row.get("confidence"),
            "prob_high":                   row.get("prob_high"),
            "prob_medium":                 row.get("prob_medium"),
            "prob_low":                    row.get("prob_low"),
            "safety_score":                row.get("safety_score"),
            "num_hotspots":                row.get("num_hotspots"),
            "num_high_hotspots":           row.get("num_high_hotspots"),
            "weather_condition":           row.get("weather_condition"),
            "temperature_f":               row.get("temperature_f"),
            "is_precipitation":            row.get("is_precipitation"),
            "is_low_visibility":           row.get("is_low_visibility"),
            "midpoint_lat":                row.get("midpoint_lat"),
            "midpoint_lng":                row.get("midpoint_lng"),
            "hour_of_day":                 row.get("hour_of_day"),
            "model_version":               row.get("model_version"),
            "response_time_ms":            row.get("response_time_ms"),
            # ── Phase B3+ fields ─────────────────────────────────────────────
            "vertex_model_id":             row.get("vertex_model_id"),
            "vertex_endpoint_id":          row.get("vertex_endpoint_id"),
            "gemini_explanation_generated": False,
        }

        errors = _client().insert_rows_json(_TABLE_PREDS, [bq_row])
        if errors:
            print(f"[bq] insert_rows_json errors: {errors}")

    except Exception as exc:
        print(f"[bq] log_prediction_bq failed (Cloud SQL unaffected): {exc}")
