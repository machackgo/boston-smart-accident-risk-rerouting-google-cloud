"""
bq_setup.py — Create the boston_rerouting BigQuery dataset and all analytics tables.

Run once (idempotent — safe to re-run, existing tables are not touched):
    python scripts/bq_setup.py

To drop and recreate all tables (dev/reset only — DESTROYS ALL DATA):
    python scripts/bq_setup.py --drop-recreate

What this creates:
    Dataset  : boston-rerouting.boston_rerouting  (us-central1)

    Tables:
      route_predictions  — one row per /predict or /predict/segmented API call
      pipeline_runs      — one row per weekly or Phase 4 pipeline execution
      model_versions     — one row per trained model (champion + all challengers)
      challenger_runs    — one row per Phase 4 retrain + comparison outcome
      user_feedback      — user thumbs-up/down after seeing a prediction (future)
      gemini_explanations— metadata about Gemini explanation calls (future)

Prerequisites:
    pip install google-cloud-bigquery
    gcloud auth application-default login   # or use service account ADC
"""

import sys
from google.cloud import bigquery

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT  = "boston-rerouting"
DATASET  = "boston_rerouting"
LOCATION = "us-central1"


# ── Table schemas ─────────────────────────────────────────────────────────────

# route_predictions: the primary analytics fact table.
# One row per API prediction call (/predict or /predict/segmented).
# Mirrors the Cloud SQL route_predictions table but adds:
#   - prediction_id (UUID, for future cross-system joins)
#   - vertex_model_id / vertex_endpoint_id (null until Phase B)
#   - gemini_explanation_generated (false until Phase E)
SCHEMA_ROUTE_PREDICTIONS = [
    bigquery.SchemaField("prediction_id",              "STRING",    mode="REQUIRED",
                         description="UUID generated at write time"),
    bigquery.SchemaField("timestamp",                  "TIMESTAMP", mode="REQUIRED",
                         description="UTC time of the prediction request"),
    bigquery.SchemaField("endpoint",                   "STRING",    description="/predict or /predict/segmented"),
    bigquery.SchemaField("origin",                     "STRING"),
    bigquery.SchemaField("destination",                "STRING"),
    bigquery.SchemaField("departure_time",             "STRING",    description="ISO 8601 string from request, or null"),
    bigquery.SchemaField("duration_minutes",           "FLOAT64"),
    bigquery.SchemaField("distance_miles",             "FLOAT64"),
    bigquery.SchemaField("num_alternatives",           "INT64"),
    bigquery.SchemaField("recommended_route_index",    "INT64",     description="null for /predict"),
    bigquery.SchemaField("risk_class",                 "STRING",    description="High / Medium / Low"),
    bigquery.SchemaField("confidence",                 "FLOAT64",   description="Probability of predicted class (0–1)"),
    bigquery.SchemaField("prob_high",                  "FLOAT64"),
    bigquery.SchemaField("prob_medium",                "FLOAT64"),
    bigquery.SchemaField("prob_low",                   "FLOAT64"),
    bigquery.SchemaField("safety_score",               "FLOAT64",   description="Composite score; null for /predict"),
    bigquery.SchemaField("num_hotspots",               "INT64",     description="Medium+High segments; null for /predict"),
    bigquery.SchemaField("num_high_hotspots",          "INT64",     description="High segments; null for /predict"),
    bigquery.SchemaField("weather_condition",          "STRING",    description="OpenWeather main condition string"),
    bigquery.SchemaField("temperature_f",              "FLOAT64"),
    bigquery.SchemaField("is_precipitation",           "BOOL"),
    bigquery.SchemaField("is_low_visibility",          "BOOL"),
    bigquery.SchemaField("midpoint_lat",               "FLOAT64",   description="null for /predict/segmented"),
    bigquery.SchemaField("midpoint_lng",               "FLOAT64",   description="null for /predict/segmented"),
    bigquery.SchemaField("hour_of_day",                "INT64",     description="Local Boston hour 0–23"),
    bigquery.SchemaField("model_version",              "STRING",    description="e.g. v4"),
    bigquery.SchemaField("vertex_model_id",            "STRING",    description="null until Phase B Vertex AI migration"),
    bigquery.SchemaField("vertex_endpoint_id",         "STRING",    description="null until Phase B Vertex AI migration"),
    bigquery.SchemaField("response_time_ms",           "INT64"),
    bigquery.SchemaField("gemini_explanation_generated","BOOL",     description="false until Phase E Gemini integration"),
]

# pipeline_runs: execution log for the weekly automated pipeline and manual Phase 4 runs.
# Replaces / complements the GCS pipeline/logs/*.json files with a queryable table.
SCHEMA_PIPELINE_RUNS = [
    bigquery.SchemaField("run_id",                "STRING",    mode="REQUIRED",
                         description="run_ts string e.g. 2026-04-21_06-00-00"),
    bigquery.SchemaField("phase",                 "STRING",    description="weekly_phases_1_3 | phase4_retrain"),
    bigquery.SchemaField("run_timestamp",         "TIMESTAMP"),
    bigquery.SchemaField("outcome",               "STRING",    description="success | no_new_data | phase1_failed | phase2_failed | phase3_failed"),
    bigquery.SchemaField("years_checked",         "STRING",    description="JSON list of years queried by Phase 1"),
    bigquery.SchemaField("years_with_new_data",   "STRING",    description="JSON list of years with detected changes"),
    bigquery.SchemaField("years_downloaded",      "STRING",    description="JSON list of years actually downloaded"),
    bigquery.SchemaField("merged_row_count",      "INT64",     description="Total rows in merged challenger dataset after Phase 3"),
    bigquery.SchemaField("merged_gcs_path",       "STRING"),
    bigquery.SchemaField("candidate_years",       "STRING",    description="JSON list of candidate years in merged dataset"),
    bigquery.SchemaField("challenger_f1",         "FLOAT64",   description="null for weekly runs; challenger macro F1 for Phase 4"),
    bigquery.SchemaField("all_gates_passed",      "BOOL",      description="null for weekly runs"),
    bigquery.SchemaField("recommended_action",    "STRING",    description="null for weekly runs; promote | do_not_promote for Phase 4"),
    bigquery.SchemaField("is_dry_run",            "BOOL"),
    bigquery.SchemaField("notification_sent",     "BOOL"),
    bigquery.SchemaField("duration_seconds",      "INT64"),
    bigquery.SchemaField("triggered_by",          "STRING",    description="cloud_scheduler | manual"),
]

# model_versions: registry of every trained model (champion + all challengers).
# Champion row is seeded by bq_backfill.py.
# Future challenger rows are written by pipeline_retrain.py after Phase 4.
SCHEMA_MODEL_VERSIONS = [
    bigquery.SchemaField("model_id",             "STRING",    mode="REQUIRED",
                         description="Short identifier e.g. v4, challenger_20260421"),
    bigquery.SchemaField("display_name",         "STRING"),
    bigquery.SchemaField("training_date",        "TIMESTAMP"),
    bigquery.SchemaField("training_dataset_gcs", "STRING",    description="GCS path to training parquet"),
    bigquery.SchemaField("training_row_count",   "INT64"),
    bigquery.SchemaField("year_range_min",       "INT64"),
    bigquery.SchemaField("year_range_max",       "INT64"),
    bigquery.SchemaField("macro_f1",             "FLOAT64"),
    bigquery.SchemaField("accuracy",             "FLOAT64"),
    bigquery.SchemaField("binary_roc_auc",       "FLOAT64"),
    bigquery.SchemaField("high_f1",              "FLOAT64"),
    bigquery.SchemaField("medium_f1",            "FLOAT64"),
    bigquery.SchemaField("low_f1",               "FLOAT64"),
    bigquery.SchemaField("test_set_size",        "INT64"),
    bigquery.SchemaField("is_champion",          "BOOL"),
    bigquery.SchemaField("is_active",            "BOOL"),
    bigquery.SchemaField("vertex_model_id",      "STRING",    description="null until Phase B"),
    bigquery.SchemaField("vertex_endpoint_id",   "STRING",    description="null until Phase B"),
    bigquery.SchemaField("traffic_pct",          "INT64",     description="0–100; champion=100 until Phase D traffic splits"),
    bigquery.SchemaField("registered_at",        "TIMESTAMP"),
    bigquery.SchemaField("promoted_at",          "TIMESTAMP"),
    bigquery.SchemaField("deprecated_at",        "TIMESTAMP"),
    bigquery.SchemaField("champion_metrics_gcs", "STRING"),
    bigquery.SchemaField("model_artifact_gcs",   "STRING"),
]

# challenger_runs: permanent record of every Phase 4 retrain + comparison outcome.
# One row per pipeline_retrain.py execution. Never updated — append-only.
SCHEMA_CHALLENGER_RUNS = [
    bigquery.SchemaField("run_id",                  "STRING",    mode="REQUIRED",
                         description="run_ts from pipeline_retrain.py"),
    bigquery.SchemaField("run_timestamp",            "TIMESTAMP"),
    bigquery.SchemaField("merged_dataset_gcs",       "STRING"),
    bigquery.SchemaField("merged_row_count",         "INT64",     description="Total rows in merged dataset"),
    bigquery.SchemaField("effective_row_count",      "INT64",     description="Rows after dropping null severity/lat/lon"),
    bigquery.SchemaField("candidate_years",          "STRING",    description="JSON list of candidate years included"),
    bigquery.SchemaField("last_retrain_row_count",   "INT64",     description="Merged rows from the previous Phase 4 run, for delta tracking"),
    bigquery.SchemaField("challenger_model_id",      "STRING"),
    bigquery.SchemaField("challenger_macro_f1",      "FLOAT64"),
    bigquery.SchemaField("challenger_roc_auc",       "FLOAT64"),
    bigquery.SchemaField("challenger_high_f1",       "FLOAT64"),
    bigquery.SchemaField("challenger_medium_f1",     "FLOAT64"),
    bigquery.SchemaField("challenger_low_f1",        "FLOAT64"),
    bigquery.SchemaField("champion_model_id",        "STRING"),
    bigquery.SchemaField("champion_macro_f1",        "FLOAT64"),
    bigquery.SchemaField("champion_roc_auc",         "FLOAT64"),
    bigquery.SchemaField("all_gates_passed",         "BOOL"),
    bigquery.SchemaField("failed_gates",             "STRING",    description="JSON list of failed gate names"),
    bigquery.SchemaField("recommended_action",       "STRING",    description="promote | do_not_promote"),
    bigquery.SchemaField("recommendation_reason",    "STRING"),
    bigquery.SchemaField("report_gcs",               "STRING"),
    bigquery.SchemaField("model_gcs",                "STRING"),
    bigquery.SchemaField("comparison_gcs",           "STRING"),
]

# user_feedback: empty for now — schema only. Populated when feedback UI is built.
SCHEMA_USER_FEEDBACK = [
    bigquery.SchemaField("feedback_id",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("prediction_id",  "STRING",    description="FK to route_predictions.prediction_id"),
    bigquery.SchemaField("timestamp",      "TIMESTAMP"),
    bigquery.SchemaField("feedback_type",  "STRING",    description="route_accepted | route_rejected | rated_accurate | rated_inaccurate | reported_incident"),
    bigquery.SchemaField("feedback_value", "STRING",    description="1-5 rating or free text"),
    bigquery.SchemaField("session_id",     "STRING"),
    bigquery.SchemaField("user_agent",     "STRING"),
]

# gemini_explanations: empty for now — schema only. Populated when Gemini is integrated.
SCHEMA_GEMINI_EXPLANATIONS = [
    bigquery.SchemaField("explanation_id",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("prediction_id",        "STRING",    description="FK to route_predictions.prediction_id"),
    bigquery.SchemaField("timestamp",            "TIMESTAMP"),
    bigquery.SchemaField("explanation_type",     "STRING",    description="route_risk | route_comparison | hotspot_briefing"),
    bigquery.SchemaField("gemini_model",         "STRING",    description="e.g. gemini-2.0-flash-001"),
    bigquery.SchemaField("prompt_tokens",        "INT64"),
    bigquery.SchemaField("completion_tokens",    "INT64"),
    bigquery.SchemaField("latency_ms",           "INT64"),
    bigquery.SchemaField("success",              "BOOL"),
    bigquery.SchemaField("failure_reason",       "STRING",    description="null if success=true"),
    bigquery.SchemaField("risk_class_explained", "STRING"),
    bigquery.SchemaField("explanation_preview",  "STRING",    description="First 200 chars of explanation for auditing"),
]


# ── Table definitions: (name, schema, partition_field, cluster_fields) ────────

TABLES = [
    (
        "route_predictions",
        SCHEMA_ROUTE_PREDICTIONS,
        "timestamp",                        # partition field
        ["risk_class", "model_version"],    # cluster fields
    ),
    (
        "pipeline_runs",
        SCHEMA_PIPELINE_RUNS,
        "run_timestamp",
        ["phase", "outcome"],
    ),
    (
        "model_versions",
        SCHEMA_MODEL_VERSIONS,
        None,   # small reference table — no partitioning needed
        None,
    ),
    (
        "challenger_runs",
        SCHEMA_CHALLENGER_RUNS,
        "run_timestamp",
        ["recommended_action"],
    ),
    (
        "user_feedback",
        SCHEMA_USER_FEEDBACK,
        "timestamp",
        None,
    ),
    (
        "gemini_explanations",
        SCHEMA_GEMINI_EXPLANATIONS,
        "timestamp",
        ["explanation_type", "success"],
    ),
]


# ── Setup functions ───────────────────────────────────────────────────────────

def create_dataset(client: bigquery.Client, drop: bool = False) -> None:
    dataset_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    dataset_ref.location = LOCATION
    if drop:
        client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
        print(f"  Dropped  : {PROJECT}.{DATASET}")
    client.create_dataset(dataset_ref, exists_ok=True)
    print(f"  Dataset  : {PROJECT}.{DATASET}  (location={LOCATION})")


def create_table(
    client: bigquery.Client,
    name: str,
    schema: list,
    partition_field: str | None,
    cluster_fields: list | None,
    drop: bool = False,
) -> None:
    table_ref = bigquery.TableReference(
        bigquery.DatasetReference(PROJECT, DATASET), name
    )

    if drop:
        client.delete_table(table_ref, not_found_ok=True)
        print(f"  Dropped  : {DATASET}.{name}")

    table = bigquery.Table(table_ref, schema=schema)

    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )

    if cluster_fields:
        table.clustering_fields = cluster_fields

    client.create_table(table, exists_ok=True)

    col_count   = len(schema)
    part_note   = f"  PARTITION BY {partition_field}" if partition_field else "  no partition"
    clust_note  = f"  CLUSTER BY {cluster_fields}" if cluster_fields else ""
    print(f"  Table    : {DATASET}.{name:<25} ({col_count} cols){part_note}{clust_note}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(drop_recreate: bool = False) -> None:
    print(f"\n{'=' * 62}")
    print(f"  BigQuery setup — {PROJECT}.{DATASET}")
    print(f"{'=' * 62}")
    if drop_recreate:
        print("  WARNING: --drop-recreate will DELETE all table data.")
        confirm = input("  Type YES to continue: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            return

    client = bigquery.Client(project=PROJECT)

    print(f"\n  Creating dataset ...")
    create_dataset(client, drop=drop_recreate)

    print(f"\n  Creating tables ...")
    for name, schema, partition_field, cluster_fields in TABLES:
        create_table(client, name, schema, partition_field, cluster_fields,
                     drop=drop_recreate)

    print(f"\n  All done.")
    print(f"\n  Verify in BigQuery console:")
    print(f"    https://console.cloud.google.com/bigquery?project={PROJECT}")
    print(f"\n  Or via CLI:")
    print(f"    bq ls {PROJECT}:{DATASET}")
    for name, _, _, _ in TABLES:
        print(f"    bq show {PROJECT}:{DATASET}.{name}")
    print()


if __name__ == "__main__":
    main(drop_recreate="--drop-recreate" in sys.argv)
