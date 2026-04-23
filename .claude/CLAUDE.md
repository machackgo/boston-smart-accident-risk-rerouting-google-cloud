# Boston Smart Accident Risk Rerouting — Google Cloud

## What this project is

A production ML system that predicts driving accident risk for routes in Boston, MA.
Users submit an origin and destination; the API returns a risk class (Low / Medium / High),
confidence, and a Gemini-generated plain-English explanation. The recommended route is
chosen by lowest safety score across default + alternative routes.

The stack is: FastAPI on Cloud Run → Vertex AI (LightGBM) with local fallback →
Gemini 2.0 Flash explanation layer → dual-write to Cloud SQL + BigQuery.

---

## Google Cloud project

| Item | Value |
|------|-------|
| Project ID | `boston-rerouting` |
| Project number | `294613088058` |
| Region | `us-central1` |
| Service account | `294613088058-compute@developer.gserviceaccount.com` |
| Cloud Run service | `boston-accident-risk-api` |
| Cloud SQL instance | `boston-rerouting:us-central1:boston-risk-postgres` |
| Database | `boston_risk_db` |
| GCS data bucket | `boston-rerouting-data` |
| Artifact Registry repo | `boston-risk` |
| BigQuery dataset | `boston_rerouting` (location `us-central1`) |
| Vertex AI endpoint | `projects/294613088058/locations/us-central1/endpoints/1226299062353920000` |

---

## Repository layout

```
api.py                          FastAPI app — all HTTP endpoints
requirements.txt
Dockerfile
cloud-run-service.yaml          Declarative Cloud Run service definition (source of truth for deploy)
cloudbuild.yaml                 CI/CD: Docker build → Artifact Registry → Cloud Run replace

src/
  secrets.py                    Secret Manager client (get_secret(name))
  database.py                   Cloud SQL pool + log_prediction()
  bigquery_logger.py            BQ async sink (log_prediction_bq())
  live/
    routes.py                   Google Maps Routes API v2
    weather.py                  OpenWeather v2.5
    geocoding.py                Google Maps Geocoding (forward + reverse)
  predict/
    feature_builder.py          Feature engineering + BallTree spatial queries
    predictor.py                Prediction orchestrator — Vertex → local fallback → Gemini
    vertex_client.py            Direct REST client to Vertex AI endpoint
  explain/
    gemini_explainer.py         Gemini 2.0 Flash explanation layer (non-blocking)
  model/
    train_v4.py / train_v3.py / train_v2.py
    preprocess_v4.py / preprocess_v2.py
    spatial_features.py

serving/                        Separate Vertex AI custom serving container
  main.py                       FastAPI inference-only server (no Maps / BQ / Gemini)
  Dockerfile
  requirements.txt

scripts/                        One-shot and pipeline automation
  pipeline_weekly.py            Cloud Scheduler trigger for weekly data pipeline
  pipeline_retrain.py           Challenger training + champion comparison
  vertex_register.py            Register model in Vertex AI Model Registry
  vertex_deploy.py              Deploy champion to Vertex AI Endpoint
  bq_setup.py                   One-time BQ dataset + table creation
  bq_pipeline.py                Weekly BQ aggregation
  export_to_gcs.py              Parquet export (called by /admin/export)
  notify.py                     Email / Slack alerts

models/                         Local model artifacts (also mirrored to GCS)
  best_model_v4.pkl             Active model (36 features, spatial)
  thresholds_v4.json            {"High": 0.15, "Low": 0.25, "Medium": 0.4}
  feature_list_v4.txt
  weather_keep_cols_v4.json

data/
  crashes_cache.parquet         47,689 Boston crashes 2015-2024 (BallTree source)
```

---

## Model versions

Three models exist in a cascade. The highest version present is loaded at startup.

| Version | Features | Notes |
|---------|----------|-------|
| v4 | 36 (spatial + weather + time) | Active. Per-class thresholds. |
| v3 | 37 (weather + time, SMOTE) | Fallback if v4 missing |
| v2 | 37 (weather + time, baseline) | Final fallback |

Classes are always `["High", "Low", "Medium"]` (alphabetical — that is the probas order).
`_CLS_IDX = {"High": 0, "Low": 1, "Medium": 2}`.

v4 thresholds: `High=0.15, Low=0.25, Medium=0.4` (from `models/thresholds_v4.json`).
Asymmetric by design — tuned to surface High-risk cases at the cost of some precision.

---

## Inference path (Cloud Run → Vertex → local)

```
predictor.py calls vertex_client.predict_single() or predict_batch()
  └─ vertex_client makes a direct REST POST (no SDK overhead, ~2-5 s when warm)
  └─ if vertex_client returns None → local _MODEL.predict_proba() fallback
```

`vertex_client.py` never raises. Catches all exceptions, returns `None`.
Response `context` always contains `inference_source`, `vertex_model_id`, `vertex_latency_ms`.

---

## Gemini explanation layer

`get_gemini_explanation()` in `src/explain/gemini_explainer.py` runs after inference.
Always non-blocking — any failure returns `None`; prediction response is unaffected.

| Env var | Required value |
|---------|---------------|
| `ENABLE_GEMINI_EXPLANATIONS` | `"true"` (exact lowercase string) |
| `GEMINI_API_KEY` | Secret Manager: `gemini-api-key` (latest) |

Model: `gemini-2.0-flash`. SDK: `google-genai` (`from google import genai`).
The deprecated `google.generativeai` package must not be used.

The Gemini API key must have: Application restrictions = None, API = Generative Language API only.
HTTP referrer restrictions break backend Python requests (no Referer header sent).

---

## Secrets (Secret Manager, project `boston-rerouting`)

| Secret name | Used by |
|-------------|---------|
| `google-maps-api-key` | Frontend HTML injection (api.py) |
| `google-server-api-key` | Routes API + Geocoding API |
| `openweather-api-key` | OpenWeather v2.5 |
| `cloudsql-postgres-password` | Cloud SQL connector |
| `export-admin-token` | /admin/export endpoint auth |
| `gemini-api-key` | Gemini explanation layer |

All fetched at request time via `src/secrets.py:get_secret(name)`. Results cached in-memory.

---

## Cloud SQL schema (operational)

**`route_predictions`** — written by every prediction call.
Key columns: `endpoint`, `origin`, `destination`, `risk_class`, `confidence`,
`prob_low/medium/high`, `model_version`, `inference_source`, `vertex_latency_ms`, `response_time_ms`.

**`boston_crashes`** — 47,689 rows, read-only by `/crashes/*` endpoints.

---

## BigQuery tables (analytics, dataset `boston_rerouting`)

| Table | Written by | Purpose |
|-------|-----------|---------|
| `route_predictions` | `bigquery_logger.py` (async background task) | Analytics replica |
| `pipeline_runs` | `bq_pipeline.py` | Weekly pipeline audit |
| `model_versions` | `vertex_register.py` | Model registry audit log |
| `challenger_runs` | `pipeline_retrain.py` | Champion-challenger comparison |

BQ writes never block the API response and never affect Cloud SQL.

---

## Deploy commands

```bash
# Full rebuild and deploy
gcloud builds submit --config cloudbuild.yaml --project=boston-rerouting

# Update config / secrets only (no image rebuild)
gcloud run services replace cloud-run-service.yaml \
  --region=us-central1 --project=boston-rerouting

# Tail live logs
gcloud run services logs read boston-accident-risk-api \
  --region=us-central1 --project=boston-rerouting --limit=50
```

---

## Hard constraints

- Do not commit or push until explicitly asked.
- `serving/` is a separate container. API changes do not affect it. Vertex serving container
  must be rebuilt and re-pushed separately via its own Dockerfile.
- `vertex_client.py` must never raise. Always return `None` on any failure.
- `get_gemini_explanation()` must always return `None` on failure, never raise.
- BQ and Cloud SQL writes are fully independent — failure in one must not affect the other.
- Class order in `_CLASSES` is `["High", "Low", "Medium"]` (alphabetical). `probas[0]=P(High)`.
  Mixing up this order corrupts all threshold logic silently.
- The minimum route guard is enforced before prediction: `< 0.3 miles or < 2.0 minutes`
  returns 400. Do not remove this guard.
