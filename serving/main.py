"""
main.py — Vertex AI custom serving container for the Boston accident-risk champion model.

Scope: model inference only.
  - No Cloud SQL writes
  - No BigQuery writes
  - No Gemini calls
  - No external API calls (Maps, Weather, etc.)

The serving container receives an already-engineered feature vector from the caller
(Cloud Run in Phase B3). It runs the LightGBM model, applies per-class decision
thresholds (identical logic to predictor.py), and returns structured predictions.

Environment variables (set in vertex_deploy.py when deploying):
  GCS_BUCKET      : GCS bucket name                (default: boston-rerouting-data)
  MODEL_BLOB      : GCS path to model pkl           (default: production/models/best_model_v4.pkl)
  THRESHOLD_BLOB  : GCS path to thresholds JSON     (default: production/models/thresholds_v4.json)
  AIP_HTTP_PORT   : Port to listen on — set by Vertex AI automatically (default: 8080)

Vertex AI prediction protocol:

  Health check   GET /
  → {"status": "ok", "model_version": "v4", "features": 36, ...}

  Prediction     POST /predict
  Request body:
    {
      "instances": [
        {
          "lat": 42.36, "lon": -71.06, "speed_limit": 30,
          "hour_of_day": 9, "day_of_week": 1, "month": 4,
          "is_weekend": 0, "is_rush_hour": 1,
          "nearby_crash_count_1km": 5.0, ...   (all 36 features)
        }
      ]
    }
  Response body:
    {
      "predictions": [
        {
          "risk_class": "Medium",
          "confidence": 0.621,
          "probabilities": {"High": 0.201, "Low": 0.178, "Medium": 0.621}
        }
      ]
    }
"""

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── GCS config (overridable via env vars set at deploy time) ──────────────────
_GCS_BUCKET     = os.environ.get("GCS_BUCKET",     "boston-rerouting-data")
_MODEL_BLOB     = os.environ.get("MODEL_BLOB",     "production/models/best_model_v4.pkl")
_THRESHOLD_BLOB = os.environ.get("THRESHOLD_BLOB", "production/models/thresholds_v4.json")

# ── Model globals — populated by _load_artifacts() at startup ─────────────────
_MODEL: Any       = None
_CLASSES: list    = []   # e.g. ['High', 'Low', 'Medium']
_FEATURES: list   = []   # ordered list of 36 feature names
_FEATURE_SET: set = set() # pre-computed set of _FEATURES for O(1) validation
_THRESHOLDS: dict = {}   # {"High": 0.15, "Low": 0.25, "Medium": 0.4}
_CLS_IDX: dict    = {}   # {"High": 0, "Low": 1, "Medium": 2}


def _load_artifacts() -> None:
    """
    Download model bundle and thresholds from GCS.
    Called from the FastAPI lifespan so the HTTP server is already bound to a port
    before any GCS access is attempted. A 403 or network error here causes health
    checks to return 503 instead of crashing the process entirely.
    """
    global _MODEL, _CLASSES, _FEATURES, _FEATURE_SET, _THRESHOLDS, _CLS_IDX

    t_start = time.monotonic()

    log.info("Connecting to GCS bucket '%s' ...", _GCS_BUCKET)
    gcs    = storage.Client()
    bucket = gcs.bucket(_GCS_BUCKET)

    # Model pkl is a dict bundle: {"model": LGBMClassifier, "features": [...], "classes": [...]}
    log.info("Downloading model: gs://%s/%s ...", _GCS_BUCKET, _MODEL_BLOB)
    t_dl = time.monotonic()
    buf = io.BytesIO()
    bucket.blob(_MODEL_BLOB).download_to_file(buf)
    model_bytes = buf.tell()
    buf.seek(0)
    t_dl_done = time.monotonic()
    log.info("Model download complete: %.1f MB in %dms",
             model_bytes / 1_048_576, int((t_dl_done - t_dl) * 1000))

    bundle    = joblib.load(buf)
    _MODEL    = bundle["model"]
    _CLASSES  = bundle["classes"]
    _FEATURES = bundle["features"]

    # Per-class decision thresholds — same file predictor.py reads
    log.info("Downloading thresholds: gs://%s/%s ...", _GCS_BUCKET, _THRESHOLD_BLOB)
    _THRESHOLDS = json.loads(bucket.blob(_THRESHOLD_BLOB).download_as_text())

    _CLS_IDX     = {c: i for i, c in enumerate(_CLASSES)}
    _FEATURE_SET = set(_FEATURES)

    log.info(
        "Artifacts loaded in %dms total — classes: %s | features: %d | thresholds: %s",
        int((time.monotonic() - t_start) * 1000), _CLASSES, len(_FEATURES), _THRESHOLDS,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="boston-accident-risk-serving", version="v4", lifespan=lifespan)


# ── Threshold classification ───────────────────────────────────────────────────

def _classify_with_thresholds(probas: np.ndarray) -> list[str]:
    """
    One-vs-rest threshold classification.
    Exact mirror of _classify_with_thresholds() in src/predict/predictor.py.
    Assigns the class whose P ≥ threshold is highest; falls back to argmax.
    """
    preds = []
    for row in probas:
        candidates = {
            c: row[_CLS_IDX[c]]
            for c in _CLASSES
            if row[_CLS_IDX[c]] >= _THRESHOLDS.get(c, 0.5)
        }
        preds.append(
            max(candidates, key=candidates.get) if candidates
            else _CLASSES[int(np.argmax(row))]
        )
    return preds


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health() -> JSONResponse:
    """Health check — Vertex AI polls this before routing any traffic."""
    if _MODEL is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return JSONResponse(content={
        "status":        "ok",
        "model_version": "v4",
        "features":      len(_FEATURES),
        "classes":       _CLASSES,
        "thresholds":    _THRESHOLDS,
    })


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """
    Vertex AI prediction endpoint.

    Accepts a batch of feature-vector instances, runs the LightGBM model,
    applies per-class thresholds, and returns structured risk predictions.
    """
    t_req = time.monotonic()

    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    instances: list[dict] = body.get("instances", [])
    if not instances:
        raise HTTPException(status_code=400, detail="'instances' is empty or missing.")

    # Validate completeness — use pre-computed set, not set(_FEATURES) per instance.
    for i, inst in enumerate(instances):
        missing = _FEATURE_SET - set(inst.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Instance {i} missing features: {sorted(missing)}",
            )

    # Build DataFrame in the exact column order the model was trained on.
    df = pd.DataFrame(instances)[_FEATURES]

    # LightGBM inference — returns probabilities in _CLASSES order.
    t_infer = time.monotonic()
    probas       = _MODEL.predict_proba(df)
    risk_classes = _classify_with_thresholds(probas)
    infer_ms = int((time.monotonic() - t_infer) * 1000)

    predictions = []
    for risk_class, proba_row in zip(risk_classes, probas):
        proba_dict = {
            cls: round(float(proba_row[_CLS_IDX[cls]]), 4)
            for cls in _CLASSES
        }
        predictions.append({
            "risk_class":    risk_class,
            "confidence":    round(float(max(proba_row)), 4),
            "probabilities": proba_dict,
        })

    total_ms = int((time.monotonic() - t_req) * 1000)
    log.info(
        "[serving] predict rows=%d infer_ms=%d total_ms=%d",
        len(instances), infer_ms, total_ms,
    )

    return JSONResponse(content={"predictions": predictions})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # AIP_HTTP_PORT is set by Vertex AI; PORT is used locally for testing.
    port = int(os.environ.get("AIP_HTTP_PORT", os.environ.get("PORT", "8080")))
    log.info("Starting server on port %d ...", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
