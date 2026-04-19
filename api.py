"""
Boston Smart Accident Risk Rerouting — FastAPI
Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

import sys
import io
import time
import sqlalchemy
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path so src.predict.predictor is importable
# (needed when the working directory is not the repo root, e.g. Cloud Run)
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from src.predict.predictor import predict_route_risk, predict_route_risk_segmented
from src.live.routes import get_route
from src.secrets import get_secret
from src.database import log_prediction, get_engine

# Minimum route thresholds — below these the model's training distribution is not met
_MIN_DISTANCE_MILES = 0.3
_MIN_DURATION_MIN   = 2.0


def _check_route_size(origin: str, destination: str):
    """
    Call get_route() and return a 400 JSONResponse if the route is too short.
    Returns None when the route is acceptable, or the parsed route dict.
    Raises HTTPException(500) on API failures.
    """
    try:
        route_data = get_route(origin, destination)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route lookup failed: {e}")
    default = route_data["default_route"]
    dist    = default["distance_miles"]
    dur     = default["duration_minutes"]
    if dist < _MIN_DISTANCE_MILES or dur < _MIN_DURATION_MIN:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Route too short for meaningful prediction",
                "message": (
                    "Routes under 0.3 miles or 2 minutes are below the model's "
                    "training distribution. Please choose more distant points."
                ),
                "distance_miles": dist,
                "duration_minutes": dur,
            },
        )
    return None

app = FastAPI(title="Boston Smart Accident Risk Rerouting API", version="2.0")

# ── CORS ──────────────────────────────────────────────────────
# allow_origins=["*"] is intentionally permissive for now; tighten before prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────
class PredictRequest(BaseModel):
    origin: str
    destination: str
    departure_time: Optional[str] = None  # ISO 8601, e.g. "2024-10-15T08:00:00Z"

class SegmentedPredictRequest(BaseModel):
    origin: str
    destination: str
    departure_time: Optional[str] = None
    num_segments: int = 12

TABLE = "boston_crashes"
_STATIC_DIR = _REPO_ROOT / "static"


def _rows(result) -> list[dict]:
    """Convert a SQLAlchemy result into a list of plain dicts for JSON serialisation."""
    return [dict(r) for r in result.mappings().all()]


# ── 0. Frontend ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    """Serve the single-page frontend app with the Google Maps API key injected."""
    api_key = get_secret("google-maps-api-key")
    html = (_STATIC_DIR / "index.html").read_text()
    return html.replace("{{GOOGLE_MAPS_API_KEY}}", api_key)

# ── 1. Get all crashes (paginated) ───────────────────────────
@app.get("/crashes")
def get_crashes(limit: int = 100, offset: int = 0):
    """Get crashes with pagination. Default 100 per page."""
    with get_engine().connect() as conn:
        data = _rows(conn.execute(
            sqlalchemy.text(
                "SELECT * FROM boston_crashes ORDER BY id LIMIT :limit OFFSET :offset"
            ),
            {"limit": limit, "offset": offset},
        ))
    return {"total_returned": len(data), "data": data}

# ── 2. Filter by year ─────────────────────────────────────────
@app.get("/crashes/year/{year}")
def get_by_year(year: int, limit: int = 100):
    """Get crashes for a specific year (2015–2024)."""
    with get_engine().connect() as conn:
        data = _rows(conn.execute(
            sqlalchemy.text(
                "SELECT * FROM boston_crashes WHERE year = :year LIMIT :limit"
            ),
            {"year": year, "limit": limit},
        ))
    return {"year": year, "total_returned": len(data), "data": data}

# ── 3. Filter by severity ─────────────────────────────────────
@app.get("/crashes/severity/{severity}")
def get_by_severity(severity: str, limit: int = 100):
    """
    Severity options: Fatal, Injury, No Injury, Unknown
    Use SEVERITY_3CLASS values.
    """
    with get_engine().connect() as conn:
        data = _rows(conn.execute(
            sqlalchemy.text(
                "SELECT * FROM boston_crashes WHERE severity_3class ILIKE :pat LIMIT :limit"
            ),
            {"pat": f"%{severity}%", "limit": limit},
        ))
    return {"severity": severity, "total_returned": len(data), "data": data}

# ── 4. Filter by city ─────────────────────────────────────────
@app.get("/crashes/city/{city}")
def get_by_city(city: str, limit: int = 100):
    """Get crashes in a specific city/town."""
    with get_engine().connect() as conn:
        data = _rows(conn.execute(
            sqlalchemy.text(
                "SELECT * FROM boston_crashes WHERE city_town_name ILIKE :pat LIMIT :limit"
            ),
            {"pat": f"%{city}%", "limit": limit},
        ))
    return {"city": city, "total_returned": len(data), "data": data}

# ── 5. Get hotspots ───────────────────────────────────────────
@app.get("/crashes/hotspots")
def get_hotspots(limit: int = 100):
    """Get crash hotspot locations (ems_hotspot_flag = 1)."""
    with get_engine().connect() as conn:
        data = _rows(conn.execute(
            sqlalchemy.text(
                "SELECT * FROM boston_crashes WHERE ems_hotspot_flag = 1 LIMIT :limit"
            ),
            {"limit": limit},
        ))
    return {"total_returned": len(data), "data": data}

# ── 6. Fatal crashes only ─────────────────────────────────────
@app.get("/crashes/fatal")
def get_fatal(limit: int = 100):
    """Get crashes with at least 1 fatality."""
    with get_engine().connect() as conn:
        data = _rows(conn.execute(
            sqlalchemy.text(
                "SELECT * FROM boston_crashes WHERE numb_fatal_injr > 0 LIMIT :limit"
            ),
            {"limit": limit},
        ))
    return {"total_returned": len(data), "data": data}

# ── 7. Summary stats by year ──────────────────────────────────
@app.get("/stats/by-year")
def stats_by_year():
    """Get crash counts grouped by year."""
    with get_engine().connect() as conn:
        rows = conn.execute(sqlalchemy.text("""
            SELECT
                year,
                COUNT(*)                          AS crashes,
                COALESCE(SUM(numb_fatal_injr), 0) AS fatalities,
                COALESCE(SUM(numb_nonfatal_injr), 0) AS injuries
            FROM boston_crashes
            GROUP BY year
            ORDER BY year
        """)).fetchall()
    data = {
        r[0]: {"crashes": r[1], "fatalities": r[2], "injuries": r[3]}
        for r in rows
    }
    return {"data": data}

# ── 8. Advanced filter ────────────────────────────────────────
@app.get("/crashes/filter")
def filter_crashes(
    year:     Optional[int] = Query(None, description="e.g. 2020"),
    city:     Optional[str] = Query(None, description="e.g. Boston"),
    severity: Optional[str] = Query(None, description="e.g. Fatal"),
    weather:  Optional[str] = Query(None, description="e.g. Rain"),
    limit:    int = 100
):
    """Filter by multiple fields at once."""
    conditions = ["1=1"]
    params: dict = {"limit": limit}
    if year:
        conditions.append("year = :year")
        params["year"] = year
    if city:
        conditions.append("city_town_name ILIKE :city")
        params["city"] = f"%{city}%"
    if severity:
        conditions.append("severity_3class ILIKE :severity")
        params["severity"] = f"%{severity}%"
    if weather:
        conditions.append("weath_cond_descr ILIKE :weather")
        params["weather"] = f"%{weather}%"
    sql = f"SELECT * FROM boston_crashes WHERE {' AND '.join(conditions)} LIMIT :limit"
    with get_engine().connect() as conn:
        data = _rows(conn.execute(sqlalchemy.text(sql), params))
    return {
        "filters_applied": {"year": year, "city": city, "severity": severity, "weather": weather},
        "total_returned": len(data),
        "data": data,
    }


# ── 9. Route risk prediction ──────────────────────────────────
@app.post("/predict")
def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict accident risk severity for a driving route.

    Orchestrates live geocoding, routing (Google Maps Routes API), and weather
    (OpenWeather API) to assemble the v2 model feature row, then runs the
    LightGBM v2 classifier to return a risk assessment.

    **Request body:**
    - `origin` — starting address or place name (e.g. "Fenway Park, Boston, MA")
    - `destination` — destination address or place name
    - `departure_time` — optional ISO 8601 datetime (e.g. "2024-10-15T08:00:00Z").
      Defaults to current time if omitted.

    **Returns:**
    - `risk_class` — "Low", "Medium", or "High"
    - `confidence` — probability of the predicted class (0–1)
    - `class_probabilities` — probabilities for all three classes
    - `route` — duration, distance, number of alternatives
    - `weather` — current conditions at the route midpoint
    - `context` — midpoint coordinates, hour of day, weather mapping used
    """
    # Tiny-route guard — reject routes below training distribution
    guard = _check_route_size(request.origin, request.destination)
    if guard is not None:
        return guard

    t0 = time.monotonic()
    try:
        result = predict_route_risk(
            origin=request.origin,
            destination=request.destination,
            departure_time=request.departure_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    background_tasks.add_task(log_prediction, {
        "endpoint":               "/predict",
        "origin":                 request.origin,
        "destination":            request.destination,
        "departure_time":         request.departure_time,
        "duration_minutes":       result["route"]["duration_minutes"],
        "distance_miles":         result["route"]["distance_miles"],
        "num_alternatives":       result["route"]["num_alternatives"],
        "risk_class":             result["risk_class"],
        "confidence":             result["confidence"],
        "prob_low":               result["class_probabilities"]["Low"],
        "prob_medium":            result["class_probabilities"]["Medium"],
        "prob_high":              result["class_probabilities"]["High"],
        "weather_condition":      result["weather"]["condition"],
        "temperature_f":          result["weather"]["temperature_f"],
        "is_precipitation":       result["weather"]["is_precipitation"],
        "is_low_visibility":      result["weather"]["is_low_visibility"],
        "midpoint_lat":           result["context"]["midpoint_lat"],
        "midpoint_lng":           result["context"]["midpoint_lng"],
        "hour_of_day":            result["context"]["hour_of_day"],
        "recommended_route_index": None,
        "safety_score":           None,
        "num_hotspots":           None,
        "num_high_hotspots":      None,
        "model_version":          result["context"].get("model_version"),
        "response_time_ms":       elapsed_ms,
    })
    return result


# ── 10. Per-segment route risk prediction ─────────────────────
@app.post("/predict/segmented")
def predict_segmented(request: SegmentedPredictRequest, background_tasks: BackgroundTasks):
    """
    Predict accident risk at multiple points along a route.

    Samples `num_segments` points evenly from the decoded polyline and runs a
    single batch LightGBM inference.  Returns per-segment risk classes, a
    hotspot list (Medium/High segments), and an overall worst-case risk class.

    **Request body:**
    - `origin` — starting address or place name
    - `destination` — destination address or place name
    - `departure_time` — optional ISO 8601 datetime (defaults to now)
    - `num_segments` — number of sample points (default 12)

    **Returns:**
    - `risk_class` — worst-case overall risk ("Low", "Medium", or "High")
    - `overall_confidence` / `overall_probabilities` — averaged across segments
    - `segments` — list of per-point predictions with lat/lng and polyline_index
    - `hotspots` — Medium/High segments sorted by distance from start
    - `route` — includes `decoded_points` list for frontend segment drawing
    - `weather` / `context` — same shape as /predict
    """
    # Tiny-route guard — reject routes below training distribution
    guard = _check_route_size(request.origin, request.destination)
    if guard is not None:
        return guard

    t0 = time.monotonic()
    try:
        result = predict_route_risk_segmented(
            origin=request.origin,
            destination=request.destination,
            departure_time=request.departure_time,
            num_segments=request.num_segments,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    rec_idx   = result["recommended_route_index"]
    rec_route = result["routes"][rec_idx]
    def_route = result["routes"][result["default_route_index"]]
    background_tasks.add_task(log_prediction, {
        "endpoint":               "/predict/segmented",
        "origin":                 request.origin,
        "destination":            request.destination,
        "departure_time":         request.departure_time,
        "duration_minutes":       def_route["duration_minutes"],
        "distance_miles":         def_route["distance_miles"],
        "num_alternatives":       len(result["routes"]) - 1,
        "risk_class":             rec_route["overall_risk_class"],
        "confidence":             rec_route["overall_confidence"],
        "prob_low":               rec_route["overall_probabilities"]["Low"],
        "prob_medium":            rec_route["overall_probabilities"]["Medium"],
        "prob_high":              rec_route["overall_probabilities"]["High"],
        "weather_condition":      result["weather"]["condition"],
        "temperature_f":          result["weather"]["temperature_f"],
        "is_precipitation":       result["weather"]["is_precipitation"],
        "is_low_visibility":      result["weather"]["is_low_visibility"],
        "midpoint_lat":           None,
        "midpoint_lng":           None,
        "hour_of_day":            result["context"]["local_hour"],
        "recommended_route_index": rec_idx,
        "safety_score":           rec_route["safety_score"],
        "num_hotspots":           rec_route["num_hotspots"],
        "num_high_hotspots":      rec_route["num_high_hotspots"],
        "model_version":          result["context"].get("model_version"),
        "response_time_ms":       elapsed_ms,
    })
    return result


# ── 11. Predict example (GET convenience endpoint) ────────────
@app.get("/predict/example")
def predict_example():
    """
    Returns a hardcoded example prediction result for Fenway Park → Boston Logan Airport.

    This is a convenience endpoint for teammates and frontend developers to
    inspect the /predict response schema without needing to send a POST request.
    The values below are real — produced by a live call on 2026-04-11 at 00:xx UTC.
    """
    return {
        "risk_class": "Low",
        "confidence": 0.7909,
        "class_probabilities": {
            "High": 0.0199,
            "Low": 0.7909,
            "Medium": 0.1892,
        },
        "route": {
            "duration_minutes": 11.95,
            "distance_miles": 5.123,
            "num_alternatives": 1,
            "best_alternative_savings_minutes": 0.0,
        },
        "weather": {
            "condition": "Clear",
            "temperature_f": 52.38,
            "is_precipitation": False,
            "is_low_visibility": False,
        },
        "context": {
            "midpoint_lat": 42.36616,
            "midpoint_lng": -71.06551,
            "hour_of_day": 0,
            "weather_mapping_used": "'Clear' → weath_cond_descr_Clear",
        },
        "_note": "Hardcoded example. Call POST /predict for a live prediction.",
    }


# ── 12. Weekly GCS export (called by Cloud Scheduler) ────────
_EXPORT_TABLES  = ["route_predictions"]
_GCS_BUCKET     = "boston-rerouting-data"

@app.post("/admin/export", include_in_schema=False)
def admin_export(authorization: Optional[str] = Header(None)):
    """
    Export both Cloud SQL tables to dated parquet files in GCS.
    Protected by a bearer token stored in Secret Manager as 'export-admin-token'.
    Intended to be called weekly by Cloud Scheduler.
    """
    expected = "Bearer " + get_secret("export-admin-token")
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from google.cloud import storage as gcs
    import pandas as pd

    run_ts     = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    gcs_client = gcs.Client()
    engine     = get_engine()
    results    = []

    def _coerce(df):
        """Convert object-dtype columns containing non-strings (e.g. UUID) to str."""
        for col in df.select_dtypes("object").columns:
            sample = df[col].dropna()
            if len(sample) and not isinstance(sample.iloc[0], str):
                df[col] = df[col].apply(lambda v: None if v is None else str(v))
        return df

    for table in _EXPORT_TABLES:
        df       = _coerce(pd.read_sql(f"SELECT * FROM {table}", engine))
        buf      = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        buf.seek(0)

        blob_path = f"exports/{table}/{run_ts}.parquet"
        blob      = gcs_client.bucket(_GCS_BUCKET).blob(blob_path)
        blob.upload_from_file(buf, content_type="application/octet-stream")

        gcs_uri = f"gs://{_GCS_BUCKET}/{blob_path}"
        print(f"[export] {table}: {len(df):,} rows → {gcs_uri}")
        results.append({
            "table":     table,
            "rows":      len(df),
            "gcs_uri":   gcs_uri,
            "timestamp": run_ts,
        })

    return {"status": "ok", "exports": results}


# ── Static files (CSS/JS assets for future use) ───────────────
# Mounted last so explicit routes above always take precedence.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
