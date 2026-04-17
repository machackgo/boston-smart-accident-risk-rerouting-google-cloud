"""
Boston Smart Accident Risk Rerouting — FastAPI
Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on sys.path so src.predict.predictor is importable
# (needed on Render where the working directory may not be on PYTHONPATH)
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import create_client
from typing import Optional

from src.predict.predictor import predict_route_risk, predict_route_risk_segmented
from src.live.routes import get_route

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

# ── Credentials ──────────────────────────────────────────────
SUPABASE_URL = "https://iizfaawqzzrnhfaihimp.supabase.co"
SUPABASE_KEY = "sb_publishable_EnEsuHwXoNl4bQLeM2221A_LBW4nnNO"  # publishable key for reading

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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

# ── 0. Frontend ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    """Serve the single-page frontend app with the Google Maps API key injected."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    html = (_STATIC_DIR / "index.html").read_text()
    return html.replace("{{GOOGLE_MAPS_API_KEY}}", api_key)

# ── 1. Get all crashes (paginated) ───────────────────────────
@app.get("/crashes")
def get_crashes(limit: int = 100, offset: int = 0):
    """Get crashes with pagination. Default 100 per page."""
    res = supabase.table(TABLE).select("*").range(offset, offset + limit - 1).execute()
    return {"total_returned": len(res.data), "data": res.data}

# ── 2. Filter by year ─────────────────────────────────────────
@app.get("/crashes/year/{year}")
def get_by_year(year: int, limit: int = 100):
    """Get crashes for a specific year (2015–2024)."""
    res = supabase.table(TABLE).select("*").eq("year", year).limit(limit).execute()
    return {"year": year, "total_returned": len(res.data), "data": res.data}

# ── 3. Filter by severity ─────────────────────────────────────
@app.get("/crashes/severity/{severity}")
def get_by_severity(severity: str, limit: int = 100):
    """
    Severity options: Fatal, Non-fatal injury, Property damage only
    Use SEVERITY_3CLASS values.
    """
    res = supabase.table(TABLE).select("*").ilike("severity_3class", f"%{severity}%").limit(limit).execute()
    return {"severity": severity, "total_returned": len(res.data), "data": res.data}

# ── 4. Filter by city ─────────────────────────────────────────
@app.get("/crashes/city/{city}")
def get_by_city(city: str, limit: int = 100):
    """Get crashes in a specific city/town."""
    res = supabase.table(TABLE).select("*").ilike("city_town_name", f"%{city}%").limit(limit).execute()
    return {"city": city, "total_returned": len(res.data), "data": res.data}

# ── 5. Get hotspots ───────────────────────────────────────────
@app.get("/crashes/hotspots")
def get_hotspots(limit: int = 100):
    """Get crash hotspot locations (ems_hotspot_flag = 1)."""
    res = supabase.table(TABLE).select("*").eq("ems_hotspot_flag", 1).limit(limit).execute()
    return {"total_returned": len(res.data), "data": res.data}

# ── 6. Fatal crashes only ─────────────────────────────────────
@app.get("/crashes/fatal")
def get_fatal(limit: int = 100):
    """Get crashes with at least 1 fatality."""
    res = supabase.table(TABLE).select("*").gt("numb_fatal_injr", 0).limit(limit).execute()
    return {"total_returned": len(res.data), "data": res.data}

# ── 7. Summary stats by year ──────────────────────────────────
@app.get("/stats/by-year")
def stats_by_year():
    """Get crash counts grouped by year."""
    res = supabase.table(TABLE).select("year, numb_fatal_injr, numb_nonfatal_injr").execute()
    from collections import defaultdict
    stats = defaultdict(lambda: {"crashes": 0, "fatalities": 0, "injuries": 0})
    for row in res.data:
        y = row["year"]
        stats[y]["crashes"]    += 1
        stats[y]["fatalities"] += (row["numb_fatal_injr"] or 0)
        stats[y]["injuries"]   += (row["numb_nonfatal_injr"] or 0)
    return {"data": dict(sorted(stats.items()))}

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
    q = supabase.table(TABLE).select("*")
    if year:     q = q.eq("year", year)
    if city:     q = q.ilike("city_town_name", f"%{city}%")
    if severity: q = q.ilike("severity_3class", f"%{severity}%")
    if weather:  q = q.ilike("weath_cond_descr", f"%{weather}%")
    res = q.limit(limit).execute()
    return {"filters_applied": {"year": year, "city": city, "severity": severity, "weather": weather},
            "total_returned": len(res.data), "data": res.data}


# ── 9. Route risk prediction ──────────────────────────────────
@app.post("/predict")
def predict(request: PredictRequest):
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

    try:
        result = predict_route_risk(
            origin=request.origin,
            destination=request.destination,
            departure_time=request.departure_time,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 10. Per-segment route risk prediction ─────────────────────
@app.post("/predict/segmented")
def predict_segmented(request: SegmentedPredictRequest):
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

    try:
        result = predict_route_risk_segmented(
            origin=request.origin,
            destination=request.destination,
            departure_time=request.departure_time,
            num_segments=request.num_segments,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ── Static files (CSS/JS assets for future use) ───────────────
# Mounted last so explicit routes above always take precedence.
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
