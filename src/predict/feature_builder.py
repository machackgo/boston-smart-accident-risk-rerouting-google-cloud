"""
feature_builder.py — Assembles the feature row required by the active model.

v4 (current production): 36 features including 6 leakage-free spatial aggregates.
Falls back to v2 feature schema if v4 artefacts are not present.

Orchestrates:
    1. get_route()    — route geometry + midpoint + per-segment speed limits
    2. get_weather()  — current weather at the route midpoint
    3. Spatial query  — nearby crash counts from historical data (BallTree)

OpenWeather condition → feature column mapping:
    "Clear"                         → weath_cond_descr_Clear
    "Clouds"                        → weath_cond_descr_Cloudy
    "Rain" / "Drizzle"              → weath_cond_descr_Rain
    "Snow"                          → weath_cond_descr_Snow
    "Fog" / "Mist" / "Haze" etc.   → weath_cond_descr_Fog__smog__smoke
                                       (falls back to Not_Reported if column pruned)
    "Thunderstorm" etc.             → weath_cond_descr_Severe_crosswinds
                                       (falls back to Not_Reported if column pruned)
    anything else                   → weath_cond_descr_Not_Reported
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

BOSTON_TZ = ZoneInfo("America/New_York")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from live.routes import get_route
from live.weather import get_weather

# ── Feature list — prefer v4, fall back to v2 ────────────────────────────────
_FEAT_V4 = _REPO_ROOT / "models" / "feature_list_v4.txt"
_FEAT_V2 = _REPO_ROOT / "models" / "feature_list_v2.txt"
_FEAT_PATH = _FEAT_V4 if _FEAT_V4.exists() else _FEAT_V2

with open(_FEAT_PATH) as _f:
    ACTIVE_FEATURES = [line.strip() for line in _f if line.strip()]

_USING_V4 = _FEAT_PATH == _FEAT_V4

# ── Weather keep-cols (v4 pruned set; all allowed when on v2 fallback) ────────
_WEATHER_KEEPCOLS_PATH = _REPO_ROOT / "models" / "weather_keep_cols_v4.json"
if _USING_V4 and _WEATHER_KEEPCOLS_PATH.exists():
    with open(_WEATHER_KEEPCOLS_PATH) as _f:
        _WEATHER_KEEP_SET = set(json.load(_f))
else:
    _WEATHER_KEEP_SET = None   # None = keep all (v2 behaviour)

# ── Spatial BallTree — loaded once at module import (v4 only) ─────────────────
_SPATIAL_TREE      = None
_SPATIAL_SCORES    = None    # severity score (0/1/2) per historical crash
_SPATIAL_IS_FATAL  = None
_SPATIAL_IS_INJURY = None

if _USING_V4:
    try:
        from sklearn.neighbors import BallTree as _BallTree

        _CACHE_PATH = _REPO_ROOT / "data" / "crashes_cache.parquet"
        _SEVERITY_SCORE_MAP = {"No Injury": 0.0, "Injury": 1.0, "Fatal": 2.0}

        print("[feature_builder] Loading historical crash data for spatial queries ...")
        _hist = pd.read_parquet(_CACHE_PATH)
        _hist = _hist.dropna(subset=["lat", "lon", "severity_3class"])
        _hist_coords = np.radians(_hist[["lat", "lon"]].values.astype(np.float64))
        _SPATIAL_TREE      = _BallTree(_hist_coords, metric="haversine")
        _SPATIAL_SCORES    = np.array(
            [_SEVERITY_SCORE_MAP.get(s, 0.0) for s in _hist["severity_3class"]],
            dtype=np.float32,
        )
        _SPATIAL_IS_FATAL  = (_hist["severity_3class"] == "Fatal").values.astype(np.float32)
        _SPATIAL_IS_INJURY = (_hist["severity_3class"] == "Injury").values.astype(np.float32)
        print(f"[feature_builder] BallTree built on {len(_hist):,} historical crashes.")
    except Exception as _e:
        print(f"[feature_builder] WARNING: could not build spatial BallTree: {_e}")
        _USING_V4 = False


_EARTH_R = 6_371_009.0  # metres


def _spatial_features_for_point(lat: float, lng: float) -> dict:
    """
    Query the historical crash BallTree for one (lat, lng) point.
    Returns the 6 spatial aggregate features as a dict.
    Returns all-zero dict if the tree was not loaded.
    """
    zeros = {
        "nearby_crash_count_1km":  0.0,
        "nearby_fatal_count_1km":  0.0,
        "nearby_injury_count_1km": 0.0,
        "nearby_crash_count_500m": 0.0,
        "nearby_fatal_count_500m": 0.0,
        "nearby_avg_severity_1km": 0.0,
    }
    if _SPATIAL_TREE is None:
        return zeros

    point = np.radians([[lat, lng]])
    r1km  = 1000.0 / _EARTH_R
    r500m =  500.0 / _EARTH_R

    hits_1km,  = _SPATIAL_TREE.query_radius(point, r=r1km,  return_distance=False)
    hits_500m, = _SPATIAL_TREE.query_radius(point, r=r500m, return_distance=False)

    n1 = len(hits_1km)
    n5 = len(hits_500m)

    return {
        "nearby_crash_count_1km":  float(n1),
        "nearby_fatal_count_1km":  float(_SPATIAL_IS_FATAL[hits_1km].sum())  if n1 else 0.0,
        "nearby_injury_count_1km": float(_SPATIAL_IS_INJURY[hits_1km].sum()) if n1 else 0.0,
        "nearby_crash_count_500m": float(n5),
        "nearby_fatal_count_500m": float(_SPATIAL_IS_FATAL[hits_500m].sum()) if n5 else 0.0,
        "nearby_avg_severity_1km": float(_SPATIAL_SCORES[hits_1km].mean())   if n1 else 0.0,
    }


def _smarter_speed_default(distance_miles: float) -> float:
    if distance_miles < 2.0:
        return 25.0
    if distance_miles < 10.0:
        return 35.0
    return 55.0


_WEATHER_MAP = {
    "Clear":        "weath_cond_descr_Clear",
    "Clouds":       "weath_cond_descr_Cloudy",
    "Rain":         "weath_cond_descr_Rain",
    "Drizzle":      "weath_cond_descr_Rain",
    "Snow":         "weath_cond_descr_Snow",
    "Fog":          "weath_cond_descr_Fog__smog__smoke",
    "Mist":         "weath_cond_descr_Fog__smog__smoke",
    "Haze":         "weath_cond_descr_Fog__smog__smoke",
    "Smoke":        "weath_cond_descr_Fog__smog__smoke",
    "Dust":         "weath_cond_descr_Fog__smog__smoke",
    "Sand":         "weath_cond_descr_Fog__smog__smoke",
    "Ash":          "weath_cond_descr_Fog__smog__smoke",
    "Thunderstorm": "weath_cond_descr_Severe_crosswinds",
    "Squall":       "weath_cond_descr_Severe_crosswinds",
    "Tornado":      "weath_cond_descr_Severe_crosswinds",
}
_WEATHER_FALLBACK = "weath_cond_descr_Not_Reported"


def _resolve_time(departure_time):
    if departure_time is None:
        return datetime.now(BOSTON_TZ)
    if isinstance(departure_time, datetime):
        if departure_time.tzinfo:
            return departure_time.astimezone(BOSTON_TZ)
        return departure_time.replace(tzinfo=BOSTON_TZ)
    dt = datetime.fromisoformat(departure_time)
    if dt.tzinfo:
        return dt.astimezone(BOSTON_TZ)
    return dt.replace(tzinfo=BOSTON_TZ)


def _time_features(dt: datetime) -> dict:
    hour       = dt.hour
    dow        = dt.weekday()
    is_weekend = int(dow >= 5)
    is_rush    = int((hour in range(7, 10) or hour in range(16, 20)) and not is_weekend)
    return {
        "hour_of_day": hour,
        "day_of_week": dow,
        "month":       dt.month,
        "is_weekend":  is_weekend,
        "is_rush_hour": is_rush,
    }


def _light_phase_features(hour: int) -> dict:
    if 7 <= hour <= 18:
        return {"light_phase_Daylight": 1, "light_phase_Dawn_Dusk": 0, "light_phase_Dark": 0}
    if hour in (5, 6, 19, 20):
        return {"light_phase_Daylight": 0, "light_phase_Dawn_Dusk": 1, "light_phase_Dark": 0}
    return {"light_phase_Daylight": 0, "light_phase_Dawn_Dusk": 0, "light_phase_Dark": 1}


def _map_weather(ow_condition: str) -> tuple[str, str]:
    """
    Map OpenWeather condition to a feature column name.
    Falls back to _WEATHER_FALLBACK if the mapped column was pruned in v4.
    """
    col = _WEATHER_MAP.get(ow_condition, _WEATHER_FALLBACK)
    # If using v4 pruned schema and this column was dropped, use fallback
    if _WEATHER_KEEP_SET and col not in _WEATHER_KEEP_SET:
        col = _WEATHER_FALLBACK
    desc = f"'{ow_condition}' → {col}"
    return col, desc


def build_segment_features(
    route: dict,
    sample_points: list,
    departure_time=None,
    speed_limits_per_point: list | None = None,
) -> tuple:
    """
    Build an N-row feature DataFrame for a list of sample points.

    Uses ONE weather call at route midpoint; spatial features queried per point.

    Args:
        route (dict): Return value of get_route().
        sample_points (list): List of (lat, lng) tuples.
        departure_time: ISO 8601 string, datetime, or None (= now).
        speed_limits_per_point (list | None): Per-point speed estimates.

    Returns:
        features_df (pd.DataFrame): N-row DataFrame in ACTIVE_FEATURES column order.
        context (dict): Transparency dict.
    """
    dt = _resolve_time(departure_time)
    print(f"[feature_builder] segment mode | Boston time: {dt.strftime('%Y-%m-%d %H:%M %Z')}, "
          f"points={len(sample_points)}, model=v{'4' if _USING_V4 else '2/3'}")

    default_route = route["default_route"]
    midpoint      = default_route["midpoint_coords"]
    if midpoint is None:
        raise ValueError("Route has no decoded polyline — cannot fetch weather.")
    mid_lat = midpoint["lat"]
    mid_lng = midpoint["lng"]

    print(f"[feature_builder] Fetching weather at midpoint ({mid_lat:.4f}, {mid_lng:.4f}) ...")
    weather              = get_weather(lat=mid_lat, lng=mid_lng)
    ow_condition         = weather["condition"]
    weather_col, weather_mapping_desc = _map_weather(ow_condition)
    print(f"[feature_builder] Weather: {weather_mapping_desc}")

    time_feats   = _time_features(dt)
    light_feats  = _light_phase_features(time_feats["hour_of_day"])
    route_dist   = default_route.get("distance_miles", 5.0)
    fallback_spd = _smarter_speed_default(route_dist)

    rows = []
    for i, (lat, lng) in enumerate(sample_points):
        row = {feat: 0 for feat in ACTIVE_FEATURES}
        row["lat"] = lat
        row["lon"] = lng

        # Speed limit
        if (speed_limits_per_point
                and i < len(speed_limits_per_point)
                and speed_limits_per_point[i] is not None):
            row["speed_limit"] = speed_limits_per_point[i]
        else:
            row["speed_limit"] = fallback_spd

        row.update(time_feats)
        row.update(light_feats)

        # Weather
        if weather_col in row:
            row[weather_col] = 1
        else:
            row[_WEATHER_FALLBACK] = 1

        # Spatial features (v4 only; no-op if tree not loaded)
        if _USING_V4:
            row.update(_spatial_features_for_point(lat, lng))

        rows.append(row)

    features_df = pd.DataFrame(rows)[ACTIVE_FEATURES]

    context = {
        "weather_raw":          weather,
        "weather_mapping_used": weather_mapping_desc,
        "weather_col":          weather_col,
        "departure_time":       dt.isoformat(),
        "time_features":        time_feats,
        "local_hour":           time_feats["hour_of_day"],
        "num_points":           len(sample_points),
        "speed_source":         "per_segment" if speed_limits_per_point else "distance_heuristic",
        "fallback_speed_mph":   fallback_spd,
        "model_version":        "v4" if _USING_V4 else "v2/v3",
    }
    return features_df, context


def build_features(origin: str, destination: str, departure_time=None) -> tuple:
    """
    Assemble the feature DataFrame for a whole-route (non-segmented) prediction.

    Spatial features are computed at the route midpoint.
    """
    dt = _resolve_time(departure_time)
    print(f"[feature_builder] Local Boston time: {dt.strftime('%Y-%m-%d %H:%M %Z')}, hour={dt.hour}")

    print(f"[feature_builder] Fetching route: '{origin}' → '{destination}' ...")
    route = get_route(origin, destination)

    default_route = route["default_route"]
    midpoint      = default_route["midpoint_coords"]
    if midpoint is None:
        raise ValueError("Route returned no decoded polyline — cannot determine midpoint.")

    mid_lat = midpoint["lat"]
    mid_lng = midpoint["lng"]
    print(f"[feature_builder] Route midpoint: lat={mid_lat:.4f}, lng={mid_lng:.4f}")

    print(f"[feature_builder] Fetching weather ...")
    weather      = get_weather(lat=mid_lat, lng=mid_lng)
    ow_condition = weather["condition"]

    weather_col, weather_mapping_desc = _map_weather(ow_condition)
    print(f"[feature_builder] Weather: {weather_mapping_desc}")

    time_feats  = _time_features(dt)
    light_feats = _light_phase_features(time_feats["hour_of_day"])

    row = {feat: 0 for feat in ACTIVE_FEATURES}

    row["lat"] = mid_lat
    row["lon"] = mid_lng
    row["speed_limit"] = _smarter_speed_default(default_route["distance_miles"])

    row.update(time_feats)
    row.update(light_feats)

    if weather_col in row:
        row[weather_col] = 1
    else:
        row[_WEATHER_FALLBACK] = 1
        weather_mapping_desc += f" (fallback to {_WEATHER_FALLBACK})"

    # Spatial features at midpoint
    if _USING_V4:
        row.update(_spatial_features_for_point(mid_lat, mid_lng))

    features_df = pd.DataFrame([row])[ACTIVE_FEATURES]

    context = {
        "origin":           origin,
        "destination":      destination,
        "departure_time":   dt.isoformat(),
        "midpoint_lat":     mid_lat,
        "midpoint_lng":     mid_lng,
        "hour_of_day":      time_feats["hour_of_day"],
        "is_rush_hour":     time_feats["is_rush_hour"],
        "is_weekend":       time_feats["is_weekend"],
        "weather_condition_raw":  ow_condition,
        "weather_mapping_used":   weather_mapping_desc,
        "weather_feature_set":    weather_col,
        "light_phase":      next(k for k, v in light_feats.items() if v == 1),
        "speed_limit_used": _smarter_speed_default(default_route["distance_miles"]),
        "route_raw":        route,
        "weather_raw":      weather,
        "model_version":    "v4" if _USING_V4 else "v2/v3",
    }
    return features_df, context
