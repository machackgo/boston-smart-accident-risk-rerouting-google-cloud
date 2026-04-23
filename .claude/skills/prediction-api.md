# Skill: Prediction API

Covers `api.py`, `src/predict/predictor.py`, `src/predict/feature_builder.py`,
`src/live/routes.py`, `src/live/weather.py`, `src/live/geocoding.py`.

---

## Endpoints

| Method | Path | Handler |
|--------|------|---------|
| POST | `/predict` | `predict()` → `predict_route_risk()` |
| POST | `/predict/segmented` | `predict_segmented()` → `predict_route_risk_segmented()` |
| GET | `/predict/example` | Hardcoded response (Fenway → Logan, Low, 0.7909) |
| POST | `/admin/export` | GCS parquet export, auth via `export-admin-token` header |
| GET | `/crashes`, `/crashes/*` | Cloud SQL read-only queries |
| GET | `/stats/by-year` | In-memory aggregation of Cloud SQL data |

**Route guard** (enforced before every prediction):
- `_MIN_DISTANCE_MILES = 0.3`
- `_MIN_DURATION_MIN = 2.0`
- Returns 400 if route is below these thresholds.

---

## Prediction flow (`predict_route_risk`)

1. `build_features(origin, destination, departure_time)` — assembles one feature row
2. `vertex_client.predict_single(features_row)` — attempts Vertex AI inference
3. If Vertex returns `None` → `_MODEL.predict_proba(features_df)` local fallback
4. `_classify_with_thresholds(probas)` — applies per-class thresholds
5. Extract spatial features from `features_df` for Gemini context
6. Build result dict with `risk_class`, `confidence`, `class_probabilities`, `route`, `weather`, `context`
7. `get_gemini_explanation({...result, spatial_features: {...}})` → adds `"explanation"` field
8. Return result (FastAPI serialises as JSON; `None` becomes `null`)

Background tasks (non-blocking, after response is sent):
- `log_prediction(row)` → Cloud SQL `route_predictions`
- `log_prediction_bq(row)` → BigQuery `route_predictions`

---

## Segmented prediction flow (`predict_route_risk_segmented`)

1. `get_route()` → fetch all routes (default + alternatives)
2. Sample `num_segments` points evenly per route (default 12)
3. `build_segment_features()` — ONE weather call, per-point spatial queries, all routes stacked
4. `vertex_client.predict_batch(all_instances)` or local `_MODEL.predict_proba(all_df)`
5. Reconstruct per-route segments + hotspots (Medium/High segments)
6. Reverse-geocode each hotspot for street name + neighbourhood
7. Safety score per route: `(n_high × 10) + (n_med × 3) + (dur_min × 0.1)`
8. Recommend route with lowest safety score
9. `get_gemini_explanation(recommended_route_data)` → adds `"explanation"` field

Response shape: `routes[]`, `recommended_route_index`, `recommendation_reason`, `explanation`, `weather`, `context`.

---

## Feature schema (v4 — 36 features)

| Group | Features |
|-------|---------|
| Location | `lat`, `lon` |
| Route | `speed_limit` |
| Time | `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour` |
| Light | `light_phase_Daylight`, `light_phase_Dawn_Dusk`, `light_phase_Dark` |
| Spatial (BallTree) | `nearby_crash_count_1km`, `nearby_fatal_count_1km`, `nearby_injury_count_1km`, `nearby_crash_count_500m`, `nearby_fatal_count_500m`, `nearby_avg_severity_1km` |
| Weather (one-hot, 19 cols) | `weath_cond_descr_Clear`, `weath_cond_descr_Cloudy`, `weath_cond_descr_Rain`, `weath_cond_descr_Snow`, `weath_cond_descr_Fog__smog__smoke`, `weath_cond_descr_Severe_crosswinds`, `weath_cond_descr_Not_Reported` + 12 other sparse cols |

BallTree is loaded once at `feature_builder` module import from `data/crashes_cache.parquet`.
Radii: 1 km = `1000 / 6_371_009` rad, 500 m = `500 / 6_371_009` rad.
If BallTree fails to load, all spatial features default to `0.0` (v4 degrades silently to v2 behaviour).

Speed limit heuristic (used when Routes API returns no advisory data):
- `< 2 miles` → 25 mph
- `2–10 miles` → 35 mph
- `≥ 10 miles` → 55 mph

---

## OpenWeather → feature column mapping

| OpenWeather `main` | Feature column |
|-------------------|---------------|
| Clear | `weath_cond_descr_Clear` |
| Clouds | `weath_cond_descr_Cloudy` |
| Rain, Drizzle | `weath_cond_descr_Rain` |
| Snow | `weath_cond_descr_Snow` |
| Fog, Mist, Haze, Smoke, Dust, Sand, Ash | `weath_cond_descr_Fog__smog__smoke` |
| Thunderstorm, Squall, Tornado | `weath_cond_descr_Severe_crosswinds` |
| anything else | `weath_cond_descr_Not_Reported` |

---

## External APIs used

| API | Secret | Endpoint |
|-----|--------|---------|
| Google Maps Routes v2 | `google-server-api-key` | `https://routes.googleapis.com/directions/v2:computeRoutes` |
| Google Maps Geocoding | `google-maps-api-key` (reverse) / `google-server-api-key` (forward) | `https://maps.googleapis.com/maps/api/geocode/json` |
| OpenWeather v2.5 | `openweather-api-key` | `https://api.openweathermap.org/data/2.5/weather` |

Routes API is called with `TRAFFIC_AWARE_OPTIMAL` routing and `TRAFFIC_ON_POLYLINE` extra computation.
`computeAlternativeRoutes: true` — may return 0, 1, or 2 alternatives depending on route.

---

## Common tasks

**Adding a new endpoint:**
- Define a Pydantic request model in `api.py`
- Add the FastAPI route decorator
- Call the relevant `predict_*` function or Cloud SQL query
- Never import `google.generativeai` — use only `google-genai`

**Changing the minimum route guard:**
- Edit `_MIN_DISTANCE_MILES` and `_MIN_DURATION_MIN` at the top of `api.py`

**Debugging a prediction:**
- Check `context.inference_source` in the response: `"vertex"` or `"local"`
- Check `context.vertex_latency_ms` for Vertex AI performance
- Check server logs for `[predictor]`, `[vertex_client]`, `[gemini_explainer]` lines

**Adding a field to the prediction response:**
- Add it to the result dict in `predictor.py` (both functions if applicable)
- It will appear in JSON automatically (FastAPI serialises plain dicts)
- If it's also needed in Cloud SQL, add column to `database.py` and `log_prediction()`
- If it's also needed in BQ, add field to `bigquery_logger.py`

---

## What NOT to touch

- `serving/main.py` — separate container, inference-only, no live APIs. Changing `predictor.py` does not affect it.
- `data/crashes_cache.parquet` — do not modify directly; it's generated by `preprocess_v4.py`
- `models/thresholds_v4.json` — tuned by grid search in `train_v4.py`; don't hand-edit
