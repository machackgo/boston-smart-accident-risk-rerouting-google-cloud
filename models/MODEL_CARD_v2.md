# Model Card — best_model_v2.pkl

## Overview

| Field            | Value |
|------------------|-------|
| Model type       | LGBMClassifier |
| Version          | v2 |
| Task             | Multi-class severity classification (High / Medium / Low) |
| Training data    | Boston crash records 2015–present (`data/crashes_cache.parquet`) |
| Training rows    | 40,438  (from 47,689 raw; 7,251 dropped for missing lat/lon/datetime/target) |
| Features         | 80 (all forward-knowable) |
| Saved to         | `models/best_model_v2.pkl` |

---

## Motivation: Why v2 Was Necessary

**v1 used post-crash features that would not be available at inference time.**

The v1 model (macro F1 = 0.4616) was trained on signals recorded *after* a crash
occurred — manner of collision, road surface condition, observed ambient light, and
number of vehicles involved. A live rerouting system needs to score a road segment
*before* a crash happens, so v1 predictions would require data that simply doesn't
exist yet at inference time.

v2 was retrained using only features a driver (or routing API) could observe in
advance: location, time of day, speed limit, and current weather.

---

## Feature List (80 features)

### Numeric / Engineered (8 features)

| Feature | Source | Notes |
|---------|--------|-------|
| `lat` | GPS coordinates | Location — road property |
| `lon` | GPS coordinates | Location — road property |
| `speed_limit` | Road database | Road property; ~22% missing → filled with median (30 mph) |
| `hour_of_day` | Crash datetime | 0–23 |
| `day_of_week` | Crash datetime | 0=Monday … 6=Sunday |
| `month` | Crash datetime | 1–12 |
| `is_weekend` | Engineered from day_of_week | 1 if Sat/Sun |
| `is_rush_hour` | Engineered from hour_of_day | 1 if 07:00–09:00 or 16:00–19:00 on a weekday |

### One-Hot: Weather Condition (69 features, `weath_cond_descr_*`)

Derived from `weath_cond_descr` — the historical weather condition recorded at crash
time. This is a pre-event feature because the driver observes weather before any
collision. Missing values filled with "Unknown".

Key categories include: `Clear`, `Cloudy`, `Rain`, `Snow`, `Fog__smog__smoke`,
`Sleet__hail__freezing_rain_or_drizzle_`, `Severe_crosswinds`, and compound values
(e.g., `Rain_Snow`) recorded when conditions changed during an incident window.

### One-Hot: Light Phase (3 features, `light_phase_*`)

Engineered from `hour_of_day` — **not** from the post-crash `ambnt_light_descr` field.

| Category | Hours |
|----------|-------|
| `light_phase_Daylight` | 07:00–18:59 |
| `light_phase_Dawn_Dusk` | 06:00–06:59 and 19:00–19:59 |
| `light_phase_Dark` | 00:00–05:59 and 20:00–23:59 |

### Features Explicitly Dropped vs. v1

| v1 Feature | Reason Dropped |
|------------|---------------|
| `manr_coll_descr_*` | Post-crash: recorded after collision |
| `road_surf_cond_descr_*` | Post-crash: recorded on-scene; weather used as proxy |
| `ambnt_light_descr_*` | Post-crash: recorded on-scene; hour-of-day used as proxy |
| `numb_vehc` | Post-crash: number of vehicles is a crash outcome |
| `rdwy_jnct_type_descr_*` | Post-crash: recorded on-scene |
| `ems_hotspot_flag` / `ems_ped_hotspot_flag` / `ems_peak_hour` | Only available for 73% of rows; EMS-derived not available for arbitrary new segments |
| `district_num` | Partially redundant with lat/lon; omitted to keep feature set minimal |
| `city_town_name_*` | Partially redundant with lat/lon |

---

## v1 vs v2 Comparison

| Metric | v1 (LightGBM) | v2 (LightGBM) | Change |
|--------|--------------|--------------|--------|
| Macro F1 | 0.4616 | 0.4025 | −0.0591 |
| Accuracy | 0.6556 | 0.6209 | −0.0347 |
| Features | 126 | 80 | −46 |
| Forward-knowable | No | Yes | Fixed |

**Why the drop in macro F1 is actually correct behaviour:**
v1's higher F1 was partly an artefact of leakage — post-crash features like manner
of collision are strongly correlated with severity *by definition* (a head-on
collision is more likely fatal than a sideswipe). Removing those features reduces
apparent performance but produces a model that reflects real-world signal available
before a crash, making it genuinely deployable.

---

## Per-Class Metrics (v2 LightGBM on 20% hold-out)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| High (Fatal) | 0.02 | 0.04 | 0.03 | 45 |
| Low (No Injury) | 0.77 | 0.66 | 0.71 | 5,621 |
| Medium (Injury) | 0.41 | 0.54 | 0.47 | 2,422 |
| **Macro avg** | **0.40** | **0.42** | **0.40** | **8,088** |

---

## Top 10 Feature Importances (LightGBM split gain)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `lat` | 4,487 |
| 2 | `lon` | 4,440 |
| 3 | `hour_of_day` | 2,196 |
| 4 | `month` | 1,983 |
| 5 | `day_of_week` | 1,448 |
| 6 | `speed_limit` | 1,308 |
| 7 | `weath_cond_descr_Clear` | 320 |
| 8 | `is_rush_hour` | 255 |
| 9 | `weath_cond_descr_Not_Reported` | 201 |
| 10 | `weath_cond_descr_Clear_Clear` | 165 |

Location and time dominate. Speed limit is a strong signal. Weather contributes
at the margin. Light phase (engineered) provides some signal but is largely captured
by `hour_of_day` already.

---

## Known Limitations

1. **Severe class imbalance on Fatal (High):** Only 224 of 40,438 training rows are
   fatal (0.6%). Despite `class_weight="balanced"`, the model barely detects High
   severity (F1 = 0.03). A dedicated anomaly detection approach or SMOTE oversampling
   may be needed for the fatal class.

2. **No road-network features:** Street type, intersection geometry, and traffic
   volume are not in the dataset. Adding OSM or HERE road attributes would likely
   improve performance significantly.

3. **Weather field is coarse:** `weath_cond_descr` reflects conditions at crash
   *report* time, which may lag the actual conditions by minutes. Live OpenWeather
   data (available via `src/live/weather.py`) would be more precise at inference time.

4. **Temporal leakage not fully addressed:** The train/test split is random, not
   time-based. A future improvement is to split by date (e.g., train on 2015–2022,
   test on 2023–2024) to simulate true forward prediction.

5. **Speed limit coverage:** ~22% of records have no speed limit — imputed with
   median. A geocoded speed-limit lookup (e.g., from OSM) would be more accurate.

---

## How to Load and Use v2

```python
import joblib
import pandas as pd

obj = joblib.load("models/best_model_v2.pkl")
model    = obj["model"]       # LGBMClassifier
features = obj["features"]    # list of 80 feature names
classes  = obj["classes"]     # ['High', 'Low', 'Medium']

# Build a single-row input dict with all 80 features (zeros for unknown OHE cols)
row = dict.fromkeys(features, 0)
row["lat"]          = 42.3467    # Fenway Park
row["lon"]          = -71.0972
row["hour_of_day"]  = 8          # 8 AM
row["day_of_week"]  = 0          # Monday
row["month"]        = 3          # March
row["is_weekend"]   = 0
row["is_rush_hour"] = 1          # 7-9 AM weekday
row["speed_limit"]  = 30.0
row["weath_cond_descr_Rain"] = 1     # raining
row["light_phase_Daylight"]  = 1    # 8 AM = daylight

X = pd.DataFrame([row])[features]
pred = model.predict(X)[0]          # e.g. "Medium"
proba = model.predict_proba(X)[0]   # [P(High), P(Low), P(Medium)]
print(f"Predicted severity: {pred}")
print(dict(zip(classes, proba.round(3))))
```
