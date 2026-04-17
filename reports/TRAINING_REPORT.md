# Training Report — Boston Smart Accident Risk Rerouting

## v1 Baseline (All Features)

**Date trained:** April 2026
**Script:** `src/model/train.py`
**Data:** `data/crashes_cache.parquet` (47,689 rows)
**Features:** 126 (numeric + time-engineered + 5 one-hot categoricals)

### Model Comparison (v1)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 0.6556 | **0.4616** |
| RandomForest | 0.7164 | 0.3689 |
| LogisticRegression | 0.5262 | 0.3635 |

**Winner:** LightGBM (macro F1 = 0.4616)

### v1 Feature Groups

- Numeric: `speed_limit`, `numb_vehc`, `ems_hotspot_flag`, `ems_ped_hotspot_flag`,
  `ems_peak_hour`, `district_num`, `lat`, `lon`
- Time-engineered: `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`
- One-hot (5 categoricals): `weath_cond_descr`, `road_surf_cond_descr`,
  `ambnt_light_descr`, `manr_coll_descr`, `rdwy_jnct_type_descr`, `city_town_name`

### v1 Known Design Flaw

v1 included post-crash features — manner of collision, road surface condition, ambient
light description, and number of vehicles involved — that are only known *after* a
crash occurs. These features are not available at live prediction time and constitute
data leakage relative to a real-world rerouting use case. The higher F1 for v1 is
partly attributable to this leakage.

---

## v2 Retraining (Forward-Knowable Features Only)

**Date trained:** April 2026
**Script:** `src/model/train_v2.py`
**Data:** `data/crashes_cache.parquet` (same cache, no API re-hit)
**Features:** 80 (all forward-knowable — see full list below)
**Model artifact:** `models/best_model_v2.pkl`

### Design Change

The core motivation for v2 was to eliminate data leakage introduced by post-crash
signals in v1. A rerouting system scores road segments **before** a crash occurs, so
all inference-time features must be knowable in advance. v2 retains only:

- **Location:** `lat`, `lon`
- **Time:** `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`
- **Road property:** `speed_limit`
- **Weather:** `weath_cond_descr` one-hot (observable by driver before any crash)
- **Light proxy:** `light_phase` engineered from `hour_of_day` (not post-crash `ambnt_light_descr`)

**Dropped from v1:**
`manr_coll_descr`, `road_surf_cond_descr`, `ambnt_light_descr`, `numb_vehc`,
`rdwy_jnct_type_descr`, `ems_hotspot_flag`, `ems_ped_hotspot_flag`, `ems_peak_hour`,
`district_num`, `city_town_name`

### Row Counts

| Stage | Rows |
|-------|------|
| Raw parquet | 47,689 |
| After dropping missing target | 43,199 |
| After dropping missing lat/lon/datetime | 40,438 |
| Train (80%) | 32,350 |
| Test (20%) | 8,088 |

### Model Comparison (v2)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 0.6209 | **0.4025** |
| RandomForest | 0.6811 | 0.3412 |
| LogisticRegression | 0.4591 | 0.3253 |

**Winner:** LightGBM (macro F1 = 0.4025)

### v1 vs v2 Macro F1 Comparison

| Version | Macro F1 | Forward-Knowable | Notes |
|---------|----------|-----------------|-------|
| v1 | 0.4616 | No | Includes post-crash leakage |
| v2 | 0.4025 | Yes | Clean — deployable |
| Delta | −0.0591 | — | Expected drop after removing leakage |

The ~6-point drop in macro F1 is expected and correct. Post-crash features in v1
(especially manner of collision) are definitionally correlated with severity, so
removing them reduces apparent performance. v2 reflects genuine signal available
before a crash.

### v2 Per-Class Metrics (LightGBM, test set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| High (Fatal) | 0.02 | 0.04 | 0.03 | 45 |
| Low (No Injury) | 0.77 | 0.66 | 0.71 | 5,621 |
| Medium (Injury) | 0.41 | 0.54 | 0.47 | 2,422 |
| Macro avg | 0.40 | 0.42 | 0.40 | 8,088 |

### v2 Top 10 Feature Importances (LightGBM split gain)

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

Location and time are the dominant signals. Speed limit is a meaningful road-level
predictor. Weather contributes at the margin.

### v2 Full Feature List (80 features)

```
lat, lon, speed_limit, hour_of_day, day_of_week, month, is_weekend, is_rush_hour,
weath_cond_descr_Blowing_sand__snow, weath_cond_descr_Blowing_sand__snow_Blowing_sand__snow,
weath_cond_descr_Clear, weath_cond_descr_Clear_Blowing_sand__snow,
weath_cond_descr_Clear_Clear, weath_cond_descr_Clear_Cloudy,
weath_cond_descr_Clear_Fog__smog__smoke, weath_cond_descr_Clear_Other,
weath_cond_descr_Clear_Rain, weath_cond_descr_Clear_Severe_crosswinds,
weath_cond_descr_Clear_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Clear_Snow, weath_cond_descr_Clear_Unknown,
weath_cond_descr_Cloudy, weath_cond_descr_Cloudy_Clear, weath_cond_descr_Cloudy_Cloudy,
weath_cond_descr_Cloudy_Fog__smog__smoke, weath_cond_descr_Cloudy_Other,
weath_cond_descr_Cloudy_Rain, weath_cond_descr_Cloudy_Severe_crosswinds,
weath_cond_descr_Cloudy_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Cloudy_Snow, weath_cond_descr_Cloudy_Unknown,
weath_cond_descr_Fog__smog__smoke, weath_cond_descr_Fog__smog__smoke_Fog__smog__smoke,
weath_cond_descr_Fog__smog__smoke_Rain, weath_cond_descr_Not_Reported,
weath_cond_descr_Other, weath_cond_descr_Other_Other, weath_cond_descr_Other_Rain,
weath_cond_descr_Other_Snow, weath_cond_descr_Other_Unknown,
weath_cond_descr_Rain, weath_cond_descr_Rain_Clear, weath_cond_descr_Rain_Cloudy,
weath_cond_descr_Rain_Fog__smog__smoke, weath_cond_descr_Rain_Other,
weath_cond_descr_Rain_Rain, weath_cond_descr_Rain_Severe_crosswinds,
weath_cond_descr_Rain_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Rain_Snow, weath_cond_descr_Rain_Unknown,
weath_cond_descr_Reported_but_invalid, weath_cond_descr_Severe_crosswinds,
weath_cond_descr_Severe_crosswinds_Blowing_sand__snow,
weath_cond_descr_Severe_crosswinds_Clear, weath_cond_descr_Severe_crosswinds_Other,
weath_cond_descr_Severe_crosswinds_Rain,
weath_cond_descr_Severe_crosswinds_Severe_crosswinds,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Blowing_sand__snow,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Cloudy,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Rain,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Severe_crosswinds,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Sleet__hail__freezing_rain_or_drizzle__Snow,
weath_cond_descr_Snow, weath_cond_descr_Snow_Blowing_sand__snow,
weath_cond_descr_Snow_Clear, weath_cond_descr_Snow_Cloudy,
weath_cond_descr_Snow_Fog__smog__smoke, weath_cond_descr_Snow_Other,
weath_cond_descr_Snow_Rain, weath_cond_descr_Snow_Severe_crosswinds,
weath_cond_descr_Snow_Sleet__hail__freezing_rain_or_drizzle_,
weath_cond_descr_Snow_Snow, weath_cond_descr_Unknown,
weath_cond_descr_Unknown_Clear, weath_cond_descr_Unknown_Unknown,
light_phase_Dark, light_phase_Dawn_Dusk, light_phase_Daylight
```

### Known Limitations

- **Fatal class barely detected** (F1 = 0.03, support = 45 in test set). The 0.6%
  base rate makes this class extremely hard to learn without oversampling or a
  dedicated anomaly detector.
- **Random train/test split** — temporal leakage not fully addressed. Future work
  should use a date-based split (train on earlier years, test on later years).
- **Speed limit imputed** for ~22% of records. A geocoded lookup from OSM would be
  more accurate.
- **No road-network features** (street type, intersection count, AADT traffic volume).
  Adding these would likely close much of the v1 → v2 performance gap without
  reintroducing data leakage.

---

## v3 Retraining (SMOTE + Real Speed Limits)

**Date trained:** April 2026
**Script:** `src/model/train_v3.py`
**Data:** `data/crashes_cache.parquet` (same 40,438 usable rows as v2)
**Features:** 80 (identical schema to v2)
**Model artifact:** `models/best_model_v3.pkl`
**Full notes:** `models/MODEL_CARD_v3.md`

### Motivation

v2 had three problems targeted by v3:

1. **Medium-bias** — almost all routes scored Medium regardless of context.
2. **High-class F1 ≈ 0.03** — only 224 Fatal crashes in 40k rows; the model rarely
   predicted High.
3. **Hardcoded speed limit** — `speed_limit = 30.0` for every inference call; the
   feature had no variance at prediction time.

### Changes

| Change | Detail |
|--------|--------|
| SMOTE oversampling | `SMOTE(random_state=42, k_neighbors=5)` applied **only to X_train/y_train** — test set untouched |
| Per-segment speed limits | `routes.py` requests `travelAdvisory.speedReadingIntervals` from Google; maps NORMAL/SLOW/TRAFFIC_JAM to 25/35/55 mph tiers |
| API tiny-route guard | `api.py` rejects routes < 0.3 miles or < 2 minutes with HTTP 400 |

### Model Comparison (v3)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LightGBM | 0.6864 | **0.3894** |
| RandomForest | 0.6459 | 0.3768 |
| LogisticRegression | 0.5825 | 0.3672 |

**Winner:** LightGBM (macro F1 = 0.3894)

### v1 vs v2 vs v3 Summary

| Version | Macro F1 | High F1 | Notes |
|---------|----------|---------|-------|
| v1 | 0.4616 | ~0.20 | Includes post-crash leakage — not valid for live use |
| v2 | 0.4025 | ~0.03 | Clean, forward-knowable features |
| v3 | 0.3894 | ~0.00 | SMOTE did not improve High recall for tree models |

### v3 Per-Class Metrics (LightGBM, test set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| High (Fatal) | 0.00 | 0.00 | 0.00 | 45 |
| Low (No Injury) | 0.74 | 0.85 | 0.79 | 5,621 |
| Medium (Injury) | 0.47 | 0.31 | 0.38 | 2,422 |
| Macro avg | 0.40 | 0.39 | 0.39 | 8,088 |

### Top 10 v3 Feature Importances (LightGBM split gain)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `hour_of_day` | 3,215 |
| 2 | `month` | 2,860 |
| 3 | `lon` | 2,695 |
| 4 | `lat` | 2,630 |
| 5 | `day_of_week` | 2,103 |
| 6 | `speed_limit` | 1,772 |
| 7 | `weath_cond_descr_Clear` | 443 |
| 8 | `is_rush_hour` | 287 |
| 9 | `weath_cond_descr_Cloudy` | 210 |
| 10 | `weath_cond_descr_Rain` | 197 |

`speed_limit` remains 6th most important. Notably, `hour_of_day` and `month` overtook
`lat`/`lon` as the top features in v3 — the SMOTE resampling appears to have shifted the
model from purely location-based patterns toward time-of-day seasonality.

### Honest Assessment

SMOTE at a 33% equal balance was too aggressive for this dataset. The 224 real Fatal
crashes in training are too sparse to generate useful synthetic neighbours — SMOTE
interpolated in a region of feature space dominated by Low/Medium points, and LightGBM
largely ignored the synthetic High examples. Macro F1 dropped by −0.013 vs v2.

The per-segment speed limit improvement (Improvement B) is the more durable change:
inference-time `speed_limit` now varies between 25, 35, and 55 mph (or intermediate
values from Google traffic categories) rather than being a constant 30. This means the
feature is no longer wasted at prediction time, even though the training data improvement
is modest.

**Next steps for future v4:**
- Try SMOTE with much lower oversampling ratio (e.g. minority = 5% rather than 33%)
- Try ADASYN instead of SMOTE
- Add road-network features (OSM street type, intersection count)
- Use temporal split for evaluation (train 2015–2021, test 2022–2024)
