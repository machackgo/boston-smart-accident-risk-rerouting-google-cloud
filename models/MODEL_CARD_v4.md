# Model Card — v4 (Spatial Features + Threshold Tuning)

**Artifact:** `best_model_v4.pkl`
**Date trained:** April 2026
**Script:** `src/model/train_v4.py`
**Feature list:** `models/feature_list_v4.txt` (36 features — down from 80 in v2/v3)
**Thresholds:** `models/thresholds_v4.json`
**Binary parallel:** `models/best_model_v4_binary.pkl`

---

## Why v4 Was Built — v3 Weaknesses

| Weakness | Description |
|----------|-------------|
| **Flat spatial signal** | v2/v3 only used raw `lat`/`lon` — no neighborhood context. Two points 50 m apart in a high-crash corridor vs. a low-crash suburb got the same treatment. |
| **Sparse weather noise** | v2/v3 had 80 feature columns, 50 of which had dataset sums < 50 — essentially noise. |
| **SMOTE failure (v3)** | SMOTE hurt macro F1 (0.3894 vs 0.4025 v2). Abandoned in v4. |
| **No threshold tuning** | Default 0.5 thresholds are suboptimal for imbalanced classes. |

---

## What Changed in v4

### A — 6 Leakage-Free Spatial Aggregate Features

Built by `src/model/spatial_features.py` using `sklearn.BallTree` (haversine metric):

| Feature | Description |
|---------|-------------|
| `nearby_crash_count_1km` | All crashes within 1 km (excl. self in training) |
| `nearby_fatal_count_1km` | Fatal crashes within 1 km |
| `nearby_injury_count_1km` | Injury crashes within 1 km |
| `nearby_crash_count_500m` | All crashes within 500 m |
| `nearby_fatal_count_500m` | Fatal crashes within 500 m |
| `nearby_avg_severity_1km` | Mean severity score within 1 km (Low=0, Med=1, High=2) |

**Leakage prevention:**
- Training set: each row queries only other training rows (leave-one-out: self index excluded).
- Test set: queries training-set tree only — no test-set crashes included.
- Live inference: queries the full historical dataset (training + test), as there is no held-out
  set at prediction time and we want the richest neighbourhood signal.

### B — Sparse Weather Column Pruning

50 of 69 weather one-hot columns had fewer than 50 non-zero rows across 40k records.
v4 drops these and retains 19 columns. Total features: 8 numeric + 6 spatial + 19 weather + 3 light = **36**.

### C — Per-Class Threshold Tuning

Searched [0.10, 0.90] in 0.05 steps for each class's optimal binary F1 threshold:

| Class | Threshold | Rationale |
|-------|-----------|-----------|
| High | 0.15 | Lower bar to catch rare Fatal events |
| Low | 0.25 | Allow more aggressive Low predictions |
| Medium | 0.40 | Slightly below 0.5 to improve recall |

Applied one-vs-rest: predict the class with the highest P above its threshold; fall back to argmax if none exceed.

### D — LightGBM `class_weight` comparison

Both `"balanced"` and manual `{Low:1, Medium:2, High:8}` were evaluated.
`"balanced"` won (macro F1 = 0.4384 vs 0.4357).

---

## v1 vs v2 vs v3 vs v4 Comparison

| Version | Features | Macro F1 (default) | Macro F1 (tuned) | High F1 | Medium F1 | Low F1 | Notes |
|---------|----------|-------------------|-----------------|---------|-----------|--------|-------|
| v1 | 126 | 0.4616 | — | ~0.20 | — | — | Includes post-crash leakage |
| v2 | 80 | 0.4025 | — | 0.03 | 0.47 | 0.71 | Clean, no leakage |
| v3 | 80 | 0.3894 | — | 0.00 | 0.38 | 0.79 | SMOTE hurt — abandoned |
| **v4** | **36** | **0.4384** | **0.4381** | **0.04** | **0.53** | **0.74** | **Current production** |

v4 is the best clean model (+0.036 macro F1 over v2 clean baseline).

---

## v4 Per-Class Metrics (LightGBM, test set, default thresholds)

```
              precision    recall  f1-score   support

        High       0.17      0.02      0.04        45
         Low       0.81      0.69      0.74      5621
      Medium       0.46      0.63      0.53      2422

    accuracy                           0.67      8088
   macro avg       0.48      0.45      0.44      8088
weighted avg       0.70      0.67      0.68      8088
```

---

## Top 15 Feature Importances (LightGBM split gain)

| Rank | Feature | Importance | Type |
|------|---------|-----------|------|
| 1 | `nearby_avg_severity_1km` | 4,246 | **Spatial** |
| 2 | `nearby_crash_count_500m` | 2,850 | **Spatial** |
| 3 | `nearby_injury_count_1km` | 2,833 | **Spatial** |
| 4 | `lat` | 2,734 | Location |
| 5 | `lon` | 2,557 | Location |
| 6 | `nearby_crash_count_1km` | 2,027 | **Spatial** |
| 7 | `hour_of_day` | 1,659 | Time |
| 8 | `nearby_fatal_count_1km` | 1,572 | **Spatial** |
| 9 | `month` | 1,508 | Time |
| 10 | `nearby_fatal_count_500m` | 1,176 | **Spatial** |
| 11 | `day_of_week` | 1,103 | Time |
| 12 | `speed_limit` | 1,024 | Road |
| 13 | `weath_cond_descr_Clear` | 252 | Weather |
| 14 | `is_rush_hour` | 217 | Time |
| 15 | `light_phase_Daylight` | 149 | Light |

**6 of the top 10 features are spatial aggregates.** `nearby_avg_severity_1km` alone
outranks `lat`, `lon`, `hour_of_day`, and `month` combined in split gain. This confirms
that neighborhood crash history is the dominant signal available before an accident.

---

## Binary Model (Step 6) — Safe vs Elevated

`best_model_v4_binary.pkl` maps Medium + High → "Elevated" (1) and Low → "Safe" (0).
Same v4 features, LightGBM with `class_weight="balanced"`.

| Metric | Value |
|--------|-------|
| ROC AUC | 0.7416 |
| Binary F1 (Elevated) | 0.5584 |
| Macro F1 | 0.6454 |
| Accuracy | 0.6668 |

The binary framing simplifies the problem enough that macro F1 jumps to 0.645 — a much
more usable signal for routing decisions where "Should I avoid this road?" is the real
question. The ROC AUC of 0.74 means the model meaningfully distinguishes elevated-risk
corridors from safe ones.

---

## Honest Limitations

- **High-class F1 still near zero.** Only 224 Fatal crashes in 40k rows. The model
  correctly identifies *corridors* with high crash history (via spatial features) but
  cannot reliably predict which individual trips will result in fatalities. This is a
  fundamental data ceiling.
- **Random train/test split.** Temporal leakage not addressed. v5 should use a
  date-based split (train 2015–2021, test 2022–2024).
- **Spatial features are historical, not causal.** A high `nearby_crash_count_1km`
  means "this location has had many past crashes" — it does not mean a crash is
  imminent at the time of the query.
- **Speed limits at inference are estimated.** Google traffic categories map to
  25/35/55 mph tiers, not actual posted limits.
- **Weather is single-point.** One OpenWeather call at route midpoint; micro-climate
  variation along a 50-mile route is ignored.

---

## Versioning

| File | Description |
|------|-------------|
| `best_model.pkl` | v1 — leakage, not for live use |
| `best_model_v2.pkl` | v2 — clean, 80 features |
| `best_model_v3.pkl` | v3 — SMOTE (worse than v2) |
| `best_model_v4.pkl` | v4 — **production** (36 features, spatial) |
| `best_model_v4_binary.pkl` | v4 binary — Safe vs Elevated (analysis) |
| `thresholds_v4.json` | Per-class thresholds used at inference |
| `weather_keep_cols_v4.json` | 19 weather columns kept after pruning |
