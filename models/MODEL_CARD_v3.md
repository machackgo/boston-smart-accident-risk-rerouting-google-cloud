# Model Card — v3 (SMOTE + Per-Segment Speed Limits)

**Artifact:** `best_model_v3.pkl`
**Date trained:** April 2026
**Script:** `src/model/train_v3.py`
**Feature list:** `models/feature_list_v3.txt` (identical to v2 — 80 features)

---

## Why v3 Was Built — v2 Weaknesses

| Weakness | Description |
|----------|-------------|
| **Medium-bias** | v2 predicted "Medium" for the vast majority of routes, even on clear-weather weekday mornings |
| **High-class F1 ≈ 0.15** | Fatal crashes are only 0.55% of training data (224 of 40,438 rows). The model rarely predicted High even when the ground truth was Fatal |
| **Hardcoded speed limit** | `speed_limit = 30.0` for every route, making it a constant rather than a discriminating feature |
| **Tiny-route nonsense** | Routes under 0.3 miles returned risk predictions that were statistically meaningless |

---

## What Changed in v3

### A — API guard for tiny routes
Added to `api.py` endpoints `/predict` and `/predict/segmented`: routes under 0.3 miles or 2 minutes duration now return HTTP 400 with a clear error message. The predictor itself is unchanged.

### B — Per-segment speed limits (real vs hardcoded 30)
`src/live/routes.py` now requests `routes.legs.travelAdvisory.speedReadingIntervals`
with `extraComputations: TRAFFIC_ON_POLYLINE` from the Google Routes API.

Speed category mapping:
| Google SpeedCategory | Estimated mph |
|----------------------|---------------|
| NORMAL | distance-based default (25 / 35 / 55) |
| SLOW | default − 10 (min 15) |
| TRAFFIC_JAM | default − 20 (min 10) |

Distance-based fallback (when Google returns no intervals):
| Route distance | Default speed |
|----------------|---------------|
| < 2 miles | 25 mph (city streets) |
| 2–10 miles | 35 mph (mixed) |
| > 10 miles | 55 mph (highway) |

The per-segment speed limits are threaded through `predictor._sample_leg` →
`feature_builder.build_segment_features` so every sampled point gets a distinct
estimated speed limit instead of 30.0.

### C — SMOTE oversampling on training set only
`imbalanced-learn` SMOTE (k_neighbors=5, random_state=42) was applied exclusively to
`(X_train, y_train)` — never to the test set.

After SMOTE, the three classes were equalised to ~22,481 each (up from 179 High, 9,626
Medium, 22,545 Low in the training split).

Three models were trained: LogisticRegression, RandomForestClassifier(n_estimators=200),
LightGBMClassifier(n_estimators=200) — all with `class_weight='balanced'`.

---

## v1 vs v2 vs v3 Comparison

| Version | Strategy | Macro F1 | High F1 | Notes |
|---------|----------|----------|---------|-------|
| v1 | All features (includes post-crash leakage) | **0.4616** | ~0.20 | Not valid for live inference |
| v2 | Forward-knowable features only, balanced weights | 0.4025 | ~0.15 | Clean, no leakage |
| v3 | v2 + SMOTE on train + per-segment speed limits | 0.3894 | ~0.00 | See honest assessment below |

---

## v3 Per-Class Metrics (LightGBM — test set)

```
              precision    recall  f1-score   support

        High       0.00      0.00      0.00        45
         Low       0.74      0.85      0.79      5621
      Medium       0.47      0.31      0.38      2422

    accuracy                           0.69      8088
   macro avg       0.40      0.39      0.39      8088
weighted avg       0.66      0.69      0.66      8088
```

---

## Honest Assessment of v3 Results

**SMOTE did not improve High-class recall for LightGBM or RandomForest.**

The root cause is the extreme imbalance: only 45 High samples in the 8,088-row test
set (0.55%). Even after SMOTE synthetically equalised the training classes, LightGBM
effectively predicts 0 High samples. Macro F1 dropped by −0.013 vs v2.

Contributing factors:
1. SMOTE equalised to 33.3% each — too aggressive. The synthetic High samples lie
   in a very sparse region of feature space (lat/lon/time), making them hard to
   separate from Low.
2. LightGBM and RandomForest are strong enough to recognise that the synthetic points
   are "off" and ignore them. The `class_weight='balanced'` parameter already
   addresses class weighting without needing SMOTE.
3. The LogisticRegression variant achieves High recall = 0.22 but macro F1 = 0.367,
   worse overall.

**Top 10 v3 Feature Importances (LightGBM):**
```
 1. hour_of_day            3215
 2. month                  2860
 3. lon                    2695
 4. lat                    2630
 5. day_of_week            2103
 6. speed_limit            1772   ← now varied (was hardcoded 30 in v2)
 7. weath_cond_descr_Clear   443
 8. is_rush_hour             287
 9. weath_cond_descr_Cloudy  210
10. weath_cond_descr_Rain    197
```

`speed_limit` is the 6th most important feature (rank unchanged from v2, but now its
value varies per segment/route instead of being a constant). The real improvement from
Improvement B will be visible in prediction quality over time as the model sees varied
speed limits at inference — but the training data itself only ever recorded the actual
speed limits from crash records, which SMOTE did not affect.

**Remaining limitations:**
- High-class (Fatal) recall is near-zero regardless of SMOTE strategy given how few
  Fatal crashes exist (224 in 40k rows).
- `speed_limit` in training data is the recorded road speed limit; at inference it is
  estimated from Google traffic categories — a proxy, not the ground truth.
- The model is trained on Boston-area crash data 2015–2024. Predictions outside that
  geography or for post-2024 infrastructure changes may degrade.
- Weather is fetched at the route midpoint only — a single weather call for a 50-mile
  route ignores micro-climate variation.

---

## Versioning

| File | Description |
|------|-------------|
| `best_model.pkl` | v1 (leakage — do not use for live inference) |
| `best_model_v2.pkl` | v2 — clean, forward-knowable features |
| `best_model_v3.pkl` | v3 — SMOTE + per-segment speed limits |
| `feature_list_v2.txt` / `feature_list_v3.txt` | Identical 80-feature list |
