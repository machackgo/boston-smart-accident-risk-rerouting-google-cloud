"""
preprocess_v4.py — Feature matrix for v4 model.

Extends v2/v3 feature set with 6 leakage-free spatial aggregate features
and prunes weather one-hot columns whose column sum < 50 (too sparse to learn from).

Input : data/crashes_with_spatial_v4.parquet  (built by spatial_features.py)
Output: (X, y, feature_names) tuple identical in shape convention to v2.
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
SPATIAL_PATH = REPO_ROOT / "data" / "crashes_with_spatial_v4.parquet"

SEVERITY_MAP = {
    "No Injury": "Low",
    "Injury":    "Medium",
    "Fatal":     "High",
}

SPATIAL_COLS = [
    "nearby_crash_count_1km",
    "nearby_fatal_count_1km",
    "nearby_injury_count_1km",
    "nearby_crash_count_500m",
    "nearby_fatal_count_500m",
    "nearby_avg_severity_1km",
]

# Minimum non-zero column sum to keep a weather one-hot column
WEATHER_MIN_SUM = 50


def _light_phase_from_hour(hour: pd.Series) -> pd.Series:
    conditions = [
        (hour < 6) | (hour >= 20),
        hour.isin([6, 19]),
    ]
    choices = ["Dark", "Dawn_Dusk"]
    return pd.Series(
        np.select(conditions, choices, default="Daylight"),
        index=hour.index,
        name="light_phase",
    )


def build_features_v4(df: pd.DataFrame, weather_keep_cols: list | None = None):
    """
    Feature engineering + encoding for v4.

    Args:
        df               : Enriched DataFrame from crashes_with_spatial_v4.parquet.
        weather_keep_cols: If provided, use this fixed list of weather columns
                           (for test-set consistency). If None, determine from data.

    Returns:
        X               : pd.DataFrame
        y               : pd.Series
        feature_names   : list[str]
        weather_keep_cols: list[str]  — pass back to re-use on test set
    """
    original_rows = len(df)
    print(f"\nOriginal row count : {original_rows:,}")

    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    df["severity_class"] = df["severity_3class"].map(SEVERITY_MAP)
    df = df.dropna(subset=["severity_class"])
    print(f"Rows after dropping missing target : {len(df):,}")

    # ── Time features ─────────────────────────────────────────────────────────
    df["crash_datetime_clean"] = pd.to_datetime(
        df["crash_datetime_clean"], utc=True, errors="coerce"
    )
    df["hour_of_day"]  = df["crash_datetime_clean"].dt.hour
    df["day_of_week"]  = df["crash_datetime_clean"].dt.dayofweek
    df["month"]        = df["crash_datetime_clean"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour_of_day"].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0
    )
    df["light_phase"]  = _light_phase_from_hour(df["hour_of_day"])

    # ── Drop rows missing critical features ───────────────────────────────────
    critical = ["lat", "lon", "crash_datetime_clean"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"Rows after dropping missing lat/lon/datetime : {len(df):,}  "
          f"(dropped {before - len(df):,})")

    # Also drop rows missing any spatial feature (shouldn't happen after spatial pass)
    before = len(df)
    df = df.dropna(subset=SPATIAL_COLS)
    if len(df) < before:
        print(f"Rows after dropping missing spatial features : {len(df):,}  "
              f"(dropped {before - len(df):,})")

    # ── Numeric base features ─────────────────────────────────────────────────
    numeric_cols = [
        "lat", "lon", "speed_limit",
        "hour_of_day", "day_of_week", "month",
        "is_weekend", "is_rush_hour",
    ] + SPATIAL_COLS

    feature_df = df[numeric_cols].copy()

    # Fill speed_limit NaN with median
    speed_median = feature_df["speed_limit"].median()
    feature_df["speed_limit"] = feature_df["speed_limit"].fillna(
        speed_median if pd.notna(speed_median) else 0
    )
    for col in numeric_cols:
        if feature_df[col].isna().any():
            med = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(med if pd.notna(med) else 0)

    # ── Weather one-hot encoding + sparse-column pruning ─────────────────────
    weather_dummies = pd.get_dummies(
        df["weath_cond_descr"].fillna("Unknown"),
        prefix="weath_cond_descr",
        drop_first=False,
    ).astype(int)

    # Sanitize column names now so we prune on clean names
    weather_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", c) for c in weather_dummies.columns
    ]

    if weather_keep_cols is None:
        col_sums = weather_dummies.sum()
        dropped_weather = col_sums[col_sums < WEATHER_MIN_SUM].index.tolist()
        weather_keep_cols = col_sums[col_sums >= WEATHER_MIN_SUM].index.tolist()
        print(f"\nWeather columns pruned (sum < {WEATHER_MIN_SUM}):"
              f" {len(dropped_weather)} dropped, {len(weather_keep_cols)} kept")
        if dropped_weather:
            for c in sorted(dropped_weather):
                print(f"    dropped: {c}  (sum={int(col_sums[c])})")
    else:
        # test-set path: align to the training-determined keep list
        missing = [c for c in weather_keep_cols if c not in weather_dummies.columns]
        for m in missing:
            weather_dummies[m] = 0

    weather_dummies = weather_dummies[weather_keep_cols]
    feature_df = pd.concat([feature_df, weather_dummies], axis=1)

    # ── Light-phase one-hot ───────────────────────────────────────────────────
    light_dummies = pd.get_dummies(
        df["light_phase"], prefix="light_phase", drop_first=False
    ).astype(int)
    light_dummies.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", c) for c in light_dummies.columns
    ]
    feature_df = pd.concat([feature_df, light_dummies], axis=1)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    feature_df = feature_df.fillna(0)

    # Sanitize all column names
    feature_df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", c) for c in feature_df.columns
    ]

    # Deduplicate collisions from sanitization
    seen = {}
    clean_cols = []
    for col in feature_df.columns:
        if col in seen:
            seen[col] += 1
            clean_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            clean_cols.append(col)
    feature_df.columns = clean_cols

    X = feature_df.reset_index(drop=True)
    y = df["severity_class"].reset_index(drop=True)

    print(f"\nFinal row count    : {len(X):,}")
    print(f"Final feature count: {X.shape[1]}")

    return X, y, list(X.columns), weather_keep_cols


if __name__ == "__main__":
    df = pd.read_parquet(SPATIAL_PATH)
    X, y, features, _ = build_features_v4(df)
    print(f"\nFeature names (first 20): {features[:20]}")
    print(f"Target distribution:\n{y.value_counts()}")
