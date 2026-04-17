"""
preprocess_v2.py
Loads cached crash data and builds a feature matrix using ONLY
forward-knowable (pre-event) features — i.e., features that would
be available at live prediction time before a crash occurs.

Design motivation:
    v1 used post-crash signals (manner of collision, road surface condition,
    ambient light description, number of vehicles) that are recorded AFTER the
    crash and are therefore unavailable at inference time. v2 corrects this.

Forward-knowable features kept:
    - Location   : lat, lon
    - Time       : hour_of_day, day_of_week, month, is_weekend, is_rush_hour
    - Road prop  : speed_limit
    - Weather    : weath_cond_descr (historical weather at crash time — the
                   driver could observe this before any collision)
    - Light proxy: is_dark_hours — engineered from hour_of_day, NOT from the
                   post-crash ambnt_light_descr field

Features explicitly dropped (post-crash / not knowable in advance):
    - manr_coll_descr       (recorded after collision)
    - road_surf_cond_descr  (recorded on-scene; weather used as proxy instead)
    - ambnt_light_descr     (recorded on-scene; hour-of-day used as proxy)
    - numb_vehc             (number of vehicles involved — a crash outcome)
    - rdwy_jnct_type_descr  (recorded on-scene)
    - ems_hotspot_flag / ems_ped_hotspot_flag / ems_peak_hour
                            (EMS-derived, only available for ~73% of rows)
    - district_num          (not in the keep list; partially covered by lat/lon)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "crashes_cache.parquet"

SEVERITY_MAP = {
    "No Injury": "Low",
    "Injury":    "Medium",
    "Fatal":     "High",
}


def _light_phase_from_hour(hour: pd.Series) -> pd.Series:
    """
    Engineer a light-phase category from hour of day.
      Dark      : 00:00–05:59 and 20:00–23:59
      Dawn_Dusk : 06:00–06:59 and 19:00–19:59
      Daylight  : 07:00–18:59
    Returns a string Series suitable for one-hot encoding.
    """
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


def build_features_v2(df: pd.DataFrame):
    """
    Feature engineering + encoding for v2 (forward-knowable features only).

    Returns:
        X            : pd.DataFrame  — feature matrix
        y            : pd.Series     — target (Low / Medium / High)
        feature_names: list[str]     — ordered feature names matching X.columns
    """
    original_rows = len(df)
    print(f"\nOriginal row count : {original_rows:,}")

    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    df["severity_class"] = df["severity_3class"].map(SEVERITY_MAP)
    df = df.dropna(subset=["severity_class"])
    print(f"Rows after dropping missing target : {len(df):,}")

    # ── Time features (from crash_datetime_clean) ─────────────────────────────
    df["crash_datetime_clean"] = pd.to_datetime(
        df["crash_datetime_clean"], utc=True, errors="coerce"
    )
    df["hour_of_day"]  = df["crash_datetime_clean"].dt.hour
    df["day_of_week"]  = df["crash_datetime_clean"].dt.dayofweek   # 0=Mon
    df["month"]        = df["crash_datetime_clean"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour_of_day"].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0
    )

    # ── Light proxy from hour (NOT ambnt_light_descr) ─────────────────────────
    df["light_phase"] = _light_phase_from_hour(df["hour_of_day"])

    # ── Drop rows missing critical forward-knowable features ──────────────────
    critical = ["lat", "lon", "crash_datetime_clean"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"Rows after dropping missing lat/lon/datetime : {len(df):,}  "
          f"(dropped {before - len(df):,})")

    # ── Numeric features ──────────────────────────────────────────────────────
    numeric_cols = ["lat", "lon", "speed_limit",
                    "hour_of_day", "day_of_week", "month",
                    "is_weekend", "is_rush_hour"]

    feature_df = df[numeric_cols].copy()

    # Fill speed_limit NaN with median (road property; ~22% missing)
    speed_median = feature_df["speed_limit"].median()
    feature_df["speed_limit"] = feature_df["speed_limit"].fillna(
        speed_median if pd.notna(speed_median) else 0
    )

    # Fill remaining numeric NaNs with median (covers unparseable datetimes)
    for col in numeric_cols:
        if feature_df[col].isna().any():
            med = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(med if pd.notna(med) else 0)

    # ── Categorical features — one-hot encode ─────────────────────────────────
    for col, series in [
        ("weath_cond_descr", df["weath_cond_descr"].fillna("Unknown")),
        ("light_phase",      df["light_phase"]),
    ]:
        dummies = pd.get_dummies(series, prefix=col, drop_first=False).astype(int)
        feature_df = pd.concat([feature_df, dummies], axis=1)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    feature_df = feature_df.fillna(0)

    # Sanitize column names (LightGBM rejects special JSON characters)
    feature_df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", c) for c in feature_df.columns
    ]

    # Deduplicate any collisions from sanitization
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

    X = feature_df
    y = df["severity_class"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    print(f"\nFinal row count    : {len(X):,}")
    print(f"Final feature count: {X.shape[1]}")
    print(f"\nAll feature names:")
    for i, name in enumerate(X.columns):
        print(f"  [{i:03d}] {name}")
    print(f"\nTarget class distribution:")
    print(y.value_counts().to_string())

    return X, y, list(X.columns)


if __name__ == "__main__":
    df = pd.read_parquet(CACHE_PATH)
    X, y, features = build_features_v2(df)
    print("\nSample features (first 3 rows):")
    print(X.head(3).to_string())
