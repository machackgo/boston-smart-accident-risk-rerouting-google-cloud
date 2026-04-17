"""
preprocess.py
Loads cached crash data, engineers features, encodes categoricals,
and creates a 3-class severity target: Low / Medium / High.
"""

import pandas as pd
import numpy as np
from load_data import fetch_all_crashes


# Severity mapping based on crash_severity_descr / severity_3class values
SEVERITY_MAP = {
    "No Injury":  "Low",
    "Injury":     "Medium",
    "Fatal":      "High",
}


def build_features(df: pd.DataFrame):
    """
    Feature engineering + encoding.
    Returns X (DataFrame), y (Series), feature_names (list).
    """

    # ── Target ────────────────────────────────────────────────────────────────
    print("\n=== Raw severity_3class value counts ===")
    print(df["severity_3class"].value_counts(dropna=False))

    df = df.copy()
    df["severity_class"] = df["severity_3class"].map(SEVERITY_MAP)

    print("\n=== Mapped severity_class distribution ===")
    print(df["severity_class"].value_counts(dropna=False))

    # Drop rows with missing target
    df = df.dropna(subset=["severity_class"])
    print(f"\nRows after dropping missing target: {len(df):,}")

    # ── Time features ─────────────────────────────────────────────────────────
    df["crash_datetime_clean"] = pd.to_datetime(
        df["crash_datetime_clean"], utc=True, errors="coerce"
    )
    df["hour_of_day"]  = df["crash_datetime_clean"].dt.hour
    df["day_of_week"]  = df["crash_datetime_clean"].dt.dayofweek   # 0=Mon
    df["month"]        = df["crash_datetime_clean"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour_of_day"].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 18) else 0
    )

    # ── Numeric features ──────────────────────────────────────────────────────
    numeric_cols = [
        "speed_limit",
        "numb_vehc",
        "ems_hotspot_flag",
        "ems_ped_hotspot_flag",
        "ems_peak_hour",
        "district_num",
        "lat",
        "lon",
    ]

    # ── Categorical features ──────────────────────────────────────────────────
    categorical_cols = [
        "weath_cond_descr",
        "road_surf_cond_descr",
        "ambnt_light_descr",
        "manr_coll_descr",
        "rdwy_jnct_type_descr",
        "city_town_name",
    ]

    # Keep only columns that exist in the DataFrame
    numeric_cols     = [c for c in numeric_cols     if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    time_cols = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour"]

    feature_df = df[numeric_cols + time_cols].copy()

    # Fill all numeric / time NaNs with median (covers unparseable datetimes)
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            med = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(med if pd.notna(med) else 0)

    # One-hot encode categoricals
    for col in categorical_cols:
        dummies = pd.get_dummies(
            df[col].fillna("Unknown"), prefix=col, drop_first=False
        )
        # Ensure bool columns become int
        dummies = dummies.astype(int)
        feature_df = pd.concat([feature_df, dummies], axis=1)

    # Final safety: fill any remaining NaNs with 0
    feature_df = feature_df.fillna(0)

    # Sanitize column names — LightGBM rejects special JSON characters
    import re
    feature_df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", col) for col in feature_df.columns
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
    y = df["severity_class"]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y


if __name__ == "__main__":
    df = fetch_all_crashes()
    X, y = build_features(df)
    print("\nSample features:")
    print(X.head(3))
