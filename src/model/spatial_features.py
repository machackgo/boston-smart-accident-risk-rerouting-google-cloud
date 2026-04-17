"""
spatial_features.py — Leakage-free neighborhood-level historical risk features.

For each crash record, engineers 6 features derived from neighboring crashes:
    nearby_crash_count_1km   — total crashes within 1 km (excl. self)
    nearby_fatal_count_1km   — Fatal crashes within 1 km
    nearby_injury_count_1km  — Injury crashes within 1 km
    nearby_crash_count_500m  — total crashes within 500 m
    nearby_fatal_count_500m  — Fatal crashes within 500 m
    nearby_avg_severity_1km  — mean severity score within 1 km (Low=0, Med=1, High=2)

Leakage rules:
    - Training rows: each row queries only OTHER training rows (self excluded via
      leave-one-out: build the tree from all training rows, query with k+1,
      drop the closest hit which is the point itself).
    - Test rows: query the training-set tree only (no test-set crashes included).

Uses scipy BallTree with haversine metric for exact great-circle distance.

Usage (as a script):
    python src/model/spatial_features.py
    → writes data/crashes_with_spatial_v4.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# BallTree lives in sklearn.neighbors — avoids a separate scipy install
from sklearn.neighbors import BallTree

REPO_ROOT  = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "crashes_cache.parquet"
OUT_PATH   = REPO_ROOT / "data" / "crashes_with_spatial_v4.parquet"

SEVERITY_MAP = {
    "No Injury": "Low",
    "Injury":    "Medium",
    "Fatal":     "High",
}
SEVERITY_SCORE = {"Low": 0.0, "Medium": 1.0, "High": 2.0}

RADIUS_1KM  = 1000.0   # metres
RADIUS_500M =  500.0   # metres
EARTH_RADIUS_M = 6_371_009.0   # metres (WGS-84 mean)


def _radians_coords(df: pd.DataFrame) -> np.ndarray:
    """Return an (N, 2) array of [lat_rad, lng_rad]."""
    return np.radians(df[["lat", "lon"]].values.astype(np.float64))


def compute_spatial_features(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    is_train: bool,
) -> pd.DataFrame:
    """
    Compute spatial aggregate features for every row in query_df.

    Args:
        train_df  : DataFrame with at least lat, lon, severity_class columns.
                    This is the reference set — the BallTree is built from it.
        query_df  : DataFrame to compute features for.
        is_train  : If True, each query row is assumed to exist in train_df
                    (leave-one-out: the closest returned neighbour is self,
                    so we drop it and use the remaining hits).
                    If False, query_df is a held-out set — all returned
                    neighbours are valid.

    Returns:
        pd.DataFrame with 6 new columns, index matching query_df.index.
    """
    train_coords = _radians_coords(train_df)
    query_coords = _radians_coords(query_df)

    severity_array  = train_df["severity_class"].values
    is_fatal_array  = (severity_array == "High").astype(np.float32)
    is_injury_array = (severity_array == "Medium").astype(np.float32)
    score_array     = np.array(
        [SEVERITY_SCORE.get(s, 0.0) for s in severity_array], dtype=np.float32
    )

    r1km  = RADIUS_1KM  / EARTH_RADIUS_M   # radians
    r500m = RADIUS_500M / EARTH_RADIUS_M

    tree = BallTree(train_coords, metric="haversine")

    # --- 1 km radius query ---
    idx_1km  = tree.query_radius(query_coords, r=r1km,  return_distance=False)
    # --- 500 m radius query ---
    idx_500m = tree.query_radius(query_coords, r=r500m, return_distance=False)

    n = len(query_df)
    nearby_crash_count_1km   = np.zeros(n, dtype=np.float32)
    nearby_fatal_count_1km   = np.zeros(n, dtype=np.float32)
    nearby_injury_count_1km  = np.zeros(n, dtype=np.float32)
    nearby_crash_count_500m  = np.zeros(n, dtype=np.float32)
    nearby_fatal_count_500m  = np.zeros(n, dtype=np.float32)
    nearby_avg_severity_1km  = np.zeros(n, dtype=np.float32)

    for i, (hits_1km, hits_500m) in enumerate(zip(idx_1km, idx_500m)):
        if is_train:
            # drop self — the exact duplicate closest hit
            # (since query row IS in the tree, the first sorted hit is self)
            hits_1km  = hits_1km[hits_1km != i]   # drop index equal to self-row
            hits_500m = hits_500m[hits_500m != i]

        n1  = len(hits_1km)
        n5  = len(hits_500m)

        nearby_crash_count_1km[i]  = n1
        nearby_crash_count_500m[i] = n5

        if n1 > 0:
            nearby_fatal_count_1km[i]  = is_fatal_array[hits_1km].sum()
            nearby_injury_count_1km[i] = is_injury_array[hits_1km].sum()
            nearby_avg_severity_1km[i] = score_array[hits_1km].mean()
        if n5 > 0:
            nearby_fatal_count_500m[i] = is_fatal_array[hits_500m].sum()

    return pd.DataFrame(
        {
            "nearby_crash_count_1km":  nearby_crash_count_1km,
            "nearby_fatal_count_1km":  nearby_fatal_count_1km,
            "nearby_injury_count_1km": nearby_injury_count_1km,
            "nearby_crash_count_500m": nearby_crash_count_500m,
            "nearby_fatal_count_500m": nearby_fatal_count_500m,
            "nearby_avg_severity_1km": nearby_avg_severity_1km,
        },
        index=query_df.index,
    )


def build_spatial_parquet(random_state: int = 42) -> pd.DataFrame:
    """
    Load crash cache, compute spatial features, and save enriched parquet.

    Returns the enriched DataFrame.
    """
    print("=" * 60)
    print("  Loading crash cache ...")
    print("=" * 60)
    df = pd.read_parquet(CACHE_PATH)
    print(f"  Loaded {len(df):,} rows")

    # ── Minimum required columns ──────────────────────────────────────────────
    df = df.copy()
    df["severity_class"] = df["severity_3class"].map(SEVERITY_MAP)
    df = df.dropna(subset=["severity_class", "lat", "lon"])
    df = df.reset_index(drop=True)
    print(f"  Rows after severity + lat/lon filter: {len(df):,}")

    # ── Stratified 80/20 split — same seed as training pipeline ──────────────
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=random_state,
        stratify=df["severity_class"],
    )
    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # ── Compute spatial features ──────────────────────────────────────────────
    print("\n  Computing spatial features for TRAINING set (leave-one-out) ...")
    train_spatial = compute_spatial_features(train_df, train_df, is_train=True)

    print("  Computing spatial features for TEST set (training-set tree only) ...")
    test_spatial  = compute_spatial_features(train_df, test_df,  is_train=False)

    # ── Attach to original rows ───────────────────────────────────────────────
    spatial_all = pd.concat([train_spatial, test_spatial]).sort_index()
    enriched_df = pd.concat([df, spatial_all], axis=1)

    # ── Stats & correlation ───────────────────────────────────────────────────
    spatial_cols = list(train_spatial.columns)
    severity_score_series = enriched_df["severity_class"].map(SEVERITY_SCORE)

    print("\n" + "=" * 60)
    print("  Spatial feature summary stats")
    print("=" * 60)
    print(f"  {'Feature':<30s}  {'Mean':>8}  {'Min':>6}  {'Max':>7}  {'Corr w/ severity':>16}")
    for col in spatial_cols:
        vals = enriched_df[col]
        corr = vals.corr(severity_score_series)
        print(f"  {col:<30s}  {vals.mean():>8.2f}  {vals.min():>6.1f}  {vals.max():>7.1f}  {corr:>16.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    enriched_df.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved enriched parquet → {OUT_PATH}")
    print(f"  Shape: {enriched_df.shape}")
    return enriched_df


if __name__ == "__main__":
    build_spatial_parquet()
