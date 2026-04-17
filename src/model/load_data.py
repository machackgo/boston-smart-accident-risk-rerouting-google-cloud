"""
load_data.py
Fetches all crashes from the live API with pagination.
Caches to data/crashes_cache.parquet to avoid re-hitting the API on every run.
"""

import os
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://boston-smart-accident-risk-rerouting.onrender.com"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "crashes_cache.parquet"


def fetch_all_crashes(limit_per_page: int = 1000, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches all crash records from the API using pagination.
    Returns a DataFrame. Uses parquet cache if available.
    """
    if CACHE_PATH.exists() and not force_refresh:
        print(f"Loading from cache: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        print(f"Loaded {len(df):,} rows from cache.")
        return df

    print("Cache not found. Fetching from API (first request may take 30-60s)...")
    all_records = []
    offset = 0

    while True:
        params = {"limit": limit_per_page, "offset": offset}
        resp = requests.get(f"{BASE_URL}/crashes", params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("data", [])
        if not records:
            break

        all_records.extend(records)
        fetched = len(all_records)
        print(f"  Fetched {fetched:,} records so far...", end="\r")

        # If fewer records than limit_per_page, we've hit the last page
        if len(records) < limit_per_page:
            break

        offset += limit_per_page

    print(f"\nTotal records fetched: {len(all_records):,}")
    df = pd.DataFrame(all_records)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"Cached to {CACHE_PATH}")

    return df


if __name__ == "__main__":
    df = fetch_all_crashes()
    print(df.shape)
    print(df.dtypes)
