"""
Boston Smart Accident Risk API - Quick Start Example
=====================================================
API Base URL: https://boston-accident-risk-api-294613088058.us-central1.run.app
Swagger Docs: https://boston-accident-risk-api-294613088058.us-central1.run.app/docs

Note: The first request may take ~15 seconds on cold start (Cloud Run scales to zero when idle).

Install dependencies:
    pip install requests pandas
"""

import requests
import pandas as pd

BASE = "https://boston-accident-risk-api-294613088058.us-central1.run.app"


def fetch(endpoint, params=None):
    """Helper: GET request, returns parsed JSON."""
    url = f"{BASE}{endpoint}"
    print(f"Fetching {url} ...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# ── 1. Fatal crashes ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. Fatal Crashes (limit=10)")
print("=" * 60)

result = fetch("/crashes/fatal", params={"limit": 10})
fatal_df = pd.DataFrame(result["data"])
print(f"Records returned: {result['total_returned']}")
print(fatal_df[["year", "city_town_name", "crash_severity_descr", "weath_cond_descr"]].to_string(index=False))

# ── 2. Crashes in 2020 with rain ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Crashes in 2020 with Rain (limit=10)")
print("=" * 60)

result = fetch("/crashes/filter", params={"year": 2020, "weather": "Rain", "limit": 10})
rain_df = pd.DataFrame(result["data"])
print(f"Records returned: {result['total_returned']}")
print(rain_df[["year", "city_town_name", "weath_cond_descr", "crash_severity_descr"]].to_string(index=False))

# ── 3. Yearly stats ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Crash Stats by Year")
print("=" * 60)

result = fetch("/stats/by-year")
stats_df = pd.DataFrame(result["data"])
print(stats_df.to_string(index=False))
