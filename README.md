# Boston Smart Accident Risk and Rerouting System

Team project for Intro to Data Science.

## Team Members
- Mohammed Mubashir Uddin Faraz
- Sandhia Maheshwari
- Himabindu Tummala
- Kamal Dalal

## Project Idea
This project uses historical Boston accident data, live weather, live traffic conditions, and route information to estimate accident/disruption risk and recommend alternate routes.

## Dataset API

We uploaded the full Boston crash dataset (47,000+ rows, 2015–2024) from MassDOT to a live REST API — no CSV download needed. The API is built with **FastAPI**, hosted on **Render**, and backed by **Supabase** as the database.

**Live API base URL:** https://boston-smart-accident-risk-rerouting.onrender.com

**Swagger docs (browser):** https://boston-smart-accident-risk-rerouting.onrender.com/docs

> **Note:** The first request may take **30–60 seconds** due to Render's free tier cold start. Subsequent requests will be fast.

### Available Endpoints

| Endpoint | Description |
|---|---|
| `GET /crashes` | All crashes (paginated) |
| `GET /crashes/year/{year}` | Filter by year (2015–2024) |
| `GET /crashes/city/{city}` | Filter by city/town name |
| `GET /crashes/severity/{severity}` | Filter by severity class |
| `GET /crashes/fatal` | Crashes with at least 1 fatality |
| `GET /crashes/hotspots` | EMS-flagged crash hotspot locations |
| `GET /crashes/filter` | Multi-field filter (year, city, severity, weather) |
| `GET /stats/by-year` | Aggregated crash counts and injuries per year |

### Response Shape

All endpoints return a JSON object like:

```json
{
  "total_returned": 2,
  "data": [ { ...crash fields... }, { ...crash fields... } ]
}
```

Access the records via `response.json()["data"]`.

---

## Quick Start

### Prerequisites

```bash
pip install requests pandas
```

### Python Example

```python
import requests
import pandas as pd

BASE = "https://boston-smart-accident-risk-rerouting.onrender.com/"
# Note: first request may take 30-60 seconds (Render free tier cold start)

# --- Fatal crashes ---
response = requests.get(f"{BASE}/crashes/fatal", params={"limit": 50})
response.raise_for_status()
fatal_df = pd.DataFrame(response.json()["data"])
print(f"Fatal crashes returned: {len(fatal_df)}")
print(fatal_df[["year", "city_town_name", "crash_severity_descr", "weath_cond_descr"]].head())

# --- Crashes in 2020 with rain ---
response = requests.get(f"{BASE}/crashes/filter", params={
    "year": 2020,
    "weather": "Rain",
    "limit": 100
})
response.raise_for_status()
rain_df = pd.DataFrame(response.json()["data"])
print(f"\nRainy crashes in 2020 returned: {len(rain_df)}")
print(rain_df[["year", "city_town_name", "weath_cond_descr", "crash_severity_descr"]].head())
```

A runnable version of this code is in [`examples/quickstart.py`](examples/quickstart.py).

For browser-based exploration, open the [Swagger docs](https://boston-smart-accident-risk-rerouting.onrender.com/docs).

---

No CSV download required — query only what you need, directly into a DataFrame.
