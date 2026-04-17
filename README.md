# Boston Smart Accident Risk and Rerouting — Google Cloud

Uses historical Boston crash data (MassDOT 2015–2024), live weather, and live Google Maps traffic to estimate accident risk on a driving route and recommend safer alternatives. Deployed on Google Cloud Platform.

**Team:** Mohammed Mubashir Uddin Faraz, Sandhia Maheshwari, Himabindu Tummala, Kamal Dalal

---

## Live Service

**Base URL:** `https://boston-accident-risk-api-qzr2qvsfqa-uc.a.run.app`

**API docs (Swagger):** `https://boston-accident-risk-api-qzr2qvsfqa-uc.a.run.app/docs`

---

## Google Cloud Architecture

| Component | Purpose |
|---|---|
| **Cloud Run** | Hosts the FastAPI service (`boston-accident-risk-api`) |
| **Artifact Registry** | Stores the Docker container image (`boston-risk` repo) |
| **Cloud Build** | Builds and pushes the image on each deploy |
| **Secret Manager** | Stores API keys; fetched at runtime by the app via the Python SDK |

Secrets are never injected as environment variables. The app fetches them directly from Secret Manager using `src/secrets.py`, which caches each value in memory after the first fetch.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend map interface |
| `/predict` | POST | Predict accident risk for a route |
| `/predict/segmented` | POST | Per-segment risk along a route |
| `/predict/example` | GET | Hardcoded example response (no API calls) |
| `/crashes` | GET | All crashes, paginated |
| `/crashes/year/{year}` | GET | Filter by year (2015–2024) |
| `/crashes/city/{city}` | GET | Filter by city/town |
| `/crashes/severity/{severity}` | GET | Filter by severity class |
| `/crashes/fatal` | GET | Crashes with at least 1 fatality |
| `/crashes/hotspots` | GET | EMS-flagged hotspot locations |
| `/crashes/filter` | GET | Multi-field filter (year, city, severity, weather) |
| `/stats/by-year` | GET | Crash counts and injuries grouped by year |

### Predict request body

```json
{
  "origin": "Fenway Park, Boston, MA",
  "destination": "Boston Logan International Airport, MA",
  "departure_time": "2024-10-15T08:00:00Z"
}
```

`departure_time` is optional — defaults to now.

---

## Deploying

### Build and push a new image

```bash
gcloud builds submit \
  --project=boston-rerouting \
  --tag=us-central1-docker.pkg.dev/boston-rerouting/boston-risk/boston-accident-risk-api:latest \
  .
```

### Deploy to Cloud Run

```bash
gcloud run services replace cloud-run-service.yaml --region=us-central1
```

---

## Local Development

```bash
pip install -r requirements.txt
```

The app fetches secrets from Google Secret Manager at runtime. For local development, authenticate with Application Default Credentials:

```bash
gcloud auth application-default login
```

Then run:

```bash
uvicorn api:app --reload
```

The app will be available at `http://localhost:8000`.

---

## Secrets

Three secrets must exist in Google Secret Manager under project `boston-rerouting`:

| Secret name | Used for |
|---|---|
| `google-maps-api-key` | Frontend map + reverse geocoding |
| `google-server-api-key` | Routes API + forward geocoding |
| `openweather-api-key` | Live weather conditions |

The Cloud Run service account (`294613088058-compute@developer.gserviceaccount.com`) must have `roles/secretmanager.secretAccessor` on all three.
