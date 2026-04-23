"""
vertex_deploy.py — Phase B2: Create Vertex AI Endpoint and deploy champion v4.

Phase B2: Serving layer deployed, user traffic still on Cloud Run.
  - Registers a NEW Vertex AI model with the custom serving container.
    (Separate from the Phase B1 sklearn-placeholder model.)
  - Creates a Vertex AI Endpoint in us-central1.
  - Deploys the serving model to the endpoint (~5–10 minutes).
  - Writes endpoint details back to champion_manifest.json.
  - Cloud Run API is completely unaffected — no live traffic changes.
  - Phase B3 (Cloud Run calling Vertex AI) is NOT implemented here.

GCS artifacts used by the serving container (loaded at endpoint startup):
  gs://boston-rerouting-data/production/models/best_model_v4.pkl   ← model pkl
  gs://boston-rerouting-data/production/models/thresholds_v4.json  ← thresholds

Versioned artifact directory (for Vertex AI model registration metadata):
  gs://boston-rerouting-data/production/models/v4/

Prerequisites (must be done before running live):
  1. Build and push the serving container image:
       docker build \\
         -t us-central1-docker.pkg.dev/boston-rerouting/boston-risk/accident-risk-serving:v4 \\
         ./serving/
       docker push \\
         us-central1-docker.pkg.dev/boston-rerouting/boston-risk/accident-risk-serving:v4

  2. Vertex AI API already enabled from Phase B1.

  3. Service account 294613088058-compute@developer.gserviceaccount.com already has
     storage.objectViewer on the GCS bucket (set during original Cloud Run setup).

Usage:
  python scripts/vertex_deploy.py            # full Phase B2 deployment
  python scripts/vertex_deploy.py --dry-run  # preview all steps, no writes
  python scripts/vertex_deploy.py --test     # smoke-test the deployed endpoint
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import vertex_helpers as vh
from google.cloud import aiplatform

# ── Config ────────────────────────────────────────────────────────────────────
GCS_BUCKET      = "boston-rerouting-data"
GCS_MANIFEST    = "production/models/champion_manifest.json"

# GCS paths passed to the serving container as env vars at deploy time.
MODEL_BLOB      = "production/models/best_model_v4.pkl"
THRESHOLD_BLOB  = "production/models/thresholds_v4.json"

# Versioned artifact directory URI for Vertex AI model registration metadata.
ARTIFACT_GCS_URI = f"gs://{GCS_BUCKET}/production/models/v4/"

# Vertex AI deployment hardware. n1-standard-2 (2 vCPU / 7.5 GB) is more
# than sufficient for LightGBM inference with 36 features.
MACHINE_TYPE  = "n1-standard-2"
MIN_REPLICAS  = 1
MAX_REPLICAS  = 1

DRY_RUN = "--dry-run" in sys.argv
TEST    = "--test"    in sys.argv


# ── Helpers ───────────────────────────────────────────────────────────────────

def sec(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


def gcs_read_json(client: storage.Client, path: str) -> dict:
    blob = client.bucket(GCS_BUCKET).blob(path)
    if not blob.exists():
        raise FileNotFoundError(f"NOT FOUND: gs://{GCS_BUCKET}/{path}")
    return json.loads(blob.download_as_text())


def gcs_write_json(client: storage.Client, obj: dict, path: str) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    client.bucket(GCS_BUCKET).blob(path).upload_from_string(
        data, content_type="application/json"
    )
    print(f"  → gs://{GCS_BUCKET}/{path}")


# ── Smoke test ────────────────────────────────────────────────────────────────

def run_smoke_test(manifest: dict) -> None:
    """
    Send one prediction to the deployed Vertex AI endpoint and print the result.
    Uses a realistic Boston location during morning rush hour on a clear day.
    All 36 features are set explicitly — no zeros that would mask missing features.
    """
    sec("Smoke test — one prediction to the Vertex AI endpoint")

    endpoint_id = manifest.get("vertex_endpoint_id")
    if not endpoint_id:
        print("  ERROR: vertex_endpoint_id not in manifest. Run deployment first.")
        sys.exit(1)

    # Load the canonical feature list so the test instance is always complete.
    feature_list_path = (
        Path(__file__).resolve().parents[1] / "models" / "feature_list_v4.txt"
    )
    if not feature_list_path.exists():
        print(f"  ERROR: feature list not found at {feature_list_path}")
        sys.exit(1)

    features = [
        line.strip()
        for line in feature_list_path.read_text().splitlines()
        if line.strip()
    ]

    # Start with a zero baseline, then set known-realistic values.
    # Zero is safe for one-hot encoded weather/light columns (all-off is valid).
    instance = {f: 0.0 for f in features}
    instance.update({
        # Location — Boston City Hall area
        "lat":                       42.3601,
        "lon":                       -71.0589,
        "speed_limit":               30.0,
        # Time — Tuesday 9 AM, April, weekday, rush hour
        "hour_of_day":               9.0,
        "day_of_week":               1.0,
        "month":                     4.0,
        "is_weekend":                0.0,
        "is_rush_hour":              1.0,
        # Spatial — moderate nearby-crash history
        "nearby_crash_count_1km":    12.0,
        "nearby_fatal_count_1km":    0.0,
        "nearby_injury_count_1km":   5.0,
        "nearby_crash_count_500m":   3.0,
        "nearby_fatal_count_500m":   0.0,
        "nearby_avg_severity_1km":   1.4,
        # Weather — clear sky
        "weath_cond_descr_Clear":    1.0,
        # Light — daytime
        "light_phase_Daylight":      1.0,
    })

    vh.init_vertex()
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)

    print(f"  endpoint_id : {endpoint_id}")
    print(f"  features    : {len(instance)} / {len(features)}")
    print("  Calling endpoint.predict() ...")
    print()

    response = endpoint.predict(instances=[instance])

    print("  Response:")
    for i, pred in enumerate(response.predictions):
        print(f"    [{i}] risk_class   : {pred.get('risk_class')}")
        print(f"         confidence   : {pred.get('confidence')}")
        print(f"         probabilities: {pred.get('probabilities')}")

    print()
    print("  Smoke test complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = storage.Client()

    if TEST:
        manifest = gcs_read_json(client, GCS_MANIFEST)
        run_smoke_test(manifest)
        return

    print(f"\n{'=' * 62}")
    print(f"  Phase B2 — Vertex AI Endpoint Deployment")
    print(f"{'=' * 62}")
    print(f"  Project        : {vh.GCP_PROJECT}")
    print(f"  Location       : {vh.GCP_LOCATION}")
    print(f"  Serving image  : {vh.SERVING_IMAGE_URI}")
    print(f"  Artifact URI   : {ARTIFACT_GCS_URI}")
    print(f"  Machine type   : {MACHINE_TYPE}  (min={MIN_REPLICAS} / max={MAX_REPLICAS})")
    if DRY_RUN:
        print("  Mode           : DRY RUN — no writes")
    else:
        print("  Mode           : LIVE — deployment takes ~5–10 minutes")

    # ── 1. Read manifest ───────────────────────────────────────────────────────
    sec("Step 1 — Reading champion manifest from GCS")
    manifest = gcs_read_json(client, GCS_MANIFEST)
    print(f"  model_version          : {manifest.get('model_version')}")
    print(f"  vertex_model_id (B1)   : {manifest.get('vertex_model_id', '[not set]')}")
    print(f"  vertex_endpoint_id     : {manifest.get('vertex_endpoint_id', '[not set]')}")

    if manifest.get("vertex_endpoint_id") and not DRY_RUN:
        print(f"\n  WARNING: vertex_endpoint_id already set:")
        print(f"    {manifest['vertex_endpoint_id']}")
        print("  Continuing will create a NEW endpoint and re-deploy.")
        try:
            input("\n  Press Enter to continue, or Ctrl-C to abort: ")
        except KeyboardInterrupt:
            print("\n  Aborted.")
            sys.exit(0)

    # ── 2. Register serving model with custom container ────────────────────────
    sec("Step 2 — Registering serving model with custom container")
    print(f"  display_name   : boston-accident-risk-v4-serving")
    print(f"  serving_image  : {vh.SERVING_IMAGE_URI}")
    print(f"  artifact_uri   : {ARTIFACT_GCS_URI}")
    print(f"  MODEL_BLOB env : gs://{GCS_BUCKET}/{MODEL_BLOB}")
    print(f"  THRESHOLD env  : gs://{GCS_BUCKET}/{THRESHOLD_BLOB}")
    print()
    print("  NOTE: This is a NEW Vertex AI model resource, separate from the Phase B1")
    print("  registry model. Phase B1 used the sklearn placeholder container.")
    print("  Phase B2 uses the custom container from serving/Dockerfile.")

    if DRY_RUN:
        print("\n  [dry-run] Would call vh.register_serving_model()")
    else:
        year_range = manifest.get("year_range", [0, 0])
        serving_model = vh.register_serving_model(
            display_name="boston-accident-risk-v4-serving",
            artifact_gcs_uri=ARTIFACT_GCS_URI,
            serving_image_uri=vh.SERVING_IMAGE_URI,
            gcs_bucket=GCS_BUCKET,
            model_blob=MODEL_BLOB,
            threshold_blob=THRESHOLD_BLOB,
            description=(
                f"Champion v4 with custom serving container + threshold logic. "
                f"Trained on {manifest.get('row_count', '?')} records "
                f"({year_range[0]}–{year_range[-1]}). "
                f"Thresholds: {manifest.get('thresholds')}."
            ),
            labels={
                "model-version": "v4",
                "phase":         "b2",
                "status":        "champion",
                "container":     "custom",
            },
        )
        serving_model_id = serving_model.resource_name
        print(f"\n  serving_model_id : {serving_model_id}")

    # ── 3. Create Vertex AI Endpoint ──────────────────────────────────────────
    sec("Step 3 — Creating Vertex AI Endpoint")
    print("  display_name : boston-accident-risk-endpoint")

    if DRY_RUN:
        print("  [dry-run] Would call vh.create_endpoint()")
    else:
        endpoint = vh.create_endpoint("boston-accident-risk-endpoint")
        endpoint_id = endpoint.resource_name
        print(f"\n  endpoint created : {endpoint_id}")

    # ── 4. Deploy model to endpoint ───────────────────────────────────────────
    sec("Step 4 — Deploying model to endpoint")
    print(f"  machine_type  : {MACHINE_TYPE}")
    print(f"  min_replicas  : {MIN_REPLICAS}")
    print(f"  max_replicas  : {MAX_REPLICAS}")
    print(f"  traffic_pct   : 100  (all endpoint traffic — no live user traffic yet)")
    print()
    print("  The Cloud Run API will not be changed.")
    print("  No live user traffic will reach this endpoint until Phase B3.")

    if DRY_RUN:
        print("\n  [dry-run] Would call vh.deploy_to_endpoint() — takes ~5–10 min live.")
        sec("Dry run complete — no writes performed")
        return

    print("\n  Deploying ... (this blocks until the deployment is fully live)")
    vh.deploy_to_endpoint(
        model=serving_model,
        endpoint=endpoint,
        machine_type=MACHINE_TYPE,
        min_replicas=MIN_REPLICAS,
        max_replicas=MAX_REPLICAS,
    )
    print("\n  Deployment complete.")

    # ── 5. Update champion_manifest.json ─────────────────────────────────────
    sec("Step 5 — Updating champion_manifest.json in GCS")
    manifest["vertex_serving_model_id"]  = serving_model_id
    manifest["vertex_endpoint_id"]       = endpoint_id
    manifest["vertex_endpoint_deployed"] = True
    manifest["vertex_b2_completed_at"]   = datetime.now(timezone.utc).isoformat()

    gcs_write_json(client, manifest, GCS_MANIFEST)
    print(f"  Fields written:")
    print(f"    vertex_serving_model_id  : {serving_model_id}")
    print(f"    vertex_endpoint_id       : {endpoint_id}")
    print(f"    vertex_b2_completed_at   : {manifest['vertex_b2_completed_at']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    sec("Phase B2 complete")
    print("  Vertex AI endpoint is live with the champion v4 model.")
    print("  Cloud Run API is completely unchanged.")
    print("  No live user traffic has been routed to the Vertex AI endpoint.")
    print()
    print("  What was written:")
    print(f"    Serving model  : {serving_model_id}")
    print(f"    Endpoint       : {endpoint_id}")
    print(f"    GCS manifest   : gs://{GCS_BUCKET}/{GCS_MANIFEST}")
    print()
    print("  Smoke test (verify the endpoint works):")
    print("    python scripts/vertex_deploy.py --test")
    print()
    print("  View in Cloud Console:")
    print(f"    https://console.cloud.google.com/vertex-ai/endpoints?project={vh.GCP_PROJECT}")
    print()
    print("  Next step (Phase B3): add Vertex AI inference path to Cloud Run")
    print("  with a local-model fallback. Phase B3 is NOT implemented here.")
    print()


if __name__ == "__main__":
    main()
