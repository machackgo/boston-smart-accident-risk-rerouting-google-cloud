"""
vertex_register.py — One-time Vertex AI Model Registry registration for champion v4.

Phase B1: Registry only.
  - No Vertex AI endpoint is created.
  - No serving traffic is changed.
  - The live Cloud Run API is completely unaffected.
  - All writes go to GCS (artifact copy + manifest update) and Vertex AI Model Registry.

What this script does, in order:
  1. Reads production/models/champion_manifest.json from GCS.
  2. Copies the four champion artifact files into a complete versioned GCS path:
         gs://boston-rerouting-data/production/models/v4/
     This path is kept for our own reference and Cloud Run use.
  3. Creates a separate Vertex AI-compatible artifact path:
         gs://boston-rerouting-data/production/models/v4-vertex/
     This path contains exactly ONE file named model.pkl (required by the pre-built
     sklearn container used as the registry serving spec). The file is a copy of
     best_model_v4.pkl, renamed. No other files go here.
  4. Calls aiplatform.Model.upload() against the v4-vertex/ path.
     This takes ~30–60 seconds and returns a vertex_model_id.
  5. Prints the vertex_model_id to stdout.
  6. Writes vertex_model_id (plus artifact URI and registration timestamp) back
     into production/models/champion_manifest.json on GCS.

Why two separate GCS paths?
  production/models/v4/         — complete versioned snapshot (4 files, original names)
                                  used by Cloud Run and our own scripts
  production/models/v4-vertex/  — Vertex AI-compatible artifact directory (1 file: model.pkl)
                                  the sklearn container validates that exactly one file named
                                  model.pkl or model.joblib exists; extra files cause rejection

GCS source artifacts (already exist from champion seeding):
  gs://boston-rerouting-data/production/models/best_model_v4.pkl
  gs://boston-rerouting-data/production/models/feature_list_v4.txt
  gs://boston-rerouting-data/production/models/thresholds_v4.json
  gs://boston-rerouting-data/production/models/weather_keep_cols_v4.json

Versioned GCS destination — complete copy (unchanged by this fix):
  gs://boston-rerouting-data/production/models/v4/best_model_v4.pkl
  gs://boston-rerouting-data/production/models/v4/feature_list_v4.txt
  gs://boston-rerouting-data/production/models/v4/thresholds_v4.json
  gs://boston-rerouting-data/production/models/v4/weather_keep_cols_v4.json

Vertex AI-compatible artifact path (created by this fix — single file, renamed):
  gs://boston-rerouting-data/production/models/v4-vertex/model.pkl

Usage:
  python scripts/vertex_register.py            # live run
  python scripts/vertex_register.py --dry-run  # preview, no writes

Prerequisites:
  pip install google-cloud-aiplatform google-cloud-storage
  gcloud auth application-default login
  gcloud services enable aiplatform.googleapis.com --project=boston-rerouting
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import vertex_helpers as vh

GCS_BUCKET          = "boston-rerouting-data"
GCS_MANIFEST        = "production/models/champion_manifest.json"
GCS_SRC_PREFIX      = "production/models"
GCS_V4_PREFIX       = "production/models/v4"
# Separate path with exactly one file (model.pkl) for the pre-built sklearn container.
# The sklearn container validates that only model.pkl or model.joblib is present.
GCS_V4_VERTEX_PREFIX = "production/models/v4-vertex"

# The four artifact files to version into production/models/v4/.
# champion_manifest.json is NOT copied — it is metadata, not a model artifact.
ARTIFACT_FILES = [
    "best_model_v4.pkl",
    "feature_list_v4.txt",
    "thresholds_v4.json",
    "weather_keep_cols_v4.json",
]

DRY_RUN = "--dry-run" in sys.argv


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


# ── Step 2a: Copy all artifacts to complete versioned path ────────────────────

def copy_artifacts_to_v4(client: storage.Client) -> None:
    """
    GCS-to-GCS copy of the four champion artifacts into production/models/v4/.
    This is the complete versioned snapshot used by Cloud Run and our own scripts.
    Skips files that already exist at the destination — safe to re-run.
    """
    bucket = client.bucket(GCS_BUCKET)

    for filename in ARTIFACT_FILES:
        src_path = f"{GCS_SRC_PREFIX}/{filename}"
        dst_path = f"{GCS_V4_PREFIX}/{filename}"
        dst_blob = bucket.blob(dst_path)

        if dst_blob.exists():
            print(f"  [skip] already exists: gs://{GCS_BUCKET}/{dst_path}")
            continue

        if DRY_RUN:
            print(f"  [dry-run] would copy: {src_path} → {dst_path}")
            continue

        src_blob = bucket.blob(src_path)
        bucket.copy_blob(src_blob, bucket, dst_path)
        dst_blob.reload()
        size_kb = round(dst_blob.size / 1024, 1) if dst_blob.size else "?"
        print(f"  copied  : gs://{GCS_BUCKET}/{dst_path}  ({size_kb} KB)")


# ── Step 2b: Prepare the Vertex AI-compatible artifact path ───────────────────

def prepare_vertex_artifact(client: storage.Client) -> str:
    """
    Create gs://boston-rerouting-data/production/models/v4-vertex/ containing
    exactly one file: model.pkl (a copy of best_model_v4.pkl, renamed).

    The pre-built sklearn container used as the Vertex AI registry serving spec
    validates that the artifact directory contains exactly one file named
    model.pkl or model.joblib. Any other files — or any other name — cause
    registration to fail with 'expected to contain exactly one of: [model.pkl, ...]'.

    This path is used ONLY for Vertex AI Model Registry registration.
    The complete production artifact folder (v4/) is unaffected.

    Returns the gs:// URI of the Vertex-compatible artifact directory.
    """
    bucket   = client.bucket(GCS_BUCKET)
    src_path = f"{GCS_SRC_PREFIX}/best_model_v4.pkl"
    dst_path = f"{GCS_V4_VERTEX_PREFIX}/model.pkl"
    vertex_uri = f"gs://{GCS_BUCKET}/{GCS_V4_VERTEX_PREFIX}/"

    dst_blob = bucket.blob(dst_path)
    if dst_blob.exists():
        print(f"  [skip] already exists: gs://{GCS_BUCKET}/{dst_path}")
        return vertex_uri

    if DRY_RUN:
        print(f"  [dry-run] would copy and rename:")
        print(f"    {src_path}")
        print(f"    → {dst_path}  (renamed to model.pkl for sklearn container)")
        return vertex_uri

    src_blob = bucket.blob(src_path)
    bucket.copy_blob(src_blob, bucket, dst_path)
    dst_blob.reload()
    size_kb = round(dst_blob.size / 1024, 1) if dst_blob.size else "?"
    print(f"  copied  : gs://{GCS_BUCKET}/{dst_path}  ({size_kb} KB)")
    print(f"  (best_model_v4.pkl → model.pkl, sklearn container naming convention)")

    return vertex_uri


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'=' * 62}")
    print(f"  Phase B1 — Vertex AI Model Registry Registration")
    print(f"{'=' * 62}")
    print(f"  Project   : {vh.GCP_PROJECT}")
    print(f"  Location  : {vh.GCP_LOCATION}")
    print(f"  Bucket    : gs://{GCS_BUCKET}")
    if DRY_RUN:
        print("  Mode      : DRY RUN — no writes to GCS or Vertex AI")
    else:
        print("  Mode      : LIVE — will write to GCS and Vertex AI")

    client = storage.Client()

    # ── 1. Read champion manifest ──────────────────────────────────────────────
    sec("Step 1 — Reading champion manifest from GCS")
    manifest = gcs_read_json(client, GCS_MANIFEST)

    print(f"  model_version  : {manifest.get('model_version')}")
    print(f"  training_date  : {manifest.get('training_date')}")
    print(f"  year_range     : {manifest.get('year_range')}")
    print(f"  row_count      : {manifest.get('row_count'):,}" if isinstance(manifest.get('row_count'), int)
          else f"  row_count      : {manifest.get('row_count')}")
    print(f"  feature_count  : {manifest.get('feature_count')}")
    print(f"  algorithm      : {manifest.get('algorithm')}")

    existing_vertex_id = manifest.get("vertex_model_id")
    if existing_vertex_id:
        print(f"\n  WARNING: vertex_model_id already set in this manifest:")
        print(f"    {existing_vertex_id}")
        print("  Continuing will register a NEW version in Vertex AI Model Registry.")
        print("  If this is unintentional, press Ctrl-C now.")
        if not DRY_RUN:
            try:
                input("\n  Press Enter to continue, or Ctrl-C to abort: ")
            except KeyboardInterrupt:
                print("\n  Aborted — manifest unchanged.")
                sys.exit(0)

    # ── 2a. Copy all artifacts to complete versioned path (v4/) ──────────────────
    sec(f"Step 2a — Complete versioned copy → gs://{GCS_BUCKET}/{GCS_V4_PREFIX}/")
    print("  Source files:")
    for f in ARTIFACT_FILES:
        print(f"    gs://{GCS_BUCKET}/{GCS_SRC_PREFIX}/{f}")
    print()
    copy_artifacts_to_v4(client)

    # ── 2b. Prepare the Vertex AI-compatible artifact path (v4-vertex/) ─────────
    sec(f"Step 2b — Vertex AI artifact → gs://{GCS_BUCKET}/{GCS_V4_VERTEX_PREFIX}/")
    print("  Copies only best_model_v4.pkl, renamed to model.pkl.")
    print("  The sklearn container requires exactly one file named model.pkl.")
    print("  The original v4/ folder is untouched.")
    print()
    artifact_uri = prepare_vertex_artifact(client)
    print(f"\n  Vertex AI artifact URI: {artifact_uri}")

    # ── 3. Register in Vertex AI Model Registry ────────────────────────────────
    sec("Step 3 — Registering in Vertex AI Model Registry")

    if DRY_RUN:
        print("  [dry-run] Would call aiplatform.Model.upload() with:")
        print(f"    display_name      : boston-accident-risk-v4")
        print(f"    artifact_uri      : {artifact_uri}")
        print(f"    serving_container : {vh._SKLEARN_CONTAINER}")
        print(f"    labels            : model_version=v4, framework=lightgbm, status=champion")
        print()
        print("  [dry-run] Would update champion_manifest.json with vertex_model_id.")
        sec("Dry run complete — no writes performed")
        return

    year_range = manifest.get("year_range", [0, 0])
    thresholds = manifest.get("thresholds", {})

    print("  Calling aiplatform.Model.upload() — this takes ~30–60 seconds ...")
    print(f"  serving_container : {vh._SKLEARN_CONTAINER}")
    print("  NOTE: This container is a registry placeholder.")
    print("        Phase B2 will deploy using a custom container with threshold logic.")
    print()

    model = vh.register_model(
        display_name="boston-accident-risk-v4",
        artifact_gcs_uri=artifact_uri,
        description=(
            f"Champion LightGBM v4 accident risk classifier. "
            f"Trained on {manifest.get('row_count', '?')} Boston crash records "
            f"({year_range[0] if year_range else '?'}–"
            f"{year_range[-1] if year_range else '?'}). "
            f"{manifest.get('feature_count', '?')} features including 6 leakage-free "
            f"spatial aggregates (BallTree haversine). "
            f"Per-class decision thresholds: "
            f"High={thresholds.get('High')}, "
            f"Medium={thresholds.get('Medium')}, "
            f"Low={thresholds.get('Low')}."
        ),
        labels={
            "model-version": "v4",
            "framework":     "lightgbm",
            "status":        "champion",
            "year-min":      str(year_range[0] if year_range else "0"),
            "year-max":      str(year_range[-1] if year_range else "0"),
        },
    )

    vertex_model_id    = model.resource_name   # full URI
    vertex_model_short = model.name            # numeric ID only

    # ── 4. Print results ───────────────────────────────────────────────────────
    sec("Step 4 — Registration results")
    print(f"  vertex_model_id   : {vertex_model_id}")
    print(f"  short model ID    : {vertex_model_short}")
    print(f"  display_name      : {model.display_name}")
    print(f"  artifact_uri      : {artifact_uri}")
    print(f"\n  View in Cloud Console:")
    print(f"    https://console.cloud.google.com/vertex-ai/models?project={vh.GCP_PROJECT}")

    # ── 5. Write vertex_model_id back to champion_manifest.json ───────────────
    sec("Step 5 — Updating champion_manifest.json in GCS")
    manifest["vertex_model_id"]       = vertex_model_id
    manifest["vertex_model_short_id"] = vertex_model_short
    manifest["vertex_artifact_uri"]   = artifact_uri
    manifest["vertex_registered_at"]  = datetime.now(timezone.utc).isoformat()

    gcs_write_json(client, manifest, GCS_MANIFEST)
    print(f"  Fields added to manifest:")
    print(f"    vertex_model_id       : {vertex_model_id}")
    print(f"    vertex_model_short_id : {vertex_model_short}")
    print(f"    vertex_artifact_uri   : {artifact_uri}")
    print(f"    vertex_registered_at  : {manifest['vertex_registered_at']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    sec("Phase B1 complete")
    print("  The champion v4 model is now registered in Vertex AI Model Registry.")
    print("  No endpoint has been created. No serving traffic has changed.")
    print("  The live Cloud Run API is completely unaffected.")
    print()
    print("  What was written:")
    print(f"    GCS versioned  : gs://{GCS_BUCKET}/{GCS_V4_PREFIX}/   (4 files, original names)")
    print(f"    GCS vertex     : gs://{GCS_BUCKET}/{GCS_V4_VERTEX_PREFIX}/   (model.pkl only)")
    print(f"    GCS manifest   : gs://{GCS_BUCKET}/{GCS_MANIFEST}")
    print(f"    Vertex AI      : {vertex_model_id}")
    print()
    print("  Next step (Phase B2): create a Vertex AI Endpoint and deploy the model.")
    print("  This requires a custom serving container that preserves threshold logic.")
    print()


if __name__ == "__main__":
    main()
