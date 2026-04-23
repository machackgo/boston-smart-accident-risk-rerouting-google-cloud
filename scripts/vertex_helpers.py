"""
vertex_helpers.py — Thin shared Vertex AI SDK helper.

Imported by vertex_register.py and (in Phase B2+) pipeline_retrain.py.
NEVER imported by api.py or any live serving code — zero impact on Cloud Run.

Design rules:
  - No side effects on import (init_vertex() must be called explicitly)
  - Never raises to callers — errors are printed and re-raised so the caller decides
  - All Vertex AI API calls are synchronous (sync=True) so scripts see results immediately
"""

from google.cloud import aiplatform

GCP_PROJECT   = "boston-rerouting"
GCP_LOCATION  = "us-central1"
GCS_BUCKET    = "boston-rerouting-data"

# ── Serving container spec ────────────────────────────────────────────────────
# Vertex AI Model.upload() requires a serving container image URI even for
# registry-only registrations where no endpoint is created yet.
#
# We use the pre-built sklearn CPU container as a Phase B1 placeholder because:
#   1. It is a valid URI the registry accepts for custom sklearn/LightGBM pickles.
#   2. No endpoint is deployed in Phase B1, so this container is never actually invoked.
#
# Phase B2 WILL replace this with a custom container image that:
#   - Loads the dict-bundle joblib format: {"model": LGBMClassifier, "features": [...]}
#   - Applies per-class decision thresholds from thresholds_v4.json
#   - Matches the exact prediction path in predictor.py
_SKLEARN_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"


def init_vertex() -> None:
    """
    Initialise the Vertex AI SDK for this process.
    Call once per script, before any other aiplatform API call.
    staging_bucket is required by the SDK for certain upload operations.
    """
    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        staging_bucket=f"gs://{GCS_BUCKET}/vertex-staging",
    )


# ── Phase B2 constants ────────────────────────────────────────────────────────
# Custom serving container image URI (built from serving/Dockerfile).
# The Artifact Registry repo "boston-risk" already exists (used by Cloud Run).
SERVING_IMAGE_URI = (
    f"us-central1-docker.pkg.dev/{GCP_PROJECT}/boston-risk/accident-risk-serving:v4"
)


def register_model(
    display_name: str,
    artifact_gcs_uri: str,
    description: str = "",
    labels: dict | None = None,
) -> aiplatform.Model:
    """
    Register a model artifact directory in Vertex AI Model Registry.

    Does NOT create an endpoint or deploy the model — registry metadata only.

    Args:
        display_name:     Human-readable name shown in Cloud Console.
        artifact_gcs_uri: GCS URI of the artifact directory, must end with '/'.
                          e.g. "gs://boston-rerouting-data/production/models/v4/"
        description:      Optional free-text description stored with the registry entry.
        labels:           Optional dict of lowercase string key-value labels.
                          GCP label constraints: keys/values must be lowercase
                          letters, digits, hyphens, underscores; max 64 chars each.

    Returns:
        aiplatform.Model — use .resource_name for the full Vertex AI resource URI,
        or .name for the numeric model ID.

    Raises:
        google.api_core.exceptions.GoogleAPICallError on Vertex AI API failures.
    """
    init_vertex()

    # Sanitise labels: GCP requires all values to be lowercase strings.
    safe_labels = {str(k): str(v).lower() for k, v in (labels or {}).items()}

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_gcs_uri,
        serving_container_image_uri=_SKLEARN_CONTAINER,
        description=description,
        labels=safe_labels,
        sync=True,   # block until registration is complete before returning
    )
    return model


# ── Phase B2 helpers ──────────────────────────────────────────────────────────

def register_serving_model(
    display_name: str,
    artifact_gcs_uri: str,
    serving_image_uri: str,
    gcs_bucket: str,
    model_blob: str,
    threshold_blob: str,
    description: str = "",
    labels: dict | None = None,
) -> aiplatform.Model:
    """
    Register a Vertex AI model with the custom serving container.

    Unlike register_model() (Phase B1, sklearn placeholder), this registration
    uses the custom container built from serving/Dockerfile. The container loads
    model artifacts from GCS via the env vars below at endpoint startup.

    Args:
        display_name:      Human-readable name in Cloud Console.
        artifact_gcs_uri:  GCS directory URI of versioned model artifacts.
        serving_image_uri: Full URI of the custom container image.
        gcs_bucket:        GCS bucket name passed to the container as GCS_BUCKET.
        model_blob:        GCS blob path for the model pkl (MODEL_BLOB env var).
        threshold_blob:    GCS blob path for thresholds JSON (THRESHOLD_BLOB env var).
        description:       Optional free-text description.
        labels:            Optional dict of lowercase string key-value labels.

    Returns:
        aiplatform.Model with custom serving container spec.
    """
    init_vertex()
    safe_labels = {str(k): str(v).lower() for k, v in (labels or {}).items()}

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_gcs_uri,
        serving_container_image_uri=serving_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/",
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "GCS_BUCKET":     gcs_bucket,
            "MODEL_BLOB":     model_blob,
            "THRESHOLD_BLOB": threshold_blob,
        },
        description=description,
        labels=safe_labels,
        sync=True,
    )
    return model


def create_endpoint(display_name: str = "boston-accident-risk-endpoint") -> aiplatform.Endpoint:
    """
    Create a Vertex AI Endpoint (empty — no model deployed yet).

    The endpoint is created in GCP_PROJECT / GCP_LOCATION.
    Returns the Endpoint object; use .resource_name for the full URI.
    """
    init_vertex()
    endpoint = aiplatform.Endpoint.create(
        display_name=display_name,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        sync=True,
    )
    return endpoint


def deploy_to_endpoint(
    model: aiplatform.Model,
    endpoint: aiplatform.Endpoint,
    machine_type: str = "n1-standard-2",
    min_replicas: int = 1,
    max_replicas: int = 1,
) -> None:
    """
    Deploy a Vertex AI model to an endpoint.

    Uses sync=True so the call blocks until the deployment is fully live
    (~5–10 minutes). The deployed model receives 100% of endpoint traffic,
    but no live user traffic reaches the endpoint until Phase B3.

    Args:
        model:        The registered Vertex AI model to deploy.
        endpoint:     The target Vertex AI Endpoint.
        machine_type: Compute machine type (default: n1-standard-2).
        min_replicas: Minimum replica count (default: 1).
        max_replicas: Maximum replica count (default: 1).
    """
    init_vertex()
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model.display_name,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        sync=True,
    )
