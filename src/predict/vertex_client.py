"""
vertex_client.py — Vertex AI Online Prediction client for Cloud Run.

Uses the Vertex AI REST prediction API directly (not the Python SDK) to avoid
the SDK's one-time per-instance initialization overhead (~3–10 s), which was
being charged to the first user request on each new Cloud Run instance.

Authentication is handled by google.auth.transport.requests.AuthorizedSession,
which refreshes tokens automatically and reuses the HTTPS connection pool.

Design rules:
  - Never raises — all exceptions are caught, logged, and return None
  - Lazy session init — credentials are not fetched until first call
  - Enabled only when VERTEX_ENDPOINT_ID env var is non-empty
  - VERTEX_TIMEOUT_SECONDS        — deadline for single-row calls (default 10.0 s)
  - VERTEX_BATCH_TIMEOUT_SECONDS  — deadline for multi-row batch calls (default 20.0 s)
  - None return always means: use local LightGBM fallback

Return envelope (on success):
  {
    "prediction":      {"risk_class": str, "confidence": float, "probabilities": dict},
    "vertex_model_id": str,   # deployedModelId from REST response
    "latency_ms":      int,   # wall-clock time of the REST call only
  }
"""

import logging
import os
import time
from typing import Optional

import google.auth
from google.auth.transport.requests import AuthorizedSession

log = logging.getLogger(__name__)

_VERTEX_ENDPOINT_ID   = os.environ.get("VERTEX_ENDPOINT_ID", "").strip()
_VERTEX_TIMEOUT       = float(os.environ.get("VERTEX_TIMEOUT_SECONDS",       "10.0"))
_VERTEX_BATCH_TIMEOUT = float(os.environ.get("VERTEX_BATCH_TIMEOUT_SECONDS", "20.0"))
_GCP_LOCATION         = os.environ.get("VERTEX_LOCATION", "us-central1")

# Lazy singletons — initialised on first call, reused for all subsequent calls.
_session:     Optional[AuthorizedSession] = None
_predict_url: str = ""


def is_enabled() -> bool:
    """Return True when VERTEX_ENDPOINT_ID is configured."""
    return bool(_VERTEX_ENDPOINT_ID)


def _get_session() -> tuple[AuthorizedSession, str]:
    """
    Return the shared AuthorizedSession and prediction URL, initialising both
    on the first call.  ~100 ms one-time cost per Cloud Run instance (credential
    fetch from the GCE metadata server), versus 3–10 s for the aiplatform SDK.
    """
    global _session, _predict_url
    if _session is None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        _session = AuthorizedSession(credentials)
        _predict_url = (
            f"https://{_GCP_LOCATION}-aiplatform.googleapis.com/v1/"
            f"{_VERTEX_ENDPOINT_ID}:predict"
        )
        log.info("[vertex_client] REST session initialised → %s", _predict_url)
    return _session, _predict_url


def _classify_exc(exc: Exception) -> str:
    """Map an exception to a short human-readable reason string for logging."""
    name = type(exc).__name__
    msg  = str(exc).lower()
    if "deadline" in name.lower() or "timeout" in msg or "deadline" in msg:
        return "timeout"
    if "forbidden" in name.lower() or "403" in msg or "permission" in msg:
        return "permission_denied"
    if "notfound" in name.lower() or "404" in msg or "not found" in msg:
        return "endpoint_not_found"
    if "unavailable" in name.lower() or "503" in msg:
        return "service_unavailable"
    return f"unexpected_error:{name}"


def _parse_pred(pred: dict) -> Optional[dict]:
    """
    Validate one prediction dict from the Vertex REST response.
    Expected shape: {"risk_class": str, "confidence": float, "probabilities": {str: float}}
    Returns None if any required field is missing or has the wrong type.
    """
    risk_class    = pred.get("risk_class")
    confidence    = pred.get("confidence")
    probabilities = pred.get("probabilities")
    if not risk_class or confidence is None or not isinstance(probabilities, dict):
        log.warning("[vertex_client] Unexpected prediction shape: %s", pred)
        return None
    return {
        "risk_class":    str(risk_class),
        "confidence":    round(float(confidence), 4),
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
    }


def predict_single(features: dict) -> Optional[dict]:
    """
    Send one feature dict to the Vertex AI endpoint.

    Args:
        features: dict of {feature_name: value} for all 36 model features.

    Returns:
        {
            "prediction":      {"risk_class": str, "confidence": float, "probabilities": dict},
            "vertex_model_id": str,
            "latency_ms":      int,
        }
        or None on any failure (triggers local-model fallback in predictor.py).
    """
    if not is_enabled():
        return None
    session, url = _get_session()
    t0 = time.monotonic()
    try:
        resp = session.post(
            url,
            json={"instances": [features]},
            timeout=_VERTEX_TIMEOUT,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        resp.raise_for_status()

        data  = resp.json()
        preds = data.get("predictions", [])
        if not preds:
            log.warning("[vertex_client] predict_single: empty predictions list.")
            return None

        parsed = _parse_pred(preds[0])
        if parsed is None:
            return None

        vertex_model_id = data.get("deployedModelId", "")
        log.info(
            "[vertex_client] predict_single OK latency=%dms model=%s",
            latency_ms, vertex_model_id,
        )
        return {
            "prediction":      parsed,
            "vertex_model_id": vertex_model_id,
            "latency_ms":      latency_ms,
        }

    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        reason = _classify_exc(exc)
        log.warning(
            "[vertex_client] predict_single FAILED reason=%s latency=%dms — falling back to local. (%s)",
            reason, latency_ms, exc,
        )
        return None


def predict_batch(features_list: list[dict]) -> Optional[dict]:
    """
    Send a batch of feature dicts to the Vertex AI endpoint.

    Used by predict_route_risk_segmented() — up to ~36 rows per call
    (3 routes × 12 segments).

    Returns:
        {
            "predictions":     [{"risk_class", "confidence", "probabilities"}, ...],
            "vertex_model_id": str,
            "latency_ms":      int,
        }
        or None on any failure (triggers local-model fallback for the entire batch).
    """
    if not is_enabled():
        return None
    session, url = _get_session()
    t0 = time.monotonic()
    try:
        resp = session.post(
            url,
            json={"instances": features_list},
            timeout=_VERTEX_BATCH_TIMEOUT,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        resp.raise_for_status()

        data  = resp.json()
        preds = data.get("predictions", [])

        if len(preds) != len(features_list):
            log.warning(
                "[vertex_client] predict_batch count mismatch: sent %d, got %d — falling back.",
                len(features_list), len(preds),
            )
            return None

        parsed = [_parse_pred(p) for p in preds]
        if any(p is None for p in parsed):
            log.warning("[vertex_client] predict_batch: validation failed on one or more rows — falling back.")
            return None

        vertex_model_id = data.get("deployedModelId", "")
        log.info(
            "[vertex_client] predict_batch OK rows=%d latency=%dms model=%s",
            len(features_list), latency_ms, vertex_model_id,
        )
        return {
            "predictions":     parsed,
            "vertex_model_id": vertex_model_id,
            "latency_ms":      latency_ms,
        }

    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        reason = _classify_exc(exc)
        log.warning(
            "[vertex_client] predict_batch FAILED reason=%s latency=%dms rows=%d — falling back to local. (%s)",
            reason, latency_ms, len(features_list), exc,
        )
        return None
