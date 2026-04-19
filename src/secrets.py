"""
secrets.py — thin wrapper around Google Cloud Secret Manager.

Fetches a secret by its GCP secret name and caches the result in memory so
each secret is retrieved from the API at most once per container instance.
"""

from google.cloud import secretmanager

_PROJECT_ID = "boston-rerouting"
_cache: dict[str, str] = {}
_client: secretmanager.SecretManagerServiceClient | None = None


def _get_client() -> secretmanager.SecretManagerServiceClient:
    global _client
    if _client is None:
        _client = secretmanager.SecretManagerServiceClient()
    return _client


def get_secret(name: str) -> str:
    """Return the latest value of a GCP secret, fetching it once and caching it.

    Args:
        name: The secret's resource name in Secret Manager (e.g. "google-maps-api-key").

    Returns:
        The secret payload as a plain string.

    Raises:
        google.api_core.exceptions.NotFound: If the secret does not exist.
        google.api_core.exceptions.PermissionDenied: If the service account lacks access.
    """
    if name not in _cache:
        secret_path = f"projects/{_PROJECT_ID}/secrets/{name}/versions/latest"
        response = _get_client().access_secret_version(name=secret_path)
        _cache[name] = response.payload.data.decode("utf-8").strip()
    return _cache[name]
