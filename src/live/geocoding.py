"""
geocoding.py — Google Maps Geocoding API integration.

Converts a human-readable address into geographic coordinates (lat/lng)
using the Google Maps Geocoding API.

Usage example:
    from src.live.geocoding import geocode

    result = geocode("Fenway Park, Boston, MA")
    print(result["lat"])               # 42.3467...
    print(result["lng"])               # -71.0972...
    print(result["formatted_address"]) # "4 Jersey St, Boston, MA 02215, USA"
    print(result["location_type"])     # "GEOMETRIC_CENTER"
"""

import requests
from src.secrets import get_secret

GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode(address: str) -> dict:
    """
    Convert a human-readable address to geographic coordinates.

    Args:
        address (str): The address or place name to geocode.

    Returns:
        dict: {
            "lat": float,             # latitude
            "lng": float,             # longitude
            "formatted_address": str, # Google's canonical address string
            "place_id": str,          # Google's unique place identifier
            "location_type": str,     # accuracy: "ROOFTOP", "RANGE_INTERPOLATED",
                                      #           "GEOMETRIC_CENTER", or "APPROXIMATE"
        }

    Raises:
        google.api_core.exceptions.NotFound: If google-server-api-key is missing from Secret Manager.
        requests.ConnectionError: On network errors reaching Google.
        requests.HTTPError: On non-200 HTTP responses from Google.
        ValueError: If Google returns ZERO_RESULTS or a non-OK status.

    Example:
        # Geocode a Boston landmark
        result = geocode("Fenway Park, Boston, MA")
        print(result["lat"], result["lng"])
        # → 42.3466764  -71.09715390000001

        # Use coordinates directly in a downstream call
        from src.live.weather import get_weather
        weather = get_weather(lat=result["lat"], lng=result["lng"])
    """
    api_key = get_secret("google-server-api-key")

    params = {
        "address": address,
        "key": api_key,
    }

    try:
        resp = requests.get(GEOCODING_API_URL, params=params, timeout=15)
    except requests.ConnectionError as e:
        raise requests.ConnectionError(f"Network error reaching Google Geocoding API: {e}")

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("error_message", resp.text)
        except Exception:
            msg = resp.text
        raise requests.HTTPError(
            f"Google Geocoding API returned HTTP {resp.status_code}: {msg}"
        )

    data = resp.json()
    status = data.get("status")

    if status == "ZERO_RESULTS":
        raise ValueError(
            f"Google Geocoding API returned no results for address: '{address}'. "
            "Check the address is valid and spelled correctly."
        )

    if status != "OK":
        error_message = data.get("error_message", "No error message provided.")
        raise ValueError(
            f"Google Geocoding API returned status '{status}' for address: '{address}'. "
            f"Details: {error_message}"
        )

    result = data["results"][0]
    location = result["geometry"]["location"]

    return {
        "lat": float(location["lat"]),
        "lng": float(location["lng"]),
        "formatted_address": result["formatted_address"],
        "place_id": result["place_id"],
        "location_type": result["geometry"]["location_type"],
    }


def reverse_geocode(lat: float, lng: float) -> dict:
    """
    Convert geographic coordinates to a human-readable street address.

    Args:
        lat (float): Latitude.
        lng (float): Longitude.

    Returns:
        dict: {
            "street_name":       str | None,  # e.g. "Massachusetts Ave"
            "neighborhood":      str | None,  # e.g. "Back Bay"
            "short_label":       str,         # e.g. "Massachusetts Ave, Back Bay"
            "formatted_address": str,         # full Google address string
        }

    Raises:
        google.api_core.exceptions.NotFound: If google-server-api-key is missing from Secret Manager.
        requests.ConnectionError: On network errors.
        requests.HTTPError: On non-200 HTTP responses.
        ValueError: If Google returns ZERO_RESULTS.
    """
    api_key = get_secret("google-server-api-key")

    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key,
    }

    try:
        resp = requests.get(GEOCODING_API_URL, params=params, timeout=10)
    except requests.ConnectionError as e:
        raise requests.ConnectionError(f"Network error reaching Google Geocoding API: {e}")

    if not resp.ok:
        try:
            msg = resp.json().get("error_message", resp.text)
        except Exception:
            msg = resp.text
        raise requests.HTTPError(
            f"Google Geocoding API returned HTTP {resp.status_code}: {msg}"
        )

    data = resp.json()
    status = data.get("status")

    if status == "ZERO_RESULTS":
        raise ValueError(f"Reverse geocode returned no results for ({lat}, {lng}).")

    if status != "OK":
        raise ValueError(
            f"Reverse geocode returned status '{status}' for ({lat}, {lng}): "
            f"{data.get('error_message', '')}"
        )

    # Pick the most detailed result (first in list)
    result = data["results"][0]
    components = result.get("address_components", [])

    def _get(types):
        """Return the long_name of the first component matching any of types."""
        for c in components:
            if any(t in c.get("types", []) for t in types):
                return c["long_name"]
        return None

    street_name  = _get(["route"])
    neighborhood = _get(["neighborhood", "sublocality_level_1", "sublocality"])

    if street_name and neighborhood:
        short_label = f"{street_name}, {neighborhood}"
    elif street_name:
        short_label = street_name
    elif neighborhood:
        short_label = neighborhood
    else:
        short_label = result.get("formatted_address", f"{lat:.4f}, {lng:.4f}")

    return {
        "street_name":       street_name,
        "neighborhood":      neighborhood,
        "short_label":       short_label,
        "formatted_address": result.get("formatted_address", ""),
    }
