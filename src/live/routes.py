"""
routes.py — Google Maps Routes API integration.

Calls the Routes API v2 to get traffic-aware driving directions
with alternative routes between two points.

Usage example:
    from src.live.routes import get_route

    # Using address strings
    result = get_route(
        origin="Fenway Park, Boston, MA",
        destination="Boston Logan International Airport, MA"
    )
    print(result["default_route"]["duration_minutes"])

    # Using lat/lng dicts
    result = get_route(
        origin={"lat": 42.3467, "lng": -71.0972},
        destination={"lat": 42.3656, "lng": -71.0096}
    )
"""

import requests
import polyline as polyline_lib
from src.secrets import get_secret

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = (
    "routes.duration,"
    "routes.distanceMeters,"
    "routes.polyline.encodedPolyline,"
    "routes.description,"
    "routes.legs.steps.navigationInstruction,"
    "routes.legs.travelAdvisory.speedReadingIntervals"
)


def _distance_based_speed(distance_miles: float) -> float:
    """Estimate a default speed limit based on route total distance."""
    if distance_miles < 2.0:
        return 25.0
    if distance_miles < 10.0:
        return 35.0
    return 55.0


def _speed_category_to_mph(category: str, base_speed: float) -> float:
    """Map Google SpeedCategory enum to an estimated speed-limit value."""
    mapping = {
        "NORMAL":      base_speed,
        "SLOW":        max(15.0, base_speed - 10.0),
        "TRAFFIC_JAM": max(10.0, base_speed - 20.0),
    }
    return mapping.get(category, base_speed)


def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into a list of (lat, lng) tuples."""
    if not encoded:
        return []
    return polyline_lib.decode(encoded)


def _build_waypoint(location):
    """
    Accepts either:
      - a dict with 'lat' and 'lng' keys  → lat/lng waypoint
      - a string                           → address waypoint
    """
    if isinstance(location, dict):
        return {
            "location": {
                "latLng": {
                    "latitude": location["lat"],
                    "longitude": location["lng"],
                }
            }
        }
    return {"address": location}


def _parse_route(route):
    """Extract the standard fields from a single route object.

    Also builds a per-polyline-point ``speed_limits`` list by reading
    ``legs[*].travelAdvisory.speedReadingIntervals`` when Google returns them
    (requires ``extraComputations: TRAFFIC_ON_POLYLINE``).  Falls back to a
    distance-based default (25 / 35 / 55 mph) when the API returns nothing.
    """
    duration_seconds = int(route.get("duration", "0s").rstrip("s"))
    distance_meters = route.get("distanceMeters", 0)
    encoded = route.get("polyline", {}).get("encodedPolyline", "")

    points = _decode_polyline(encoded)
    num_points = len(points)

    if points:
        start_coords  = {"lat": points[0][0],               "lng": points[0][1]}
        end_coords    = {"lat": points[-1][0],               "lng": points[-1][1]}
        mid_idx       = num_points // 2
        midpoint_coords = {"lat": points[mid_idx][0],        "lng": points[mid_idx][1]}
    else:
        start_coords = midpoint_coords = end_coords = None

    distance_miles = round(distance_meters / 1609.344, 3)
    base_speed = _distance_based_speed(distance_miles)

    # ── Per-point speed limits ─────────────────────────────────────────────
    speed_limits = [base_speed] * num_points

    for leg in route.get("legs", []):
        advisory = leg.get("travelAdvisory", {})
        intervals = advisory.get("speedReadingIntervals", [])
        for interval in intervals:
            start_idx = interval.get("startPolylinePointIndex", 0)
            end_idx   = interval.get("endPolylinePointIndex", 0)
            category  = interval.get("speed", "NORMAL")
            speed_val = _speed_category_to_mph(category, base_speed)
            for i in range(start_idx, end_idx + 1):
                if 0 <= i < num_points:
                    speed_limits[i] = speed_val

    return {
        "duration_seconds": duration_seconds,
        "duration_minutes": round(duration_seconds / 60, 2),
        "distance_meters": distance_meters,
        "distance_km": round(distance_meters / 1000, 3),
        "distance_miles": distance_miles,
        "polyline": encoded,
        "start_coords": start_coords,
        "midpoint_coords": midpoint_coords,
        "end_coords": end_coords,
        "decoded_points": points,
        "num_points": num_points,
        "speed_limits": speed_limits,      # parallel list to decoded_points
        "base_speed_mph": base_speed,       # which distance-tier was used
    }


def get_route(origin, destination, departure_time=None):
    """
    Get traffic-aware driving directions from origin to destination.

    Args:
        origin (str | dict): Address string OR dict with 'lat' and 'lng' keys.
        destination (str | dict): Address string OR dict with 'lat' and 'lng' keys.
        departure_time (str, optional): ISO 8601 datetime string, e.g.
            "2024-10-15T08:00:00Z". If omitted, Google defaults to now.

    Returns:
        dict: {
            "default_route": {
                "duration_seconds": int,
                "duration_minutes": float,
                "distance_meters": int,
                "distance_km": float,
                "distance_miles": float,
                "polyline": str,
            },
            "alternative_routes": [ { same fields }, ... ],
            "best_alternative_savings_minutes": float,
            "num_alternatives": int,
        }

    Raises:
        google.api_core.exceptions.NotFound: If google-server-api-key is missing from Secret Manager.
        requests.HTTPError: On non-200 responses from Google.
        ValueError: If the API returns no routes.
    """
    api_key = get_secret("google-server-api-key")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    body = {
        "origin": _build_waypoint(origin),
        "destination": _build_waypoint(destination),
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
        "computeAlternativeRoutes": True,
        "languageCode": "en-US",
        "units": "IMPERIAL",
        # Request per-segment traffic speed data so we can estimate speed limits
        "extraComputations": ["TRAFFIC_ON_POLYLINE"],
    }

    if departure_time:
        body["departureTime"] = departure_time

    try:
        resp = requests.post(ROUTES_API_URL, json=body, headers=headers, timeout=15)
    except requests.ConnectionError as e:
        raise requests.ConnectionError(f"Network error reaching Google Routes API: {e}")

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise requests.HTTPError(
            f"Google Routes API returned HTTP {resp.status_code}: {msg}"
        )

    data = resp.json()
    routes = data.get("routes", [])

    if not routes:
        raise ValueError(
            "Google Routes API returned no routes. "
            "Check that the origin/destination are valid and the API is enabled."
        )

    default = _parse_route(routes[0])
    alternatives = [_parse_route(r) for r in routes[1:]]

    # Best saving = how many minutes faster the quickest alternative is (if any)
    savings = 0.0
    if alternatives:
        fastest_alt = min(alternatives, key=lambda r: r["duration_seconds"])
        diff = default["duration_seconds"] - fastest_alt["duration_seconds"]
        savings = round(max(diff, 0) / 60, 2)

    return {
        "default_route": default,
        "alternative_routes": alternatives,
        "best_alternative_savings_minutes": savings,
        "num_alternatives": len(alternatives),
    }
