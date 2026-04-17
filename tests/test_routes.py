"""
Smoke test for src/live/routes.py

Tests a real API call: Fenway Park → Boston Logan Airport.
Run with:
    python tests/test_routes.py
"""

import sys
import os

# Allow running from repo root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.live.routes import get_route


def test_fenway_to_logan():
    print("=" * 60)
    print("Smoke Test: Fenway Park → Boston Logan Airport")
    print("=" * 60)

    result = get_route(
        origin="Fenway Park, Boston, MA",
        destination="Boston Logan International Airport, MA",
    )

    default = result["default_route"]
    alts = result["alternative_routes"]
    savings = result["best_alternative_savings_minutes"]
    num_alts = result["num_alternatives"]

    print(f"\nDefault Route:")
    print(f"  Duration    : {default['duration_minutes']} minutes")
    print(f"  Distance    : {default['distance_miles']} miles  ({default['distance_km']} km)")
    print(f"  Polyline    : {default['polyline'][:40]}..." if default['polyline'] else "  Polyline    : (none)")
    print(f"  Num points  : {default['num_points']}")
    print(f"  Start       : {default['start_coords']}")
    print(f"  Midpoint    : {default['midpoint_coords']}")
    print(f"  End         : {default['end_coords']}")

    print(f"\nAlternative Routes: {num_alts}")
    for i, alt in enumerate(alts, 1):
        print(f"  Alt {i}: {alt['duration_minutes']} min  /  {alt['distance_miles']} miles  |  "
              f"start={alt['start_coords']}  mid={alt['midpoint_coords']}  end={alt['end_coords']}")

    if savings > 0:
        print(f"\nBest alternative saves: {savings} minutes over the default route")
    else:
        print("\nNo faster alternative found (default is already optimal)")

    print("\nFull result dict (decoded_points omitted, shown as length):")
    import json

    def _summarise(route_dict):
        d = {k: v for k, v in route_dict.items() if k != "decoded_points"}
        d["polyline"] = (route_dict["polyline"][:40] + "...") if route_dict["polyline"] else ""
        d["decoded_points_len"] = route_dict["num_points"]
        return d

    display = {
        "default_route": _summarise(default),
        "alternative_routes": [_summarise(a) for a in alts],
        "best_alternative_savings_minutes": savings,
        "num_alternatives": num_alts,
    }
    print(json.dumps(display, indent=2))
    print("\nSMOKE TEST PASSED")
    return result


if __name__ == "__main__":
    test_fenway_to_logan()
