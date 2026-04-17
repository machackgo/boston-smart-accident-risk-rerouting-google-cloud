"""
Smoke test for src/live/geocoding.py

Geocodes three addresses (two Boston landmarks + one Worcester sanity check)
and prints the results.

Run directly:
    python tests/test_geocoding.py
"""

import sys
import os

# Allow running from repo root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.live.geocoding import geocode

ADDRESSES = [
    "Fenway Park, Boston, MA",
    "Boston Logan International Airport, MA",
    "Worcester Polytechnic Institute, Worcester, MA",
]


def run_smoke_test():
    print("=" * 60)
    print("Geocoding Smoke Test — Boston area addresses")
    print("=" * 60)

    results = []
    for address in ADDRESSES:
        print(f"\nGeocoding: {address}")
        result = geocode(address)
        print(f"  Formatted address : {result['formatted_address']}")
        print(f"  Coordinates       : lat={result['lat']}, lng={result['lng']}")
        print(f"  Place ID          : {result['place_id']}")
        print(f"  Location type     : {result['location_type']}")
        results.append(result)

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    return results


if __name__ == "__main__":
    run_smoke_test()
