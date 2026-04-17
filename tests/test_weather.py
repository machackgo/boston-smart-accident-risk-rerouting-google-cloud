"""
Smoke test for src/live/weather.py

Calls get_weather with Fenway Park coordinates and prints the result.

Run directly:
    python tests/test_weather.py
"""

import sys
import os

# Allow running from repo root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.live.weather import get_weather

FENWAY_LAT = 42.3467
FENWAY_LNG = -71.0972


def run_smoke_test():
    print("=" * 55)
    print("OpenWeather Smoke Test — Fenway Park, Boston MA")
    print(f"Coordinates: lat={FENWAY_LAT}, lng={FENWAY_LNG}")
    print("=" * 55)

    result = get_weather(lat=FENWAY_LAT, lng=FENWAY_LNG)

    print(f"\nCondition    : {result['condition']} — {result['description']}")
    print(f"Temperature  : {result['temperature_f']}°F  (feels like {result['feels_like_f']}°F)")
    print(f"Humidity     : {result['humidity_pct']}%")
    print(f"Pressure     : {result['pressure_hpa']} hPa")
    print(f"\nWind         : {result['wind_speed_mph']} mph @ {result['wind_direction_deg']}°", end="")
    if result['wind_gust_mph'] > 0:
        print(f"  (gusts {result['wind_gust_mph']} mph)", end="")
    print()
    print(f"Clouds       : {result['clouds_pct']}%")
    print(f"Visibility   : {result['visibility_meters']} m")

    if result['rain_1h_mm'] > 0:
        print(f"Rain (1h)    : {result['rain_1h_mm']} mm")
    if result['snow_1h_mm'] > 0:
        print(f"Snow (1h)    : {result['snow_1h_mm']} mm")

    print(f"\nEngineered flags:")
    print(f"  is_precipitation  : {result['is_precipitation']}")
    print(f"  is_low_visibility : {result['is_low_visibility']}")
    print("=" * 55)
    print("Smoke test PASSED")

    return result


if __name__ == "__main__":
    run_smoke_test()
