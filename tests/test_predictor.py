"""
Smoke test for src/predict/predictor.py

Runs three Boston-area route risk predictions and prints full results.

Run directly:
    python tests/test_predictor.py
"""

import sys
import os

# Allow imports from repo root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predict.predictor import predict_route_risk

SCENARIOS = [
    {
        "label":       "Scenario 1: Fenway Park → Boston Logan Airport",
        "origin":      "Fenway Park, Boston, MA",
        "destination": "Boston Logan International Airport, MA",
    },
    {
        "label":       "Scenario 2: Harvard University → Boston Common",
        "origin":      "Harvard University, Cambridge, MA",
        "destination": "Boston Common, Boston, MA",
    },
    {
        "label":       "Scenario 3: WPI Worcester → Fenway Park (longer trip outside Boston)",
        "origin":      "Worcester Polytechnic Institute, Worcester, MA",
        "destination": "Fenway Park, Boston, MA",
    },
]


def run_smoke_test():
    print("=" * 65)
    print("Route Risk Prediction Smoke Test — 3 Boston-area scenarios")
    print("=" * 65)

    all_passed = True

    for scenario in SCENARIOS:
        print(f"\n{'─' * 65}")
        print(f"  {scenario['label']}")
        print(f"{'─' * 65}")
        print(f"  Origin      : {scenario['origin']}")
        print(f"  Destination : {scenario['destination']}")
        print()

        try:
            result = predict_route_risk(
                origin=scenario["origin"],
                destination=scenario["destination"],
            )

            # Risk summary
            print(f"  Risk Class  : {result['risk_class']}")
            print(f"  Confidence  : {result['confidence']:.1%}")
            print(f"  Probabilities:")
            for cls, p in sorted(result["class_probabilities"].items()):
                bar = "█" * int(p * 30)
                print(f"      {cls:<8} {p:.4f}  {bar}")

            # Route info
            r = result["route"]
            print(f"\n  Route:")
            print(f"    Duration     : {r['duration_minutes']} min")
            print(f"    Distance     : {r['distance_miles']} miles")
            print(f"    Alternatives : {r['num_alternatives']}")
            if r["best_alternative_savings_minutes"] > 0:
                print(f"    Best alt saves: {r['best_alternative_savings_minutes']} min")

            # Weather info
            w = result["weather"]
            print(f"\n  Weather (at route midpoint):")
            print(f"    Condition    : {w['condition']}")
            print(f"    Temperature  : {w['temperature_f']}°F")
            print(f"    Precipitation: {w['is_precipitation']}")
            print(f"    Low visibility: {w['is_low_visibility']}")

            # Context / mapping decisions
            ctx = result["context"]
            print(f"\n  Context:")
            print(f"    Midpoint     : lat={ctx['midpoint_lat']}, lng={ctx['midpoint_lng']}")
            print(f"    Hour of day  : {ctx['hour_of_day']:02d}:xx")
            print(f"    Weather map  : {ctx['weather_mapping_used']}")

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            all_passed = False

    print(f"\n{'=' * 65}")
    if all_passed:
        print("All 3 scenarios PASSED")
    else:
        print("One or more scenarios FAILED — see errors above")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
