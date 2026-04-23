"""
gemini_explainer.py — Post-prediction Gemini explanation layer.

Called AFTER model inference completes. If Gemini is unavailable,
disabled, or fails for any reason, returns None so the caller can set
explanation=None in the response without breaking the prediction result.

Required env vars:
  GEMINI_API_KEY               — Google AI Studio API key
  ENABLE_GEMINI_EXPLANATIONS   — must be exactly "true" to enable (default: off)
"""

import os

# Defensive import: the app must start cleanly even when google-genai
# is not installed. If the import fails, every call to get_gemini_explanation()
# will return None immediately.
try:
    from google import genai as _genai_module
    _GENAI_AVAILABLE = True
except ImportError:
    _genai_module = None
    _GENAI_AVAILABLE = False
    print("[gemini_explainer] google-genai not installed — explanation layer disabled")

_MODEL_NAME = "gemini-2.0-flash"
_client = None


def _is_enabled() -> bool:
    return os.environ.get("ENABLE_GEMINI_EXPLANATIONS", "false").lower() == "true"


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[gemini_explainer] GEMINI_API_KEY is empty — cannot configure")
        return None
    _client = _genai_module.Client(api_key=api_key)
    print("[gemini_explainer] Gemini client created successfully")
    return _client


def _build_prompt(prediction: dict) -> str:
    risk_class = prediction.get("risk_class", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    proba      = prediction.get("class_probabilities", {})
    weather    = prediction.get("weather", {})
    route      = prediction.get("route", {})
    context    = prediction.get("context", {})
    spatial    = prediction.get("spatial_features", {})
    hotspots   = prediction.get("hotspots", [])

    hour = context.get("hour_of_day", context.get("local_hour", "unknown"))

    dur  = route.get("duration_minutes")
    dist = route.get("distance_miles")
    route_line = (
        f"{dur:.1f} min, {dist:.2f} miles"
        if dur is not None and dist is not None
        else "unknown"
    )

    weather_cond = weather.get("condition", "unknown")
    temp         = weather.get("temperature_f")
    temp_str     = f"{temp:.0f}°F" if temp is not None else "unknown temperature"
    precip       = "precipitation present" if weather.get("is_precipitation") else "no precipitation"
    vis          = ", low visibility" if weather.get("is_low_visibility") else ""
    weather_line = f"{weather_cond}, {temp_str}, {precip}{vis}"

    lines = [
        "You are a driving safety assistant explaining a machine learning route risk prediction.",
        "",
        "Prediction data:",
        f"- Risk level: {risk_class} (confidence {confidence:.1%})",
        f"- Probabilities: Low={proba.get('Low', 0):.1%}, "
        f"Medium={proba.get('Medium', 0):.1%}, High={proba.get('High', 0):.1%}",
        f"- Route: {route_line}, departing at hour {hour}",
        f"- Weather: {weather_line}",
    ]

    if spatial:
        crashes = spatial.get("nearby_crash_count_1km")
        fatals  = spatial.get("nearby_fatal_count_1km")
        if crashes is not None:
            crash_line = f"- Historical crashes within 1 km of route: {int(crashes)}"
            if fatals:
                crash_line += f" ({int(fatals)} fatal)"
            lines.append(crash_line)
        crashes_500 = spatial.get("nearby_crash_count_500m")
        if crashes_500 is not None:
            lines.append(f"- Historical crashes within 500 m: {int(crashes_500)}")

    if hotspots:
        n_high = sum(1 for h in hotspots if h.get("risk_class") == "High")
        n_med  = sum(1 for h in hotspots if h.get("risk_class") == "Medium")
        names  = [
            h.get("short_label") or h.get("street_name")
            for h in hotspots[:3]
            if h.get("short_label") or h.get("street_name")
        ]
        hs_line = f"- Route hotspots: {n_high} high-risk, {n_med} medium-risk"
        if names:
            hs_line += f". Locations: {', '.join(names)}"
        lines.append(hs_line)

    lines += [
        "",
        f"Write exactly 2 concise sentences explaining why this route received a "
        f"{risk_class} risk rating. Base your explanation only on the data above. "
        "Do not give driving advice, warnings, or caveats.",
    ]

    return "\n".join(lines)


def get_gemini_explanation(prediction: dict) -> str | None:
    """
    Returns a 2-sentence plain-English explanation of the prediction.
    Returns None if Gemini is disabled, unconfigured, or fails for any reason.
    """
    print(
        f"[gemini_explainer] get_gemini_explanation called "
        f"(enabled={_is_enabled()}, genai_available={_GENAI_AVAILABLE})"
    )

    if not _GENAI_AVAILABLE:
        print("[gemini_explainer] Returning None — google-genai not installed")
        return None

    if not _is_enabled():
        print("[gemini_explainer] Returning None — ENABLE_GEMINI_EXPLANATIONS is not 'true'")
        return None

    try:
        client = _get_client()
        if client is None:
            print("[gemini_explainer] Returning None — client not configured (check GEMINI_API_KEY)")
            return None

        prompt = _build_prompt(prediction)
        print(f"[gemini_explainer] Calling Gemini model={_MODEL_NAME} ...")
        response = client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        text = response.text.strip()
        print(f"[gemini_explainer] Gemini returned {len(text)} chars")
        return text

    except Exception as e:
        print(f"[gemini_explainer] Gemini call failed (non-fatal): {type(e).__name__}: {e}")
        return None
