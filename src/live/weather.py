"""
weather.py — OpenWeather Current Weather API integration.

Calls the OpenWeather API v2.5 to get current weather conditions
at a given latitude/longitude.

Usage example:
    from src.live.weather import get_weather

    # Fenway Park, Boston, MA
    result = get_weather(lat=42.3467, lng=-71.0972)
    print(result["condition"])          # e.g. "Rain"
    print(result["temperature_f"])      # e.g. 48.2
    print(result["is_precipitation"])   # True if rain or snow is falling
    print(result["is_low_visibility"])  # True if visibility < 5000 meters
"""

import requests
from src.secrets import get_secret

WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(lat: float, lng: float) -> dict:
    """
    Get current weather conditions at the given coordinates.

    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.

    Returns:
        dict: {
            "condition": str,           # weather[0].main, e.g. "Rain", "Clear", "Snow"
            "description": str,         # weather[0].description, e.g. "light rain"
            "temperature_f": float,     # main.temp
            "feels_like_f": float,      # main.feels_like
            "humidity_pct": int,        # main.humidity
            "pressure_hpa": int,        # main.pressure
            "visibility_meters": int,   # visibility (top-level field)
            "wind_speed_mph": float,    # wind.speed
            "wind_direction_deg": int,  # wind.deg
            "wind_gust_mph": float,     # wind.gust if present, else 0.0
            "clouds_pct": int,          # clouds.all
            "rain_1h_mm": float,        # rain["1h"] if present, else 0.0
            "snow_1h_mm": float,        # snow["1h"] if present, else 0.0
            "is_precipitation": bool,   # True if rain_1h > 0 or snow_1h > 0
            "is_low_visibility": bool,  # True if visibility < 5000 meters
        }

    Raises:
        EnvironmentError: If OPENWEATHER_API_KEY is not set in .env.
        requests.ConnectionError: On network errors reaching OpenWeather.
        requests.HTTPError: On non-200 responses from OpenWeather.
        KeyError: If expected fields are missing in the response.

    Example:
        # Fenway Park, Boston, MA
        weather = get_weather(lat=42.3467, lng=-71.0972)
        if weather["is_precipitation"]:
            print(f"Wet conditions: {weather['description']}")
        if weather["is_low_visibility"]:
            print(f"Low visibility: {weather['visibility_meters']}m")
    """
    api_key = get_secret("openweather-api-key")

    params = {
        "lat": lat,
        "lon": lng,
        "appid": api_key,
        "units": "imperial",  # Fahrenheit for temp, mph for wind — US conventions
    }

    try:
        resp = requests.get(WEATHER_API_URL, params=params, timeout=15)
    except requests.ConnectionError as e:
        raise requests.ConnectionError(f"Network error reaching OpenWeather API: {e}")

    if not resp.ok:
        try:
            err = resp.json()
            msg = err.get("message", resp.text)
        except Exception:
            msg = resp.text
        print(f"OpenWeather error: {msg}")
        raise requests.HTTPError(
            f"OpenWeather API returned HTTP {resp.status_code}: {msg}"
        )

    data = resp.json()

    try:
        weather_list = data["weather"]
        condition = weather_list[0]["main"] if weather_list else "Unknown"
        description = weather_list[0]["description"] if weather_list else "unknown"

        main = data["main"]
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        snow = data.get("snow", {})

        rain_1h = float(rain.get("1h", 0.0))
        snow_1h = float(snow.get("1h", 0.0))
        visibility = int(data.get("visibility", 0))

        return {
            "condition": condition,
            "description": description,
            "temperature_f": float(main["temp"]),
            "feels_like_f": float(main["feels_like"]),
            "humidity_pct": int(main["humidity"]),
            "pressure_hpa": int(main["pressure"]),
            "visibility_meters": visibility,
            "wind_speed_mph": float(wind.get("speed", 0.0)),
            "wind_direction_deg": int(wind.get("deg", 0)),
            "wind_gust_mph": float(wind.get("gust", 0.0)),
            "clouds_pct": int(clouds.get("all", 0)),
            "rain_1h_mm": rain_1h,
            "snow_1h_mm": snow_1h,
            "is_precipitation": rain_1h > 0 or snow_1h > 0,
            "is_low_visibility": visibility < 5000,
        }
    except KeyError as e:
        raise KeyError(
            f"Unexpected OpenWeather response structure — missing field: {e}. "
            f"Raw response: {data}"
        )
