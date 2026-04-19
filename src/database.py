"""
database.py — Cloud SQL (PostgreSQL) connection and prediction logging.

Uses the Cloud SQL Python Connector for IAM-based authentication from Cloud Run.
The connection pool is created once (lazy) and reused across background tasks.
A DB failure never raises to the caller — errors are printed and swallowed so
the prediction response is always returned to the user.
"""

import sqlalchemy
from google.cloud.sql.connector import Connector
from src.secrets import get_secret

_INSTANCE  = "boston-rerouting:us-central1:boston-risk-postgres"
_DB_NAME   = "boston_risk_db"
_DB_USER   = "postgres"

_engine: sqlalchemy.engine.Engine | None = None


def _get_engine() -> sqlalchemy.engine.Engine:
    global _engine
    if _engine is None:
        connector = Connector()

        def _getconn():
            return connector.connect(
                _INSTANCE,
                "pg8000",
                user=_DB_USER,
                password=get_secret("cloudsql-postgres-password"),
                db=_DB_NAME,
            )

        _engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=_getconn,
            pool_size=2,       # small pool — logging writes are infrequent
            max_overflow=0,    # never exceed pool_size connections
            pool_pre_ping=True,
        )
    return _engine


_INSERT = sqlalchemy.text("""
    INSERT INTO route_predictions (
        endpoint, origin, destination, departure_time,
        duration_minutes, distance_miles, num_alternatives,
        risk_class, confidence, prob_low, prob_medium, prob_high,
        weather_condition, temperature_f, is_precipitation, is_low_visibility,
        midpoint_lat, midpoint_lng, hour_of_day,
        recommended_route_index, safety_score, num_hotspots, num_high_hotspots,
        model_version, response_time_ms
    ) VALUES (
        :endpoint, :origin, :destination, :departure_time,
        :duration_minutes, :distance_miles, :num_alternatives,
        :risk_class, :confidence, :prob_low, :prob_medium, :prob_high,
        :weather_condition, :temperature_f, :is_precipitation, :is_low_visibility,
        :midpoint_lat, :midpoint_lng, :hour_of_day,
        :recommended_route_index, :safety_score, :num_hotspots, :num_high_hotspots,
        :model_version, :response_time_ms
    )
""")


def log_prediction(row: dict) -> None:
    """Insert one row into route_predictions. Errors are logged, never raised."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(_INSERT, row)
            conn.commit()
    except Exception as exc:
        print(f"[database] log_prediction failed (non-fatal): {exc}")


# Public alias so api.py can share the same engine for crash queries.
get_engine = _get_engine
