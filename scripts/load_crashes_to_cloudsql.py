"""
load_crashes_to_cloudsql.py
────────────────────────────
One-time migration script: reads data/crashes_cache.parquet and loads
all 47,689 rows into the boston_crashes table in Cloud SQL PostgreSQL.

Run from the repo root:
    python scripts/load_crashes_to_cloudsql.py

Prerequisites:
    - pip install cloud-sql-python-connector[pg8000] sqlalchemy pandas pyarrow
    - gcloud auth application-default login
    - The Cloud SQL instance must be reachable (either via Cloud SQL Auth Proxy
      locally, or run this script from a Cloud Shell / VM inside GCP)

The script is fully idempotent: running it again drops and recreates the table.
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import sqlalchemy
from google.cloud.sql.connector import Connector

from src.secrets import get_secret

# ── Config ────────────────────────────────────────────────────────────────────
_INSTANCE = "boston-rerouting:us-central1:boston-risk-postgres"
_DB_NAME  = "boston_risk_db"
_DB_USER  = "postgres"
_PARQUET  = _REPO_ROOT / "data" / "crashes_cache.parquet"
_TABLE    = "boston_crashes"

# ── DDL ───────────────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS boston_crashes (
    id                   BIGINT PRIMARY KEY,
    crash_numb           BIGINT,
    crash_datetime_clean TEXT,
    year                 SMALLINT,
    crash_hour           TEXT,
    city_town_name       TEXT,
    lat                  DOUBLE PRECISION,
    lon                  DOUBLE PRECISION,
    crash_severity_descr TEXT,
    severity_3class      TEXT,
    numb_fatal_injr      INTEGER,
    numb_nonfatal_injr   INTEGER,
    numb_vehc            INTEGER,
    weath_cond_descr     TEXT,
    road_surf_cond_descr TEXT,
    ambnt_light_descr    TEXT,
    manr_coll_descr      TEXT,
    rdwy_jnct_type_descr TEXT,
    rdwy                 TEXT,
    speed_limit          REAL,
    hit_run_descr        TEXT,
    ems_hotspot_flag     REAL,
    ems_ped_hotspot_flag REAL,
    ems_peak_hour        REAL,
    district_num         INTEGER,
    cnty_name            TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_boston_crashes_year     ON boston_crashes (year);",
    "CREATE INDEX IF NOT EXISTS idx_boston_crashes_severity ON boston_crashes (severity_3class);",
    "CREATE INDEX IF NOT EXISTS idx_boston_crashes_city     ON boston_crashes (city_town_name);",
    "CREATE INDEX IF NOT EXISTS idx_boston_crashes_hotspot  ON boston_crashes (ems_hotspot_flag);",
    "CREATE INDEX IF NOT EXISTS idx_boston_crashes_fatal    ON boston_crashes (numb_fatal_injr);",
]


def _build_engine() -> sqlalchemy.engine.Engine:
    connector = Connector()

    def _getconn():
        return connector.connect(
            _INSTANCE,
            "pg8000",
            user=_DB_USER,
            password=get_secret("cloudsql-postgres-password"),
            db=_DB_NAME,
        )

    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=_getconn,
        pool_size=1,
        max_overflow=0,
    )


def main() -> None:
    print(f"[load] Reading {_PARQUET} ...")
    df = pd.read_parquet(_PARQUET)
    print(f"[load] Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Coerce numpy int64 → Python int so pg8000 accepts them without complaint
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("Int64")  # nullable integer

    print("[load] Connecting to Cloud SQL ...")
    engine = _build_engine()

    with engine.begin() as conn:
        print(f"[load] Creating table '{_TABLE}' if it does not exist ...")
        conn.execute(sqlalchemy.text(_CREATE_TABLE))

        print(f"[load] Truncating existing rows (safe to re-run) ...")
        conn.execute(sqlalchemy.text(f"TRUNCATE TABLE {_TABLE} RESTART IDENTITY;"))

    print(f"[load] Bulk-inserting {len(df):,} rows ...")
    df.to_sql(
        _TABLE,
        engine,
        if_exists="append",   # table already exists — append into the empty table
        index=False,
        chunksize=500,         # insert 500 rows per batch; safe for pg8000
        method="multi",        # single INSERT ... VALUES (...),(...)
    )

    with engine.begin() as conn:
        print("[load] Creating indexes ...")
        for ddl in _CREATE_INDEXES:
            conn.execute(sqlalchemy.text(ddl))

    # Verification
    with engine.connect() as conn:
        row_count = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM {_TABLE}")
        ).scalar()
        year_range = conn.execute(
            sqlalchemy.text(f"SELECT MIN(year), MAX(year) FROM {_TABLE}")
        ).fetchone()
        severity_counts = conn.execute(
            sqlalchemy.text(
                f"SELECT severity_3class, COUNT(*) FROM {_TABLE} "
                f"GROUP BY severity_3class ORDER BY severity_3class"
            )
        ).fetchall()

    print()
    print("=" * 50)
    print(f"  Rows loaded : {row_count:,}")
    print(f"  Year range  : {year_range[0]} – {year_range[1]}")
    print("  Severity breakdown:")
    for sev, cnt in severity_counts:
        print(f"    {sev:<20} {cnt:>6,}")
    print("=" * 50)

    expected = len(df)
    if row_count == expected:
        print(f"[load] SUCCESS — all {expected:,} rows verified in Cloud SQL.")
    else:
        print(f"[load] WARNING — expected {expected:,} rows but found {row_count:,}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
