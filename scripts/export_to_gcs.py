"""
export_to_gcs.py
─────────────────
Manually export Cloud SQL tables to Google Cloud Storage as dated parquet files.
Run from the repo root whenever you want an on-demand backup:

    python scripts/export_to_gcs.py

Prerequisites:
    - pip install google-cloud-storage cloud-sql-python-connector[pg8000] pandas pyarrow
    - gcloud auth application-default login

Writes to:
    gs://boston-rerouting-data/exports/route_predictions/YYYY-MM-DD_HH-MM-SS.parquet
"""

import sys
import io
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import sqlalchemy
from google.cloud import storage
from google.cloud.sql.connector import Connector

from src.secrets import get_secret

_INSTANCE   = "boston-rerouting:us-central1:boston-risk-postgres"
_DB_NAME    = "boston_risk_db"
_DB_USER    = "postgres"
_GCS_BUCKET = "boston-rerouting-data"
_TABLES     = ["route_predictions"]


def _build_engine() -> sqlalchemy.engine.Engine:
    connector = Connector()

    def _getconn():
        return connector.connect(
            _INSTANCE, "pg8000",
            user=_DB_USER,
            password=get_secret("cloudsql-postgres-password"),
            db=_DB_NAME,
        )

    return sqlalchemy.create_engine(
        "postgresql+pg8000://", creator=_getconn, pool_size=1, max_overflow=0,
    )


def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-dtype columns containing non-strings (e.g. UUID) to str."""
    for col in df.select_dtypes("object").columns:
        sample = df[col].dropna()
        if len(sample) and not isinstance(sample.iloc[0], str):
            df[col] = df[col].apply(lambda v: None if v is None else str(v))
    return df


def export_table(engine, gcs_client, table: str, run_ts: str) -> dict:
    print(f"[export] Reading '{table}' from Cloud SQL ...")
    df = _coerce_for_parquet(pd.read_sql(f"SELECT * FROM {table}", engine))
    print(f"[export] {len(df):,} rows read.")

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)

    blob_path = f"exports/{table}/{run_ts}.parquet"
    bucket = gcs_client.bucket(_GCS_BUCKET)
    blob   = bucket.blob(blob_path)
    blob.upload_from_file(buf, content_type="application/octet-stream")

    gcs_uri = f"gs://{_GCS_BUCKET}/{blob_path}"
    size_kb  = round(buf.tell() / 1024, 1)
    print(f"[export] Uploaded {size_kb} KB → {gcs_uri}")
    return {"table": table, "rows": len(df), "gcs_uri": gcs_uri, "size_kb": size_kb}


def main() -> None:
    run_ts     = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    engine     = _build_engine()
    gcs_client = storage.Client()

    print(f"[export] Export run: {run_ts}")
    results = [export_table(engine, gcs_client, t, run_ts) for t in _TABLES]

    print()
    print("=" * 55)
    for r in results:
        print(f"  {r['table']:<25} {r['rows']:>6,} rows  {r['size_kb']} KB")
        print(f"    → {r['gcs_uri']}")
    print("=" * 55)
    print("[export] Done.")


if __name__ == "__main__":
    main()
