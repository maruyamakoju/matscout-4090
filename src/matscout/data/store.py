"""Parquet + DuckDB persistence for CandidateRecords."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import CandidateRecord


def records_to_df(records: list[CandidateRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


def _clean_row(row: dict) -> dict:
    """Parquet stores None as NaN and lists as ndarrays; restore them for pydantic."""
    out = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        elif v is pd.NaT:
            out[k] = None
        else:
            out[k] = v
    return out


def df_to_records(df: pd.DataFrame) -> list[CandidateRecord]:
    return [CandidateRecord(**_clean_row(row)) for row in df.to_dict(orient="records")]


def save_parquet(records: list[CandidateRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_df(records)
    # list[str] columns serialize fine to parquet via pyarrow
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: str | Path) -> list[CandidateRecord]:
    df = pd.read_parquet(path)
    return df_to_records(df)


def query_duckdb(parquet_path: str | Path, sql: str) -> pd.DataFrame:
    """Run a SQL query against a parquet file. Use table name `c`."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"CREATE VIEW c AS SELECT * FROM read_parquet('{Path(parquet_path).as_posix()}')")
    return con.execute(sql).fetchdf()
