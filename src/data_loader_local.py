"""Local parquet-based ETF data loader for notebook testing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_features import ensure_etf_feature_columns, standardize_etf_ohlcv_frame


def load_etf_data_local(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    parquet_path: str = "data/parquet/local_etf_daily.parquet",
    benchmark_parquet_path: str | None = None,
    volume_col: str = "volume",
    turnover_col: str = "turn",
    **_: object,
) -> pd.DataFrame:
    """Load ETF data from a local parquet snapshot and add derived features."""
    data = _read_parquet(parquet_path)
    benchmark = _read_parquet(benchmark_parquet_path) if benchmark_parquet_path else None
    data = standardize_etf_ohlcv_frame(data, source="local_parquet")
    if "amount" not in data.columns and {"close", "volume"}.issubset(data.columns):
        data["amount"] = data["close"] * data["volume"]
    data = ensure_etf_feature_columns(data, benchmark_df=benchmark)
    return _filter_by_date(data, start_date, end_date)


def load_condition_research_data_local(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    parquet_path: str = "data/parquet/local_etf_daily.parquet",
    benchmark_parquet_path: str | None = None,
    volume_col: str = "volume",
    turnover_col: str = "turn",
    **kwargs: object,
) -> pd.DataFrame:
    """Load the local data snapshot for condition research."""
    data = load_etf_data_local(
        start_date=start_date,
        end_date=end_date,
        parquet_path=parquet_path,
        benchmark_parquet_path=benchmark_parquet_path,
        volume_col=volume_col,
        turnover_col=turnover_col,
        **kwargs,
    )
    return data


def _read_parquet(path: str | None) -> pd.DataFrame:
    if path is None:
        raise ValueError("parquet path must not be None.")
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Local parquet file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    mask = (data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))
    data = data.loc[mask].sort_values(["code", "date"]).reset_index(drop=True)
    return data.dropna(subset=["date", "code", "close"])
