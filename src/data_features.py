"""Shared ETF schema and feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_STANDARD_COLUMNS = [
    "date",
    "code",
    "instrument",
    "name",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turn",
    "source",
    "adjust_type",
]


NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turn",
    "pre_close",
]


def standardize_etf_ohlcv_frame(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    code_col: str = "code",
    name_col: str | None = "name",
    source: str | None = None,
    adjust_type: str | None = None,
) -> pd.DataFrame:
    """Normalize any ETF daily frame into the project standard schema."""
    if df.empty:
        return df.copy()

    data = df.copy()
    rename_map = {}
    if date_col in data.columns and date_col != "date":
        rename_map[date_col] = "date"
    if code_col in data.columns and code_col != "code":
        rename_map[code_col] = "code"
    if "instrument" in data.columns and "code" not in data.columns:
        rename_map["instrument"] = "code"
    if name_col and name_col in data.columns and name_col != "name":
        rename_map[name_col] = "name"
    data = data.rename(columns=rename_map)

    if "code" not in data.columns:
        raise ValueError("ETF frame must include a code/instrument column.")
    if "date" not in data.columns:
        raise ValueError("ETF frame must include a date column.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["code"] = data["code"].astype(str).str.strip()
    data["instrument"] = data["code"]

    if "name" not in data.columns:
        data["name"] = None

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if source is not None:
        data["source"] = source
    elif "source" not in data.columns:
        data["source"] = None

    if adjust_type is not None:
        data["adjust_type"] = adjust_type
    elif "adjust_type" not in data.columns:
        data["adjust_type"] = None

    ordered = [col for col in DEFAULT_STANDARD_COLUMNS if col in data.columns]
    remaining = [col for col in data.columns if col not in ordered]
    return data[ordered + remaining]


def add_etf_derived_features(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    code_col: str = "code",
) -> pd.DataFrame:
    """Add the shared technical features used by the agent."""
    if df.empty:
        return df.copy()

    data = df.copy()
    if code_col not in data.columns:
        if "instrument" in data.columns:
            code_col = "instrument"
        else:
            raise ValueError("DataFrame must include a code or instrument column.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values([code_col, "date"]).reset_index(drop=True)

    grouped = data.groupby(code_col, group_keys=False)
    close = pd.to_numeric(data["close"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else close
    high = pd.to_numeric(data["high"], errors="coerce") if "high" in data.columns else close
    low = pd.to_numeric(data["low"], errors="coerce") if "low" in data.columns else close
    volume = pd.to_numeric(data["volume"], errors="coerce") if "volume" in data.columns else pd.Series(np.nan, index=data.index)
    amount = pd.to_numeric(data["amount"], errors="coerce") if "amount" in data.columns else pd.Series(np.nan, index=data.index)

    data["return_1d"] = grouped["close"].pct_change()
    data["return_5d"] = grouped["close"].transform(lambda s: s / s.shift(5) - 1)
    data["momentum_5d"] = data["return_5d"]
    data["momentum_20d"] = grouped["close"].transform(lambda s: s / s.shift(20) - 1)
    data["momentum_60d"] = grouped["close"].transform(lambda s: s / s.shift(60) - 1)
    data["future_return_1d"] = grouped["close"].transform(lambda s: s.shift(-1) / s - 1)
    data["future_return_5d"] = grouped["close"].transform(lambda s: s.shift(-5) / s - 1)
    data["future_return_10d"] = grouped["close"].transform(lambda s: s.shift(-10) / s - 1)
    data["future_return_20d"] = grouped["close"].transform(lambda s: s.shift(-20) / s - 1)

    data["ma_5d"] = grouped["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    data["ma_20d"] = grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    data["ma_60d"] = grouped["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    data["ma_gap_20d"] = data["close"] / data["ma_20d"] - 1
    data["ma_gap_60d"] = data["close"] / data["ma_60d"] - 1
    data["trend_strength"] = data["ma_5d"] / data["ma_20d"] - 1
    data["_above_ma20"] = (data["close"] > data["ma_20d"]).astype(float)
    data["trend_persistence_20d"] = grouped["_above_ma20"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    data["volume_ratio_5d"] = grouped["volume"].transform(lambda s: s / s.rolling(5, min_periods=5).mean())
    data["volume_ratio_20d"] = grouped["volume"].transform(lambda s: s / s.rolling(20, min_periods=20).mean())
    data["amount_ratio_5d"] = grouped["amount"].transform(lambda s: s / s.rolling(5, min_periods=5).mean())
    data["amount_ratio_20d"] = grouped["amount"].transform(lambda s: s / s.rolling(20, min_periods=20).mean())
    data["crowding_amount_ratio_5d"] = data["amount_ratio_5d"] / data["amount_ratio_20d"]
    data["volume_price_divergence_20d"] = data["amount_ratio_20d"] / (data["momentum_20d"].abs() + 1e-3)
    data["volume_breakout_20d"] = data["amount_ratio_20d"] * data["ma_gap_20d"]

    data["volatility_20d"] = grouped["return_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    data["volatility_60d"] = grouped["return_1d"].transform(lambda s: s.rolling(60, min_periods=60).std())

    # BBI (Bull-Bear Index) = average of MA(3) + MA(6) + MA(12) + MA(24)
    bbi_ma3 = grouped["close"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    bbi_ma6 = grouped["close"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    bbi_ma12 = grouped["close"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    bbi_ma24 = grouped["close"].transform(lambda s: s.rolling(24, min_periods=24).mean())
    data["bbi_indicator"] = (bbi_ma3 + bbi_ma6 + bbi_ma12 + bbi_ma24) / 4.0
    data["bbi_gap"] = data["close"] / data["bbi_indicator"] - 1
    data["risk_adjusted_momentum_20d"] = data["momentum_20d"] / (data["volatility_20d"] + 1e-9)
    data["drawdown_20d"] = data["close"] / grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).max()) - 1
    data["drawdown_60d"] = data["close"] / grouped["close"].transform(lambda s: s.rolling(60, min_periods=60).max()) - 1
    data["new_high_distance_60d"] = data["close"] / grouped["close"].transform(lambda s: s.rolling(60, min_periods=60).max()) - 1
    data["breakout_60d"] = data["close"] / grouped["high"].transform(lambda s: s.rolling(60, min_periods=60).max()) - 1
    data["ma_alignment_score"] = (data["ma_5d"] / data["ma_20d"] - 1) + (data["ma_20d"] / data["ma_60d"] - 1)
    data["reversal_5d"] = -data["return_5d"]
    data["ma_deviation_reversal_20d"] = -data["ma_gap_20d"]
    data["downside_volatility_20d"] = grouped["return_1d"].transform(
        lambda s: s.clip(upper=0).rolling(20, min_periods=20).std()
    )

    delta = grouped["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(data[code_col]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(data[code_col]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = avg_gain / (avg_loss + 1e-9)
    data["rsi_14d"] = 100 - (100 / (1 + rs))

    data["_signed_volume"] = np.sign(data["close"].diff()).fillna(0) * data["volume"].fillna(0)
    data["obv_20d"] = grouped["_signed_volume"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    data["obv_trend_20d"] = grouped["_signed_volume"].transform(lambda s: s.cumsum().diff(20))

    if benchmark_df is not None and not benchmark_df.empty:
        data = _merge_benchmark_features(data, benchmark_df, code_col=code_col)
    else:
        if "relative_strength_20d" not in data.columns:
            data["relative_strength_20d"] = pd.NA
        if "relative_strength_60d" not in data.columns:
            data["relative_strength_60d"] = pd.NA
        if "benchmark_return_1d" not in data.columns:
            data["benchmark_return_1d"] = pd.NA
        if "beta_60d" not in data.columns:
            data["beta_60d"] = pd.NA

    data = data.drop(columns=[col for col in ["_above_ma20", "_signed_volume"] if col in data.columns])
    return data


def _merge_benchmark_features(
    etf_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    code_col: str,
) -> pd.DataFrame:
    benchmark = benchmark_df.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark = benchmark.sort_values(["date"]).reset_index(drop=True)
    benchmark["benchmark_return_1d"] = benchmark["close"].pct_change()
    benchmark["benchmark_return_20d"] = benchmark["close"] / benchmark["close"].shift(20) - 1
    benchmark["benchmark_return_60d"] = benchmark["close"] / benchmark["close"].shift(60) - 1
    benchmark = benchmark[["date", "benchmark_return_1d", "benchmark_return_20d", "benchmark_return_60d", "close"]].rename(
        columns={"close": "benchmark_close"}
    )

    merged = etf_df.merge(benchmark, on="date", how="left")
    merged["relative_strength_20d"] = merged["momentum_20d"] - merged["benchmark_return_20d"]
    merged["relative_strength_60d"] = merged["momentum_60d"] - merged["benchmark_return_60d"]

    def _calc_beta(frame: pd.DataFrame) -> pd.Series:
        cov = frame["return_1d"].rolling(60, min_periods=20).cov(frame["benchmark_return_1d"])
        var = frame["benchmark_return_1d"].rolling(60, min_periods=20).var()
        return cov / (var + 1e-9)

    merged["beta_60d"] = merged.groupby(code_col, group_keys=False).apply(_calc_beta).reset_index(level=0, drop=True)
    return merged


def ensure_etf_feature_columns(
    df: pd.DataFrame,
    *,
    benchmark_df: pd.DataFrame | None = None,
    code_col: str = "code",
) -> pd.DataFrame:
    """Normalize raw ETF data then add all supported derived features."""
    standardized = standardize_etf_ohlcv_frame(df, code_col=code_col)
    return add_etf_derived_features(standardized, benchmark_df=benchmark_df, code_col="code")
