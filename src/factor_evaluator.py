"""Whitelist factor evaluator.

Given a raw ETF frame and a set of requested factor names, ensure each
requested column exists in the frame. Returns a structured result
(`FactorAvailability`) instead of raising on the missing path so
upstream callers can render graceful `unmet_research_plan` reports.

Contract:

- A factor is "supported" iff it appears in
  `src.factor_registry.get_supported_factor_names()` AND has a `compute`
  function registered in `FACTOR_COMPUTERS`.
- If the column already exists in the DataFrame, it's marked `existing`.
- If supported and the column is missing but the required fields exist,
  the column is computed with the whitelisted pandas function (no
  `eval`, no BigQuant DSL runtime).
- If the factor is unknown / not-implemented, it's recorded as
  `not_implemented`.
- If the factor is supported but required fields are not in the
  DataFrame, it's recorded as `missing_required_fields`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from src.factor_registry import (
    get_factor_spec,
    get_supported_factor_names,
)


# ---------------------------------------------------------------------------
# Compute functions. Each compute function takes a DataFrame whose `code`
# column contains the instrument id, returns the Series for the requested
# factor. Whitelist only — no formula strings.
# ---------------------------------------------------------------------------


def _require_code_col(df: pd.DataFrame) -> str:
    if "code" in df.columns:
        return "code"
    if "instrument" in df.columns:
        return "instrument"
    raise ValueError("DataFrame must include a 'code' or 'instrument' column.")


def _grouped_close(df: pd.DataFrame, code_col: str):
    return df.groupby(code_col, group_keys=False)["close"]


def _compute_momentum(df: pd.DataFrame, period: int) -> pd.Series:
    code_col = _require_code_col(df)
    return _grouped_close(df, code_col).transform(lambda s: s / s.shift(period) - 1)


def _compute_ma_gap(df: pd.DataFrame, period: int) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    ma = grouped["close"].transform(lambda s: s.rolling(period, min_periods=period).mean())
    return df["close"] / ma - 1


def _compute_ma_rolling(df: pd.DataFrame, period: int) -> pd.Series:
    code_col = _require_code_col(df)
    return df.groupby(code_col, group_keys=False)["close"].transform(
        lambda s: s.rolling(period, min_periods=period).mean()
    )


def _compute_trend_strength(df: pd.DataFrame) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    ma5 = grouped["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma20 = grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    return ma5 / ma20 - 1


def _compute_volume_ratio(df: pd.DataFrame, period: int) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    avg = grouped["volume"].transform(lambda s: s.rolling(period, min_periods=period).mean())
    return df["volume"] / avg


def _compute_amount_ratio(df: pd.DataFrame, period: int) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    avg = grouped["amount"].transform(lambda s: s.rolling(period, min_periods=period).mean())
    return df["amount"] / avg


def _compute_volatility(df: pd.DataFrame, period: int) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    return_1d = grouped["close"].transform(lambda s: s.pct_change())
    return return_1d.groupby(df[code_col]).transform(
        lambda s: s.rolling(period, min_periods=period).std()
    )


def _compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    delta = grouped["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df[code_col]).transform(
        lambda s: s.rolling(period, min_periods=period).mean()
    )
    avg_loss = loss.groupby(df[code_col]).transform(
        lambda s: s.rolling(period, min_periods=period).mean()
    )
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_bbi(df: pd.DataFrame) -> pd.Series:
    code_col = _require_code_col(df)
    grouped = df.groupby(code_col, group_keys=False)
    ma3 = grouped["close"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    ma6 = grouped["close"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    ma12 = grouped["close"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    ma24 = grouped["close"].transform(lambda s: s.rolling(24, min_periods=24).mean())
    return (ma3 + ma6 + ma12 + ma24) / 4.0


# Whitelist of supported factors -> compute function. Only the 10 whitelist
# factors are registered here. New factors must be added explicitly.
FACTOR_COMPUTERS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "momentum_20d": lambda df: _compute_momentum(df, 20),
    "momentum_60d": lambda df: _compute_momentum(df, 60),
    "ma_gap_20d": lambda df: _compute_ma_gap(df, 20),
    "trend_strength": _compute_trend_strength,
    "amount_ratio_20d": lambda df: _compute_amount_ratio(df, 20),
    "volume_ratio_5d": lambda df: _compute_volume_ratio(df, 5),
    "volume_ratio_20d": lambda df: _compute_volume_ratio(df, 20),
    "volatility_20d": lambda df: _compute_volatility(df, 20),
    "rsi_14d": lambda df: _compute_rsi(df, 14),
    "bbi_indicator": _compute_bbi,
}


WHITELIST_FACTOR_NAMES = frozenset(FACTOR_COMPUTERS.keys())


# ---------------------------------------------------------------------------
# Structured availability result
# ---------------------------------------------------------------------------


@dataclass
class FactorAvailability:
    """Outcome of an `ensure_factors_available` call.

    Attributes
    ----------
    df:
        The DataFrame after computed columns have been added.
    requested_factors:
        Input factor name list (preserves order, deduped).
    available_factors:
        Subset that is now actually present in `df.columns`. Use this
        list as the input to downstream IC / backtest stages.
    existing_factors:
        Subset that already existed in the input DataFrame (no work done).
    computed_factors:
        Subset that was missing and has now been computed via whitelist.
    not_implemented:
        Names that the registry does not know about.
    missing_required_fields:
        Names that are supported but the input DataFrame lacks required
        raw fields (e.g. `volume` missing for `volume_ratio_5d`).
    ok:
        True iff there are no entries in `not_implemented` and no
        entries in `missing_required_fields`. Existing / computed
        factors are still allowed.
    """

    df: pd.DataFrame
    requested_factors: list[str] = field(default_factory=list)
    available_factors: list[str] = field(default_factory=list)
    existing_factors: list[str] = field(default_factory=list)
    computed_factors: list[str] = field(default_factory=list)
    not_implemented: list[str] = field(default_factory=list)
    missing_required_fields: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.not_implemented and not self.missing_required_fields


def ensure_factors_available(
    df: pd.DataFrame,
    factor_names: list[str] | None,
) -> FactorAvailability:
    """Ensure each requested factor is present in `df.columns`.

    See module docstring for the contract.
    """
    result = FactorAvailability(df=df)
    if not factor_names:
        return result

    seen: set[str] = set()
    requested: list[str] = []
    for name in factor_names:
        if not name or name in seen:
            continue
        seen.add(name)
        requested.append(name)
    result.requested_factors = requested

    supported_names = set(get_supported_factor_names()) | WHITELIST_FACTOR_NAMES

    work_df = df
    for name in requested:
        if name in work_df.columns:
            result.existing_factors.append(name)
            result.available_factors.append(name)
            continue

        if name not in supported_names or name not in FACTOR_COMPUTERS:
            result.not_implemented.append(name)
            continue

        spec = get_factor_spec(name) or {}
        required = list(spec.get("required_fields") or [])
        missing_fields = [f for f in required if f and f not in work_df.columns]
        if missing_fields:
            result.missing_required_fields.append(
                {"name": name, "missing_fields": missing_fields}
            )
            continue

        # Compute via whitelisted function — no eval, no formula string.
        compute = FACTOR_COMPUTERS[name]
        try:
            work_df = work_df.copy()
            work_df[name] = compute(work_df)
            result.computed_factors.append(name)
            result.available_factors.append(name)
        except Exception as exc:  # pragma: no cover - surface as missing
            result.missing_required_fields.append(
                {"name": name, "missing_fields": [f"compute_error: {type(exc).__name__}"]}
            )

    result.df = work_df
    return result


__all__ = [
    "FACTOR_COMPUTERS",
    "WHITELIST_FACTOR_NAMES",
    "FactorAvailability",
    "ensure_factors_available",
]
