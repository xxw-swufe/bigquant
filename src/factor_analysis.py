"""Single factor validation utilities."""

import numpy as np
import pandas as pd


def calc_daily_ic(df: pd.DataFrame, factor_col: str, target_col: str) -> pd.Series:
    if factor_col not in df.columns or target_col not in df.columns:
        return pd.Series(dtype=float)
    data = df[["date", factor_col, target_col]].dropna()
    if data.empty:
        return pd.Series(dtype=float)
    return (
        data.groupby("date")
        .apply(lambda x: x[factor_col].corr(x[target_col], method="spearman"))
        .dropna()
    )


def calc_quantile_return(
    df: pd.DataFrame,
    factor_col: str,
    target_col: str,
    n_quantiles: int = 5,
) -> pd.Series:
    if factor_col not in df.columns or target_col not in df.columns:
        return pd.Series(dtype=float)
    data = df[["date", "instrument", factor_col, target_col]].dropna().copy()
    if data.empty:
        return pd.Series(dtype=float)

    def assign_quantile(x: pd.Series) -> pd.Series:
        try:
            return pd.qcut(x, n_quantiles, labels=False, duplicates="drop") + 1
        except Exception:
            return pd.Series(index=x.index, data=np.nan)

    data["quantile"] = data.groupby("date")[factor_col].transform(assign_quantile)
    return data.groupby("quantile")[target_col].mean()


def summarize_ic(ic: pd.Series) -> dict:
    if ic.empty:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_positive_ratio": np.nan,
            "ic_count": 0,
        }
    ic_std = ic.std()
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic_std),
        "icir": float(ic.mean() / ic_std) if ic_std and not np.isnan(ic_std) else np.nan,
        "ic_positive_ratio": float((ic > 0).mean()),
        "ic_count": int(ic.count()),
    }


def run_factor_analysis(
    df: pd.DataFrame,
    factor_cols: list[str],
    target_cols: list[str],
) -> dict:
    result = {}
    for factor_col in factor_cols:
        if factor_col not in df.columns:
            result[factor_col] = {}
            for target_col in target_cols:
                result[factor_col][target_col] = {
                    "ic_summary": summarize_ic(pd.Series(dtype=float)),
                    "ic_series": pd.Series(dtype=float),
                    "quantile_return": pd.Series(dtype=float),
                    "coverage": 0.0,
                    "skipped": True,
                    "skip_reason": f"Missing factor column: {factor_col}",
                }
            continue
        result[factor_col] = {}
        for target_col in target_cols:
            if target_col not in df.columns:
                result[factor_col][target_col] = {
                    "ic_summary": summarize_ic(pd.Series(dtype=float)),
                    "ic_series": pd.Series(dtype=float),
                    "quantile_return": pd.Series(dtype=float),
                    "coverage": float(df[factor_col].notna().mean()) if factor_col in df else 0.0,
                    "skipped": True,
                    "skip_reason": f"Missing target column: {target_col}",
                }
                continue
            ic = calc_daily_ic(df, factor_col, target_col)
            quantile_return = calc_quantile_return(df, factor_col, target_col)
            result[factor_col][target_col] = {
                "ic_summary": summarize_ic(ic),
                "ic_series": ic,
                "quantile_return": quantile_return,
                "coverage": float(df[factor_col].notna().mean()) if factor_col in df else 0.0,
            }
    return result
