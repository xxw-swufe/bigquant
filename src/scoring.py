"""Factor normalization and composite scoring."""

import numpy as np
import pandas as pd


def normalize_factors(
    df: pd.DataFrame,
    factor_directions: dict[str, str],
    method: str = "rank",
) -> pd.DataFrame:
    """Normalize factors cross-sectionally by date."""
    data = df.copy()
    for factor_col, direction in factor_directions.items():
        score_col = f"{factor_col}_score"
        if method != "rank":
            raise ValueError(f"Unsupported normalization method: {method}")
        data[score_col] = data.groupby("date")[factor_col].rank(pct=True)
        if direction == "lower_better":
            data[score_col] = 1 - data[score_col]
    return data


def build_weight_schemes(factor_cols: list[str], factor_result: dict | None = None) -> dict[str, dict[str, float]]:
    equal_weight = {factor: 1 / len(factor_cols) for factor in factor_cols}
    hypothesis_weight = {
        "relative_strength_20d": 0.35,
        "momentum_20d": 0.25,
        "amount_ratio_20d": 0.20,
        "trend_strength": 0.10,
        "volatility_20d": 0.10,
    }
    hypothesis_weight = {factor: hypothesis_weight.get(factor, equal_weight[factor]) for factor in factor_cols}
    hypothesis_weight = normalize_weights(hypothesis_weight)

    schemes = {
        "equal_weight": equal_weight,
        "hypothesis_weight": hypothesis_weight,
    }
    if factor_result:
        schemes["icir_weight"] = build_icir_weight(factor_cols, factor_result)
    return schemes


def build_icir_weight(factor_cols: list[str], factor_result: dict) -> dict[str, float]:
    raw = {}
    for factor in factor_cols:
        target_results = factor_result.get(factor, {})
        icirs = [
            abs(item.get("ic_summary", {}).get("icir", np.nan))
            for item in target_results.values()
        ]
        valid_icirs = [value for value in icirs if not np.isnan(value)]
        raw[factor] = float(np.mean(valid_icirs)) if valid_icirs else 0.0
    if sum(raw.values()) <= 0:
        return {factor: 1 / len(factor_cols) for factor in factor_cols}
    return normalize_weights(raw)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(abs(value) for value in weights.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return {key: value / total for key, value in weights.items()}


def compute_composite_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    data = df.copy()
    score = pd.Series(0.0, index=data.index)
    for factor_col, weight in weights.items():
        score_col = f"{factor_col}_score"
        if score_col not in data.columns:
            raise ValueError(f"Missing normalized factor score column: {score_col}")
        score = score + data[score_col].fillna(0.0) * weight
    data["composite_score"] = score
    return data

