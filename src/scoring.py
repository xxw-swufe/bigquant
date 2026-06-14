"""Factor normalization and composite scoring."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def normalize_factors(
    df: pd.DataFrame,
    factor_directions: dict[str, str],
    method: str = "rank",
) -> pd.DataFrame:
    """Normalize factors cross-sectionally by date.

    This is the legacy, column-name based normalization entry point. It keeps
    compatibility with the existing notebook flow.
    """
    data = df.copy()
    for factor_col, direction in factor_directions.items():
        score_col = f"{factor_col}_score"
        if method != "rank":
            raise ValueError(f"Unsupported normalization method: {method}")
        data[score_col] = data.groupby("date")[factor_col].rank(pct=True, method="average")
        if direction == "lower_better":
            data[score_col] = 1 - data[score_col]
        data[score_col] = data[score_col].clip(0.0, 1.0)
    return data


def score_series(
    series: pd.Series,
    *,
    direction: str = "higher_better",
    score_shape: str = "linear",
    optimal_range: Optional[Tuple[float, float]] = None,
) -> pd.Series:
    """Convert a raw factor series to a 0-1 style score.

    Supported shapes:
    - linear: higher/lower is better
    - range_better: best within an interval
    - u_shape: both low and high are better, mid is worse
    - inverted_u: mid is better, extremes are worse
    """
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(np.nan, index=series.index)

    if score_shape == "linear":
        ranked = values.rank(pct=True, method="average")
        if direction == "lower_better":
            ranked = 1 - ranked
        return ranked.clip(0.0, 1.0)

    if score_shape == "range_better":
        if optimal_range is None:
            raise ValueError("range_better scoring requires optimal_range.")
        low, high = optimal_range
        midpoint = (low + high) / 2
        span = max(high - low, 1e-9)
        score = pd.Series(1.0, index=values.index, dtype=float)
        score = score.where((values >= low) & (values <= high), 0.0)
        edge_distance = (values - midpoint).abs() / (span / 2)
        score = score.where(score == 0.0, 1 - 0.25 * edge_distance)
        return score.clip(0.0, 1.0)

    if score_shape == "u_shape":
        center = values.median()
        spread = max((values.quantile(0.75) - values.quantile(0.25)) / 2, 1e-9)
        distance = (values - center).abs() / spread
        score = distance
        if direction == "lower_better":
            score = 1 - score.rank(pct=True, method="average")
        else:
            score = score.rank(pct=True, method="average")
        return score.clip(0.0, 1.0)

    if score_shape == "inverted_u":
        center = values.median()
        spread = max((values.quantile(0.75) - values.quantile(0.25)) / 2, 1e-9)
        distance = (values - center).abs() / spread
        score = 1 - distance
        return score.clip(0.0, 1.0)

    raise ValueError(f"Unsupported score shape: {score_shape}")


def normalize_factor_frame(
    df: pd.DataFrame,
    factor_metadata: dict[str, dict],
    method: str = "rank",
) -> pd.DataFrame:
    """Normalize a factor frame using metadata-driven score shapes."""
    data = df.copy()
    for factor_name, metadata in factor_metadata.items():
        if factor_name not in data.columns:
            continue
        score_col = f"{factor_name}_score"
        if method == "rank" and metadata.get("score_shape", "linear") == "linear":
            data[score_col] = data.groupby("date")[factor_name].rank(pct=True, method="average")
            if metadata.get("direction") == "lower_better":
                data[score_col] = 1 - data[score_col]
            data[score_col] = data[score_col].clip(0.0, 1.0)
            continue

        data[score_col] = data.groupby("date")[factor_name].transform(
            lambda s: score_series(
                s,
                direction=metadata.get("direction", "higher_better"),
                score_shape=metadata.get("score_shape", "linear"),
                optimal_range=metadata.get("optimal_range"),
            )
        )
    return data


def build_weight_schemes(factor_cols: list[str], factor_result: Optional[dict] = None) -> dict[str, dict[str, float]]:
    if not factor_cols:
        raise ValueError("factor_cols must not be empty.")

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


def compute_metadata_composite_score(
    df: pd.DataFrame,
    factor_metadata: dict[str, dict],
    weights: dict[str, float],
    method: str = "rank",
) -> pd.DataFrame:
    """Normalize factors using metadata then compute the weighted score."""
    normalized = normalize_factor_frame(df, factor_metadata, method=method)
    return compute_composite_score(normalized, weights)
