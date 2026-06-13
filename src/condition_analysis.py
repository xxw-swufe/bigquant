"""Conditional probability event-study utilities."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable

import numpy as np
import pandas as pd


OPERATORS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_condition_features(
    df: pd.DataFrame,
    volume_window: int = 5,
) -> pd.DataFrame:
    """Compute the first-MVP event-study features when raw columns exist."""
    data = df.copy()
    require_columns(data, ["date", "instrument", "close", "volume"])
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["instrument", "date"])

    grouped = data.groupby("instrument", group_keys=False)
    if "return_1d" not in data.columns:
        data["return_1d"] = grouped["close"].pct_change()
    if "volume_ratio_5d" not in data.columns:
        rolling_volume = grouped["volume"].transform(
            lambda x: x.rolling(volume_window, min_periods=volume_window).mean()
        )
        data["volume_ratio_5d"] = data["volume"] / rolling_volume
    if "future_return_1d" not in data.columns:
        data["future_return_1d"] = grouped["close"].shift(-1) / data["close"] - 1
    return data


def build_condition_mask(df: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    """Build an event mask from structured conditions."""
    if not conditions:
        raise ValueError("At least one condition is required.")

    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        field = condition["field"]
        op = condition["operator"]
        value = condition["value"]
        if op not in OPERATORS:
            raise ValueError(f"Unsupported operator: {op}")
        require_columns(df, [field])
        mask = mask & OPERATORS[op](df[field], value)
    return mask.fillna(False)


def run_conditional_probability_test(
    df: pd.DataFrame,
    conditions: list[dict],
    target: dict | None = None,
    min_event_count: int = 200,
) -> dict:
    """Evaluate whether condition events have better next-period outcomes."""
    target = target or {"field": "future_return_1d", "operator": ">", "value": 0.0}
    target_col = target["field"]
    require_columns(df, ["date", "instrument", target_col])

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    event_mask = build_condition_mask(data, conditions)
    valid_mask = data[target_col].notna()
    data = data.loc[valid_mask].copy()
    event_mask = event_mask.loc[valid_mask]

    target_up = OPERATORS[target["operator"]](data[target_col], target["value"])
    event_data = data.loc[event_mask].copy()
    event_target_up = target_up.loc[event_mask]

    event_count = int(event_mask.sum())
    total_count = int(len(data))
    event_up_probability = _safe_mean(event_target_up)
    baseline_up_probability = _safe_mean(target_up)
    event_mean_return = _safe_mean(event_data[target_col])
    baseline_mean_return = _safe_mean(data[target_col])

    yearly = _calc_group_stats(data, event_mask, target_up, target_col, group_key=data["date"].dt.year)
    by_instrument = _calc_group_stats(data, event_mask, target_up, target_col, group_key=data["instrument"])

    return {
        "research_type": "conditional_probability_test",
        "conditions": conditions,
        "target": target,
        "event_count": event_count,
        "total_count": total_count,
        "event_ratio": event_count / total_count if total_count else math.nan,
        "event_up_probability": event_up_probability,
        "baseline_up_probability": baseline_up_probability,
        "probability_lift": event_up_probability - baseline_up_probability,
        "event_mean_return": event_mean_return,
        "baseline_mean_return": baseline_mean_return,
        "mean_return_lift": event_mean_return - baseline_mean_return,
        "min_event_count": min_event_count,
        "is_sample_sufficient": event_count >= min_event_count,
        "yearly_stats": yearly,
        "instrument_stats": by_instrument,
    }


def diagnose_condition_result(result: dict) -> dict:
    """Diagnose sample size, probability lift, return lift, and stability."""
    # [AI-CORE]
    strengths = []
    risks = []
    suggestions = []

    if not result["is_sample_sufficient"]:
        risks.append(
            f"满足条件的样本数为 {result['event_count']}，低于最小样本阈值 {result['min_event_count']}，统计稳定性不足。"
        )

    probability_lift = result["probability_lift"]
    mean_return_lift = result["mean_return_lift"]
    if probability_lift > 0.03:
        strengths.append("条件样本的次日上涨概率相对全样本提升超过 3 个百分点，具备进一步研究价值。")
    elif probability_lift > 0.01:
        strengths.append("条件样本的次日上涨概率相对全样本有小幅提升。")
        suggestions.append("建议继续观察分年份稳定性和交易成本后的有效性。")
    else:
        risks.append("条件样本的次日上涨概率相对全样本提升不明显。")

    if mean_return_lift > 0:
        strengths.append("条件样本的次日平均收益高于全样本平均水平。")
    else:
        risks.append("条件样本的次日平均收益没有高于全样本平均水平。")

    yearly_stats = result.get("yearly_stats", pd.DataFrame())
    if not yearly_stats.empty and "probability_lift" in yearly_stats:
        positive_year_ratio = float((yearly_stats["probability_lift"] > 0).mean())
        if positive_year_ratio >= 0.6:
            strengths.append("该条件在多数年份的概率提升为正，稳定性相对更好。")
        else:
            risks.append("该条件分年份表现不稳定，可能依赖特定市场环境。")

    if not suggestions:
        suggestions.append("建议后续将该条件作为二值信号，与趋势、相对强度、风险因子共同检验。")

    decision = "谨慎继续"
    if result["is_sample_sufficient"] and probability_lift > 0.03 and mean_return_lift > 0:
        decision = "继续研究"
    elif (not result["is_sample_sufficient"]) or (probability_lift <= 0 and mean_return_lift <= 0):
        decision = "暂不建议继续"

    return {
        "summary": "本诊断基于条件事件样本数量、次日上涨概率提升、次日平均收益提升和分年份稳定性。",
        "strengths": strengths,
        "risks": risks,
        "improvement_suggestions": suggestions,
        "research_decision": decision,
        "not_investment_advice": True,
    }


def _calc_group_stats(
    data: pd.DataFrame,
    event_mask: pd.Series,
    target_up: pd.Series,
    target_col: str,
    group_key: pd.Series,
) -> pd.DataFrame:
    rows = []
    for group_value in sorted(pd.Series(group_key).dropna().unique()):
        group_mask = group_key == group_value
        group_event_mask = event_mask & group_mask
        group_valid_mask = group_mask
        event_count = int(group_event_mask.sum())
        total_count = int(group_valid_mask.sum())
        if total_count == 0:
            continue
        event_up_probability = _safe_mean(target_up.loc[group_event_mask])
        baseline_up_probability = _safe_mean(target_up.loc[group_valid_mask])
        event_mean_return = _safe_mean(data.loc[group_event_mask, target_col])
        baseline_mean_return = _safe_mean(data.loc[group_valid_mask, target_col])
        rows.append(
            {
                "group": group_value,
                "event_count": event_count,
                "total_count": total_count,
                "event_up_probability": event_up_probability,
                "baseline_up_probability": baseline_up_probability,
                "probability_lift": event_up_probability - baseline_up_probability,
                "event_mean_return": event_mean_return,
                "baseline_mean_return": baseline_mean_return,
                "mean_return_lift": event_mean_return - baseline_mean_return,
            }
        )
    return pd.DataFrame(rows)


def _safe_mean(values) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.nanmean(values))

