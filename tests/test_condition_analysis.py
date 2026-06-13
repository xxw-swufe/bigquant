import pandas as pd

from src.condition_analysis import (
    build_condition_mask,
    ensure_condition_features,
    run_conditional_probability_test,
)
from src.condition_parser import parse_condition_research


def test_condition_probability_pipeline_runs():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=8).tolist() * 2,
            "instrument": ["ETF001"] * 8 + ["ETF002"] * 8,
            "close": [1.00, 1.01, 1.02, 1.01, 1.03, 1.04, 1.03, 1.05] * 2,
            "volume": [100, 110, 120, 90, 150, 160, 130, 180] * 2,
            "turnover": [3, 4, 4, 6, 3, 2, 4, 3] * 2,
        }
    )
    hypothesis = parse_condition_research("量比 > 1，换手率 < 5，今日上涨，明天上涨概率大吗？")
    featured = ensure_condition_features(df)
    mask = build_condition_mask(featured, hypothesis["conditions"])
    assert mask.dtype == bool

    result = run_conditional_probability_test(
        featured,
        conditions=hypothesis["conditions"],
        target=hypothesis["target"],
        min_event_count=1,
    )
    assert result["total_count"] > 0
    assert "event_up_probability" in result
    assert "yearly_stats" in result

