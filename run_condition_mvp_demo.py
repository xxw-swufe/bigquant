"""Run the first condition-study MVP with synthetic ETF data.

This script is only for local workflow verification. Formal research data must
come from BigQuant in the notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.condition_analysis import (
    diagnose_condition_result,
    ensure_condition_features,
    run_conditional_probability_test,
)
from src.condition_parser import parse_condition_research
from src.condition_report import generate_condition_report, save_condition_report


def make_mock_etf_data(n_days: int = 260, n_etfs: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for idx in range(n_etfs):
        instrument = f"ETF{idx + 1:03d}"
        returns = rng.normal(0.0004, 0.012, size=n_days)
        close = 1.0 * np.cumprod(1 + returns)
        volume = rng.lognormal(mean=12, sigma=0.35, size=n_days)
        turnover = rng.uniform(0.5, 7.0, size=n_days)
        rows.extend(
            {
                "date": date,
                "instrument": instrument,
                "close": close[i],
                "volume": volume[i],
                "turnover": turnover[i],
            }
            for i, date in enumerate(dates)
        )
    return pd.DataFrame(rows)


def main() -> dict:
    user_idea = "当单日量比 > 1、换手率 < 5、今日上涨时，明天这个 ETF 上涨几率大吗？"
    hypothesis = parse_condition_research(user_idea)
    df = ensure_condition_features(make_mock_etf_data())
    result = run_conditional_probability_test(
        df=df,
        conditions=hypothesis["conditions"],
        target=hypothesis["target"],
        min_event_count=100,
    )
    diagnosis = diagnose_condition_result(result)
    report = generate_condition_report(hypothesis, result, diagnosis)
    save_condition_report(report)
    Path("outputs").mkdir(exist_ok=True)
    result["yearly_stats"].to_csv("outputs/condition_yearly_stats.csv", index=False)
    return {
        "event_count": result["event_count"],
        "event_up_probability": result["event_up_probability"],
        "baseline_up_probability": result["baseline_up_probability"],
        "probability_lift": result["probability_lift"],
        "research_decision": diagnosis["research_decision"],
    }


if __name__ == "__main__":
    print(main())

