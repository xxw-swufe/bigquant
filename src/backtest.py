"""Simplified ETF rotation backtest."""

import numpy as np
import pandas as pd


def run_top_pct_backtest(
    df: pd.DataFrame,
    score_col: str = "composite_score",
    target_col: str = "future_return_5d",
    top_pct: float = 0.10,
    min_holdings: int = 3,
    cost_bps: float = 10.0,
) -> dict:
    """Run a research-oriented Top Percent equal-weight backtest."""
    data = df[["date", "instrument", score_col, target_col]].dropna().copy()
    if data.empty:
        raise ValueError("Backtest input is empty after dropping missing values.")

    portfolio_rows = []
    previous_holdings: set[str] = set()
    for date, group in data.groupby("date"):
        group = group.sort_values(score_col, ascending=False)
        n_holdings = max(min_holdings, int(np.ceil(len(group) * top_pct)))
        selected = group.head(n_holdings).copy()
        holdings = set(selected["instrument"])
        turnover = len(holdings.symmetric_difference(previous_holdings)) / max(len(holdings), 1)
        cost = turnover * cost_bps / 10000
        daily_return = selected[target_col].mean() - cost
        portfolio_rows.append(
            {
                "date": date,
                "return": daily_return,
                "gross_return": selected[target_col].mean(),
                "turnover": turnover,
                "cost": cost,
                "holding_count": len(selected),
                "holdings": list(selected["instrument"]),
            }
        )
        previous_holdings = holdings

    daily_returns = pd.DataFrame(portfolio_rows).set_index("date").sort_index()
    performance = calc_performance(daily_returns["return"])
    performance.update(
        {
            "average_turnover": float(daily_returns["turnover"].mean()),
            "total_cost": float(daily_returns["cost"].sum()),
            "average_holding_count": float(daily_returns["holding_count"].mean()),
        }
    )
    return {
        "daily_returns": daily_returns,
        "cumulative_returns": (1 + daily_returns["return"]).cumprod() - 1,
        "performance": performance,
    }


def calc_performance(returns: pd.Series, periods_per_year: int = 252) -> dict:
    if returns.empty:
        return {}
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1
    max_drawdown = ((cumulative / cumulative.cummax()) - 1).min()
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / volatility if volatility and not np.isnan(volatility) else np.nan
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "win_rate": float((returns > 0).mean()),
        "volatility": float(volatility),
    }

