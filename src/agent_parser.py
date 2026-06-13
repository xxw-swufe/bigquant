"""Natural language strategy parser."""


def parse_user_strategy(user_idea: str) -> dict:
    """Parse a user's ETF research idea into a structured hypothesis.

    MVP uses a deterministic template. This function is intentionally marked as
    an AI-core extension point for later LLM integration.
    """
    # [AI-CORE]
    return {
        "user_idea": user_idea,
        "research_goal": "检验行业 ETF 的趋势延续、相对强度和资金关注度信号是否有效",
        "asset_universe": "行业 ETF / 主题 ETF",
        "prediction_targets": ["future_return_5d", "future_return_20d"],
        "core_variables": [
            "momentum",
            "relative_strength",
            "amount_ratio",
            "trend_strength",
            "volatility",
        ],
        "rebalance_frequency": "weekly",
        "portfolio_rule": "按综合得分选择 Top 10% ETF 等权持有",
        "not_investment_advice": True,
    }

