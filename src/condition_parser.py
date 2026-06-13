"""Condition-rule parser for the first ETF event-study MVP."""


def parse_condition_research(user_idea: str) -> dict:
    """Parse a simple natural-language ETF condition study.

    First MVP intentionally supports the canonical example:
    volume ratio > 1, turnover < 5%, today's return > 0, and asks whether the
    next trading day return is positive.
    """
    # [AI-CORE]
    return {
        "research_type": "conditional_probability_test",
        "user_idea": user_idea,
        "asset_universe": "ETF",
        "research_goal": "检验满足单日量价条件后，ETF 下一交易日上涨概率是否高于基准概率",
        "conditions": [
            {
                "field": "volume_ratio_5d",
                "operator": ">",
                "value": 1.0,
                "description": "当日成交量大于过去 5 日平均成交量",
            },
            {
                "field": "turnover",
                "operator": "<",
                "value": 5.0,
                "description": "当日换手率低于 5%",
            },
            {
                "field": "return_1d",
                "operator": ">",
                "value": 0.0,
                "description": "当日收盘价较上一交易日上涨",
            },
        ],
        "target": {
            "field": "future_return_1d",
            "operator": ">",
            "value": 0.0,
            "description": "下一交易日收益率为正",
        },
        "required_columns": [
            "date",
            "instrument",
            "close",
            "volume",
            "turnover",
            "volume_ratio_5d",
            "return_1d",
            "future_return_1d",
        ],
        "not_investment_advice": True,
    }

