"""Routing priority tests for src.factor_resolver.resolve_factor_intent.

The router's job is to map a free-form Chinese research question onto the
right research pipeline. These tests pin the contract:

- Questions that name multiple factors and ask for excess return /
  ranking / backtest / rotation -> factor_research.
- Questions that ask for next-day / same-day / future-N-day up-probability
  after a named event -> conditional_event_study.
- A multi-factor signal (>=2 factor names) is enough to lift a question
  out of the "condition_expression" branch.
- Condition cues ("上涨概率", "反弹", "下一日" etc.) win over incidental
  factor-name matches from a too-greedy KB.
"""

from __future__ import annotations

from src.factor_resolver import resolve_factor_intent


# ---------------------------------------------------------------------------
# Multi-factor -> factor_research
# ---------------------------------------------------------------------------


def test_multi_factor_with_keywords_routes_to_factor_research():
    """The smoke-test #2 question: amount ratio + momentum + volatility, future 5d."""
    intent = resolve_factor_intent(
        "研究成交额放大、20日动量强、波动率低的ETF，未来5日是否有超额收益？"
    )
    assert intent["research_type"] == "factor_research"
    # Should resolve at least 2 factor names (momentum_20d, volatility_20d, amount_ratio_20d)
    assert len(intent.get("factor_names", [])) >= 2


def test_two_factor_phrase_routes_to_factor_research():
    intent = resolve_factor_intent("研究20日动量和低波动ETF未来5日表现")
    assert intent["research_type"] == "factor_research"
    assert "momentum_20d" in intent.get("factor_names", [])
    assert "volatility_20d" in intent.get("factor_names", [])


def test_rotation_strategy_routes_to_factor_research():
    intent = resolve_factor_intent("用动量、趋势、成交额和低波动因子构建 ETF 轮动策略")
    assert intent["research_type"] == "factor_research"


# ---------------------------------------------------------------------------
# Condition cue -> conditional_event_study
# ---------------------------------------------------------------------------


def test_up_probability_question_routes_to_condition_study():
    intent = resolve_factor_intent(
        "研究成交额放大超过1.5倍、当日上涨但涨幅小于2%的ETF，隔日上涨概率是否更高？"
    )
    assert intent["research_type"] == "conditional_event_study"


def test_rebound_probability_routes_to_condition_study():
    intent = resolve_factor_intent("研究连续3天下跌后，ETF 次日反弹概率是否提高？")
    assert intent["research_type"] == "conditional_event_study"


def test_short_chitchat_with_volume_routes_to_condition_study():
    """Even when factor names incidentally match, condition cue wins."""
    intent = resolve_factor_intent("缩量上涨后下一日怎么样")
    assert intent["research_type"] == "conditional_event_study"


def test_tomorrow_up_chance_routes_to_condition_study():
    intent = resolve_factor_intent("明日上涨情况如何")
    assert intent["research_type"] == "conditional_event_study"


# ---------------------------------------------------------------------------
# Priority invariant: condition cue beats incidental factor matching
# ---------------------------------------------------------------------------


def test_condition_cue_beats_many_factor_matches():
    """The "连续3天下跌" question matches 12 factor names but is a condition study."""
    intent = resolve_factor_intent("研究连续3天下跌后，ETF 次日反弹概率是否提高？")
    assert intent["research_type"] == "conditional_event_study"
    # KB can still surface many factor names — that's fine, the type wins.
    assert len(intent.get("factor_names", [])) >= 1


def test_research_rsi_routes_to_factor_research_not_not_implemented():
    intent = resolve_factor_intent("研究 RSI")
    assert "rsi_14d" in intent.get("factor_names", []), intent
    assert intent["research_type"] == "factor_research", intent
    assert "RSI" not in intent.get("recognized_not_implemented_terms", []), intent


def test_research_rsi_supplement_phrase_routes_to_factor_research():
    intent = resolve_factor_intent("研究一下 RSI")
    assert "rsi_14d" in intent.get("factor_names", []), intent
    assert intent["research_type"] == "factor_research", intent
    assert "RSI" not in intent.get("recognized_not_implemented_terms", []), intent


def test_what_is_rsi_routes_to_factor_research():
    intent = resolve_factor_intent("rsi 是什么")
    assert "rsi_14d" in intent.get("factor_names", []), intent
    assert intent["research_type"] == "factor_research", intent

