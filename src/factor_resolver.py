"""Resolve natural language factor intents into factor candidates and rules."""

from __future__ import annotations

from src.etf_factor_library import get_factor_library, search_factors
from src.factor_availability import select_factor_plan


def resolve_factor_intent(user_idea: str) -> dict:
    """Resolve a user's idea into searchable factor candidates.

    The MVP uses keyword search and lightweight rule heuristics. Later we can
    swap in LLM-based parsing without changing downstream tool contracts.
    """
    text = user_idea.strip()
    factors = search_factors(text, limit=20)
    tokens = [token for token in _split_tokens(text) if token]
    library_index = {factor["name"]: factor for factor in get_factor_library()}

    conditions = []
    research_type = "factor_score"
    if _looks_like_condition_query(text):
        research_type = "conditional_event_study"
        conditions = _build_condition_rules(tokens)

    factor_plan = select_factor_plan(factors, max_factors=8) if factors else []
    factor_names = [factor["name"] for factor in factor_plan]

    return {
        "user_idea": user_idea,
        "research_type": research_type,
        "matched_tokens": tokens,
        "matched_factors": factors,
        "factor_plan": factor_plan,
        "factor_names": factor_names,
        "conditions": conditions,
        "available_factor_count": len(library_index),
        "notes": _build_notes(factors),
    }


def _split_tokens(text: str) -> list[str]:
    for sep in ["，", ",", "。", "；", ";", " "]:
        text = text.replace(sep, "|")
    return [token for token in text.split("|") if token]


def _looks_like_condition_query(text: str) -> bool:
    keywords = ["上涨", "下跌", "放量", "缩量", "突破", "反弹", "超卖", "超买", "金叉", "死叉"]
    return any(keyword in text for keyword in keywords)


def _build_condition_rules(tokens: list[str]) -> list[dict]:
    rules = []
    joined = " ".join(tokens)
    if "放量" in joined or "量比" in joined:
        rules.append({"field": "amount_ratio_20d", "operator": ">", "value": 1.0})
    if "缩量" in joined:
        rules.append({"field": "amount_ratio_20d", "operator": "<", "value": 1.0})
    if "上涨" in joined:
        rules.append({"field": "return_1d", "operator": ">", "value": 0.0})
    if "下跌" in joined:
        rules.append({"field": "return_1d", "operator": "<", "value": 0.0})
    if "OBV" in joined.upper():
        rules.append({"field": "obv_trend_20d", "operator": ">", "value": 0.0})
    if "缩量下跌" in joined or ("缩量" in joined and "下跌" in joined):
        rules.append({"field": "amount_ratio_20d", "operator": "<", "value": 1.0})
        rules.append({"field": "return_1d", "operator": "<", "value": 0.0})
        rules.append({"field": "obv_trend_20d", "operator": "<", "value": 0.0})
    return rules


def _build_notes(factors: list[dict]) -> list[str]:
    if not factors:
        return ["未直接匹配到标准因子，建议扩充 aliases 或补充原生指标映射。"]
    notes = [f"matched: {factor['name']}" for factor in factors[:5]]
    if len(factors) > 5:
        notes.append(f"还有 {len(factors) - 5} 个候选因子未展开。")
    return notes
