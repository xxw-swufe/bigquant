"""Resolve natural language factor intents into factor candidates and rules."""

from __future__ import annotations

from src.expression_knowledge_base import classify_expression, search_expressions
from src.etf_factor_library import get_factor_library, search_factors
from src.factor_availability import select_factor_plan


def resolve_factor_intent(user_idea: str) -> dict:
    """Resolve a user's idea into searchable factor candidates.

    The MVP uses keyword search and lightweight rule heuristics. Later we can
    swap in LLM-based parsing without changing downstream tool contracts.
    """
    text = user_idea.strip()
    kb_result = classify_expression(text)
    matched_expression = kb_result.get("best_match") or {}
    factors = search_factors(text, limit=20)
    tokens = [token for token in _split_tokens(text) if token]
    library_index = {factor["name"]: factor for factor in get_factor_library()}

    conditions = []
    research_type = "factor_score"
    route_intent = ["factor"]
    if matched_expression.get("expression_type") == "condition_expression":
        research_type = "conditional_event_study"
        route_intent = ["condition"]
        conditions = _build_condition_rules(tokens, matched_expression)
    elif matched_expression.get("expression_type") == "target_expression":
        research_type = "target_analysis"
        route_intent = ["target"]
    elif matched_expression.get("expression_type") == "metric_expression":
        research_type = "metric_analysis"
        route_intent = ["metric"]
    elif matched_expression.get("phrase") in {"量价分", "动量分", "趋势分", "反转分", "风险分", "估值分", "质量分", "成长分"}:
        research_type = "composite_score_analysis"
        route_intent = ["metric", "factor"]

    factor_plan = select_factor_plan(factors, max_factors=8) if factors else []
    factor_names = [factor["name"] for factor in factor_plan]

    return {
        "user_idea": user_idea,
        "research_type": research_type,
        "matched_tokens": tokens,
        "matched_factors": factors,
        "matched_expressions": kb_result.get("matched", []),
        "expression_match": matched_expression,
        "factor_plan": factor_plan,
        "factor_names": factor_names,
        "conditions": conditions,
        "available_factor_count": len(library_index),
        "route_intent": route_intent,
        "route_to": matched_expression.get("route_to", []),
        "notes": _build_notes(factors),
    }


def _split_tokens(text: str) -> list[str]:
    for sep in ["，", ",", "。", "；", ";", " "]:
        text = text.replace(sep, "|")
    return [token for token in text.split("|") if token]


def _looks_like_condition_query(text: str) -> bool:
    keywords = ["上涨", "下跌", "放量", "缩量", "突破", "反弹", "超卖", "超买", "金叉", "死叉"]
    return any(keyword in text for keyword in keywords)


def _build_condition_rules(tokens: list[str], matched_expression: dict | None = None) -> list[dict]:
    rules = []
    joined = " ".join(tokens)
    if matched_expression:
        rules.append(
            {
                "field": (matched_expression.get("derived_columns") or [matched_expression.get("canonical_name")])[0],
                "operator": matched_expression.get("operator", ">"),
                "value": matched_expression.get("threshold", 0),
                "description": matched_expression.get("meaning", matched_expression.get("phrase")),
            }
        )
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
    if "趋势走强" in joined or "走强" in joined or "强趋势" in joined:
        rules.append({"field": "trend_persistence_20d", "operator": ">", "value": 0.6})
        rules.append({"field": "ma_gap_20d", "operator": ">", "value": 0.0})
    if "过热" in joined or "高位" in joined:
        rules.append({"field": "rsi_overbought_20d", "operator": ">", "value": 0.0})
    if "缩量下跌" in joined or ("缩量" in joined and "下跌" in joined):
        rules.append({"field": "amount_ratio_20d", "operator": "<", "value": 1.0})
        rules.append({"field": "return_1d", "operator": "<", "value": 0.0})
        rules.append({"field": "obv_trend_20d", "operator": "<", "value": 0.0})
    deduped = []
    seen = set()
    for rule in rules:
        key = (rule["field"], rule["operator"], repr(rule["value"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def _build_notes(factors: list[dict]) -> list[str]:
    if not factors:
        return ["未直接匹配到标准因子，建议扩充 aliases 或补充原生指标映射。"]
    notes = [f"matched: {factor['name']}" for factor in factors[:5]]
    if len(factors) > 5:
        notes.append(f"还有 {len(factors) - 5} 个候选因子未展开。")
    return notes
