"""Condition-rule parser for the first ETF event-study MVP."""

from __future__ import annotations

from src.expression_knowledge_base import classify_expression, get_expression_by_phrase


def parse_condition_research(user_idea: str) -> dict:
    """Parse a simple natural-language ETF condition study.

    First MVP intentionally supports the canonical example:
    volume ratio > 1, turnover < 5%, today's return > 0, and asks whether the
    next trading day return is positive.
    """
    # [AI-CORE]
    kb_result = classify_expression(user_idea)
    best_match = kb_result.get("best_match") or {}
    matched_expressions = kb_result.get("matched", [])
    condition_matches = _select_condition_entries(matched_expressions)
    if best_match.get("expression_type") == "condition_expression" and best_match.get("layer") == "expression_template":
        return _build_kb_condition_research(user_idea, best_match, kb_result)
    if len(condition_matches) >= 2:
        return _build_multi_kb_condition_research(user_idea, condition_matches, kb_result)
    if best_match.get("expression_type") == "condition_expression":
        return _build_kb_condition_research(user_idea, best_match, kb_result)
    if condition_matches:
        return _build_multi_kb_condition_research(user_idea, condition_matches, kb_result)

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


def _select_condition_entries(entries: list[dict]) -> list[dict]:
    condition_entries = [item for item in entries if item.get("expression_type") == "condition_expression"]
    template_entries = [item for item in condition_entries if item.get("layer") == "expression_template"]
    if template_entries:
        return template_entries
    return condition_entries


def _build_kb_condition_research(user_idea: str, entry: dict, kb_result: dict) -> dict:
    conditions = _dedupe_conditions(_build_conditions_from_entry(entry))
    target = _infer_target_from_query(user_idea, kb_result)
    required_columns = sorted(set(entry.get("required_columns", [])) | set(target.get("required_columns", [])))
    return {
        "research_type": "conditional_probability_test",
        "user_idea": user_idea,
        "asset_universe": "ETF",
        "research_goal": f"检验{entry.get('phrase')}对应条件下，ETF 后续表现是否优于基准。",
        "conditions": conditions,
        "target": target["target"],
        "required_columns": required_columns + ["date", "instrument"],
        "expression_match": entry,
        "matched_expressions": kb_result.get("matched", []),
        "route_intent": kb_result.get("route_intent", ["condition"]),
        "route_to": kb_result.get("route_to", ["condition_research"]),
        "not_investment_advice": True,
    }


def _build_multi_kb_condition_research(user_idea: str, entries: list[dict], kb_result: dict) -> dict:
    condition_entries = [item for item in entries if item.get("expression_type") == "condition_expression"]
    conditions = []
    required_columns = {"date", "instrument"}
    for entry in condition_entries:
        conditions.extend(_build_conditions_from_entry(entry))
        required_columns.update(entry.get("required_columns", []))
    conditions = _dedupe_conditions(conditions)
    target = _infer_target_from_query(user_idea, kb_result)
    required_columns.update(target.get("required_columns", []))
    return {
        "research_type": "conditional_probability_test",
        "user_idea": user_idea,
        "asset_universe": "ETF",
        "research_goal": "检验多个条件同时成立时，ETF 后续表现是否优于基准。",
        "conditions": conditions,
        "target": target["target"],
        "required_columns": sorted(required_columns),
        "expression_match": kb_result.get("best_match"),
        "matched_expressions": condition_entries,
        "route_intent": kb_result.get("route_intent", ["condition"]),
        "route_to": kb_result.get("route_to", ["condition_research"]),
        "not_investment_advice": True,
    }


def _dedupe_conditions(conditions: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for condition in conditions:
        key = (condition.get("field"), condition.get("operator"), condition.get("value"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(condition)
    return deduped


def _build_conditions_from_entry(entry: dict) -> list[dict]:
    operator = entry.get("operator", ">")
    threshold = entry.get("threshold", 0)
    derived_columns = entry.get("derived_columns", [])
    field = _infer_condition_field(entry, derived_columns)
    if entry.get("threshold_type") == "compound_rule" and isinstance(threshold, dict):
        return [
            {
                "field": key,
                "operator": _infer_compound_operator(key, value, entry),
                "value": value,
                "description": entry.get("meaning", entry.get("phrase")),
            }
            for key, value in threshold.items()
        ]
    return [
        {
            "field": field,
            "operator": operator or "=",
            "value": threshold if threshold is not None else True,
            "description": entry.get("meaning", entry.get("phrase")),
        }
    ]


def _infer_condition_field(entry: dict, derived_columns: list[str]) -> str:
    canonical_name = entry.get("canonical_name")
    if canonical_name in derived_columns:
        return canonical_name
    if entry.get("threshold_type") in {"breakout", "indicator_level"} and derived_columns:
        return derived_columns[-1]
    if entry.get("operator") == "" and derived_columns:
        return derived_columns[-1]
    return derived_columns[0] if derived_columns else canonical_name


def _infer_compound_operator(field: str, value: object, entry: dict) -> str:
    phrase = entry.get("phrase", "")
    formula = entry.get("formula", "")
    if f"{field} <" in formula:
        return "<"
    if f"{field} >" in formula:
        return ">"
    if field == "return_1d" and ("下跌" in phrase or "回调" in phrase):
        return "<"
    if field == "amount_ratio_20d" and ("缩量" in phrase or "萎缩" in phrase):
        return "<"
    if field == "volume_ratio_20d" and ("缩量" in phrase or "萎缩" in phrase):
        return "<"
    if isinstance(value, bool):
        return "="
    if isinstance(value, (int, float)):
        return ">"
    return "="


def _infer_target_from_query(user_idea: str, kb_result: dict) -> dict:
    query = user_idea.lower()
    for item in kb_result.get("matched", []):
        if item.get("expression_type") == "target_expression":
            return {"target": _build_target(item), "required_columns": item.get("required_columns", [])}
    if "未来5" in query or "5日" in query or "5天" in query:
        entry = get_expression_by_phrase("未来5日收益")
        if entry:
            return {"target": _build_target(entry), "required_columns": entry.get("required_columns", [])}
    if "未来1" in query or "次日" in query or "明天" in query or "下一日" in query:
        entry = get_expression_by_phrase("未来1日上涨")
        if entry:
            return {"target": _build_target(entry), "required_columns": entry.get("required_columns", [])}
    entry = get_expression_by_phrase("未来1日上涨")
    return {"target": _build_target(entry), "required_columns": entry.get("required_columns", []) if entry else []}


def _build_target(entry: dict | None) -> dict:
    if not entry:
        return {
            "field": "future_return_1d",
            "operator": ">",
            "value": 0.0,
            "description": "下一交易日收益率为正",
        }
    return {
        "field": entry.get("derived_columns", ["future_return_1d"])[0],
        "operator": entry.get("operator", ">"),
        "value": entry.get("threshold", 0.0),
        "description": entry.get("meaning", entry.get("phrase")),
    }
