"""Conversation state helpers for notebook-based AutoETF chat."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.condition_parser import parse_condition_research
from src.expression_knowledge_base import classify_expression
from src.factor_resolver import resolve_factor_intent


DEFAULT_CURRENT_CONTEXT: dict[str, Any] = {
    "intent": "factor_research",
    "asset_scope": "ETF",
    "conditions": [],
    "factors": [],
    "target": None,
    "metrics": ["win_rate", "avg_return"],
    "sort": None,
    "filters": [],
    "last_result_summary": None,
    "last_question": None,
    "last_action": None,
}


def create_chat_state(config=None) -> dict[str, Any]:
    """Create a fresh notebook chat state."""
    state = {
        "conversation_history": [],
        "current_context": deepcopy(DEFAULT_CURRENT_CONTEXT),
    }
    if config is not None:
        state["current_context"]["asset_scope"] = "ETF"
        if getattr(config, "top_pct", None) is not None:
            state["current_context"]["sort"] = {
                "field": "composite_score",
                "direction": "desc",
                "top_pct": getattr(config, "top_pct", 0.10),
            }
    return state


def update_chat_state(
    state: dict[str, Any],
    user_input: str,
    tool_result: dict | None = None,
    assistant_reply: str | None = None,
) -> dict[str, Any]:
    """Update conversation history and structured current context."""
    new_state = deepcopy(state or create_chat_state())
    current_context = new_state.setdefault("current_context", deepcopy(DEFAULT_CURRENT_CONTEXT))
    conversation_history = new_state.setdefault("conversation_history", [])

    conversation_history.append({"role": "user", "content": user_input})

    parsed = _parse_user_intent(user_input, current_context)
    current_context = _merge_context(current_context, parsed, user_input)
    current_context["last_question"] = user_input

    if tool_result is not None:
        current_context["last_result_summary"] = _summarize_tool_result(tool_result)
    current_context["last_action"] = parsed.get("action")
    new_state["current_context"] = current_context

    if assistant_reply is not None:
        conversation_history.append({"role": "assistant", "content": assistant_reply})

    return new_state


def parse_user_context(user_input: str, current_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse a user turn into a structured intent update without mutating state."""
    return _parse_user_intent(user_input, current_context or deepcopy(DEFAULT_CURRENT_CONTEXT))


def format_context_for_display(state: dict[str, Any]) -> str:
    """Pretty-format the structured current context for notebook debugging."""
    import json

    return json.dumps(state.get("current_context", {}), ensure_ascii=False, indent=2, default=str)


def suggest_follow_up(state: dict[str, Any]) -> list[str]:
    """Suggest a few next questions based on current context."""
    context = state.get("current_context", {})
    suggestions = []
    if context.get("conditions"):
        suggestions.append("加上成交额上升")
        suggestions.append("换成未来10日")
        suggestions.append("改成放量突破")
    if context.get("target"):
        suggestions.append("看次日胜率")
    if context.get("sort"):
        suggestions.append("按前10%重新排序")
    return suggestions[:4]


def _parse_user_intent(user_input: str, current_context: dict[str, Any]) -> dict[str, Any]:
    text = user_input.strip()
    kb_result = classify_expression(text)
    intent_result = resolve_factor_intent(text)
    parsed_conditions = []
    best_match = kb_result.get("best_match") or {}

    if best_match.get("expression_type") == "condition_expression":
        parsed = parse_condition_research(text)
        parsed_conditions = parsed.get("conditions", [])
        return {
            "action": _detect_action(text, current_context, target=parsed.get("target")),
            "research_type": "condition_research",
            "matched_expressions": kb_result.get("matched", []),
            "conditions": parsed_conditions,
            "target": parsed.get("target"),
            "factors": [],
            "metrics": _extract_metrics(text),
            "sort": _extract_sort(text),
            "route_intent": parsed.get("route_intent", kb_result.get("route_intent", [])),
        }

    parsed_conditions = intent_result.get("conditions", [])
    target = _extract_target_from_intent(text, kb_result, intent_result)
    factors = _extract_factor_names(intent_result)
    metrics = _extract_metrics(text, intent_result)
    sort_rule = _extract_sort(text, intent_result)
    action = _detect_action(text, current_context, target=target)
    research_type = intent_result.get("research_type", "factor_score")
    if target and target.get("field", "").startswith("future_"):
        research_type = "condition_research" if parsed_conditions else research_type
    return {
        "action": action,
        "research_type": research_type,
        "matched_expressions": kb_result.get("matched", []),
        "conditions": parsed_conditions,
        "target": target,
        "factors": factors,
        "metrics": metrics,
        "sort": sort_rule,
        "route_intent": intent_result.get("route_intent", kb_result.get("route_intent", [])),
    }


def _merge_context(current_context: dict[str, Any], parsed: dict[str, Any], user_input: str) -> dict[str, Any]:
    context = deepcopy(current_context)
    action = parsed.get("action") or "append"
    if action in {"replace", "reset"}:
        context["conditions"] = []
        context["factors"] = []
        context["target"] = None
        context["sort"] = None
        context["filters"] = []

    if parsed.get("conditions"):
        if action == "replace":
            context["conditions"] = _dedupe_conditions(parsed["conditions"])
        else:
            context["conditions"] = _dedupe_conditions(context.get("conditions", []) + parsed["conditions"])

    if parsed.get("factors"):
        if action == "replace":
            context["factors"] = list(parsed["factors"])
        else:
            context["factors"] = _dedupe_list(context.get("factors", []) + parsed["factors"])

    if parsed.get("target"):
        if action in {"replace", "reset", "modify_target"} or context.get("target") is None:
            context["target"] = parsed["target"]

    if parsed.get("metrics"):
        context["metrics"] = _dedupe_list(parsed["metrics"] + context.get("metrics", []))

    if parsed.get("sort"):
        context["sort"] = parsed["sort"]

    context["filters"] = _derive_filters(context)
    context["last_action"] = action
    return context


def _detect_action(text: str, current_context: dict[str, Any], target: dict | None = None) -> str:
    if any(keyword in text for keyword in ["重置", "清空", "重新开始"]):
        return "reset"
    if any(keyword in text for keyword in ["换成", "改成", "替换", "不要", "去掉"]):
        return "replace"
    if any(keyword in text for keyword in ["那未来", "改未来", "换未来", "未来10", "未来5", "次日", "明天", "10日", "5日"]):
        if current_context.get("conditions"):
            return "modify_target"
    if any(keyword in text for keyword in ["加上", "再加", "同时", "并且", "还要", "新增"]):
        return "append"
    if target and target.get("field", "").startswith("future_") and current_context.get("conditions"):
        return "append"
    return "append"


def _extract_metrics(text: str, intent_result: dict | None = None) -> list[str]:
    metrics = []
    if "胜率" in text or "概率" in text:
        metrics.append("win_rate")
    if "收益" in text or "平均收益" in text:
        metrics.append("avg_return")
    if "IC" in text.upper():
        metrics.append("ic")
    if "回撤" in text:
        metrics.append("drawdown")
    if "分层" in text or "排序" in text:
        metrics.append("ranking")
    if intent_result:
        if intent_result.get("research_type") == "composite_score_analysis":
            metrics.append("composite_score")
        elif intent_result.get("research_type") == "metric_analysis":
            metrics.append("metric")
    return _dedupe_list(metrics)


def _extract_sort(text: str, intent_result: dict | None = None) -> dict | None:
    if any(keyword in text for keyword in ["前10%", "前 10%", "Top10", "top10", "选前10"]):
        return {"field": "composite_score", "direction": "desc", "top_pct": 0.10}
    if any(keyword in text for keyword in ["前20%", "前 20%", "Top20", "top20", "选前20"]):
        return {"field": "composite_score", "direction": "desc", "top_pct": 0.20}
    if intent_result and intent_result.get("research_type") == "composite_score_analysis":
        return {"field": "composite_score", "direction": "desc", "top_pct": 0.10}
    return None


def _extract_target_from_intent(text: str, kb_result: dict, intent_result: dict) -> dict | None:
    for item in kb_result.get("matched", []):
        if item.get("expression_type") == "target_expression":
            return {
                "field": item.get("derived_columns", ["future_return_1d"])[0],
                "operator": item.get("operator", ">"),
                "value": item.get("threshold", 0.0),
                "description": item.get("meaning", item.get("phrase")),
            }
    target = intent_result.get("expression_match")
    if target and target.get("expression_type") == "target_expression":
        return {
            "field": target.get("derived_columns", ["future_return_1d"])[0],
            "operator": target.get("operator", ">"),
            "value": target.get("threshold", 0.0),
            "description": target.get("meaning", target.get("phrase")),
        }
    if any(keyword in text for keyword in ["未来10", "10日", "10天"]):
        return {"field": "future_return_10d", "operator": ">", "value": 0.0, "description": "未来10日收益率为正"}
    if any(keyword in text for keyword in ["未来5", "5日", "5天"]):
        return {"field": "future_return_5d", "operator": ">", "value": 0.0, "description": "未来5日收益率为正"}
    if any(keyword in text for keyword in ["次日", "明天", "下一日"]):
        return {"field": "future_return_1d", "operator": ">", "value": 0.0, "description": "下一交易日收益率为正"}
    return None


def _extract_factor_names(intent_result: dict) -> list[str]:
    factors = intent_result.get("factor_names") or []
    if factors:
        return list(factors)
    factor_plan = intent_result.get("factor_plan") or []
    return [item.get("name") for item in factor_plan if item.get("name")]


def _derive_filters(context: dict[str, Any]) -> list[dict[str, Any]]:
    filters = []
    for condition in context.get("conditions", []):
        filters.append(
            {
                "field": condition.get("field"),
                "operator": condition.get("operator"),
                "value": condition.get("value"),
            }
        )
    if context.get("target"):
        filters.append({"field": context["target"].get("field"), "operator": context["target"].get("operator")})
    return filters


def _summarize_tool_result(tool_result: dict) -> dict[str, Any]:
    summary = {
        "tool_name": tool_result.get("tool_name"),
        "research_type": tool_result.get("hypothesis", {}).get("research_goal") or tool_result.get("research_type"),
        "data_shape": tool_result.get("data_shape"),
    }
    if "result" in tool_result:
        result = tool_result["result"]
        summary.update(
            {
                "event_count": result.get("event_count"),
                "total_count": result.get("total_count"),
                "probability_lift": result.get("probability_lift"),
                "mean_return_lift": result.get("mean_return_lift"),
            }
        )
    if "backtest_result" in tool_result:
        performance = tool_result["backtest_result"].get("performance", {})
        summary.update(
            {
                "total_return": performance.get("total_return"),
                "annual_return": performance.get("annual_return"),
                "max_drawdown": performance.get("max_drawdown"),
            }
        )
    return summary


def _dedupe_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for condition in conditions:
        key = (condition.get("field"), condition.get("operator"), condition.get("value"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(condition)
    return deduped


def _dedupe_list(items: list[Any]) -> list[Any]:
    deduped = []
    seen = set()
    for item in items:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
