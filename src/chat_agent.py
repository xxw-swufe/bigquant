"""Notebook-friendly chat agent for AutoETF research.

This module keeps the LLM layer thin: DeepSeek decides which research tool to
run and summarizes the result, while all quantitative work stays in agent_tools.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from src.agent_tools import call_agent_tool
from src.chat_state import create_chat_state, parse_user_context, update_chat_state
from src.factor_resolver import resolve_factor_intent


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def call_deepseek(
    messages: list[dict],
    model: str = DEFAULT_DEEPSEEK_MODEL,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    temperature: float = 0.2,
    timeout: int = 60,
) -> str:
    """Call DeepSeek's OpenAI-compatible chat completions API."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in the notebook or environment.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for DeepSeek API calls.") from exc

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def decide_tool(
    user_input: str,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    parsed_context: Optional[dict[str, Any]] = None,
) -> dict:
    """Decide which research pipeline should handle the user request."""
    fallback = _rule_based_decision(user_input)
    if not use_llm:
        return fallback

    prompt = f"""
你是一个 ETF 量化研究智能体的工具路由器。
只能在下面两个工具中选择一个：
1. run_condition_research_pipeline: 用户描述明确条件，并询问概率、胜率、是否上涨时使用。
2. run_factor_research_pipeline: 用户研究因子、评分、轮动、超额收益、回测时使用。

如果用户是在追问上一轮结果，请结合上下文判断：
- 追加条件：偏向 condition_research
- 修改目标或排序：仍可用 condition_research 或 factor_research，但不要丢失上下文

请只输出 JSON，不要输出解释。JSON 格式：
{{"tool_name": "...", "reason": "..."}}

用户问题：{user_input}
上下文摘要：{json.dumps(parsed_context or {}, ensure_ascii=False, default=str)}
""".strip()
    try:
        content = call_deepseek(
            [
                {"role": "system", "content": "你只输出严格 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            api_key=api_key,
            temperature=0.0,
        )
        decision = _extract_json(content)
        if decision.get("tool_name") in {"run_condition_research_pipeline", "run_factor_research_pipeline"}:
            return decision
    except Exception as exc:
        fallback["llm_error"] = str(exc)
    return fallback


def run_research_chat(
    user_input: str,
    config,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run a one-turn notebook chat interaction."""
    state = state or create_chat_state(config)
    parsed = parse_user_context(user_input, state.get("current_context", {}))
    decision = decide_tool(user_input, use_llm=use_llm, api_key=api_key, model=model, parsed_context=parsed)
    tool_name = decision["tool_name"]

    tool_result = call_agent_tool(
        tool_name,
        user_idea=parsed.get("user_input", user_input),
        start_date=config.start_date,
        end_date=config.end_date,
        table_name=config.fund_table,
        top_pct=getattr(config, "top_pct", 0.10),
        min_holdings=getattr(config, "min_holdings", 3),
        cost_bps=getattr(config, "cost_bps", 10.0),
    ) if tool_name == "run_factor_research_pipeline" else call_agent_tool(
        tool_name,
        user_idea=parsed.get("user_input", user_input),
        start_date=config.start_date,
        end_date=config.end_date,
        table_name=config.fund_table,
    )

    reply = summarize_tool_result(
        parsed.get("user_input", user_input),
        tool_name,
        tool_result,
        use_llm=use_llm,
        api_key=api_key,
        model=model,
    )
    next_state = update_chat_state(state, user_input=user_input, tool_result=tool_result, assistant_reply=reply)
    return {
        "user_input": user_input,
        "parsed_context": parsed,
        "decision": decision,
        "tool_name": tool_name,
        "tool_result": tool_result,
        "reply": reply,
        "state": next_state,
    }


def summarize_tool_result(
    user_input: str,
    tool_name: str,
    tool_result: dict,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
) -> str:
    """Summarize a tool result in plain Chinese."""
    fallback = _template_summary(tool_name, tool_result)
    if not use_llm:
        return fallback

    compact = _compact_tool_result(tool_name, tool_result)
    prompt = f"""
你是 ETF 量化研究助理。请基于工具结果，用中文给用户一个简洁研究结论。
要求：
- 先回答用户问题
- 说明样本、概率/回测表现、主要风险
- 不要给投资建议
- 不要编造工具结果里没有的数据

用户问题：{user_input}
调用工具：{tool_name}
工具结果摘要：
{json.dumps(compact, ensure_ascii=False, default=str)}
""".strip()
    try:
        return call_deepseek(
            [
                {"role": "system", "content": "你是严谨的量化研究报告助理。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            api_key=api_key,
            temperature=0.2,
        )
    except Exception as exc:
        return f"{fallback}\n\n[LLM 总结失败，已使用模板总结] {exc}"


def _rule_based_decision(user_input: str) -> dict:
    intent = resolve_factor_intent(user_input)
    if intent.get("research_type") == "conditional_event_study":
        return {
            "tool_name": "run_condition_research_pipeline",
            "reason": "规则判断为条件事件研究。",
        }
    return {
        "tool_name": "run_factor_research_pipeline",
        "reason": "规则判断为因子研究。",
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def _compact_tool_result(tool_name: str, tool_result: dict) -> dict:
    if tool_name == "run_condition_research_pipeline":
        result = tool_result.get("result", {})
        return {
            "data_shape": tool_result.get("data_shape"),
            "event_count": result.get("event_count"),
            "total_count": result.get("total_count"),
            "event_up_probability": result.get("event_up_probability"),
            "baseline_up_probability": result.get("baseline_up_probability"),
            "probability_lift": result.get("probability_lift"),
            "mean_return_lift": result.get("mean_return_lift"),
            "diagnosis": tool_result.get("diagnosis"),
        }

    performance = tool_result.get("backtest_result", {}).get("performance", {})
    return {
        "data_shape": tool_result.get("data_shape"),
        "factor_names": [factor.get("name") for factor in tool_result.get("factors", [])],
        "performance": performance,
        "diagnosis": tool_result.get("diagnosis"),
    }


def _template_summary(tool_name: str, tool_result: dict) -> str:
    if tool_name == "run_condition_research_pipeline":
        result = tool_result.get("result", {})
        diagnosis = tool_result.get("diagnosis", {})
        return (
            f"条件研究完成。满足条件样本数为 {result.get('event_count')}，"
            f"全样本数为 {result.get('total_count')}，"
            f"概率提升为 {result.get('probability_lift')}。"
            f"诊断结论：{diagnosis.get('research_decision')}。"
        )

    performance = tool_result.get("backtest_result", {}).get("performance", {})
    diagnosis = tool_result.get("diagnosis", {})
    return (
        f"因子研究完成。组合总收益为 {performance.get('total_return')}，"
        f"年化收益为 {performance.get('annual_return')}，"
        f"最大回撤为 {performance.get('max_drawdown')}。"
        f"诊断结论：{diagnosis.get('research_decision')}。"
    )
