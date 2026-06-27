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
    if _looks_like_chitchat(user_input):
        return {
            "tool_name": "direct_reply",
            "reason": "识别为闲聊/身份问答，不需要调用研究工具。",
        }
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
    if tool_name == "direct_reply":
        reply = _direct_reply(user_input)
        next_state = update_chat_state(state, user_input=user_input, tool_result={"reply": reply, "direct_reply": True}, assistant_reply=reply)
        return {
            "user_input": user_input,
            "parsed_context": parsed,
            "decision": decision,
            "tool_name": tool_name,
            "tool_result": {"reply": reply, "direct_reply": True},
            "reply": reply,
            "state": next_state,
        }
    if tool_name == "run_factor_research_pipeline" and _is_empty_factor_request(parsed, state):
        reply = "请先明确要研究的因子或类别，我才能启动研究。"
        next_state = update_chat_state(
            state,
            user_input=user_input,
            tool_result={"reply": reply, "direct_reply": True},
            assistant_reply=reply,
        )
        return {
            "user_input": user_input,
            "parsed_context": parsed,
            "decision": decision,
            "tool_name": "direct_reply",
            "tool_result": {"reply": reply, "direct_reply": True},
            "reply": reply,
            "state": next_state,
        }
    mutation_block = _mutation_block_message(parsed, state)
    if tool_name == "run_factor_research_pipeline" and mutation_block is not None:
        next_state = update_chat_state(
            state,
            user_input=user_input,
            tool_result={"reply": mutation_block, "direct_reply": True},
            assistant_reply=mutation_block,
        )
        return {
            "user_input": user_input,
            "parsed_context": parsed,
            "decision": decision,
            "tool_name": "direct_reply",
            "tool_result": {"reply": mutation_block, "direct_reply": True},
            "reply": mutation_block,
            "state": next_state,
        }
    not_implemented_block = _recognized_not_implemented_block(parsed, user_input)
    if tool_name == "run_factor_research_pipeline" and not_implemented_block is not None:
        next_state = update_chat_state(
            state,
            user_input=user_input,
            tool_result={"reply": not_implemented_block, "direct_reply": True},
            assistant_reply=not_implemented_block,
        )
        return {
            "user_input": user_input,
            "parsed_context": parsed,
            "decision": decision,
            "tool_name": "direct_reply",
            "tool_result": {"reply": not_implemented_block, "direct_reply": True},
            "reply": not_implemented_block,
            "state": next_state,
        }
    data_backend = getattr(config, "data_backend", "bigquant")
    akshare_data_source = getattr(config, "akshare_data_source", "auto")
    akshare_proxy_patch_token = getattr(config, "akshare_proxy_patch_token", None)
    akshare_proxy_patch_gateway = getattr(config, "akshare_proxy_patch_gateway", "101.201.173.125")
    akshare_proxy_patch_hook_domains = getattr(config, "akshare_proxy_patch_hook_domains", None)
    akshare_proxy_patch_retry = getattr(config, "akshare_proxy_patch_retry", 30)
    akshare_proxy_patch_fast = getattr(config, "akshare_proxy_patch_fast", True)
    local_etf_parquet = getattr(config, "local_etf_parquet", "data/parquet/local_etf_daily.parquet")
    local_benchmark_parquet = getattr(config, "local_benchmark_parquet", None)

    pipeline_kwargs = _build_factor_pipeline_kwargs(parsed, user_input)
    tool_result = call_agent_tool(
        tool_name,
        **pipeline_kwargs,
        start_date=config.start_date,
        end_date=config.end_date,
        table_name=config.fund_table,
        data_backend=data_backend,
        akshare_data_source=akshare_data_source,
        akshare_proxy_patch_token=akshare_proxy_patch_token,
        akshare_proxy_patch_gateway=akshare_proxy_patch_gateway,
        akshare_proxy_patch_hook_domains=akshare_proxy_patch_hook_domains,
        akshare_proxy_patch_retry=akshare_proxy_patch_retry,
        akshare_proxy_patch_fast=akshare_proxy_patch_fast,
        local_etf_parquet=local_etf_parquet,
        local_benchmark_parquet=local_benchmark_parquet,
        top_pct=getattr(config, "top_pct", 0.10),
        min_holdings=getattr(config, "min_holdings", 3),
        cost_bps=getattr(config, "cost_bps", 10.0),
        volume_col=getattr(config, "volume_col", "volume"),
        turnover_col=getattr(config, "turnover_col", "turn"),
    ) if tool_name == "run_factor_research_pipeline" else call_agent_tool(
        tool_name,
        user_idea=parsed.get("effective_user_idea", parsed.get("user_input", user_input)),
        start_date=config.start_date,
        end_date=config.end_date,
        table_name=config.fund_table,
        data_backend=data_backend,
        akshare_data_source=akshare_data_source,
        akshare_proxy_patch_token=akshare_proxy_patch_token,
        akshare_proxy_patch_gateway=akshare_proxy_patch_gateway,
        akshare_proxy_patch_hook_domains=akshare_proxy_patch_hook_domains,
        akshare_proxy_patch_retry=akshare_proxy_patch_retry,
        akshare_proxy_patch_fast=akshare_proxy_patch_fast,
        local_etf_parquet=local_etf_parquet,
        local_benchmark_parquet=local_benchmark_parquet,
        volume_col=getattr(config, "volume_col", "volume"),
        turnover_col=getattr(config, "turnover_col", "turn"),
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
    selection_result = tool_result.get("selection_result") or {}
    if selection_result:
        status = selection_result.get("status")
        if status == "missing_context":
            return _missing_context_reply(selection_result) or tool_result.get("report") or _template_summary(tool_name, tool_result)
    if tool_result.get("selection_result") and not tool_result.get("backtest_result"):
        return tool_result.get("report") or _template_summary(tool_name, tool_result)
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
    except Exception:
        return fallback


def _build_factor_pipeline_kwargs(parsed: dict[str, Any], user_input: str) -> dict[str, Any]:
    """Pass structured draft-plan factors to the pipeline instead of re-parsing rewritten text."""
    display_query = parsed.get("user_input", user_input)
    kwargs: dict[str, Any] = {
        "user_idea": display_query,
        "display_query": display_query,
    }
    planned_names = parsed.get("pipeline_factor_names") or (parsed.get("draft_plan") or {}).get("selected_factor_names") or []
    if planned_names:
        kwargs["planned_factor_names"] = list(planned_names)
        kwargs["planned_target"] = parsed.get("pipeline_target") or (parsed.get("draft_plan") or {}).get("target")
    return kwargs


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


def _looks_like_chitchat(user_input: str) -> bool:
    text = (user_input or "").strip().lower()
    if not text:
        return True
    chitchat_keywords = [
        "你是谁",
        "你叫什么",
        "自我介绍",
        "你好",
        "在吗",
        "hello",
        "hi",
        "who are you",
    ]
    return any(keyword in text for keyword in chitchat_keywords)


def _direct_reply(user_input: str) -> str:
    text = (user_input or "").strip()
    if "你是谁" in text or "你叫什么" in text or "自我介绍" in text:
        return "我是你的 AutoETF 研究助手，主要帮你做条件研究、因子分析、回测和报告。"
    if "你好" in text or text.lower() in {"hello", "hi"}:
        return "你好，我可以帮你分析 ETF 条件、因子和回测结果。"
    return "我在。你可以直接问我 ETF 条件、因子、回测或者报告相关的问题。"


def _is_empty_factor_request(parsed_context: dict[str, Any], state: dict[str, Any] | None = None) -> bool:
    plan_mutation = parsed_context.get("plan_mutation")
    if plan_mutation is not None and getattr(plan_mutation, "mutation_type", None) is not None and plan_mutation.mutation_type.value != "no_op":
        return False
    draft_plan = parsed_context.get("draft_plan") or {}
    if draft_plan.get("selected_factor_names"):
        return False
    committed = (state or {}).get("current_context", {}).get("committed_plan") or parsed_context.get("committed_plan") or {}
    committed_names = committed.get("selected_factor_names", []) if isinstance(committed, dict) else []
    has_factor_cues = bool(parsed_context.get("factors") or parsed_context.get("selected_factors"))
    has_unresolved = bool(parsed_context.get("unresolved_terms"))
    has_not_implemented = bool(parsed_context.get("unavailable_factors"))
    return (
        not committed_names
        and parsed_context.get("selection_status") == "factor_research"
        and not has_factor_cues
        and not has_unresolved
        and not has_not_implemented
    )


def _recognized_not_implemented_block(parsed_context: dict[str, Any], user_input: str) -> str | None:
    terms = list(parsed_context.get("recognized_not_implemented_terms") or [])
    if not terms:
        return None
    lines = [
        "# AutoETF Research Plan",
        "",
        "## 研究无法执行",
        "",
        f"- 用户问题：{user_input}",
        "- 研究状态：recognized_not_implemented",
        "",
        "### 不可执行因子",
    ]
    lines.extend(f"- {term}: recognized_not_implemented" for term in terms)
    lines.extend(
        [
            "",
            "### 处理结果",
            "- 未启动 IC 分析",
            "- 未启动分层收益分析",
            "- 未启动回测",
            "- 未使用默认因子替代",
        ]
    )
    return "\n".join(lines)


def _mutation_block_message(parsed_context: dict[str, Any], state: dict[str, Any] | None = None) -> str | None:
    mutation = parsed_context.get("plan_mutation")
    if mutation is None or getattr(mutation, "mutation_type", None) is None:
        return None
    mutation_type = mutation.mutation_type.value
    if mutation_type not in {"remove_factors", "replace_factors"}:
        return None

    committed = (state or {}).get("current_context", {}).get("committed_plan") or parsed_context.get("committed_plan") or {}
    committed_names = committed.get("selected_factor_names", []) if isinstance(committed, dict) else []
    draft_plan = parsed_context.get("draft_plan") or {}
    draft_names = draft_plan.get("selected_factor_names", [])

    if mutation_type == "remove_factors":
        if not getattr(mutation, "remove_factor_names", []):
            return "当前计划中没有要删除的因子。"
        if committed_names and not draft_names:
            return "删除后计划为空，请先明确新的研究因子。"
    if mutation_type == "replace_factors" and not draft_names:
        return "替换后计划为空，请先明确新的研究因子。"
    return None


def _missing_context_reply(selection_result: dict) -> str | None:
    unavailable = selection_result.get("unavailable_factors") or []
    for item in unavailable:
        missing_context = item.get("missing_context") if isinstance(item, dict) else getattr(item, "missing_context", [])
        if "benchmark_series" in (missing_context or []):
            return "相对强度因子需要 benchmark 基准，请告诉我用沪深300、中证全指还是其他基准。"
    return None


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
    selection_result = tool_result.get("selection_result") or {}
    target = selection_result.get("target") or {}
    return {
        "data_shape": tool_result.get("data_shape"),
        "factor_names": [factor.get("name") for factor in tool_result.get("factors", [])],
        "target_horizon": target.get("horizon"),
        "target_metric": target.get("metric"),
        "selection_status": selection_result.get("status"),
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
