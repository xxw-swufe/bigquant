"""Agent-facing tool wrappers for the ETF research workflow."""

from __future__ import annotations

from typing import Any
from typing import Optional

from src.condition_analysis import run_conditional_probability_test
from src.condition_parser import parse_condition_research
from src.condition_report import generate_condition_report
from src.data_access import load_condition_research_data, load_etf_data
from src.data_probe_bigquant import run_data_probe
from src.factor_analysis import run_factor_analysis
from src.factor_availability import build_factor_tool_payload, get_available_factors
from src.factor_generator import generate_factor_candidates
from src.factor_resolver import resolve_factor_intent
from src.research_plan import build_factor_intent_from_plan, target_return_column
from src.report import generate_report
from src.scoring import build_weight_schemes, compute_composite_score, compute_metadata_composite_score, normalize_factors
from src.backtest import run_top_pct_backtest
from src.condition_analysis import diagnose_condition_result
from src.diagnosis import diagnose_strategy


def tool_run_data_probe(**kwargs):
    return run_data_probe(**kwargs)


def tool_resolve_factor_intent(user_idea: str):
    return resolve_factor_intent(user_idea)


def tool_build_factor_plan(user_idea: str, required_columns: Optional[list[str]] = None):
    return build_factor_tool_payload(user_idea, required_columns=required_columns)


def tool_get_available_factors(required_columns: Optional[list[str]] = None):
    return get_available_factors(required_columns)


def tool_load_etf_data(**kwargs):
    return load_etf_data(**kwargs)


def tool_run_factor_analysis(df, factor_cols, target_cols):
    return run_factor_analysis(df, factor_cols, target_cols)


def tool_run_conditional_probability_test(df, conditions, target=None, min_event_count=200):
    return run_conditional_probability_test(
        df,
        conditions=conditions,
        target=target,
        min_event_count=min_event_count,
    )


def tool_build_weight_schemes(factor_cols, factor_result=None):
    return build_weight_schemes(factor_cols, factor_result)


def tool_compute_composite_score(df, factor_metadata, weights):
    return compute_metadata_composite_score(df, factor_metadata, weights)


def tool_diagnose_strategy(factor_result, backtest_result, weights):
    return diagnose_strategy(factor_result, backtest_result, weights)


def tool_generate_report(hypothesis, factors, factor_result, backtest_result, diagnosis, weights):
    return generate_report(
        hypothesis=hypothesis,
        factors=factors,
        factor_result=factor_result,
        backtest_result=backtest_result,
        diagnosis=diagnosis,
        weights=weights,
    )


def tool_run_factor_research_pipeline(
    user_idea: str,
    start_date: str,
    end_date: str,
    planned_factor_names: list[str] | None = None,
    planned_target: dict | None = None,
    display_query: str | None = None,
    data_backend: str = "bigquant",
    akshare_data_source: str = "auto",
    akshare_proxy_patch_token: str | None = None,
    akshare_proxy_patch_gateway: str = "101.201.173.125",
    akshare_proxy_patch_hook_domains: list[str] | None = None,
    akshare_proxy_patch_retry: int = 30,
    akshare_proxy_patch_fast: bool = True,
    table_name: str = "cn_fund_bar1d",
    local_etf_parquet: str = "data/parquet/local_etf_daily.parquet",
    local_benchmark_parquet: str | None = None,
    top_pct: float = 0.10,
    min_holdings: int = 3,
    cost_bps: float = 10.0,
    volume_col: str = "volume",
    turnover_col: str | None = None,
    **_: object,
) -> dict[str, Any]:
    """Run the full factor research pipeline as a single tool call."""
    display_query = (display_query or user_idea).strip()
    if planned_factor_names:
        hypothesis = build_factor_intent_from_plan(
            display_query,
            planned_factor_names,
            planned_target,
            selection_source="committed_plan",
        )
    else:
        hypothesis = resolve_factor_intent(user_idea)
        if display_query and display_query != user_idea:
            hypothesis.raw_query = display_query
    probe_results = run_data_probe() if data_backend == "bigquant" else {}
    df = load_etf_data(
        data_backend=data_backend,
        data_source=akshare_data_source,
        start_date=start_date,
        end_date=end_date,
        table_name=table_name,
        volume_col=volume_col,
        parquet_path=local_etf_parquet,
        benchmark_parquet_path=local_benchmark_parquet,
        proxy_patch_token=akshare_proxy_patch_token,
        proxy_patch_gateway=akshare_proxy_patch_gateway,
        proxy_patch_hook_domains=akshare_proxy_patch_hook_domains,
        proxy_patch_retry=akshare_proxy_patch_retry,
        proxy_patch_fast=akshare_proxy_patch_fast,
    )
    available_fields = set(df.columns)
    available_context = _build_available_context(df)
    selection = generate_factor_candidates(
        hypothesis,
        data_backend=data_backend,
        available_fields=available_fields,
        available_context=available_context,
        max_candidates=5,
        allow_partial_execution=False,
    )
    if not selection.can_execute:
        return {
            "hypothesis": _with_selection_payload(hypothesis, selection),
            "selection_result": selection.to_dict(),
            "probe_tables": list(probe_results.keys()),
            "data_shape": df.shape,
            "status": selection.status.value if hasattr(selection.status, "value") else str(selection.status),
            "report": _build_selection_failure_report(hypothesis, selection),
        }

    factors = selection.selected_factors
    factor_cols = [factor["name"] for factor in factors if factor["name"] in df.columns]
    factor_directions = {factor["name"]: factor["direction"] for factor in factors if factor["name"] in factor_cols}

    target_col = target_return_column(selection.target)
    if target_col not in df.columns:
        from src.research_plan import FactorSelectionResult, SelectionStatus

        invalid_selection = FactorSelectionResult(
            status=SelectionStatus.INVALID_TARGET,
            selected_factors=factors,
            target=selection.target,
            selection_source=selection.selection_source,
        )
        return {
            "hypothesis": _with_selection_payload(hypothesis, invalid_selection),
            "selection_result": invalid_selection.to_dict(),
            "probe_tables": list(probe_results.keys()),
            "data_shape": df.shape,
            "status": SelectionStatus.INVALID_TARGET.value,
            "report": _build_selection_failure_report(hypothesis, invalid_selection),
        }

    analysis_targets = _analysis_target_columns(df.columns, target_col)
    factor_result = run_factor_analysis(df, factor_cols, analysis_targets)
    weight_schemes = build_weight_schemes(factor_cols, factor_result)
    weights = weight_schemes["hypothesis_weight"]
    scored_df = normalize_factors(df, factor_directions)
    scored_df = compute_composite_score(scored_df, weights)
    backtest_result = run_top_pct_backtest(
        scored_df,
        score_col="composite_score",
        target_col=target_col,
        top_pct=top_pct,
        min_holdings=min_holdings,
        cost_bps=cost_bps,
    )
    diagnosis = diagnose_strategy(factor_result, backtest_result, weights)
    report = generate_report(_with_selection_payload(hypothesis, selection), factors, factor_result, backtest_result, diagnosis, weights)

    return {
        "hypothesis": _with_selection_payload(hypothesis, selection),
        "selection_result": selection.to_dict(),
        "factors": factors,
        "probe_tables": list(probe_results.keys()),
        "data_shape": df.shape,
        "factor_result": factor_result,
        "weight_schemes": weight_schemes,
        "backtest_result": backtest_result,
        "diagnosis": diagnosis,
        "report": report,
    }


def tool_run_condition_research_pipeline(
    user_idea: str,
    start_date: str,
    end_date: str,
    data_backend: str = "bigquant",
    akshare_data_source: str = "auto",
    akshare_proxy_patch_token: str | None = None,
    akshare_proxy_patch_gateway: str = "101.201.173.125",
    akshare_proxy_patch_hook_domains: list[str] | None = None,
    akshare_proxy_patch_retry: int = 30,
    akshare_proxy_patch_fast: bool = True,
    table_name: str = "cn_fund_bar1d",
    local_etf_parquet: str = "data/parquet/local_etf_daily.parquet",
    local_benchmark_parquet: str | None = None,
    volume_col: str = "volume",
    turnover_col: str = "turn",
    min_event_count: int = 200,
) -> dict[str, Any]:
    """Run the full condition-study pipeline as a single tool call."""
    hypothesis = parse_condition_research(user_idea)
    probe_results = run_data_probe() if data_backend == "bigquant" else {}
    df = load_condition_research_data(
        data_backend=data_backend,
        data_source=akshare_data_source,
        start_date=start_date,
        end_date=end_date,
        table_name=table_name,
        volume_col=volume_col,
        turnover_col=turnover_col,
        parquet_path=local_etf_parquet,
        benchmark_parquet_path=local_benchmark_parquet,
        proxy_patch_token=akshare_proxy_patch_token,
        proxy_patch_gateway=akshare_proxy_patch_gateway,
        proxy_patch_hook_domains=akshare_proxy_patch_hook_domains,
        proxy_patch_retry=akshare_proxy_patch_retry,
        proxy_patch_fast=akshare_proxy_patch_fast,
    )
    result = run_conditional_probability_test(
        df=df,
        conditions=hypothesis["conditions"],
        target=hypothesis["target"],
        min_event_count=min_event_count,
    )
    diagnosis = diagnose_condition_result(result)
    report = generate_condition_report(hypothesis, result, diagnosis)

    return {
        "hypothesis": hypothesis,
        "probe_tables": list(probe_results.keys()),
        "data_shape": df.shape,
        "result": result,
        "diagnosis": diagnosis,
        "report": report,
    }


def _benchmark_context_available(df) -> bool:
    for column in ("benchmark_close", "benchmark_return_20d", "benchmark_return_60d"):
        if column in df.columns and df[column].notna().any():
            return True
    return False


def _build_available_context(df) -> set[str]:
    context = set()
    if _benchmark_context_available(df):
        context.add("benchmark_series")
    return context


def _analysis_target_columns(columns, primary_target: str) -> list[str]:
    candidates = [primary_target]
    for column in ("future_return_5d", "future_return_10d", "future_return_20d"):
        if column in columns and column not in candidates:
            candidates.append(column)
    return candidates


def _intent_to_dict(intent) -> dict[str, Any]:
    if hasattr(intent, "to_dict"):
        return intent.to_dict()
    if isinstance(intent, dict):
        return intent
    return {"raw_query": str(intent)}


def _with_selection_payload(intent, selection) -> dict[str, Any]:
    payload = _intent_to_dict(intent)
    payload["selection_result"] = selection.to_dict()
    payload["selected_factors"] = [factor.get("name") for factor in getattr(selection, "selected_factors", [])]
    return payload


def _build_selection_failure_report(hypothesis, selection) -> str:
    intent = _intent_to_dict(hypothesis)
    lines = [
        "# AutoETF Research Plan",
        "",
        "## 研究无法执行",
        "",
        f"- 用户问题：{intent.get('raw_query', '')}",
        f"- 研究状态：{selection.status.value if hasattr(selection.status, 'value') else selection.status}",
        "",
        "### 已识别因子",
    ]
    selected_factors = getattr(selection, "selected_factors", None) or []
    if selected_factors:
        for factor in selected_factors:
            name = factor.get("name") if isinstance(factor, dict) else str(factor)
            if name:
                lines.append(f"- {name}")
    elif intent.get("factor_names"):
        for name in intent.get("factor_names", []):
            lines.append(f"- {name}")
    else:
        lines.append("- 无")

    if getattr(selection, "ambiguous_terms", None):
        lines.extend(["", "### 歧义项"])
        lines.extend(f"- {item}" for item in selection.ambiguous_terms)
    if getattr(selection, "unresolved_terms", None):
        lines.extend(["", "### 未识别项"])
        lines.extend(f"- {item}" for item in selection.unresolved_terms)
    if getattr(selection, "unavailable_factors", None):
        lines.extend(["", "### 不可执行因子"])
        for item in selection.unavailable_factors:
            lines.append(f"- {item.name}: {item.reason}")
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


TOOL_REGISTRY = {
    "run_data_probe": tool_run_data_probe,
    "resolve_factor_intent": tool_resolve_factor_intent,
    "build_factor_plan": tool_build_factor_plan,
    "get_available_factors": tool_get_available_factors,
    "load_etf_data": tool_load_etf_data,
    "run_factor_analysis": tool_run_factor_analysis,
    "run_conditional_probability_test": tool_run_conditional_probability_test,
    "build_weight_schemes": tool_build_weight_schemes,
    "compute_composite_score": tool_compute_composite_score,
    "diagnose_strategy": tool_diagnose_strategy,
    "generate_report": tool_generate_report,
    "run_factor_research_pipeline": tool_run_factor_research_pipeline,
    "run_condition_research_pipeline": tool_run_condition_research_pipeline,
}


def list_agent_tools() -> list[str]:
    return sorted(TOOL_REGISTRY)


def call_agent_tool(name: str, *args, **kwargs):
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown agent tool: {name}")
    return TOOL_REGISTRY[name](*args, **kwargs)
