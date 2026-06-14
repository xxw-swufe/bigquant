"""Agent-facing tool wrappers for the ETF research workflow."""

from __future__ import annotations

from typing import Optional

from src.condition_analysis import run_conditional_probability_test
from src.data_probe_bigquant import run_data_probe
from src.data_loader_bigquant import load_etf_data_bigquant
from src.diagnosis import diagnose_strategy
from src.factor_analysis import run_factor_analysis
from src.factor_availability import build_factor_tool_payload, get_available_factors
from src.factor_resolver import resolve_factor_intent
from src.report import generate_report
from src.scoring import build_weight_schemes, compute_metadata_composite_score


def tool_run_data_probe(**kwargs):
    return run_data_probe(**kwargs)


def tool_resolve_factor_intent(user_idea: str):
    return resolve_factor_intent(user_idea)


def tool_build_factor_plan(user_idea: str, required_columns: Optional[list[str]] = None):
    return build_factor_tool_payload(user_idea, required_columns=required_columns)


def tool_get_available_factors(required_columns: Optional[list[str]] = None):
    return get_available_factors(required_columns)


def tool_load_etf_data(**kwargs):
    return load_etf_data_bigquant(**kwargs)


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
}


def list_agent_tools() -> list[str]:
    return sorted(TOOL_REGISTRY)


def call_agent_tool(name: str, *args, **kwargs):
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown agent tool: {name}")
    return TOOL_REGISTRY[name](*args, **kwargs)
