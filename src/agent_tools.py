"""Agent-facing tool wrappers for the ETF research workflow."""

from __future__ import annotations

from typing import Any
from typing import Optional

from src.condition_analysis import run_conditional_probability_test
from src.condition_parser import parse_condition_research
from src.condition_report import generate_condition_report
from src.data_probe_bigquant import run_data_probe
from src.data_loader_bigquant import load_etf_data_bigquant
from src.diagnosis import diagnose_strategy
from src.factor_analysis import run_factor_analysis
from src.factor_availability import build_factor_tool_payload, get_available_factors
from src.factor_generator import generate_factor_candidates
from src.factor_resolver import resolve_factor_intent
from src.report import generate_report
from src.scoring import build_weight_schemes, compute_composite_score, normalize_factors
from src.backtest import run_top_pct_backtest
from src.condition_analysis import diagnose_condition_result


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


def tool_run_factor_research_pipeline(
    user_idea: str,
    start_date: str,
    end_date: str,
    table_name: str = "cn_fund_bar1d",
    top_pct: float = 0.10,
    min_holdings: int = 3,
    cost_bps: float = 10.0,
    volume_col: str = "volume",
) -> dict[str, Any]:
    """Run the full factor research pipeline as a single tool call."""
    hypothesis = resolve_factor_intent(user_idea)
    factors = generate_factor_candidates(hypothesis)
    probe_results = run_data_probe()
    df = load_etf_data_bigquant(
        start_date=start_date,
        end_date=end_date,
        table_name=table_name,
        volume_col=volume_col,
    )

    factor_cols = [factor["name"] for factor in factors if factor["name"] in df.columns]
    factor_directions = {
        factor["name"]: factor["direction"]
        for factor in factors
        if factor["name"] in factor_cols
    }

    factor_result = run_factor_analysis(df, factor_cols, ["future_return_5d", "future_return_20d"])
    weight_schemes = build_weight_schemes(factor_cols, factor_result)
    weights = weight_schemes["hypothesis_weight"]
    scored_df = normalize_factors(df, factor_directions)
    scored_df = compute_composite_score(scored_df, weights)
    backtest_result = run_top_pct_backtest(
        scored_df,
        score_col="composite_score",
        target_col="future_return_5d",
        top_pct=top_pct,
        min_holdings=min_holdings,
        cost_bps=cost_bps,
    )
    diagnosis = diagnose_strategy(factor_result, backtest_result, weights)
    report = generate_report(hypothesis, factors, factor_result, backtest_result, diagnosis, weights)

    return {
        "hypothesis": hypothesis,
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
    table_name: str = "cn_fund_bar1d",
    volume_col: str = "volume",
    turnover_col: str = "turn",
    min_event_count: int = 200,
) -> dict[str, Any]:
    """Run the full condition-study pipeline as a single tool call."""
    hypothesis = parse_condition_research(user_idea)
    probe_results = run_data_probe()
    df = load_etf_data_bigquant(
        start_date=start_date,
        end_date=end_date,
        table_name=table_name,
        volume_col=volume_col,
    )
    if turnover_col in df.columns and "turnover" not in df.columns:
        df["turnover"] = df[turnover_col]
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
