from src.factor_generator import generate_factor_candidates
from src.etf_factor_library import get_factor_library
from src.factor_resolver import resolve_factor_intent
from src.factor_availability import build_factor_tool_payload, select_factor_plan
from src.scoring import compute_metadata_composite_score, normalize_factor_frame, score_series
from src.agent_tools import list_agent_tools, call_agent_tool
from src.chat_agent import decide_tool, summarize_tool_result
from src.expression_knowledge_base import classify_expression, get_expression_by_phrase, search_expressions, list_expression_layers
from src.condition_parser import parse_condition_research


def test_generated_factors_do_not_use_future_data():
    factors = generate_factor_candidates({})
    assert factors
    assert all(not factor["uses_future_data"] for factor in factors)
    assert all("m_lead" not in factor["formula"] for factor in factors)


def test_reference_library_size_is_mvp_friendly():
    factors = get_factor_library()
    assert 10 <= len(factors) <= 40
    assert all(not factor["uses_future_data"] for factor in factors)


def test_factor_library_has_enriched_metadata():
    factors = get_factor_library()
    sample = factors[0]
    assert "aliases" in sample
    assert "correlated_with" in sample
    assert "score_shape" in sample
    assert "data_dependency" in sample


def test_factor_resolver_maps_common_language():
    result = resolve_factor_intent("OBV上涨，缩量下跌")
    assert result["matched_factors"]
    assert result["research_type"] in {"conditional_event_study", "factor_score"}


def test_factor_plan_and_agent_tools_exist():
    tools = list_agent_tools()
    assert "resolve_factor_intent" in tools
    assert "build_factor_plan" in tools
    payload = build_factor_tool_payload("成交额放大，趋势走强")
    assert "selected_factors" in payload
    assert payload["candidate_count"] >= payload["selected_count"]


def test_factor_selection_deduplicates_redundancy():
    factors = get_factor_library()
    selected = select_factor_plan(factors, max_factors=5)
    names = [factor["name"] for factor in selected]
    assert len(names) == len(set(names))


def test_non_linear_scoring_and_metadata_composite_score():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "instrument": ["A", "B", "C"],
            "turnover_rate": [1.0, 4.0, 8.0],
            "close": [10.0, 11.0, 12.0],
        }
    )
    scored = score_series(df["turnover_rate"], score_shape="range_better", optimal_range=(3.0, 5.0))
    assert scored.notna().all()
    metadata = {
        "turnover_rate": {
            "direction": "higher_better",
            "score_shape": "range_better",
            "optimal_range": (3.0, 5.0),
        }
    }
    normalized = normalize_factor_frame(df, metadata)
    normalized["turnover_rate_score"] = scored
    weighted = compute_metadata_composite_score(
        normalized,
        metadata,
        {"turnover_rate": 1.0},
    )
    assert "composite_score" in weighted
    assert weighted["composite_score"].notna().all()


def test_agent_tool_dispatch_works():
    result = call_agent_tool("resolve_factor_intent", "obv上涨，缩量下跌")
    assert "matched_factors" in result


def test_search_and_range_better_scoring():
    from src.etf_factor_library import search_factors

    assert search_factors("OBV上涨", limit=5)
    import pandas as pd

    scored = score_series(pd.Series([1, 3, 4, 5, 7]), score_shape="range_better", optimal_range=(3, 5))
    assert scored.iloc[1] >= scored.iloc[0]
    assert scored.iloc[2] >= scored.iloc[0]


def test_chat_agent_rule_based_routing_and_summary():
    condition_decision = decide_tool("量比大于1，今日上涨，明天上涨概率大吗？", use_llm=False)
    factor_decision = decide_tool("成交额放大趋势走强，做因子回测", use_llm=False)
    assert condition_decision["tool_name"] == "run_condition_research_pipeline"
    assert factor_decision["tool_name"] == "run_factor_research_pipeline"

    reply = summarize_tool_result(
        "测试",
        "run_condition_research_pipeline",
        {
            "result": {"event_count": 10, "total_count": 100, "probability_lift": 0.02},
            "diagnosis": {"research_decision": "谨慎继续"},
        },
        use_llm=False,
    )
    assert "条件研究完成" in reply


def test_expression_knowledge_base_routes_common_phrases():
    match = get_expression_by_phrase("今日上涨")
    assert match is not None
    assert match["canonical_name"] == "return_1d_positive"

    results = search_expressions("成交额放大", limit=5)
    assert results
    assert any(item["canonical_name"] == "amount_ratio_expansion" for item in results)

    routed = classify_expression("未来5日收益")
    assert routed["best_match"]["canonical_name"] == "future_return_5d"
    assert "target" in routed["route_intent"]

    combo = classify_expression("今日上涨且成交额放大")
    assert combo["best_match"] is not None
    assert combo["matched"]
    assert "processed_factor" in combo["matched_layers"]


def test_expression_knowledge_base_has_full_layer_structure():
    layers = list_expression_layers()
    assert layers["raw_data"] >= 1
    assert layers["native_indicator"] >= 1
    assert layers["processed_factor"] >= 1
    assert layers["composite_score"] >= 1
    assert layers["intent_modifier"] >= 1


def test_condition_parser_handles_compound_expression():
    result = parse_condition_research("今日上涨且成交额放大")
    assert len(result["conditions"]) >= 2
    fields = {item["field"] for item in result["conditions"]}
    assert "return_1d" in fields
    assert "amount_ratio_20d" in fields


def test_expression_kb_routes_composite_scores():
    match = classify_expression("趋势分")
    assert match["best_match"]["canonical_name"] == "trend_score"
    assert "metric" in match["route_intent"] or "factor" in match["route_intent"]


def test_expression_search_does_not_match_opposite_related_terms():
    combo = classify_expression("今日上涨且成交额放大")
    names = {item["canonical_name"] for item in combo["matched"]}
    assert "return_1d_positive" in names
    assert "amount_ratio_expansion" in names
    assert "return_1d_negative" not in names
    assert "volume_contraction" not in names


def test_compound_templates_keep_correct_operators():
    down = parse_condition_research("放量下跌")
    down_rules = {(item["field"], item["operator"], item["value"]) for item in down["conditions"]}
    assert ("return_1d", "<", 0.0) in down_rules
    assert ("amount_ratio_20d", ">", 1.0) in down_rules

    pullback = parse_condition_research("缩量回调")
    pullback_rules = {(item["field"], item["operator"], item["value"]) for item in pullback["conditions"]}
    assert ("return_1d", "<", 0.0) in pullback_rules
    assert ("amount_ratio_20d", "<", 1.0) in pullback_rules


def test_target_expression_stays_out_of_condition_rules():
    result = parse_condition_research("缩量上涨，未来5日收益")
    condition_fields = {item["field"] for item in result["conditions"]}
    assert "future_return_5d" not in condition_fields
    assert result["target"]["field"] == "future_return_5d"
