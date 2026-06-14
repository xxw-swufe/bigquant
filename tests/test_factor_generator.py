from src.factor_generator import generate_factor_candidates
from src.etf_factor_library import get_factor_library
from src.factor_resolver import resolve_factor_intent
from src.factor_availability import build_factor_tool_payload, select_factor_plan
from src.scoring import compute_metadata_composite_score, normalize_factor_frame, score_series
from src.agent_tools import list_agent_tools, call_agent_tool


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
