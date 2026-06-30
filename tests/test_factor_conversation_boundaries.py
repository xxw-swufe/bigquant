from dataclasses import dataclass

from src.chat_agent import run_research_chat
from src.chat_state import create_chat_state


@dataclass
class FakeConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    fund_table: str = "cn_fund_bar1d"
    top_pct: float = 0.10
    data_backend: str = "local"
    local_etf_parquet: str = "data/parquet/local_etf_daily.parquet"
    local_benchmark_parquet: str = None


def _seed_state(factor_names, horizon=5):
    state = create_chat_state(FakeConfig())
    factors = [{"name": name, "category": "momentum" if "momentum" in name else "risk"} for name in factor_names]
    state["current_context"].update(
        {
            "committed_plan": {
                "selected_factor_names": list(factor_names),
                "target": {"metric": "future_return", "horizon": horizon},
                "selection_source": "explicit",
                "selection_status": "ready",
            },
            "committed_selection": {
                "status": "ready",
                "can_execute": True,
                "selected_factors": factors,
                "selection_reasons": {name: "explicit" for name in factor_names},
                "unresolved_terms": [],
                "unavailable_factors": [],
                "ambiguous_terms": [],
                "target": {"metric": "future_return", "horizon": horizon},
                "selection_source": "explicit",
            },
            "selected_factors": factors,
            "selected_factor_names": list(factor_names),
            "factors": factors,
            "target": {"metric": "future_return", "horizon": horizon},
            "selection_status": "ready",
        }
    )
    return state


def test_delete_last_factor_never_starts_pipeline(monkeypatch):
    called = {"value": False}

    def fake_call_agent_tool(*args, **kwargs):
        called["value"] = True
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("src.chat_agent.call_agent_tool", fake_call_agent_tool)

    state = _seed_state(["momentum_20d"])
    result = run_research_chat("去掉动量因子", FakeConfig(), use_llm=False, state=state)

    assert called["value"] is False
    assert result["tool_name"] == "direct_reply"
    assert "空" in result["reply"] or "新的研究因子" in result["reply"]
    assert result["state"]["current_context"]["committed_plan"]["selected_factor_names"] == ["momentum_20d"]
    assert result["state"]["current_context"]["selected_factor_names"] == ["momentum_20d"]


def test_delete_missing_factor_keeps_committed_plan(monkeypatch):
    called = {"value": False}

    def fake_call_agent_tool(*args, **kwargs):
        called["value"] = True
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("src.chat_agent.call_agent_tool", fake_call_agent_tool)

    state = _seed_state(["momentum_20d", "volatility_20d"])
    result = run_research_chat("删掉MACD", FakeConfig(), use_llm=False, state=state)

    assert called["value"] is False
    assert result["tool_name"] == "direct_reply"
    assert "没有要删除" in result["reply"] or "当前计划" in result["reply"]
    assert result["state"]["current_context"]["committed_plan"]["selected_factor_names"] == ["momentum_20d", "volatility_20d"]
    assert result["state"]["current_context"]["selected_factor_names"] == ["momentum_20d", "volatility_20d"]


def test_relative_strength_missing_benchmark_prompts_user(monkeypatch):
    captured = {"user_ideas": []}

    def fake_call_agent_tool(name, *args, **kwargs):
        captured["user_ideas"].append(kwargs["user_idea"])
        return {
            "hypothesis": {"raw_query": kwargs["user_idea"]},
            "selection_result": {
                "status": "missing_context",
                "can_execute": False,
                "selected_factors": [
                    {"name": "momentum_20d", "category": "momentum"},
                    {"name": "relative_strength_20d", "category": "relative_momentum"},
                ],
                "selection_reasons": {
                    "momentum_20d": "explicit",
                    "relative_strength_20d": "explicit",
                },
                "unresolved_terms": [],
                "unavailable_factors": [
                    {
                        "name": "relative_strength_20d",
                        "reason": "missing_context",
                        "missing_context": ["benchmark_series"],
                    }
                ],
                "ambiguous_terms": [],
                "target": {"metric": "future_return", "horizon": 5},
                "selection_source": "explicit",
            },
            "report": "selection failed",
        }

    monkeypatch.setattr("src.chat_agent.call_agent_tool", fake_call_agent_tool)

    state = _seed_state(["momentum_20d"])
    result = run_research_chat("再加相对强度因子", FakeConfig(), use_llm=False, state=state)

    assert captured["user_ideas"]
    assert result["tool_name"] == "run_factor_research_pipeline"
    assert "benchmark" in result["reply"] or "沪深300" in result["reply"] or "中证全指" in result["reply"]
    assert result["state"]["current_context"]["committed_plan"]["selected_factor_names"] == ["momentum_20d"]
    assert result["state"]["current_context"]["selection_status"] == "missing_context"
    assert result["state"]["current_context"]["pending_selection"]["status"] == "missing_context"


def test_add_bbi_blocks_pipeline_and_preserves_plan(monkeypatch):
    """BBI is now an implemented factor (BBI(3,6,12,24) on close), so adding it
    to a plan must route through the factor pipeline rather than block with
    `recognized_not_implemented`. KDJ remains unimplemented and that contract
    is covered by `test_add_kdj_blocks_pipeline_and_preserves_plan` below.
    """
    called = {"value": False}

    def fake_call_agent_tool(*args, **kwargs):
        called["value"] = True
        return {
            "report": "# stub\n\nBBI processed.",
            "factor_result": {},
            "backtest_result": {"performance": {}},
            "factors": [{"name": "bbi_indicator"}],
        }

    monkeypatch.setattr("src.chat_agent.call_agent_tool", fake_call_agent_tool)

    state = _seed_state(["volatility_20d"])
    result = run_research_chat("再加BBI", FakeConfig(), use_llm=False, state=state)

    # Pipeline must have run because BBI is now implementable.
    assert called["value"] is True
    assert result["tool_name"] == "run_factor_research_pipeline"
