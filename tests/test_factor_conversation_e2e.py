from dataclasses import dataclass
from typing import Optional

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
    local_benchmark_parquet: Optional[str] = None


def _selection_result(names, horizon=5):
    selected_factors = [{"name": name, "category": "momentum" if "momentum" in name else "risk"} for name in names]
    return {
        "status": "ready",
        "can_execute": True,
        "selected_factors": selected_factors,
        "selection_reasons": {name: "explicit" for name in names},
        "unresolved_terms": [],
        "unavailable_factors": [],
        "ambiguous_terms": [],
        "target": {"metric": "future_return", "horizon": horizon},
        "selection_source": "explicit",
    }


def test_pipeline_receives_committed_plan_only(monkeypatch):
    captured_user_ideas = []
    captured_planned = []

    def fake_call_agent_tool(name, *args, **kwargs):
        captured_user_ideas.append(kwargs.get("user_idea"))
        captured_planned.append(list(kwargs.get("planned_factor_names") or []))
        planned_names = list(kwargs.get("planned_factor_names") or [])
        if planned_names:
            names = planned_names
        else:
            text = kwargs.get("user_idea", "")
            if "volatility_20d" in text or "低波动" in text or "波动率" in text:
                names = ["momentum_20d", "volatility_20d"] if "momentum_20d" in text or "动量" in text else ["volatility_20d"]
            else:
                names = ["momentum_20d"]
        horizon = int((kwargs.get("planned_target") or {}).get("horizon", 5))
        if horizon == 5 and "未来10日" in kwargs.get("user_idea", ""):
            horizon = 10
        text = kwargs.get("user_idea", "")
        return {
            "selection_result": _selection_result(names, horizon=horizon),
            "report": f"report for {text}",
            "factors": [{"name": name} for name in names],
            "hypothesis": {"raw_query": text},
        }

    monkeypatch.setattr("src.chat_agent.call_agent_tool", fake_call_agent_tool)

    state = create_chat_state(FakeConfig())
    first = run_research_chat("研究20日动量", FakeConfig(), use_llm=False, state=state)
    second = run_research_chat("再加低波动", FakeConfig(), use_llm=False, state=first["state"])
    third = run_research_chat("改成未来10日", FakeConfig(), use_llm=False, state=second["state"])

    assert first["state"]["current_context"]["selected_factor_names"] == ["momentum_20d"]
    assert second["state"]["current_context"]["selected_factor_names"] == ["momentum_20d", "volatility_20d"]
    assert third["state"]["current_context"]["selected_factor_names"] == ["momentum_20d", "volatility_20d"]
    assert third["state"]["current_context"]["target"]["horizon"] == 10
    assert captured_user_ideas[1] == "再加低波动"
    assert captured_planned[1] == ["momentum_20d", "volatility_20d"]
    assert captured_planned[2] == ["momentum_20d", "volatility_20d"]


def test_empty_plan_never_starts_pipeline(monkeypatch):
    called = {"value": False}

    def fake_call_agent_tool(*args, **kwargs):
        called["value"] = True
        raise AssertionError("pipeline should not be called")

    monkeypatch.setattr("src.chat_agent.call_agent_tool", fake_call_agent_tool)

    result = run_research_chat("帮我做一个ETF因子研究", FakeConfig(), use_llm=False, state=create_chat_state(FakeConfig()))

    assert called["value"] is False
    assert result["tool_name"] == "direct_reply"
    assert "请先明确" in result["reply"]
