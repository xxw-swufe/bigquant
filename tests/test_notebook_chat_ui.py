from dataclasses import dataclass

from src.chat_state import create_chat_state
from src.notebook_chat_ui import run_chat_turn_with_captured_logs


@dataclass
class FakeConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    fund_table: str = "cn_fund_bar1d"
    top_pct: float = 0.10


def test_chat_turn_captures_tool_logs_across_followups():
    calls = []

    def fake_runner(user_input, config, use_llm, api_key, model, state):
        calls.append(user_input)
        print("===== fake_probe_table =====")
        print("shape: (5, 17)")
        next_state = dict(state)
        next_state["conversation_history"] = state.get("conversation_history", []) + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"answer for {user_input}"},
        ]
        return {
            "tool_name": "run_condition_research_pipeline",
            "decision": {"reason": "fake route"},
            "reply": f"answer for {user_input}",
            "state": next_state,
        }

    config = FakeConfig()
    state = create_chat_state(config)
    first = run_chat_turn_with_captured_logs(
        fake_runner,
        user_input="缩量上涨后，下一日ETF涨的概率如何",
        config=config,
        use_llm=False,
        api_key=None,
        model="fake-model",
        state=state,
    )
    second = run_chat_turn_with_captured_logs(
        fake_runner,
        user_input="加上成交额上升",
        config=config,
        use_llm=False,
        api_key=None,
        model="fake-model",
        state=first["result"]["state"],
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert "fake_probe_table" in first["logs"]
    assert "fake_probe_table" in second["logs"]
    assert first["result"]["reply"] == "answer for 缩量上涨后，下一日ETF涨的概率如何"
    assert second["result"]["reply"] == "answer for 加上成交额上升"
    assert calls == ["缩量上涨后，下一日ETF涨的概率如何", "加上成交额上升"]


def test_chat_turn_returns_error_instead_of_raising():
    def failing_runner(**kwargs):
        print("log before failure")
        raise ValueError("fake failure")

    turn = run_chat_turn_with_captured_logs(
        failing_runner,
        user_input="缩量下跌呢",
        config=FakeConfig(),
        use_llm=False,
        api_key=None,
        model="fake-model",
        state=create_chat_state(),
    )

    assert turn["ok"] is False
    assert "log before failure" in turn["logs"]
    assert turn["error"] == "ValueError: fake failure"
    assert "failing_runner" in turn["traceback"]
