"""Tests for the chat-panel report renderer (minimal: banner + Markdown)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.notebook_report_render import _markdown_to_pre, build_report_html


def _fake_factor_tool_result() -> dict:
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    ic_series = pd.Series([0.01] * 29, index=dates[1:])
    quantile_return = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=[1, 2, 3, 4, 5])
    daily_returns = pd.Series([0.001] * 29, index=dates[1:])
    factor_result = {
        "rsi_14d": {
            "future_return_5d": {
                "ic_series": ic_series,
                "quantile_return": quantile_return,
            }
        }
    }
    backtest_result = {
        "daily_returns": pd.DataFrame({"return": daily_returns}),
        "cumulative_return": 0.05,
        "annual_return": 0.06,
        "max_drawdown": -0.05,
        "sharpe": 0.5,
        "win_rate": 0.55,
        "annualized_turnover": 1.0,
        "total_cost": 0.001,
    }
    return {
        "factor_result": factor_result,
        "backtest_result": backtest_result,
        "diagnosis": {"summary": "ok", "research_decision": "continue"},
        "report": (
            "# AutoETF Research Report\n\n"
            "## 5. factor analysis\n"
            "## 6. backtest\n"
        ),
    }


def test_build_report_html_returns_banner_and_pre_for_factor_research(tmp_path):
    html = build_report_html(_fake_factor_tool_result(), "factor_research", outputs_dir=str(tmp_path))
    assert html, "renderer should produce non-empty HTML for factor_research"
    assert "<pre" in html, "should include a Markdown <pre> block"
    assert "已生成" in html and "张图表" in html, "should announce chart count"
    assert str(tmp_path) in html, "banner should point at the resolved outputs dir"
    assert "<img" not in html, "should NOT inline images"
    assert "<a href=" not in html, "should NOT link PNGs"
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, f"expected PNGs in {tmp_path}, got {pngs}"


def test_build_report_html_returns_empty_for_non_research():
    assert build_report_html({}, "direct_reply", outputs_dir="/tmp") == ""
    assert build_report_html({}, "error", outputs_dir="/tmp") == ""
    assert build_report_html({}, "chitchat", outputs_dir="/tmp") == ""


def test_build_report_html_for_condition_research(tmp_path):
    condition_result = {
        "result": {
            "event_count": 100,
            "event_up_probability": 0.55,
            "baseline_up_probability": 0.50,
            "probability_lift": 0.05,
            "event_mean_return": 0.001,
            "baseline_mean_return": 0.0005,
            "yearly_stats": pd.DataFrame({"year": [2023], "lift": [0.04]}),
            "instrument_stats": pd.DataFrame({"instrument": ["ETF.A"], "lift": [0.03]}),
        },
        "report": "# condition\n\n## 4. probability\n## 6. yearly\n",
    }
    html = build_report_html(condition_result, "condition_research", outputs_dir=str(tmp_path))
    assert html
    assert "<pre" in html
    assert "已生成" in html
    assert "<img" not in html
    assert "<a href=" not in html
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "condition panel should produce PNGs"


def test_markdown_to_pre_escapes_html_chars():
    out = _markdown_to_pre("<script>alert(1)</script>")
    assert "<script>" not in out, "must escape HTML in markdown"
    assert "&lt;script&gt;" in out
