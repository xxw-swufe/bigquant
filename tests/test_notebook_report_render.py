"""Tests for the chat-panel report renderer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.notebook_report_render import (
    _image_grid_html,
    _markdown_to_pre,
    build_report_html,
)


def _fake_factor_tool_result() -> dict:
    """Return a minimal but valid factor pipeline tool_result."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    ic_series = pd.Series([0.01] * 29, index=dates[1:])
    quantile_return = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=[1, 2, 3, 4, 5])
    daily_returns = pd.Series([0.001] * 29, index=dates[1:])
    # plot_factor_analysis_panel expects: {factor_name: {target_name: {ic_series, quantile_return}}}
    factor_result = {
        "rsi_14d": {
            "future_return_5d": {
                "ic_series": ic_series,
                "quantile_return": quantile_return,
            }
        }
    }
    # plot_backtest_panel needs daily_returns as a DataFrame with a "return" column.
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
        "diagnosis": {"summary": "ok", "research_decision": "继续研究"},
        "report": (
            "# AutoETF Research Report\n\n"
            "## 5. 单因子分析结果\n"
            "## 6. Top 10% 简化回测结果\n"
        ),
    }


def test_build_report_html_returns_html_for_factor_research(tmp_path: Path) -> None:
    html = build_report_html(_fake_factor_tool_result(), "factor_research", outputs_dir=str(tmp_path))
    assert html, "renderer should produce non-empty HTML for factor_research"
    assert "<img" in html, "should embed plot images"
    assert "file://" in html, "should reference local PNG files via file:// URI"
    assert "<pre" in html, "should include a Markdown <pre> block"
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, f"expected PNGs in {tmp_path}, got {pngs}"


def test_build_report_html_returns_empty_for_non_research() -> None:
    """`直接回复 / error / 闲聊` 不应该渲染图表或报告。"""
    assert build_report_html({}, "direct_reply", outputs_dir="/tmp") == ""
    assert build_report_html({}, "error", outputs_dir="/tmp") == ""
    assert build_report_html({}, "chitchat", outputs_dir="/tmp") == ""


def test_build_report_html_for_condition_research(tmp_path: Path) -> None:
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
        "report": "# condition\n\n## 4. 次日上涨概率检验\n## 6. 分年份稳定性\n",
    }
    html = build_report_html(condition_result, "condition_research", outputs_dir=str(tmp_path))
    assert html
    assert "<img" in html
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "condition panel should produce PNGs"


def test_markdown_to_pre_escapes_html_chars() -> None:
    out = _markdown_to_pre("<script>alert(1)</script>")
    assert "<script>" not in out, "must escape HTML in markdown"
    assert "&lt;script&gt;" in out


def test_image_grid_html_empty_for_no_paths() -> None:
    assert _image_grid_html([]) == ""


def test_image_grid_html_renders_one_img_per_path(tmp_path: Path) -> None:
    fake_pngs = [str(tmp_path / f"chart_{i}.png") for i in range(3)]
    for p in fake_pngs:
        Path(p).write_bytes(b"")
    out = _image_grid_html(fake_pngs)
    assert out.count("<img") == 3
    assert "file://" in out
