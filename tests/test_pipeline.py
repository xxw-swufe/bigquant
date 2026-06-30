"""Unit tests for ``src/pipeline.py``.

These tests use a fake ``chat_runner`` so the pipeline can be exercised
without invoking the LLM or any data loader. Each test asserts on the
public contract of ``run_research_pipeline``:

- correct ``task_type`` classification
- correct report persistence path
- correct handling of ``direct_reply`` / error cases
- timestamps don't collide with each other across the test file
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    CONDITION_TOOL,
    DIRECT_REPLY_TOOL,
    FACTOR_TOOL,
    TASK_TYPE_CONDITION,
    TASK_TYPE_DIRECT,
    TASK_TYPE_ERROR,
    TASK_TYPE_FACTOR,
    run_research_pipeline,
)


@dataclass
class FakeConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    fund_table: str = "cn_fund_bar1d"
    top_pct: float = 0.10
    data_backend: str = "local"
    local_etf_parquet: str = "data/parquet/local_etf_daily.parquet"
    local_benchmark_parquet: str = None


FACTOR_REPORT = """# 因子研究报告

## 1. 用户策略想法
研究 20 日动量

## 2. 研究计划
因子: momentum_20d
目标: future_return_5d

## 3. 候选因子定义

## 4. 因子权重

## 5. 单因子分析结果

## 6. Top 10% 简化回测结果

## 7. 权重合理性与风险诊断
"""

CONDITION_REPORT = """# 条件研究报告

## 1. 用户问题
缩量上涨后下一日怎么样

## 2. 条件定义
amount_ratio_20d < 1.0

## 3. 样本与事件统计

## 4. 次日上涨概率检验

## 5. 次日平均收益检验

## 6. 分年份稳定性

## 7. 诊断结论
"""


def _make_runner(tool_name, tool_result):
    """Return a fake chat_runner producing the given tool_name/tool_result.

    The ``reply`` is taken from ``tool_result['reply']`` when present (so
    tests can assert the runner's reply flows through the pipeline),
    otherwise a default placeholder is used.
    """

    reply = tool_result.get("reply") if isinstance(tool_result, dict) else None

    def runner(*, user_input, config, use_llm, api_key, model, state):
        return {
            "user_input": user_input,
            "parsed_context": {},
            "decision": {"tool_name": tool_name, "reason": "test"},
            "tool_name": tool_name,
            "tool_result": tool_result,
            "reply": reply if reply is not None else f"fake-reply-for-{tool_name}",
            "state": state,
        }

    return runner


def _make_failing_runner(error_message):
    """Return a chat_runner that always raises."""

    def runner(*, user_input, config, use_llm, api_key, model, state):
        raise RuntimeError(error_message)

    return runner


def test_factor_research_persists_markdown_with_timestamp(tmp_path):
    runner = _make_runner(
        FACTOR_TOOL,
        {"report": FACTOR_REPORT, "data_shape": (100, 6), "factors": []},
    )

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_FACTOR
    assert result["tool_name"] == FACTOR_TOOL
    assert result["report_markdown"] == FACTOR_REPORT
    assert result["report_path"] is not None

    report_path = Path(result["report_path"])
    assert report_path.is_file()
    assert report_path.parent == tmp_path.resolve()
    assert report_path.name.startswith("research_report_")
    assert report_path.name.endswith(".md")
    assert report_path.read_text(encoding="utf-8") == FACTOR_REPORT
    assert result["error"] is None


def test_condition_research_persists_markdown_with_timestamp(tmp_path):
    runner = _make_runner(
        CONDITION_TOOL,
        {"report": CONDITION_REPORT, "data_shape": (200, 5), "result": {}},
    )

    result = run_research_pipeline(
        "缩量上涨后下一日怎么样",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_CONDITION
    assert result["tool_name"] == CONDITION_TOOL
    assert result["report_markdown"] == CONDITION_REPORT

    report_path = Path(result["report_path"])
    assert report_path.is_file()
    assert report_path.parent == tmp_path.resolve()
    assert report_path.name.startswith("condition_research_report_")
    assert report_path.name.endswith(".md")
    assert report_path.read_text(encoding="utf-8") == CONDITION_REPORT
    assert result["error"] is None


def test_direct_reply_does_not_write_report(tmp_path):
    runner = _make_runner(DIRECT_REPLY_TOOL, {"reply": "no research needed", "direct_reply": True})

    result = run_research_pipeline(
        "帮我做一个ETF研究",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_DIRECT
    assert result["report_path"] is None
    assert result["report_markdown"] == ""
    assert result["reply"] == "no research needed"
    assert result["error"] is None
    # outputs_dir is still created (mkdir), but no markdown file is written
    assert tmp_path.resolve().exists()
    assert list(tmp_path.iterdir()) == []


def test_error_when_runner_raises(tmp_path):
    runner = _make_failing_runner("kaboom")

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_ERROR
    assert result["report_path"] is None
    assert result["report_markdown"] == ""
    assert "kaboom" in (result["error"] or "")
    # On error we must NOT have created any markdown file.
    assert list(tmp_path.iterdir()) == []


def test_error_when_tool_result_missing_report(tmp_path):
    runner = _make_runner(FACTOR_TOOL, {"factors": []})  # no "report" key

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_ERROR
    assert result["report_path"] is None
    assert "markdown report" in (result["error"] or "")
    assert list(tmp_path.iterdir()) == []


def test_error_when_tool_name_unknown(tmp_path):
    runner = _make_runner("run_what_now", {"report": FACTOR_REPORT})

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_ERROR
    assert result["report_path"] is None
    assert result["report_markdown"] == ""


def test_outputs_dir_is_created_if_missing(tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    assert not nested.exists()

    runner = _make_runner(
        FACTOR_TOOL,
        {"report": FACTOR_REPORT},
    )

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=nested,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_FACTOR
    assert nested.exists()
    assert Path(result["report_path"]).is_file()


def test_default_outputs_dir_is_relative_outputs(tmp_path, monkeypatch):
    """When no outputs_dir is provided, files land under ./outputs (cwd-relative)."""
    monkeypatch.chdir(tmp_path)

    runner = _make_runner(FACTOR_TOOL, {"report": FACTOR_REPORT})

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_FACTOR
    assert Path(result["report_path"]).is_file()
    # Relative `outputs_dir` is anchored at the project root (parent of src/),
    # not at cwd. The report_path is therefore a sibling of src/, regardless
    # of where the test process was launched from.
    assert result["report_path"].endswith(".md")
    assert "/outputs/" in result["report_path"]
    assert Path(result["report_path"]).parent.name == "outputs"


def test_relative_outputs_dir_anchors_at_project_root(tmp_path, monkeypatch):
    """Relative outputs_dir must be anchored at the project root, NOT at cwd.

    This is the contract that lets the same Notebook work both on a laptop
    (cwd anywhere) and inside the BigQuant cloud container (cwd != project).
    """
    import os
    project_root = Path(__file__).resolve().parent.parent  # tests/ → project root
    monkeypatch.chdir(tmp_path)  # cwd is now somewhere unrelated

    runner = _make_runner(FACTOR_TOOL, {"report": FACTOR_REPORT})

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir="outputs",
        chat_runner=runner,
    )

    expected = (project_root / "outputs").resolve()
    assert result["report_path"] is not None
    assert Path(result["report_path"]).resolve().parent == expected
    assert Path(result["report_path"]).is_file()


def test_report_path_is_absolute(tmp_path):
    runner = _make_runner(FACTOR_TOOL, {"report": FACTOR_REPORT})

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["report_path"] is not None
    assert Path(result["report_path"]).is_absolute()


def test_repeated_runs_do_not_overwrite(tmp_path):
    """Each invocation should create a uniquely-named file."""
    runner = _make_runner(FACTOR_TOOL, {"report": FACTOR_REPORT})

    paths = []
    for _ in range(3):
        result = run_research_pipeline(
            "研究20日动量",
            config=FakeConfig(),
            use_llm=False,
            outputs_dir=tmp_path,
            chat_runner=runner,
        )
        paths.append(result["report_path"])
        # Force a second-resolution gap to be safe; the timestamp format
        # is %Y%m%d_%H%M%S so back-to-back calls within the same second
        # could collide in principle. Sleep covers that in the test.
        time.sleep(1.05)

    assert len(set(paths)) == 3, f"expected 3 distinct report paths, got {paths}"
    for p in paths:
        assert Path(p).is_file()


def test_user_idea_echoed_in_result(tmp_path):
    runner = _make_runner(FACTOR_TOOL, {"report": FACTOR_REPORT})

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["user_idea"] == "研究20日动量"


def test_default_config_is_used_when_none_provided(monkeypatch, tmp_path):
    """When config=None, the pipeline must still run via DEFAULT_CONFIG."""
    monkeypatch.chdir(tmp_path)
    runner = _make_runner(FACTOR_TOOL, {"report": FACTOR_REPORT})

    result = run_research_pipeline(
        "研究20日动量",
        config=None,
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_FACTOR
    assert result["report_path"] is not None
    assert Path(result["report_path"]).is_file()


# ---------------------------------------------------------------------------
# Stage B: chart integration tests
# ---------------------------------------------------------------------------


def _factor_tool_result(n_factors: int = 1) -> dict:
    """Build a fake factor tool_result with factor_result + backtest_result populated."""
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=30)
    daily_returns = pd.DataFrame({
        "return": np.random.normal(0.001, 0.01, 30),
        "gross_return": np.random.normal(0.0015, 0.01, 30),
        "turnover": np.random.uniform(0.05, 0.20, 30),
        "cost": np.random.uniform(0.0001, 0.001, 30),
        "holding_count": np.random.randint(3, 8, 30),
    }, index=dates)
    backtest_result = {
        "daily_returns": daily_returns,
        "performance": {
            "total_return": 0.10,
            "annual_return": 0.12,
            "max_drawdown": -0.05,
            "sharpe": 1.2,
            "win_rate": 0.55,
            "average_turnover": 0.10,
            "total_cost": 0.01,
        },
    }
    factor_result = {}
    for i in range(n_factors):
        factor_result[f"f{i}"] = {
            "future_return_5d": {
                "ic_summary": {"ic_mean": 0.02, "icir": 0.5, "ic_positive_ratio": 0.6, "ic_count": 30},
                "ic_series": pd.Series(np.random.normal(0.02, 0.05, 30), index=dates),
                "quantile_return": pd.Series(
                    [0.005, 0.008, 0.011, 0.014, 0.018], index=[1, 2, 3, 4, 5]
                ),
            }
        }
    return {
        "report": FACTOR_REPORT,
        "factor_result": factor_result,
        "backtest_result": backtest_result,
        "factors": [{"name": f"f{i}", "description": "x", "formula": "x", "direction": 1, "uses_future_data": False} for i in range(n_factors)],
    }


def _condition_tool_result() -> dict:
    """Build a fake condition tool_result with a populated result dict."""
    np.random.seed(11)
    yearly = pd.DataFrame({
        "group": [2022, 2023, 2024],
        "event_count": [50, 60, 70],
        "event_up_probability": [0.55, 0.58, 0.60],
        "baseline_up_probability": [0.50, 0.50, 0.50],
        "probability_lift": [0.05, 0.08, 0.10],
        "event_mean_return": [0.01] * 3,
        "baseline_mean_return": [0.005] * 3,
        "mean_return_lift": [0.005] * 3,
    })
    instrument = pd.DataFrame({
        "group": ["510300", "159915"],
        "event_count": [30, 25],
        "event_up_probability": [0.6, 0.55],
        "baseline_up_probability": [0.5, 0.5],
        "probability_lift": [0.1, 0.05],
        "event_mean_return": [0.01, 0.01],
        "baseline_mean_return": [0.005, 0.005],
        "mean_return_lift": [0.005, 0.005],
    })
    return {
        "report": CONDITION_REPORT,
        "result": {
            "event_up_probability": 0.567,
            "baseline_up_probability": 0.513,
            "yearly_stats": yearly,
            "instrument_stats": instrument,
        },
    }


def test_factor_research_emits_plot_paths(tmp_path):
    runner = _make_runner(FACTOR_TOOL, _factor_tool_result(n_factors=2))

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_FACTOR
    plot_paths = result["plot_paths"]
    assert isinstance(plot_paths, list)
    # At minimum: IC + quantile per factor (4 charts) + cumulative + drawdown (2) + correlation (1) + IC distribution per factor (2) = 9
    assert len(plot_paths) >= 4
    for p in plot_paths:
        assert Path(p).is_file()
        assert p.endswith(".png")


def test_factor_report_markdown_contains_image_references(tmp_path):
    runner = _make_runner(FACTOR_TOOL, _factor_tool_result(n_factors=2))

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert "![research_report_" in result["report_markdown"]
    assert "(research_report_" in result["report_markdown"]
    # References should be relative (basename), not absolute paths.
    assert "/Users/" not in result["report_markdown"]
    # Persistence: the markdown file on disk should also contain the references.
    persisted = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "![research_report_" in persisted


def test_factor_plot_paths_match_disk(tmp_path):
    """Every returned plot path must exist on disk and live under outputs_dir."""
    runner = _make_runner(FACTOR_TOOL, _factor_tool_result(n_factors=2))

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    out = Path(result["report_path"]).parent.resolve()
    for p in result["plot_paths"]:
        assert Path(p).is_file()
        assert Path(p).resolve().parent == out


def test_condition_research_emits_plot_paths(tmp_path):
    runner = _make_runner(CONDITION_TOOL, _condition_tool_result())

    result = run_research_pipeline(
        "缩量上涨后下一日怎么样",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_CONDITION
    plot_paths = result["plot_paths"]
    assert isinstance(plot_paths, list)
    # 4 expected: probability_bar, yearly_stability, instrument_stability, return_distribution.
    assert len(plot_paths) >= 3
    for p in plot_paths:
        assert Path(p).is_file()
        assert p.endswith(".png")


def test_condition_report_markdown_contains_image_references(tmp_path):
    runner = _make_runner(CONDITION_TOOL, _condition_tool_result())

    result = run_research_pipeline(
        "缩量上涨后下一日怎么样",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert "![condition_research_report_" in result["report_markdown"]
    assert "(condition_research_report_" in result["report_markdown"]
    assert "/Users/" not in result["report_markdown"]


def test_direct_reply_has_no_plot_paths(tmp_path):
    runner = _make_runner(DIRECT_REPLY_TOOL, {"reply": "hi", "direct_reply": True})

    result = run_research_pipeline(
        "帮我做一个ETF研究",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_DIRECT
    assert result["plot_paths"] == []


def test_direct_reply_research_plan_is_persisted(tmp_path):
    """When the runner returns an `# AutoETF Research Plan` (e.g. an
    `recognized_not_implemented` failure), the pipeline must still drop a
    markdown file under outputs/ so reviewers can inspect it later."""
    plan_reply = (
        "# AutoETF Research Plan\n\n"
        "## 研究无法执行\n\n"
        "- 用户问题：研究BBI指标选ETF\n"
        "- 研究状态：recognized_not_implemented\n\n"
        "### 不可执行因子\n- BBI: recognized_not_implemented\n"
    )
    runner = _make_runner(DIRECT_REPLY_TOOL, {
        "reply": plan_reply,
        "direct_reply": True,
    })

    result = run_research_pipeline(
        "研究BBI指标选ETF",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_DIRECT
    assert result["plot_paths"] == []
    assert result["report_path"] is not None
    assert Path(result["report_path"]).is_file()
    assert Path(result["report_path"]).name.startswith("unmet_research_plan_")
    # File content matches the runner reply verbatim.
    persisted = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "研究BBI指标选ETF" in persisted
    assert "recognized_not_implemented" in persisted
    # report_markdown mirrors the reply for in-process consumers.
    assert result["report_markdown"] == plan_reply


def test_direct_reply_pure_chitchat_is_not_persisted(tmp_path):
    """Chitchat (e.g. '你好') returns no research plan → no markdown file."""
    runner = _make_runner(DIRECT_REPLY_TOOL, {
        "reply": "你好，我是 AutoETF 研究助手。",
        "direct_reply": True,
    })

    result = run_research_pipeline(
        "你好",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_DIRECT
    assert result["report_path"] is None
    assert result["report_markdown"] == ""


def test_error_path_has_no_plot_paths(tmp_path):
    runner = _make_failing_runner("kaboom")

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_ERROR
    assert result["plot_paths"] == []


def test_image_injection_silent_when_anchor_missing(tmp_path):
    """When the report markdown has no matching anchor, pipeline still succeeds silently."""
    runner = _make_runner(FACTOR_TOOL, {
        "report": "# No anchors here\n\njust text.\n",
        "factor_result": {},
        "backtest_result": {"daily_returns": pd.DataFrame({"return": [0.01, 0.02]}, index=pd.date_range("2024-01-01", periods=2))},
    })

    result = run_research_pipeline(
        "研究20日动量",
        config=FakeConfig(),
        use_llm=False,
        outputs_dir=tmp_path,
        chat_runner=runner,
    )

    assert result["task_type"] == TASK_TYPE_FACTOR
    # Markdown is unchanged — no images injected.
    assert "![research_report_" not in result["report_markdown"]


def test_inject_images_helper_directly():
    """Unit test for the small injection helper."""
    from src.pipeline import _inject_images

    md = "# Title\n\n## 5. 单因子分析结果\n\nbody\n"
    out = _inject_images(md, "5. 单因子分析结果", [("cap", "x.png")])
    assert "![cap](x.png)" in out
    # Anchor before body — image is between the heading and body.
    assert out.index("![cap](x.png)") < out.index("body")


def test_inject_images_helper_no_anchor():
    from src.pipeline import _inject_images

    md = "# Title\n\nbody\n"
    out = _inject_images(md, "99. missing", [("cap", "x.png")])
    assert out == md