"""Unit tests for ``src/plotting.py``.

Goals:

- matplotlib ``Agg`` backend is forced when the module is imported.
- Every public helper writes a PNG when given valid data.
- Every helper is robust to empty / malformed inputs (returns ``None``
  for single plotters, returns ``[]`` for panel dispatchers).
- Panel dispatchers skip empty panels rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

import numpy as np
import pandas as pd
import pytest

from src import plotting


# ---------------------------------------------------------------------------
# Backend sanity
# ---------------------------------------------------------------------------


def test_matplotlib_agg_backend_is_forced():
    assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# Existing single-series plotters
# ---------------------------------------------------------------------------


def test_plot_cumulative_return(tmp_path):
    series = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015], index=pd.date_range("2024-01-01", periods=5))
    out = tmp_path / "cum.png"
    plotting.plot_cumulative_return(series, str(out))
    assert out.is_file()
    assert out.stat().st_size > 100


def test_plot_drawdown(tmp_path):
    series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005], index=pd.date_range("2024-01-01", periods=5))
    out = tmp_path / "dd.png"
    plotting.plot_drawdown(series, str(out))
    assert out.is_file()


def test_plot_ic_series(tmp_path):
    series = pd.Series(np.linspace(-0.05, 0.05, 20), index=pd.date_range("2024-01-01", periods=20))
    out = tmp_path / "ic.png"
    plotting.plot_ic_series(series, str(out))
    assert out.is_file()


def test_plot_quantile_return(tmp_path):
    series = pd.Series([0.01, 0.012, 0.014, 0.018, 0.022], index=[1, 2, 3, 4, 5])
    out = tmp_path / "q.png"
    plotting.plot_quantile_return(series, str(out))
    assert out.is_file()


# ---------------------------------------------------------------------------
# New condition-research plotters
# ---------------------------------------------------------------------------


def test_plot_condition_probability_bar(tmp_path):
    out = tmp_path / "bar.png"
    plotting.plot_condition_probability_bar(0.567, 0.513, str(out))
    assert out.is_file()


def test_plot_condition_probability_bar_nan_safe(tmp_path):
    """NaN probabilities should not crash the plotter."""
    out = tmp_path / "bar.png"
    plotting.plot_condition_probability_bar(float("nan"), float("nan"), str(out))
    assert out.is_file()


def test_plot_yearly_stability_writes_png(tmp_path):
    df = pd.DataFrame({
        "group": [2021, 2022, 2023, 2024],
        "event_count": [50, 60, 55, 70],
        "event_up_probability": [0.55, 0.58, 0.52, 0.60],
        "baseline_up_probability": [0.50, 0.50, 0.50, 0.50],
        "probability_lift": [0.05, 0.08, 0.02, 0.10],
        "event_mean_return": [0.01, 0.02, 0.005, 0.015],
        "baseline_mean_return": [0.005, 0.005, 0.005, 0.005],
        "mean_return_lift": [0.005, 0.015, 0.0, 0.010],
    })
    out = tmp_path / "yearly.png"
    plotting.plot_yearly_stability(df, str(out))
    assert out.is_file()


def test_plot_yearly_stability_skips_empty(tmp_path):
    out = tmp_path / "yearly.png"
    plotting.plot_yearly_stability(pd.DataFrame(), str(out))
    assert not out.exists()


def test_plot_yearly_stability_skips_missing_columns(tmp_path):
    df = pd.DataFrame({"foo": [1, 2]})
    out = tmp_path / "yearly.png"
    plotting.plot_yearly_stability(df, str(out))
    assert not out.exists()


def test_plot_return_distribution(tmp_path):
    np.random.seed(0)
    ev = pd.Series(np.random.normal(0.02, 0.05, 200))
    base = pd.Series(np.random.normal(0.005, 0.05, 500))
    out = tmp_path / "dist.png"
    plotting.plot_return_distribution(ev, base, str(out))
    assert out.is_file()


def test_plot_return_distribution_with_empty_event(tmp_path):
    """Empty event series still produces a baseline-only histogram."""
    base = pd.Series([0.01, 0.02, 0.005])
    out = tmp_path / "dist.png"
    plotting.plot_return_distribution(pd.Series(dtype=float), base, str(out))
    assert out.is_file()


def test_plot_instrument_stability_writes_png(tmp_path):
    df = pd.DataFrame({
        "group": ["510300", "510500", "159915", "512100"],
        "event_count": [40, 35, 30, 20],
        "event_up_probability": [0.58, 0.55, 0.60, 0.62],
        "baseline_up_probability": [0.50, 0.50, 0.50, 0.50],
        "probability_lift": [0.08, 0.05, 0.10, 0.12],
        "event_mean_return": [0.01] * 4,
        "baseline_mean_return": [0.005] * 4,
        "mean_return_lift": [0.005] * 4,
    })
    out = tmp_path / "inst.png"
    plotting.plot_instrument_stability(df, str(out))
    assert out.is_file()


def test_plot_instrument_stability_skips_empty(tmp_path):
    out = tmp_path / "inst.png"
    plotting.plot_instrument_stability(pd.DataFrame(), str(out))
    assert not out.exists()


# ---------------------------------------------------------------------------
# Panel dispatchers
# ---------------------------------------------------------------------------


def _fake_factor_result(n_factors: int = 2) -> dict:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=20)
    out = {}
    for i in range(n_factors):
        name = f"f{i}_x"
        out[name] = {
            "future_return_5d": {
                "ic_series": pd.Series(np.random.normal(0.02, 0.1, 20), index=dates),
                "quantile_return": pd.Series(
                    [0.005, 0.008, 0.011, 0.014, 0.018], index=[1, 2, 3, 4, 5]
                ),
            }
        }
    return out


def test_plot_factor_analysis_panel_writes_two_per_factor(tmp_path):
    fr = _fake_factor_result(n_factors=2)
    written = plotting.plot_factor_analysis_panel(fr, tmp_path, basename="fa")
    assert len(written) == 4  # 2 factors * (IC + quantile)
    assert all(Path(p).is_file() for p in written)
    # 2 IC charts and 2 quantile charts across the 2 factors.
    names = [Path(p).name for p in written]
    assert sum("fa_ic_f" in n for n in names) == 2
    assert sum("fa_quantile_f" in n for n in names) == 2


def test_plot_factor_analysis_panel_respects_primary_target(tmp_path):
    fr = {
        "factor_a": {
            "future_return_5d": {
                "ic_series": pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2024-01-01", periods=3)),
                "quantile_return": pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3]),
            },
            "future_return_10d": {
                "ic_series": pd.Series([0.05, 0.06, 0.07], index=pd.date_range("2024-01-01", periods=3)),
                "quantile_return": pd.Series([0.02, 0.03, 0.04], index=[1, 2, 3]),
            },
        }
    }
    written = plotting.plot_factor_analysis_panel(fr, tmp_path, primary_target="future_return_5d", basename="p")
    assert len(written) == 2  # only the 5d target is rendered
    assert any("future_return_5d" in p for p in written)
    assert not any("future_return_10d" in p for p in written)


def test_plot_factor_analysis_panel_handles_empty(tmp_path):
    assert plotting.plot_factor_analysis_panel({}, tmp_path) == []
    assert plotting.plot_factor_analysis_panel(None, tmp_path) == []


def test_plot_backtest_panel_writes_two_files(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30)
    daily_returns = pd.DataFrame({
        "return": np.random.normal(0.001, 0.01, 30),
        "gross_return": np.random.normal(0.0015, 0.01, 30),
        "turnover": np.random.uniform(0.05, 0.20, 30),
        "cost": np.random.uniform(0.0001, 0.001, 30),
        "holding_count": np.random.randint(3, 8, 30),
    }, index=dates)
    backtest_result = {"daily_returns": daily_returns}
    written = plotting.plot_backtest_panel(backtest_result, tmp_path, basename="bt")
    assert len(written) == 2
    paths = [Path(p) for p in written]
    assert any("cumulative_return" in p.name for p in paths)
    assert any("drawdown" in p.name for p in paths)
    assert all(p.is_file() for p in paths)


def test_plot_backtest_panel_handles_empty(tmp_path):
    assert plotting.plot_backtest_panel({}, tmp_path) == []
    assert plotting.plot_backtest_panel(None, tmp_path) == []


def test_plot_factor_correlation_heatmap(tmp_path):
    fr = _fake_factor_result(n_factors=3)
    written = plotting.plot_factor_correlation_heatmap(fr, tmp_path)
    assert len(written) == 1
    assert Path(written[0]).is_file()


def test_plot_factor_correlation_heatmap_skips_single_factor(tmp_path):
    fr = _fake_factor_result(n_factors=1)
    written = plotting.plot_factor_correlation_heatmap(fr, tmp_path)
    assert written == []


def test_plot_ic_distribution_writes_per_factor(tmp_path):
    fr = _fake_factor_result(n_factors=2)
    written = plotting.plot_ic_distribution(fr, tmp_path)
    assert len(written) == 2
    assert all(Path(p).is_file() for p in written)


def test_plot_ic_distribution_handles_empty(tmp_path):
    assert plotting.plot_ic_distribution({}, tmp_path) == []


def test_plot_condition_panel_writes_all(tmp_path):
    np.random.seed(1)
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
    result = {
        "event_up_probability": 0.567,
        "baseline_up_probability": 0.513,
        "yearly_stats": yearly,
        "instrument_stats": instrument,
    }
    ev = pd.Series(np.random.normal(0.02, 0.05, 100))
    base = pd.Series(np.random.normal(0.005, 0.05, 300))
    written = plotting.plot_condition_panel(
        result, tmp_path, basename="cond",
        event_returns=ev, baseline_returns=base,
    )
    assert len(written) == 4
    names = sorted(Path(p).name for p in written)
    assert "cond_probability_bar.png" in names
    assert "cond_yearly_stability.png" in names
    assert "cond_instrument_stability.png" in names
    assert "cond_return_distribution.png" in names


def test_plot_condition_panel_handles_empty_result(tmp_path):
    assert plotting.plot_condition_panel({}, tmp_path) == []
    assert plotting.plot_condition_panel(None, tmp_path) == []


def test_plot_condition_panel_partial_data(tmp_path):
    """When only some fields are present, the panel renders only those charts."""
    result = {"event_up_probability": 0.55, "baseline_up_probability": 0.50}
    written = plotting.plot_condition_panel(result, tmp_path, basename="p")
    assert len(written) == 1
    assert "probability_bar" in Path(written[0]).name


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------


def test_safe_filename_token_strips_special_chars():
    assert plotting._safe_filename_token("factor/with:special?chars") == "factor_with_special_chars"
    assert plotting._safe_filename_token("momentum_20d") == "momentum_20d"