"""Plotting helpers for AutoETF research reports.

This module centralises every chart the project produces. All functions save
PNG files to disk and never display interactively — the matplotlib ``Agg``
backend is forced at import time so the module is safe to use inside
headless contexts (Notebooks, CI, plain scripts).

Public API surface:

* Per-figure helpers (return ``None``):
    - ``plot_cumulative_return``
    - ``plot_drawdown``
    - ``plot_ic_series``
    - ``plot_quantile_return``
    - ``plot_condition_probability_bar``
    - ``plot_yearly_stability``
    - ``plot_return_distribution``
    - ``plot_instrument_stability``
    - ``plot_factor_correlation_heatmap``
    - ``plot_ic_distribution``

* Panel dispatchers (return ``list[str]`` of written PNG paths):
    - ``plot_factor_analysis_panel``
    - ``plot_backtest_panel``
    - ``plot_condition_panel``
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must run before importing pyplot

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Existing single-series plotters (signatures unchanged)
# ---------------------------------------------------------------------------


def plot_cumulative_return(daily_returns: pd.Series, path: str = "outputs/cumulative_return.png") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    cumret = (1 + daily_returns).cumprod() - 1
    plt.figure(figsize=(10, 5))
    plt.plot(cumret.index, cumret.values)
    plt.title("Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_drawdown(daily_returns: pd.Series, path: str = "outputs/drawdown.png") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    cumulative = (1 + daily_returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    plt.figure(figsize=(10, 5))
    plt.plot(drawdown.index, drawdown.values)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ic_series(ic_series: pd.Series, path: str = "outputs/ic_series.png") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(ic_series.index, ic_series.values)
    plt.title("IC Series")
    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_quantile_return(quantile_return: pd.Series, path: str = "outputs/quantile_return.png") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    quantile_return.plot(kind="bar")
    plt.title("Quantile Return")
    plt.xlabel("Quantile")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Condition-research plotters
# ---------------------------------------------------------------------------


def plot_condition_probability_bar(
    event_up_probability: float,
    baseline_up_probability: float,
    path: str = "outputs/condition_probability_bar.png",
) -> None:
    """Bar chart comparing event-sample vs baseline up-probability."""
    Path(path).parent.mkdir(exist_ok=True)
    labels = ["Event Sample", "Baseline"]
    values = [event_up_probability, baseline_up_probability]
    colors = ["#3a7bd5", "#b0b7bf"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, max(0.01, max(values) * 1.25))
    for bar, v in zip(bars, values):
        if pd.isna(v):
            continue
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.title("Up-Probability: Event vs Baseline")
    plt.ylabel("Up Probability")
    plt.grid(True, axis="y")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_yearly_stability(
    yearly_stats: pd.DataFrame,
    path: str = "outputs/yearly_stability.png",
) -> None:
    """Bar chart of yearly probability lift. No-op when yearly_stats is empty."""
    if yearly_stats is None or yearly_stats.empty:
        return
    if "group" not in yearly_stats.columns or "probability_lift" not in yearly_stats.columns:
        return
    Path(path).parent.mkdir(exist_ok=True)
    df = yearly_stats.copy()
    df["group"] = df["group"].astype(str)
    plt.figure(figsize=(10, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["probability_lift"].fillna(0)]
    plt.bar(df["group"], df["probability_lift"].fillna(0), color=colors)
    plt.axhline(0, color="#666", linewidth=0.8)
    plt.title("Yearly Probability Lift")
    plt.xlabel("Year")
    plt.ylabel("Probability Lift (pp)")
    plt.grid(True, axis="y")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_return_distribution(
    event_returns: pd.Series,
    baseline_returns: pd.Series,
    path: str = "outputs/return_distribution.png",
) -> None:
    """Histogram (with KDE) of event vs baseline target returns."""
    Path(path).parent.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    bins = 40

    def _series_or_empty(s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series(dtype=float)
        return s.dropna()

    ev = _series_or_empty(event_returns)
    base = _series_or_empty(baseline_returns)

    if not ev.empty:
        plt.hist(ev, bins=bins, alpha=0.55, label="Event Sample", color="#3a7bd5", density=True)
    if not base.empty:
        plt.hist(base, bins=bins, alpha=0.45, label="Baseline", color="#b0b7bf", density=True)
    plt.title("Target Return Distribution: Event vs Baseline")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_instrument_stability(
    instrument_stats: pd.DataFrame,
    path: str = "outputs/instrument_stability.png",
    top_n: int = 15,
) -> None:
    """Top-N instruments by event_count, horizontal bar of probability lift."""
    if instrument_stats is None or instrument_stats.empty:
        return
    required = {"group", "event_count", "probability_lift"}
    if not required.issubset(instrument_stats.columns):
        return
    Path(path).parent.mkdir(exist_ok=True)
    df = instrument_stats.sort_values("event_count", ascending=False).head(top_n).copy()
    df["group"] = df["group"].astype(str)
    plt.figure(figsize=(10, max(4, 0.35 * len(df) + 1)))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["probability_lift"].fillna(0)]
    plt.barh(df["group"], df["probability_lift"].fillna(0), color=colors)
    plt.axvline(0, color="#666", linewidth=0.8)
    plt.title(f"Per-Instrument Probability Lift (Top {min(top_n, len(df))} by Event Count)")
    plt.xlabel("Probability Lift (pp)")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Factor-research plotters (panel + extra)
# ---------------------------------------------------------------------------


def _safe_filename_token(token: str) -> str:
    """Make a string safe to embed in a PNG filename."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(token))


def plot_factor_analysis_panel(
    factor_result: dict,
    output_dir: str | Path = "outputs",
    *,
    primary_target: str | None = None,
    basename: str = "factor_analysis",
) -> list[str]:
    """Render IC series + quantile return PNGs for every (factor, target) pair.

    Parameters
    ----------
    factor_result:
        Output of :func:`src.factor_analysis.run_factor_analysis`. Structure:
        ``{factor_name: {target_name: {"ic_series": ..., "quantile_return": ...}}}``.
    output_dir:
        Directory where PNGs are written. Created if missing.
    primary_target:
        If provided, only pairs with this target are rendered. Otherwise all
        targets are rendered.
    basename:
        Prefix used in the output filename: ``{basename}_ic_{factor}_{target}.png``
        and ``{basename}_quantile_{factor}_{target}.png``.

    Returns
    -------
    list[str]
        Absolute paths of every PNG successfully written.
    """
    if not isinstance(factor_result, dict) or not factor_result:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    safe_base = _safe_filename_token(basename)
    for factor_name, target_dict in factor_result.items():
        if not isinstance(target_dict, dict):
            continue
        targets_to_render: Iterable[tuple[str, dict]] = target_dict.items()
        if primary_target is not None:
            if primary_target not in target_dict:
                continue
            targets_to_render = [(primary_target, target_dict[primary_target])]
        for target_name, payload in targets_to_render:
            if not isinstance(payload, dict):
                continue
            ic_series = payload.get("ic_series")
            quantile_return = payload.get("quantile_return")
            f_token = _safe_filename_token(factor_name)
            t_token = _safe_filename_token(target_name)
            if isinstance(ic_series, pd.Series) and not ic_series.empty:
                p = out / f"{safe_base}_ic_{f_token}_{t_token}.png"
                plot_ic_series(ic_series, str(p))
                written.append(str(p.resolve()))
            if isinstance(quantile_return, pd.Series) and not quantile_return.empty:
                p = out / f"{safe_base}_quantile_{f_token}_{t_token}.png"
                plot_quantile_return(quantile_return, str(p))
                written.append(str(p.resolve()))
    return written


def plot_backtest_panel(
    backtest_result: dict,
    output_dir: str | Path = "outputs",
    *,
    basename: str = "backtest",
) -> list[str]:
    """Render cumulative return + drawdown PNGs from a backtest result dict.

    Returns
    -------
    list[str]
        Absolute paths of PNG files successfully written.
    """
    if not isinstance(backtest_result, dict):
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    daily_returns = backtest_result.get("daily_returns")
    series: pd.Series | None = None
    if isinstance(daily_returns, pd.DataFrame) and "return" in daily_returns.columns:
        series = daily_returns["return"]
    if series is None:
        # Some paths expose cumulative_returns directly.
        cum = backtest_result.get("cumulative_returns")
        if isinstance(cum, pd.Series) and not cum.empty:
            series = cum
    safe_base = _safe_filename_token(basename)
    if isinstance(series, pd.Series) and not series.empty:
        p_cum = out / f"{safe_base}_cumulative_return.png"
        plot_cumulative_return(series, str(p_cum))
        written.append(str(p_cum.resolve()))
        p_dd = out / f"{safe_base}_drawdown.png"
        plot_drawdown(series, str(p_dd))
        written.append(str(p_dd.resolve()))
    return written


def plot_factor_correlation_heatmap(
    factor_result: dict,
    output_dir: str | Path = "outputs",
    *,
    primary_target: str | None = None,
    basename: str = "factor_correlation",
) -> list[str]:
    """Render a heatmap of pairwise IC correlations across factors."""
    if not isinstance(factor_result, dict) or not factor_result:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    series_map: dict[str, pd.Series] = {}
    for factor_name, target_dict in factor_result.items():
        if not isinstance(target_dict, dict):
            continue
        if primary_target is not None and primary_target in target_dict:
            payload = target_dict[primary_target]
        elif target_dict:
            # pick the first target with a usable ic_series
            payload = next(
                (v for v in target_dict.values() if isinstance(v, dict) and isinstance(v.get("ic_series"), pd.Series) and not v["ic_series"].empty),
                None,
            )
        else:
            payload = None
        if payload is None:
            continue
        ic_series = payload.get("ic_series")
        if isinstance(ic_series, pd.Series) and not ic_series.empty:
            series_map[factor_name] = ic_series

    if len(series_map) < 2:
        return []
    df = pd.DataFrame(series_map).dropna()
    if df.empty or df.shape[1] < 2:
        return []
    corr = df.corr()
    safe_base = _safe_filename_token(basename)
    path = out / f"{safe_base}.png"
    plt.figure(figsize=(max(6, 0.6 * len(corr) + 2), max(5, 0.6 * len(corr) + 1)))
    im = plt.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="Pearson corr")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Factor IC Correlation Matrix")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.values[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(value) > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return [str(path.resolve())]


def plot_ic_distribution(
    factor_result: dict,
    output_dir: str | Path = "outputs",
    *,
    primary_target: str | None = None,
    basename: str = "ic_distribution",
) -> list[str]:
    """One histogram PNG per factor showing the distribution of daily IC."""
    if not isinstance(factor_result, dict) or not factor_result:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    safe_base = _safe_filename_token(basename)
    for factor_name, target_dict in factor_result.items():
        if not isinstance(target_dict, dict):
            continue
        payload: dict | None = None
        if primary_target is not None and primary_target in target_dict:
            payload = target_dict[primary_target]
        else:
            for v in target_dict.values():
                if isinstance(v, dict) and isinstance(v.get("ic_series"), pd.Series) and not v["ic_series"].empty:
                    payload = v
                    break
        if payload is None:
            continue
        ic_series = payload.get("ic_series")
        if not isinstance(ic_series, pd.Series) or ic_series.empty:
            continue
        f_token = _safe_filename_token(factor_name)
        path = out / f"{safe_base}_{f_token}.png"
        plt.figure(figsize=(8, 5))
        plt.hist(ic_series.dropna().values, bins=30, color="#3a7bd5", alpha=0.85, density=True)
        mean_val = ic_series.mean()
        if not np.isnan(mean_val):
            plt.axvline(mean_val, color="#e74c3c", linestyle="--", label=f"mean={mean_val:.3f}")
        plt.title(f"IC Distribution - {factor_name}")
        plt.xlabel("IC")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, axis="y")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(str(path.resolve()))
    return written


# ---------------------------------------------------------------------------
# Condition-research panel dispatcher
# ---------------------------------------------------------------------------


def plot_condition_panel(
    result: dict,
    output_dir: str | Path = "outputs",
    *,
    basename: str = "condition",
    event_returns: pd.Series | None = None,
    baseline_returns: pd.Series | None = None,
) -> list[str]:
    """Render every condition-research figure in one call.

    The dispatcher is resilient: each sub-plotter is skipped silently when
    its data is missing or malformed, so the returned list always contains
    only the charts that actually wrote a PNG.

    Returns
    -------
    list[str]
        Absolute paths of every PNG written.
    """
    if not isinstance(result, dict):
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_base = _safe_filename_token(basename)
    written: list[str] = []

    event_up = result.get("event_up_probability")
    baseline_up = result.get("baseline_up_probability")
    if event_up is not None and baseline_up is not None:
        path = out / f"{safe_base}_probability_bar.png"
        plot_condition_probability_bar(event_up, baseline_up, str(path))
        written.append(str(path.resolve()))

    yearly = result.get("yearly_stats")
    if isinstance(yearly, pd.DataFrame) and not yearly.empty:
        path = out / f"{safe_base}_yearly_stability.png"
        plot_yearly_stability(yearly, str(path))
        if path.exists():
            written.append(str(path.resolve()))

    instrument = result.get("instrument_stats")
    if isinstance(instrument, pd.DataFrame) and not instrument.empty:
        path = out / f"{safe_base}_instrument_stability.png"
        plot_instrument_stability(instrument, str(path))
        if path.exists():
            written.append(str(path.resolve()))

    if event_returns is not None or baseline_returns is not None:
        path = out / f"{safe_base}_return_distribution.png"
        ev = event_returns if event_returns is not None else pd.Series(dtype=float)
        base = baseline_returns if baseline_returns is not None else pd.Series(dtype=float)
        plot_return_distribution(ev, base, str(path))
        if path.exists():
            written.append(str(path.resolve()))

    return written