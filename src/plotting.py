"""Plotting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

