"""Backend-agnostic ETF data access helpers."""

from __future__ import annotations

from typing import Any

from src.data_loader_bigquant import load_condition_research_data_bigquant, load_etf_data_bigquant
from src.data_loader_local import load_condition_research_data_local, load_etf_data_local


def load_etf_data(
    *,
    data_backend: str = "bigquant",
    **kwargs: Any,
):
    """Load ETF data from the configured backend."""
    backend = (data_backend or "bigquant").lower().strip()
    if backend == "bigquant":
        return load_etf_data_bigquant(**kwargs)
    if backend == "local":
        return load_etf_data_local(**kwargs)
    raise ValueError(f"Unsupported data_backend: {data_backend}")


def load_condition_research_data(
    *,
    data_backend: str = "bigquant",
    **kwargs: Any,
):
    """Load condition-study data from the configured backend."""
    backend = (data_backend or "bigquant").lower().strip()
    if backend == "bigquant":
        return load_condition_research_data_bigquant(**kwargs)
    if backend == "local":
        return load_condition_research_data_local(**kwargs)
    raise ValueError(f"Unsupported data_backend: {data_backend}")

