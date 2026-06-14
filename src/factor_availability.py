"""Factor availability and deduplication helpers for the agent layer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from src.etf_factor_library import get_factor_library, search_factors


def build_factor_index() -> dict[str, dict]:
    """Build a name -> metadata index for the reference factor library."""
    return {factor["name"]: factor for factor in get_factor_library()}


def get_factor_metadata(factor_name: str) -> Optional[dict]:
    """Return a factor metadata record by exact name."""
    return build_factor_index().get(factor_name)


def search_factor_candidates(query: str, limit: int = 10) -> list[dict]:
    """Search factor candidates for a user query."""
    return search_factors(query, limit=limit)


def get_available_factors(
    required_columns: Optional[Iterable[str]] = None,
    *,
    include_non_ready: bool = True,
) -> list[dict]:
    """Return factors that are feasible under a given column set."""
    required = set(required_columns or [])
    factors = get_factor_library()
    available = []
    for factor in factors:
        if not include_non_ready and not factor.get("bigquant_ready"):
            continue
        required_factor_cols = set(factor.get("required_columns", []))
        if required and not required_factor_cols.issubset(required):
            continue
        available.append(factor)
    return available


def filter_factors_by_dependencies(
    factors: list[dict],
    available_columns: Iterable[str],
) -> list[dict]:
    """Filter factor candidates by available data columns."""
    available = set(available_columns)
    result = []
    for factor in factors:
        required = set(factor.get("required_columns", []))
        if required.issubset(available):
            result.append(factor)
    return result


def select_factor_plan(
    factors: list[dict],
    *,
    max_factors: int = 8,
    prefer_ready: bool = True,
) -> list[dict]:
    """Select a compact factor plan with redundancy control."""
    if not factors:
        return []

    selected: list[dict] = []
    selected_names: set[str] = set()
    correlated_seen: set[str] = set()

    ranked = sorted(
        factors,
        key=lambda item: (
            1 if item.get("mvp_default") else 0,
            1 if item.get("bigquant_ready") is True else 0,
            1 if item.get("enter_final_score") else 0,
        ),
        reverse=True,
    )

    if prefer_ready:
        ranked = sorted(
            ranked,
            key=lambda item: (
                1 if item.get("bigquant_ready") is True else 0,
                1 if item.get("mvp_default") else 0,
                1 if item.get("enter_final_score") else 0,
            ),
            reverse=True,
        )

    for factor in ranked:
        name = factor["name"]
        if name in selected_names:
            continue
        if prefer_ready and factor.get("bigquant_ready") is not True:
            continue
        correlated = set(factor.get("correlated_with", [])) | set(factor.get("is_redundant_with", []))
        if selected and correlated_seen.intersection(correlated):
            continue
        selected.append(factor)
        selected_names.add(name)
        correlated_seen.update(correlated)
        correlated_seen.add(name)
        if len(selected) >= max_factors:
            break
    return selected


def build_factor_tool_payload(
    query: str,
    *,
    required_columns: Optional[Iterable[str]] = None,
    max_factors: int = 8,
) -> dict:
    """Build a compact agent-facing payload for factor research."""
    candidates = search_factor_candidates(query, limit=20)
    if required_columns is not None:
        candidates = filter_factors_by_dependencies(candidates, required_columns)
    selected = select_factor_plan(candidates, max_factors=max_factors)
    return {
        "query": query,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_factors": selected,
    }
