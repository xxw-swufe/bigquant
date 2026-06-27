"""ETF factor candidate generation."""

from __future__ import annotations

from collections.abc import Iterable

from src.etf_factor_library import get_factor_library
from src.research_plan import (
    AvailabilityStatus,
    FactorIntent,
    FactorSelectionResult,
    ResearchTarget,
    SelectionStatus,
    UnavailableFactor,
    coerce_target,
)


def generate_factor_candidates(
    intent: FactorIntent | dict | None,
    *,
    data_backend: str = "local",
    available_fields: Iterable[str] | None = None,
    available_context: Iterable[str] | None = None,
    max_candidates: int = 5,
    allow_partial_execution: bool = False,
) -> FactorSelectionResult:
    """Generate executable factor candidates from a structured intent."""
    normalized_intent = _coerce_intent(intent)
    available_fields_set = set(available_fields or [])
    factor_index = {factor["name"]: factor for factor in get_factor_library()}

    if not normalized_intent.raw_query.strip():
        return FactorSelectionResult(
            status=SelectionStatus.EMPTY_INTENT,
            target=normalized_intent.target,
            selection_source="empty_intent",
        )

    if normalized_intent.unresolved_terms:
        return FactorSelectionResult(
            status=SelectionStatus.UNRECOGNIZED_FACTOR,
            unresolved_terms=list(normalized_intent.unresolved_terms),
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    if normalized_intent.ambiguous_terms and not normalized_intent.factor_names and not normalized_intent.categories:
        return FactorSelectionResult(
            status=SelectionStatus.AMBIGUOUS_FACTOR,
            ambiguous_terms=list(normalized_intent.ambiguous_terms),
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    selected: list[dict] = []
    selection_reasons: dict[str, str] = {}
    seen_names: set[str] = set()

    for name in normalized_intent.factor_names:
        factor = factor_index.get(name)
        if factor is None:
            continue
        selected.append(factor)
        seen_names.add(name)
        selection_reasons[name] = "用户明确指定或知识库直接匹配"

    for category in normalized_intent.categories:
        expanded = select_category_representatives(
            category=category,
            factors=list(factor_index.values()),
            existing_names=seen_names,
            data_backend=data_backend,
            max_count=min(3, max(0, max_candidates - len(selected))) if len(selected) < max_candidates else 0,
        )
        for factor in expanded:
            name = factor["name"]
            if name in seen_names:
                continue
            selected.append(factor)
            seen_names.add(name)
            selection_reasons[name] = f"用户指定了 {category} 类别"

    if not selected and normalized_intent.recognized_not_implemented_terms:
        return FactorSelectionResult(
            status=SelectionStatus.RECOGNIZED_NOT_IMPLEMENTED,
            selected_factors=[],
            unresolved_terms=[],
            unavailable_factors=[
                UnavailableFactor(
                    name=term,
                    reason="knowledge_known_but_not_implemented",
                    status=AvailabilityStatus.NOT_IMPLEMENTED,
                )
                for term in normalized_intent.recognized_not_implemented_terms
            ],
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    if normalized_intent.recognized_not_implemented_terms and not allow_partial_execution:
        return FactorSelectionResult(
            status=SelectionStatus.RECOGNIZED_NOT_IMPLEMENTED,
            selected_factors=selected,
            selection_reasons=selection_reasons,
            unresolved_terms=list(normalized_intent.unresolved_terms),
            unavailable_factors=[
                UnavailableFactor(
                    name=term,
                    reason="knowledge_known_but_not_implemented",
                    status=AvailabilityStatus.NOT_IMPLEMENTED,
                )
                for term in normalized_intent.recognized_not_implemented_terms
            ],
            ambiguous_terms=list(normalized_intent.ambiguous_terms),
            can_execute=False,
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    if not selected:
        return FactorSelectionResult(
            status=SelectionStatus.EMPTY_INTENT,
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    unavailable = check_factor_availability(
        selected,
        data_backend=data_backend,
        available_fields=available_fields_set,
        available_context=set(available_context or []),
    )

    if unavailable and not allow_partial_execution:
        status = _classify_failure(unavailable)
        return FactorSelectionResult(
            status=status,
            selected_factors=selected,
            selection_reasons=selection_reasons,
            unresolved_terms=list(normalized_intent.unresolved_terms),
            unavailable_factors=unavailable,
            ambiguous_terms=list(normalized_intent.ambiguous_terms),
            can_execute=False,
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    executable_names = {item.name for item in unavailable}
    executable = [factor for factor in selected if factor["name"] not in executable_names]
    can_execute = bool(executable)
    if not can_execute:
        return FactorSelectionResult(
            status=SelectionStatus.DATA_UNAVAILABLE,
            selected_factors=selected,
            selection_reasons=selection_reasons,
            unresolved_terms=list(normalized_intent.unresolved_terms),
            unavailable_factors=unavailable,
            ambiguous_terms=list(normalized_intent.ambiguous_terms),
            can_execute=False,
            target=normalized_intent.target,
            selection_source=normalized_intent.selection_source,
        )

    return FactorSelectionResult(
        status=SelectionStatus.READY,
        selected_factors=executable,
        selection_reasons=selection_reasons,
        unresolved_terms=list(normalized_intent.unresolved_terms),
        unavailable_factors=unavailable,
        ambiguous_terms=list(normalized_intent.ambiguous_terms),
        can_execute=True,
        target=normalized_intent.target,
        selection_source=normalized_intent.selection_source,
    )


def select_category_representatives(
    *,
    category: str,
    factors: list[dict],
    existing_names: set[str],
    data_backend: str,
    max_count: int = 3,
) -> list[dict]:
    """Select representative factors for an expanded category query."""
    ranked = [
        factor
        for factor in factors
        if factor.get("category") == category and factor.get("name") not in existing_names
    ]
    ranked.sort(
        key=lambda item: (
            1 if item.get("implementation_status") == "implemented" else 0,
            1 if data_backend in item.get("supported_backends", []) else 0,
            int(item.get("selection_priority", 0)),
        ),
        reverse=True,
    )

    selected: list[dict] = []
    used_groups: set[str] = set()
    for factor in ranked:
        if len(selected) >= max_count:
            break
        group = factor.get("selection_group") or factor.get("category") or factor["name"]
        if group in used_groups and factor.get("name") not in existing_names:
            continue
        selected.append(factor)
        used_groups.add(group)
    return selected


BENCHMARK_DERIVED_FIELDS = {
    "benchmark_close",
    "benchmark_date",
    "benchmark_return_1d",
    "benchmark_return_20d",
    "benchmark_return_60d",
}


def check_factor_availability(
    factors: list[dict],
    *,
    data_backend: str,
    available_fields: set[str],
    available_context: set[str] | None = None,
) -> list[UnavailableFactor]:
    """Check whether each selected factor can be executed."""
    unavailable: list[UnavailableFactor] = []
    context_set = set(available_context or [])
    for factor in factors:
        name = factor["name"]
        if factor.get("implementation_status") != "implemented":
            unavailable.append(
                UnavailableFactor(
                    name=name,
                    reason="recognized_not_implemented",
                    status=AvailabilityStatus.NOT_IMPLEMENTED,
                )
            )
            continue

        supported_backends = set(factor.get("supported_backends", []))
        if supported_backends and data_backend not in supported_backends:
            unavailable.append(
                UnavailableFactor(
                    name=name,
                    reason="unsupported_backend",
                    status=AvailabilityStatus.UNSUPPORTED_BACKEND,
                )
            )
            continue

        required_context = list(factor.get("required_context", []))
        missing_context = [item for item in required_context if item not in context_set]
        if missing_context:
            unavailable.append(
                UnavailableFactor(
                    name=name,
                    reason="missing_context",
                    missing_context=missing_context,
                    status=AvailabilityStatus.MISSING_CONTEXT,
                )
            )
            continue

        required_fields = set(factor.get("required_fields", []))
        if "benchmark_series" in required_context:
            required_fields -= BENCHMARK_DERIVED_FIELDS
        missing_fields = sorted(required_fields - available_fields)
        if missing_fields:
            unavailable.append(
                UnavailableFactor(
                    name=name,
                    reason="missing_fields",
                    missing_fields=missing_fields,
                    missing_context=missing_context,
                    status=AvailabilityStatus.MISSING_FIELDS,
                )
            )
            continue

        if any(field.startswith("future_") for field in required_fields):
            unavailable.append(
                UnavailableFactor(
                    name=name,
                    reason="future_data_forbidden",
                    status=AvailabilityStatus.FUTURE_DATA_FORBIDDEN,
                )
            )
    return unavailable


def validate_factor_candidates(factors: list[dict]) -> None:
    """Validate generated factor metadata."""
    validate_no_future_leakage([factor["name"] for factor in factors])
    bad_formulas = [
        factor["name"]
        for factor in factors
        if "m_lead" in factor.get("formula", "").lower()
        or "future_return" in factor.get("formula", "").lower()
    ]
    if bad_formulas:
        raise ValueError(f"Factor formulas contain future data: {bad_formulas}")


def validate_no_future_leakage(feature_cols: list[str]) -> None:
    """Raise if feature columns contain future labels."""
    bad_cols = [col for col in feature_cols if col.startswith("future_")]
    if bad_cols:
        raise ValueError(f"Feature columns contain future labels: {bad_cols}")


def _coerce_intent(intent: FactorIntent | dict | None) -> FactorIntent:
    if isinstance(intent, FactorIntent):
        return intent
    if isinstance(intent, dict):
        return FactorIntent(
            raw_query=str(intent.get("raw_query") or intent.get("user_idea") or ""),
            research_mode=str(intent.get("research_mode") or intent.get("research_type") or "factor"),
            explicit_terms=list(intent.get("explicit_terms", [])),
            factor_names=list(intent.get("factor_names", [])),
            categories=list(intent.get("categories", [])),
            target=coerce_target(intent.get("target")),
            unresolved_terms=list(intent.get("unresolved_terms", [])),
            ambiguous_terms=list(intent.get("ambiguous_terms", [])),
            recognized_not_implemented_terms=list(intent.get("recognized_not_implemented_terms", [])),
            confidence=float(intent.get("confidence", 0.0) or 0.0),
            selection_source=str(intent.get("selection_source") or "semantic_match"),
            route_intent=list(intent.get("route_intent", [])),
            route_to=list(intent.get("route_to", [])),
            matched_expressions=list(intent.get("matched_expressions", [])),
        )
    return FactorIntent(raw_query="", research_mode="factor")


def _classify_failure(unavailable: list[UnavailableFactor]) -> SelectionStatus:
    reasons = {item.status for item in unavailable}
    if AvailabilityStatus.NOT_IMPLEMENTED in reasons:
        return SelectionStatus.RECOGNIZED_NOT_IMPLEMENTED
    if AvailabilityStatus.UNSUPPORTED_BACKEND in reasons:
        return SelectionStatus.UNSUPPORTED_BACKEND
    if AvailabilityStatus.MISSING_CONTEXT in reasons:
        return SelectionStatus.MISSING_CONTEXT
    if AvailabilityStatus.MISSING_FIELDS in reasons:
        return SelectionStatus.DATA_UNAVAILABLE
    if AvailabilityStatus.FUTURE_DATA_FORBIDDEN in reasons:
        return SelectionStatus.DATA_UNAVAILABLE
    return SelectionStatus.DATA_UNAVAILABLE
