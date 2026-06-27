"""Structured research plan objects for AutoETF factor routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SelectionStatus(str, Enum):
    READY = "ready"
    EMPTY_INTENT = "empty_intent"
    UNRECOGNIZED_FACTOR = "unrecognized_factor"
    AMBIGUOUS_FACTOR = "ambiguous_factor"
    RECOGNIZED_NOT_IMPLEMENTED = "recognized_not_implemented"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    DATA_UNAVAILABLE = "data_unavailable"
    MISSING_CONTEXT = "missing_context"
    INVALID_TARGET = "invalid_target"


class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    MISSING_FIELDS = "missing_fields"
    MISSING_CONTEXT = "missing_context"
    FUTURE_DATA_FORBIDDEN = "future_data_forbidden"


class MutationType(str, Enum):
    ADD_FACTORS = "add_factors"
    REMOVE_FACTORS = "remove_factors"
    REPLACE_FACTORS = "replace_factors"
    UPDATE_TARGET = "update_target"
    RESET_PLAN = "reset_plan"
    NO_OP = "no_op"


@dataclass
class ResearchTarget:
    metric: str = "future_return"
    horizon: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ResearchPlan:
    selected_factor_names: list[str] = field(default_factory=list)
    target: ResearchTarget = field(default_factory=ResearchTarget)
    selection_source: str = "semantic_match"
    selection_status: SelectionStatus = SelectionStatus.EMPTY_INTENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_factor_names": list(self.selected_factor_names),
            "target": self.target.to_dict() if isinstance(self.target, ResearchTarget) else self.target,
            "selection_source": self.selection_source,
            "selection_status": self.selection_status.value if isinstance(self.selection_status, SelectionStatus) else str(self.selection_status),
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


@dataclass
class UnavailableFactor:
    name: str
    reason: str
    missing_fields: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    status: AvailabilityStatus = AvailabilityStatus.MISSING_FIELDS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class PlanMutation:
    mutation_type: MutationType
    add_factor_names: list[str] = field(default_factory=list)
    remove_factor_names: list[str] = field(default_factory=list)
    target_horizon: int | None = None
    target_metric: str | None = None
    unresolved_terms: list[str] = field(default_factory=list)
    raw_query: str = ""
    reset_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_type": self.mutation_type.value if isinstance(self.mutation_type, MutationType) else str(self.mutation_type),
            "add_factor_names": list(self.add_factor_names),
            "remove_factor_names": list(self.remove_factor_names),
            "target_horizon": self.target_horizon,
            "target_metric": self.target_metric,
            "unresolved_terms": list(self.unresolved_terms),
            "raw_query": self.raw_query,
            "reset_reason": self.reset_reason,
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


@dataclass
class FactorResearchState:
    committed_plan: ResearchPlan | None = None
    committed_selection: FactorSelectionResult | None = None
    pending_mutation: PlanMutation | None = None
    pending_selection: FactorSelectionResult | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed_plan": self.committed_plan.to_dict() if self.committed_plan else None,
            "committed_selection": self.committed_selection.to_dict() if self.committed_selection else None,
            "pending_mutation": self.pending_mutation.to_dict() if self.pending_mutation else None,
            "pending_selection": self.pending_selection.to_dict() if self.pending_selection else None,
            "last_error": self.last_error,
        }


@dataclass
class FactorIntent:
    raw_query: str
    research_mode: str = "factor"
    research_type: str = "factor_research"
    explicit_terms: list[str] = field(default_factory=list)
    factor_names: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    target: ResearchTarget = field(default_factory=ResearchTarget)
    unresolved_terms: list[str] = field(default_factory=list)
    ambiguous_terms: list[str] = field(default_factory=list)
    recognized_not_implemented_terms: list[str] = field(default_factory=list)
    confidence: float = 0.0
    selection_source: str = "semantic_match"
    route_intent: list[str] = field(default_factory=list)
    route_to: list[str] = field(default_factory=list)
    matched_factors: list[dict[str, Any]] = field(default_factory=list)
    matched_expressions: list[dict[str, Any]] = field(default_factory=list)
    factor_plan: list[dict[str, Any]] = field(default_factory=list)
    expression_match: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = self.target.to_dict() if isinstance(self.target, ResearchTarget) else self.target
        return data

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


@dataclass
class FactorSelectionResult:
    status: SelectionStatus
    selected_factors: list[dict[str, Any]] = field(default_factory=list)
    selection_reasons: dict[str, str] = field(default_factory=dict)
    unresolved_terms: list[str] = field(default_factory=list)
    unavailable_factors: list[UnavailableFactor] = field(default_factory=list)
    ambiguous_terms: list[str] = field(default_factory=list)
    can_execute: bool = False
    target: ResearchTarget = field(default_factory=ResearchTarget)
    selection_source: str = "semantic_match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, SelectionStatus) else str(self.status),
            "selected_factors": self.selected_factors,
            "selection_reasons": self.selection_reasons,
            "unresolved_terms": self.unresolved_terms,
            "unavailable_factors": [item.to_dict() for item in self.unavailable_factors],
            "ambiguous_terms": self.ambiguous_terms,
            "can_execute": self.can_execute,
            "target": self.target.to_dict() if isinstance(self.target, ResearchTarget) else self.target,
            "selection_source": self.selection_source,
        }

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)


def coerce_target(target: Any | None) -> ResearchTarget:
    if isinstance(target, ResearchTarget):
        return target
    if isinstance(target, dict):
        return ResearchTarget(metric=target.get("metric", "future_return"), horizon=int(target.get("horizon", 5)))
    return ResearchTarget()


def target_return_column(target: Any | None, *, default_horizon: int = 5) -> str:
    """Map a research target to the corresponding future-return column name."""
    horizon = int(coerce_target(target).horizon or default_horizon)
    return f"future_return_{horizon}d"


def build_factor_intent_from_plan(
    raw_query: str,
    factor_names: list[str],
    target: Any | None = None,
    *,
    selection_source: str = "committed_plan",
) -> FactorIntent:
    """Build a FactorIntent from structured plan state without re-parsing natural language."""
    return FactorIntent(
        raw_query=raw_query,
        factor_names=list(factor_names),
        target=coerce_target(target),
        selection_source=selection_source,
    )

