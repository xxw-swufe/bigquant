"""Single source of truth view over the ETF factor library.

`factor_registry` is a thin aggregation layer on top of
`src.etf_factor_library`. It does not duplicate factor metadata; it
exposes a few convenience helpers for the evaluator / pipeline:

- `get_factor_registry()`         : dict[name, factor_spec] for all implemented factors
- `get_supported_factor_names()`  : set of names with at least one implemented backend
- `get_factor_spec(name)`         : one factor spec, or None

The contract is intentionally narrow so that the rest of the system
talks to *one* registry view rather than re-implementing filters over
the raw factor library.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.etf_factor_library import (
    LOCAL_SUPPORTED_FACTOR_NAMES,
    get_factor_library,
)


def _is_implemented(factor: dict) -> bool:
    status = factor.get("implementation_status")
    if status:
        return status == "implemented"
    backends = factor.get("supported_backends") or []
    return bool(backends)


@lru_cache(maxsize=1)
def get_factor_registry() -> dict[str, dict]:
    """Return {name: factor_spec} for every factor whose implementation_status
    is 'implemented' (or which has at least one supported backend and a
    positive name in LOCAL_SUPPORTED_FACTOR_NAMES)."""
    lib = get_factor_library()
    by_name = {f["name"]: f for f in lib}
    for name in LOCAL_SUPPORTED_FACTOR_NAMES:
        if name not in by_name:
            # Library set says it's supported but metadata is missing.
            # Surface a minimal stub so callers can see *why* it failed
            # to load rather than silently disappearing.
            by_name[name] = {
                "name": name,
                "implementation_status": "missing_metadata",
                "supported_backends": [],
                "required_fields": [],
                "required_context": [],
                "data_dependency": None,
                "is_derived_factor": True,
            }
    return {
        name: spec
        for name, spec in by_name.items()
        if _is_implemented(spec) and spec.get("name") in LOCAL_SUPPORTED_FACTOR_NAMES
    }


@lru_cache(maxsize=1)
def get_supported_factor_names() -> frozenset[str]:
    return frozenset(get_factor_registry().keys())


def get_factor_spec(name: str) -> Optional[dict]:
    """Return one factor spec by name, or None if not implemented."""
    return get_factor_registry().get(name)


def is_supported(name: str) -> bool:
    return name in get_supported_factor_names()


__all__ = [
    "get_factor_registry",
    "get_supported_factor_names",
    "get_factor_spec",
    "is_supported",
]
