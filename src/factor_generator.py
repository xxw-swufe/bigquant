"""ETF factor candidate generation."""

from src.etf_factor_library import get_default_factor_candidates


def generate_factor_candidates(hypothesis: dict) -> list[dict]:
    """Generate ETF factor candidates from a structured hypothesis."""
    # [AI-CORE]
    factors = get_default_factor_candidates()
    validate_factor_candidates(factors)
    return factors


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
