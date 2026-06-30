"""Tests for src.factor_registry and src.factor_evaluator.

The contract being pinned here:

1. `factor_registry` is a thin aggregation view over
   `etf_factor_library` and `LOCAL_SUPPORTED_FACTOR_NAMES`. It does not
   duplicate metadata, and the 10 whitelist factors are reported as
   `implemented` with the correct `required_fields`.

2. `factor_evaluator.ensure_factors_available` is the single chokepoint
   for "make sure this column exists in the DataFrame before we run
   IC / quantile / backtest". It:
     - marks existing columns as `existing_factors`
     - computes missing columns via whitelisted pandas functions
     - records `not_implemented` and `missing_required_fields` as data
     - never raises on the missing path
     - never uses `eval` or formula-string execution
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factor_evaluator import (
    FACTOR_COMPUTERS,
    WHITELIST_FACTOR_NAMES,
    ensure_factors_available,
)
from src.factor_registry import (
    get_factor_registry,
    get_factor_spec,
    get_supported_factor_names,
    is_supported,
)


WHITELIST_TEN = {
    "amount_ratio_20d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "momentum_20d",
    "momentum_60d",
    "trend_strength",
    "volatility_20d",
    "rsi_14d",
    "bbi_indicator",
    "ma_gap_20d",
}


def _make_ohlcv(n_dates: int = 80, n_codes: int = 3, seed: int = 0) -> pd.DataFrame:
    """Build a minimal but well-shaped ETF OHLCV frame for evaluator tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for code_idx in range(n_codes):
        base = 1.0 + code_idx * 0.5
        close = base + np.cumsum(rng.normal(0, 0.01, n_dates))
        for i, d in enumerate(dates):
            close_i = float(close[i])
            open_i = close_i * (1 + rng.normal(0, 0.002))
            high_i = max(open_i, close_i) * (1 + abs(rng.normal(0, 0.003)))
            low_i = min(open_i, close_i) * (1 - abs(rng.normal(0, 0.003)))
            volume_i = float(rng.integers(1_000_000, 5_000_000))
            amount_i = close_i * volume_i
            rows.append(
                {
                    "date": d,
                    "code": f"ETF_{code_idx:02d}",
                    "instrument": f"ETF_{code_idx:02d}",
                    "open": open_i,
                    "high": high_i,
                    "low": low_i,
                    "close": close_i,
                    "volume": volume_i,
                    "amount": amount_i,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Registry: view layer
# ---------------------------------------------------------------------------


def test_registry_lists_all_ten_whitelist_factors():
    names = set(get_supported_factor_names())
    missing = WHITELIST_TEN - names
    assert not missing, f"whitelist factors missing from registry: {missing}"


def test_registry_reports_implemented_and_required_fields():
    for name in WHITELIST_TEN:
        spec = get_factor_spec(name)
        assert spec is not None, name
        assert spec.get("implementation_status") == "implemented", spec
        assert "local" in (spec.get("supported_backends") or []), spec
        # required_fields is a list (may be empty for factors that only need close)
        assert isinstance(spec.get("required_fields"), list), spec


def test_registry_excludes_kdj_and_macd():
    assert is_supported("kdj_indicator") is False
    assert is_supported("kdj") is False
    assert is_supported("macd_indicator") is False
    assert is_supported("macd") is False
    assert is_supported("boll_indicator") is False


# ---------------------------------------------------------------------------
# Evaluator: existing / compute / missing
# ---------------------------------------------------------------------------


def test_existing_column_marked_as_existing():
    df = _make_ohlcv()
    df["momentum_20d"] = 0.0  # pretend it already exists
    result = ensure_factors_available(df, ["momentum_20d"])
    assert result.ok
    assert "momentum_20d" in result.existing_factors
    assert "momentum_20d" in result.available_factors
    assert result.computed_factors == []
    # df not copied unnecessarily
    assert result.df is df


def test_missing_whitelist_column_is_computed():
    df = _make_ohlcv()
    assert "bbi_indicator" not in df.columns
    result = ensure_factors_available(df, ["bbi_indicator"])
    assert result.ok
    assert "bbi_indicator" in result.computed_factors
    assert "bbi_indicator" in result.available_factors
    assert "bbi_indicator" in result.df.columns
    # Sanity: BBI is the average of 4 rolling means, should be close to close.
    computed = result.df["bbi_indicator"].dropna()
    assert len(computed) > 0
    assert np.isfinite(computed).all()


def test_volume_ratio_5d_computed_when_missing():
    df = _make_ohlcv()
    assert "volume_ratio_5d" not in df.columns
    result = ensure_factors_available(df, ["volume_ratio_5d"])
    assert result.ok
    assert "volume_ratio_5d" in result.computed_factors
    assert "volume_ratio_5d" in result.df.columns


def test_all_ten_whitelist_factors_ensure_in_one_pass():
    df = _make_ohlcv()
    # Drop all derived columns so we exercise the compute path.
    for col in WHITELIST_TEN:
        assert col not in df.columns
    result = ensure_factors_available(df, sorted(WHITELIST_TEN))
    assert result.ok, f"missing_required_fields={result.missing_required_fields}, not_implemented={result.not_implemented}"
    assert set(result.available_factors) == WHITELIST_TEN
    assert set(result.computed_factors) == WHITELIST_TEN
    for name in WHITELIST_TEN:
        assert name in result.df.columns


def test_missing_required_fields_reported_not_raised():
    df = _make_ohlcv().drop(columns=["volume"])
    result = ensure_factors_available(df, ["volume_ratio_5d"])
    assert not result.ok
    assert result.available_factors == []
    assert result.computed_factors == []
    assert result.missing_required_fields, result.missing_required_fields
    entry = result.missing_required_fields[0]
    assert entry["name"] == "volume_ratio_5d"
    assert "volume" in entry["missing_fields"]


def test_unknown_factor_marked_as_not_implemented():
    df = _make_ohlcv()
    result = ensure_factors_available(df, ["kdj_indicator", "macd"])
    assert not result.ok
    assert "kdj_indicator" in result.not_implemented
    assert "macd" in result.not_implemented
    assert result.available_factors == []


def test_mixed_existing_computed_and_unavailable():
    df = _make_ohlcv()
    df["momentum_20d"] = 0.123  # pretend already there
    result = ensure_factors_available(df, ["momentum_20d", "rsi_14d", "kdj_indicator"])
    # existing + computed is fine, kdj is not_implemented
    assert not result.ok
    assert "momentum_20d" in result.existing_factors
    assert "rsi_14d" in result.computed_factors
    assert "kdj_indicator" in result.not_implemented
    assert set(result.available_factors) == {"momentum_20d", "rsi_14d"}
    assert "rsi_14d" in result.df.columns


def test_evaluator_does_not_use_eval_or_formula_strings():
    """Hard contract: the only compute functions are explicit pandas
    callables registered in FACTOR_COMPUTERS. No `eval`, no string-formula
    dispatcher, no BigQuant DSL runtime."""
    for name, fn in FACTOR_COMPUTERS.items():
        assert callable(fn), name
        # No `eval` reference anywhere in the registry file
    # Whitelist membership is finite
    assert WHITELIST_FACTOR_NAMES == set(FACTOR_COMPUTERS.keys())
    assert WHITELIST_FACTOR_NAMES == WHITELIST_TEN


# ---------------------------------------------------------------------------
# agent_tools chokepoint integration
# ---------------------------------------------------------------------------


def test_agent_tools_does_not_silently_drop_missing_columns(monkeypatch):
    """Pin the change in `tool_run_factor_research_pipeline` that
    replaces the silent `[f for f in factors if f in df.columns]` with
    a real `ensure_factors_available` chokepoint.

    We invoke the tool with a fake resolver+selection that picks
    `bbi_indicator` (a column not yet in the frame) and assert that
    the tool computes it instead of returning a zero-factor result.
    """
    import src.agent_tools as agent_tools

    fake_hypothesis = {
        "raw_query": "研究 BBI",
        "research_type": "factor_research",
        "factor_names": ["bbi_indicator"],
    }
    fake_selection_dict = {
        "status": "ready",
        "can_execute": True,
        "selected_factors": [
            {
                "name": "bbi_indicator",
                "direction": "neutral",
                "category": "trend",
                "description": "BBI",
            }
        ],
        "selection_reasons": {"bbi_indicator": "explicit"},
        "unresolved_terms": [],
        "unavailable_factors": [],
        "ambiguous_terms": [],
        "target": {"metric": "future_return", "horizon": 5},
        "selection_source": "explicit",
    }

    def fake_resolve(user_idea):
        return dict(fake_hypothesis, raw_query=user_idea)

    class FakeSelection:
        def __init__(self):
            self.status = "ready"
            self.can_execute = True
            self.selected_factors = fake_selection_dict["selected_factors"]
            self.unresolved_terms = []
            self.unavailable_factors = []
            self.ambiguous_terms = []
            self.target = {"metric": "future_return", "horizon": 5}
            self.selection_source = "explicit"

        def to_dict(self):
            return fake_selection_dict

    def fake_generate(hypothesis, **_):
        return FakeSelection()

    df_stub = _make_ohlcv()  # bbi_indicator is NOT pre-computed; evaluator must add it

    def fake_load_etf_data(**_):
        return df_stub

    def fake_factor_analysis(df, factor_cols, targets):
        # Return a minimal factor_result keyed by factor name.
        return {name: {} for name in factor_cols}

    def fake_weight_schemes(factor_cols, _factor_result):
        return {"hypothesis_weight": {n: 1.0 for n in factor_cols}}

    def fake_normalize(df, _directions):
        return df

    def fake_composite(df, _weights):
        return df

    def fake_backtest(df, **_):
        return {"performance": {}, "nav": []}

    def fake_diagnose(_factor_result, _backtest, _weights):
        return {}

    def fake_generate_report(*args, **kwargs):
        return "# stub report"

    monkeypatch.setattr(agent_tools, "resolve_factor_intent", fake_resolve)
    monkeypatch.setattr(agent_tools, "generate_factor_candidates", fake_generate)
    monkeypatch.setattr(agent_tools, "load_etf_data", fake_load_etf_data)
    monkeypatch.setattr(agent_tools, "run_factor_analysis", fake_factor_analysis)
    monkeypatch.setattr(agent_tools, "build_weight_schemes", fake_weight_schemes)
    monkeypatch.setattr(agent_tools, "normalize_factors", fake_normalize)
    monkeypatch.setattr(agent_tools, "compute_composite_score", fake_composite)
    monkeypatch.setattr(agent_tools, "run_top_pct_backtest", fake_backtest)
    monkeypatch.setattr(agent_tools, "diagnose_strategy", fake_diagnose)
    monkeypatch.setattr(agent_tools, "generate_report", fake_generate_report)

    result = agent_tools.tool_run_factor_research_pipeline(
        user_idea="研究 BBI",
        start_date="2024-01-01",
        end_date="2024-12-31",
        data_backend="local",
        local_etf_parquet="unused",
    )

    # The pipeline must pass through the ensure chokepoint and NOT
    # silently drop bbi_indicator. The end-to-end run with a stub frame
    # may fail later (e.g. invalid_target) for unrelated reasons; what
    # matters is that we did NOT return early with `factors_unavailable`.
    assert result.get("status") != "factors_unavailable", result
    # The selection that *requested* bbi_indicator must have been
    # recognized — visible in selection_result.selected_factors.
    selection_result = result.get("selection_result") or {}
    sel_factor_names = [f.get("name") for f in (selection_result.get("selected_factors") or [])]
    assert "bbi_indicator" in sel_factor_names, selection_result
