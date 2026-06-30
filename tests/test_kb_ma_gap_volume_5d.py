"""Tests for the new KB terms: MA20* -> ma_gap_20d, 5d-volume -> volume_ratio_5d.

These terms are wired into `FACTOR_TERM_MAP` (query-time) and routed via
`match_factor_terms` (longest-term-first). The tests below pin:

- "研究 MA20偏离" / "20日均线偏离" / "价格偏离MA20" / "收盘价偏离20日均线" /
  "MA20乖离"  all resolve to `ma_gap_20d` via `factor_research`.
- "研究 5日成交量放大" / "5日量比" / "成交量比5日均值" all resolve to
  `volume_ratio_5d` via `factor_research` (and do NOT pick up
  `volume_ratio_20d` for the "5日" prefixed forms).
- Generalized "乖离率" / "BIAS" is intentionally NOT mapped, to keep the
  door open for a future BIAS factor.
- "研究 KDJ" still falls through to `recognized_not_implemented`.
- "成交量放大" (no period prefix) still defaults to `volume_ratio_20d`.
"""

from __future__ import annotations

from src.expression_knowledge_base import match_factor_terms
from src.factor_resolver import resolve_factor_intent


# --- ma_gap_20d mappings --------------------------------------------------


def test_resolve_ma20_deviation_phrases_route_to_ma_gap_20d():
    phrases = [
        "研究 MA20偏离",
        "研究 20日均线偏离",
        "研究 价格偏离MA20",
        "研究 收盘价偏离20日均线",
        "研究 MA20乖离",
    ]
    for phrase in phrases:
        intent = resolve_factor_intent(phrase)
        assert intent["research_type"] == "factor_research", phrase
        assert "ma_gap_20d" in intent.get("factor_names", []), phrase
        # The narrow MA20 cleanup in factor_resolver must strip the
        # false-positive 'MA' token so the chat pipeline does not bail
        # out at `_recognized_not_implemented_block`.
        assert not intent.get("recognized_not_implemented_terms"), (phrase, intent)


def test_match_factor_terms_for_ma20_phrases():
    for phrase in [
        "MA20偏离",
        "20日均线偏离",
        "价格偏离MA20",
        "收盘价偏离20日均线",
        "MA20乖离",
    ]:
        assert match_factor_terms(phrase) == ["ma_gap_20d"], phrase


# --- volume_ratio_5d mappings ---------------------------------------------


def test_resolve_5d_volume_phrases_route_to_volume_ratio_5d():
    phrases = [
        "研究 5日成交量放大",
        "研究 5日量比",
        "研究 成交量比5日均值",
    ]
    for phrase in phrases:
        intent = resolve_factor_intent(phrase)
        assert intent["research_type"] == "factor_research", phrase
        names = intent.get("factor_names", [])
        assert "volume_ratio_5d" in names, phrase
        # The "5日" prefix must NOT also pull in volume_ratio_20d:
        # we want the longer / more specific term to win.
        assert "volume_ratio_20d" not in names, phrase


def test_match_factor_terms_5d_volume_longest_wins():
    assert match_factor_terms("5日成交量放大") == ["volume_ratio_5d"]
    assert match_factor_terms("5日量比") == ["volume_ratio_5d"]
    assert match_factor_terms("成交量比5日均值") == ["volume_ratio_5d"]


def test_default_volume_amplification_still_20d():
    """Plain '成交量放大' (no period) should still default to 20d."""
    intent = resolve_factor_intent("研究 成交量放大")
    assert "volume_ratio_20d" in intent.get("factor_names", []), intent
    assert "volume_ratio_5d" not in intent.get("factor_names", []), intent


# --- Negative: generalized BIAS / 乖离率 must NOT collide with ma_gap ---


def test_generalized_bias_phrase_does_not_collapse_to_ma_gap_only():
    """Pin the boundary: generalized '乖离率' / 'BIAS' must NOT silently
    collapse to a single-factor ma_gap_20d study. The resolver's
    semantic search may surface ma_gap_* as candidates — that's fine —
    but the result must either:
      (a) include recognized_not_implemented_terms to flag the manual
          review path, OR
      (b) include BIAS / factor candidates broader than just ma_gap_*.
    """
    for phrase in ["研究 乖离率", "研究 BIAS", "研究 bias"]:
        intent = resolve_factor_intent(phrase)
        names = intent.get("factor_names") or []
        # Hard pin: never collapse to a singleton ma_gap_* factor study.
        if names:
            assert names != ["ma_gap_20d"], (phrase, names)
            assert names != ["ma_gap_60d"], (phrase, names)
        # If there's no recognized notion in the KB, route to manual review.
        # Currently 'BIAS' / 'bias' land here via recognized_not_implemented;
        # '乖离率' currently goes through semantic search, so we only pin
        # the ma_gap-only collapse case.


# --- KDJ still not implemented -------------------------------------------


def test_resolve_kdj_still_not_implemented():
    intent = resolve_factor_intent("研究 KDJ")
    assert intent["research_type"] == "recognized_not_implemented"
    assert "kdj" not in (intent.get("factor_names") or [])


# --- Chat pipeline end-to-end: MA20偏离 must NOT short-circuit to
# direct_reply. Without the narrow MA20 cleanup the chat agent's
# `_recognized_not_implemented_block` would intercept the intent because
# the KB surfaces a leftover 'MA' token. Run the chat agent against the
# local parquet snapshot and assert the tool actually executed.


def test_chat_pipeline_ma20_deviation_runs_research_pipeline():
    """End-to-end: '研究 MA20偏离' must execute the factor pipeline,
    not be short-circuited to direct_reply by the chat agent's
    recognized_not_implemented interceptor."""
    from dataclasses import dataclass

    from src.chat_agent import run_research_chat

    @dataclass
    class Cfg:
        start_date: str = "2024-01-01"
        end_date: str = "2024-12-31"
        fund_table: str = "cn_fund_bar1d"
        top_pct: float = 0.10
        data_backend: str = "local"
        local_etf_parquet: str = "data/parquet/local_etf_daily.parquet"
        local_benchmark_parquet: str = None

    result = run_research_chat("研究 MA20偏离", Cfg(), use_llm=False)
    assert result["tool_name"] == "run_factor_research_pipeline", result
    # selection_status should reach `ready`, not `recognized_not_implemented`.
    assert (
        result.get("state", {}).get("current_context", {}).get("selection_status")
        == "ready"
    ), result


# --- Boundary: generalized MA / 乖离率 / BIAS must NOT be silently
# narrowed to a single-factor ma_gap_20d study. The narrow cleanup
# inside factor_resolver is *only* applied for the listed MA20
# deviation phrases; for any other query, the `recognized_not_implemented`
# branch must still be authoritative when the KB surfaces a false token.


def test_generalized_ma_does_not_enter_factor_research():
    for phrase in ["研究 MA", "研究 均线"]:
        intent = resolve_factor_intent(phrase)
        assert intent["research_type"] != "factor_research", (phrase, intent)


def test_generalized_bias_phrase_does_not_enter_factor_research():
    for phrase in ["研究 乖离率", "研究 BIAS", "研究 bias"]:
        intent = resolve_factor_intent(phrase)
        assert intent["research_type"] != "factor_research", (phrase, intent)


def test_ma_up_does_not_resolve_to_factor_research():
    """'MA向上' is a directional condition cue, not a single factor study.
    It must not be silently routed as a factor_research outcome."""
    intent = resolve_factor_intent("研究 MA向上")
    assert intent["research_type"] != "factor_research", intent


# --- trend_strength natural language entry point -------------------------


def test_resolve_trend_strength_phrase_routes_to_trend_strength():
    intent = resolve_factor_intent("研究 trend_strength")
    assert intent["research_type"] == "factor_research"
    assert "trend_strength" in (intent.get("factor_names") or []), intent


def test_resolve_trend_strength_chinese_phrase_includes_factor():
    """'研究 趋势强度' uses Chinese phrasing; the resolver currently maps
    this to a composite_score_analysis (multiple trend factors via
    KB's score_trend entry). We pin: trend_strength must be among the
    chosen factors AND the research_type must be a `factor_research`
    family outcome (factor_research / composite_score_analysis), never
    recognized_not_implemented."""
    intent = resolve_factor_intent("研究 趋势强度")
    assert intent["research_type"] in {"factor_research", "composite_score_analysis"}, intent
    assert "trend_strength" in (intent.get("factor_names") or []), intent


def test_resolve_short_vs_mid_ma_trend_strength_does_not_blow_up():
    """Phrases like '短期均线相对中期均线强度' contain directional MA
    language. The pin is *non-explosive* behavior: must be a recognized
    research_type (either factor_research or recognized_not_implemented),
    never raise."""
    for phrase in [
        "研究 短期均线相对中期均线强度",
        "研究 MA5相对MA20趋势强度",
    ]:
        intent = resolve_factor_intent(phrase)
        assert intent["research_type"] in {
            "factor_research",
            "recognized_not_implemented",
        }, (phrase, intent)
