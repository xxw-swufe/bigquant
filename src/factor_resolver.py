"""Resolve natural language factor intents into structured research plans."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.etf_factor_library import get_factor_library, search_factors
from src.expression_knowledge_base import (
    CATEGORY_TERM_MAP,
    FACTOR_TERM_MAP,
    TARGET_HORIZON_PATTERNS,
    classify_expression,
    extract_target_horizon,
    find_ambiguous_terms,
    match_category_terms,
    match_factor_terms,
)
from src.research_plan import FactorIntent, ResearchTarget


def resolve_factor_intent(user_idea: str) -> FactorIntent:
    """Map a natural-language request onto a structured factor intent."""
    text = (user_idea or "").strip()
    kb_result = classify_expression(text)
    best_match = kb_result.get("best_match") or {}
    matched_expressions = kb_result.get("matched", [])
    factor_index = {factor["name"]: factor for factor in get_factor_library()}

    target = ResearchTarget(horizon=extract_target_horizon(text) or 5)
    factor_names: list[str] = []
    categories: list[str] = []
    explicit_terms: list[str] = []
    unresolved_terms: list[str] = []
    ambiguous_terms = find_ambiguous_terms(text)
    recognized_not_implemented_terms: list[str] = []
    selection_source = "semantic_match"
    research_type = "factor_research"

    # Match factor names & categories FIRST so the multi-factor signal can
    # override an ambiguous "condition_expression" / "target_expression" tag
    # coming from a single phrase like "成交额放大" (which KB may tag as
    # condition_expression even when the user wants it as a scoring factor).
    factor_names.extend(_match_canonical_factor_names(text, factor_index))
    factor_names.extend(match_factor_terms(text))
    if factor_names:
        explicit_terms.extend(_factor_names_to_terms(factor_names))
        selection_source = "explicit"

    categories.extend(match_category_terms(text))
    if categories and not factor_names:
        selection_source = "category_expansion"

    # Determine research_type. Priority order:
    #   1. Condition-research lexical cues win over factor matching.
    #      Phrases like "概率", "胜率", "反弹", "上涨概率" + condition
    #      phrasing ("X 后", "突破", "放量") clearly point at an event study.
    #   2. >=2 factor_names OR any factor_name + category  -> factor_research
    #      (user is asking for a multi-factor study; condition-style
    #      phrasing like "成交额放大" describes direction, not filtering)
    #   3. explicit single factor_name                        -> factor_research
    #      (same reasoning: a named factor implies a factor study)
    #   4. best_match expression type                        -> condition / target / metric
    #   5. fallback                                           -> factor_research
    condition_cue_terms = (
        "上涨概率", "胜率", "反弹", "下跌概率", "上涨情况", "下跌情况",
        "下一日", "次日", "隔日", "未来1日", "未来1日上涨", "概率是否",
    )
    has_condition_cue = any(cue in text for cue in condition_cue_terms)

    if has_condition_cue:
        # Condition-style question — condition cue wins over factor
        # matching. The user is asking "X conditions are met → what is the
        # up-probability" which is fundamentally a conditional study, not
        # a multi-factor scoring study, regardless of how many factor
        # names incidentally match the lexical KB.
        research_type = "conditional_event_study"
    else:
        # No condition cue — lean on factor signals.
        if len(factor_names) >= 2 or (factor_names and categories):
            research_type = "factor_research"
        elif factor_names:
            research_type = "factor_research"
        elif best_match.get("expression_type") == "condition_expression":
            research_type = "conditional_event_study"
        elif best_match.get("expression_type") == "target_expression":
            research_type = "target_analysis"
        elif best_match.get("expression_type") == "metric_expression":
            research_type = "composite_score_analysis"

    # INTENT_EXPRESSION_ALIASES maps KB intent names (e.g. "rsi_indicator")
    # onto actual factor names in the factor library (e.g. "rsi_14d"). Keep
    # this table tiny on purpose: today it only maps RSI, so that expanding
    # the bridge is an explicit, audited decision per factor.
    intent_expression_aliases = {
        "rsi_indicator": "rsi_14d",
    }
    for expr in matched_expressions:
        if expr.get("expression_type") == "target_expression":
            target = ResearchTarget(
                metric="future_return",
                horizon=int(_expression_horizon(expr, text)),
            )
            continue
        phrase = str(expr.get("phrase") or "").strip()
        canonical_name = str(expr.get("canonical_name") or "").strip()
        if expr.get("expression_type") == "intent_expression" and canonical_name in intent_expression_aliases:
            mapped = intent_expression_aliases[canonical_name]
            if mapped in factor_index and mapped not in factor_names:
                factor_names.append(mapped)
                if phrase and phrase not in explicit_terms:
                    explicit_terms.append(phrase)
            continue
        if expr.get("implementation_status") == "not_implemented" or (
            expr.get("expression_type") == "intent_expression" and canonical_name not in factor_index
        ):
            term = phrase or canonical_name
            if term and term not in recognized_not_implemented_terms:
                recognized_not_implemented_terms.append(term)

    if not factor_names and not categories:
        semantic_candidates = _semantic_factor_candidates(text, factor_index)
        if semantic_candidates:
            factor_names.extend(semantic_candidates)
            selection_source = "semantic_match"

    if not factor_names and not categories and not recognized_not_implemented_terms and not ambiguous_terms:
        unresolved_terms = _extract_unknown_terms(text)

    if not unresolved_terms and not factor_names and not categories and not recognized_not_implemented_terms:
        selection_source = "empty_intent"

    if not ambiguous_terms and _looks_ambiguous(text):
        ambiguous_terms = _extract_ambiguous_candidates(text)

    confidence = _estimate_confidence(
        factor_names=factor_names,
        categories=categories,
        unresolved_terms=unresolved_terms,
        ambiguous_terms=ambiguous_terms,
        recognized_not_implemented_terms=recognized_not_implemented_terms,
    )

    route_intent = _dedupe_list(
        list(kb_result.get("route_intent", []))
        + (["factor"] if factor_names or categories else [])
        + (["manual_review"] if recognized_not_implemented_terms or ambiguous_terms else [])
    )
    route_to = _dedupe_list(
        list(kb_result.get("route_to", []))
        + (["factor_research"] if factor_names or categories else [])
        + (["manual_review"] if recognized_not_implemented_terms or ambiguous_terms else [])
    )

    factor_plan = [factor_index[name] for name in factor_names if name in factor_index]
    expression_match = kb_result.get("best_match") or None

    return FactorIntent(
        raw_query=text,
        research_mode="factor",
        research_type="recognized_not_implemented" if recognized_not_implemented_terms else research_type,
        explicit_terms=explicit_terms,
        factor_names=_dedupe_list(factor_names),
        categories=_dedupe_list(categories),
        target=target,
        unresolved_terms=_dedupe_list(unresolved_terms),
        ambiguous_terms=_dedupe_list(ambiguous_terms),
        recognized_not_implemented_terms=_dedupe_list(recognized_not_implemented_terms),
        confidence=confidence,
        selection_source=selection_source,
        route_intent=route_intent,
        route_to=route_to,
        matched_factors=list(factor_plan),
        matched_expressions=matched_expressions,
        factor_plan=factor_plan,
        expression_match=expression_match,
    )


def _semantic_factor_candidates(text: str, factor_index: dict[str, dict]) -> list[str]:
    candidates: list[str] = []
    for factor in search_factors(text, limit=20):
        name = factor.get("name")
        if name not in factor_index:
            continue
        candidates.append(name)
    return _dedupe_list(candidates)


def _match_canonical_factor_names(text: str, factor_index: dict[str, dict]) -> list[str]:
    normalized = text.lower()
    matched = []
    for name in factor_index:
        if name.lower() in normalized:
            matched.append(name)
    return _dedupe_list(matched)


def _factor_names_to_terms(factor_names: Iterable[str]) -> list[str]:
    names = set(factor_names)
    terms = []
    for term, mapped_names in FACTOR_TERM_MAP.items():
        if names.intersection(mapped_names):
            terms.append(term)
    return _dedupe_list(terms)


def _extract_unknown_terms(text: str) -> list[str]:
    stop_terms = {
        "帮我做一个",
        "帮我",
        "ETF",
        "ETF因子研究",
        "因子研究",
        "研究",
        "做一个",
        "分析",
        "看看",
        "一下",
        "这个",
        "一个",
    }
    candidates = []
    for token in re.findall(r"[A-Z]{2,}|[\u4e00-\u9fa5]{2,}(?:指标|因子|均线|线|分)?", text):
        normalized = token.strip()
        if not normalized or normalized in {"指标", "因子", "均线", "线", "分"}:
            continue
        if normalized in stop_terms or any(term in normalized for term in stop_terms):
            continue
        if normalized in FACTOR_TERM_MAP or normalized in CATEGORY_TERM_MAP or normalized in TARGET_HORIZON_PATTERNS:
            continue
        candidates.append(normalized)
    return _dedupe_list(candidates)


def _extract_ambiguous_candidates(text: str) -> list[str]:
    candidates = []
    if "强弱" in text:
        candidates.extend(["RSI", "相对强度", "动量", "趋势强度"])
    if "趋势" in text and "因子" in text:
        candidates.extend(["trend_strength", "ma_gap_20d", "breakout_60d"])
    if "动量" in text and "因子" in text:
        candidates.extend(["momentum_5d", "momentum_20d", "momentum_60d"])
    return _dedupe_list(candidates)


def _looks_ambiguous(text: str) -> bool:
    return any(keyword in text for keyword in ["强弱指标", "强弱", "趋势指标", "动量指标"])


def _expression_horizon(entry: dict, text: str) -> int:
    if entry.get("default_window") is not None:
        return int(entry["default_window"])
    horizon = extract_target_horizon(text)
    return horizon or 5


def _estimate_confidence(
    *,
    factor_names: list[str],
    categories: list[str],
    unresolved_terms: list[str],
    ambiguous_terms: list[str],
    recognized_not_implemented_terms: list[str],
) -> float:
    score = 0.0
    score += min(len(factor_names) * 0.35, 0.7)
    score += min(len(categories) * 0.2, 0.3)
    score -= min(len(unresolved_terms) * 0.3, 0.6)
    score -= min(len(ambiguous_terms) * 0.25, 0.5)
    score -= min(len(recognized_not_implemented_terms) * 0.1, 0.2)
    return max(0.0, min(0.99, score))


def _dedupe_list(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
