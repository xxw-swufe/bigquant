"""Parse user follow-up requests into atomic research plan mutations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.etf_factor_library import get_factor_library
from src.expression_knowledge_base import extract_target_horizon, match_category_terms
from src.factor_resolver import resolve_factor_intent
from src.research_plan import MutationType, PlanMutation, ResearchPlan, ResearchTarget


RESET_KEYWORDS = ("重新研究", "换一个研究", "从头开始", "不要前面的", "重置", "清空", "重新开始")
REPLACE_KEYWORDS = ("改成", "改为", "替换")
ADD_KEYWORDS = ("再加", "加上", "同时", "并且", "还要", "新增")
REMOVE_KEYWORDS = ("去掉", "删除", "移除", "不要", "删掉")
TARGET_KEYWORDS = ("未来", "次日", "下一日", "明天", "收益", "周期", "周收益", "日收益")


def parse_plan_mutation(user_input: str, current_context: dict[str, Any] | None = None) -> PlanMutation:
    """Convert a user turn into an atomic plan mutation."""
    text = (user_input or "").strip()
    context = current_context or {}
    if not text:
        return PlanMutation(mutation_type=MutationType.NO_OP, raw_query=text)

    if any(keyword in text for keyword in RESET_KEYWORDS):
        return _build_reset_mutation(text)

    current_plan = _extract_committed_plan(context)
    current_names = set(current_plan.selected_factor_names)
    current_selected_factors = _extract_selected_factors(context)
    current_factor_index = {factor["name"]: factor for factor in current_selected_factors}

    if any(keyword in text for keyword in REPLACE_KEYWORDS):
        return _build_replace_mutation(text, current_names, current_factor_index, context)

    if any(keyword in text for keyword in TARGET_KEYWORDS) and not any(keyword in text for keyword in ADD_KEYWORDS + REMOVE_KEYWORDS + REPLACE_KEYWORDS):
        target_horizon = extract_target_horizon(text)
        if target_horizon is None:
            return PlanMutation(
                mutation_type=MutationType.UPDATE_TARGET,
                unresolved_terms=[text],
                raw_query=text,
            )
        return PlanMutation(
            mutation_type=MutationType.UPDATE_TARGET,
            target_horizon=target_horizon,
            target_metric="future_return",
            raw_query=text,
        )

    if any(keyword in text for keyword in REMOVE_KEYWORDS):
        return _build_remove_mutation(text, current_names, current_factor_index, context)

    if any(keyword in text for keyword in ADD_KEYWORDS):
        return _build_add_mutation(text, current_names, current_factor_index, context)

    intent = resolve_factor_intent(text)
    if not current_names and (intent.factor_names or intent.categories or intent.recognized_not_implemented_terms):
        return PlanMutation(
            mutation_type=MutationType.ADD_FACTORS,
            add_factor_names=_dedupe_list(intent.factor_names),
            target_horizon=intent.target.horizon if intent.target else None,
            target_metric=intent.target.metric if intent.target else None,
            unresolved_terms=list(intent.unresolved_terms),
            raw_query=text,
        )

    return PlanMutation(mutation_type=MutationType.NO_OP, raw_query=text)


def build_effective_user_idea(
    plan: ResearchPlan,
    *,
    factor_lookup: dict[str, dict] | None = None,
) -> str:
    """Serialize the committed plan into a stable natural-language request."""
    factor_lookup = factor_lookup or {factor["name"]: factor for factor in get_factor_library()}
    factor_phrases = []
    for name in plan.selected_factor_names:
        factor = factor_lookup.get(name)
        if factor is None:
            factor_phrases.append(name)
            continue
        factor_phrases.append(factor.get("name") or name)
    parts = []
    if factor_phrases:
        parts.append("研究" + "和".join(factor_phrases))
    else:
        parts.append("空计划")
    if plan.target and plan.target.horizon:
        parts.append(f"未来{plan.target.horizon}日")
    return "，".join(parts)


def apply_plan_mutation_to_plan(plan: ResearchPlan, mutation: PlanMutation) -> ResearchPlan:
    """Apply a mutation to a committed plan copy."""
    if mutation.mutation_type == MutationType.RESET_PLAN:
        target = ResearchTarget(
            metric=mutation.target_metric or plan.target.metric,
            horizon=mutation.target_horizon or 5,
        )
        return ResearchPlan(
            selected_factor_names=[],
            target=target,
            selection_source="reset_plan",
            selection_status=plan.selection_status,
        )

    selected_names = list(plan.selected_factor_names)
    if mutation.mutation_type in {MutationType.REMOVE_FACTORS, MutationType.REPLACE_FACTORS}:
        selected_names = [name for name in selected_names if name not in set(mutation.remove_factor_names)]
    if mutation.mutation_type in {MutationType.ADD_FACTORS, MutationType.REPLACE_FACTORS}:
        for name in mutation.add_factor_names:
            if name not in selected_names:
                selected_names.append(name)

    target = ResearchTarget(metric=plan.target.metric, horizon=plan.target.horizon)
    if mutation.target_horizon is not None:
        target.horizon = mutation.target_horizon
    if mutation.target_metric is not None:
        target.metric = mutation.target_metric

    return ResearchPlan(
        selected_factor_names=selected_names,
        target=target,
        selection_source=plan.selection_source,
        selection_status=plan.selection_status,
    )


def _build_reset_mutation(text: str) -> PlanMutation:
    return PlanMutation(mutation_type=MutationType.RESET_PLAN, raw_query=text, reset_reason=text)


def _build_add_mutation(
    text: str,
    current_names: set[str],
    current_factor_index: dict[str, dict],
    context: dict[str, Any],
) -> PlanMutation:
    intent = resolve_factor_intent(text)
    add_factor_names = _dedupe_list(intent.factor_names)
    target_horizon = intent.target.horizon if intent.target else None
    unresolved_terms = list(intent.unresolved_terms)
    if not add_factor_names and not unresolved_terms:
        unresolved_terms = [text]
    return PlanMutation(
        mutation_type=MutationType.ADD_FACTORS,
        add_factor_names=add_factor_names,
        target_horizon=target_horizon,
        target_metric=intent.target.metric if intent.target else None,
        unresolved_terms=unresolved_terms,
        raw_query=text,
    )


def _build_remove_mutation(
    text: str,
    current_names: set[str],
    current_factor_index: dict[str, dict],
    context: dict[str, Any],
) -> PlanMutation:
    intent = resolve_factor_intent(text)
    remove_factor_names = _match_existing_factor_names(intent.factor_names, current_names)
    remove_factor_names.extend(_match_existing_category_names(match_category_terms(text), current_factor_index, current_names))
    remove_factor_names = _dedupe_list(remove_factor_names)
    unresolved_terms = list(intent.unresolved_terms)
    if not remove_factor_names and not unresolved_terms:
        unresolved_terms = [text]
    return PlanMutation(
        mutation_type=MutationType.REMOVE_FACTORS,
        remove_factor_names=remove_factor_names,
        unresolved_terms=unresolved_terms,
        raw_query=text,
    )


def _build_replace_mutation(
    text: str,
    current_names: set[str],
    current_factor_index: dict[str, dict],
    context: dict[str, Any],
) -> PlanMutation:
    left, right = _split_replace_segments(text)
    remove_intent = resolve_factor_intent(left or text)
    add_intent = resolve_factor_intent(right or text)
    remove_factor_names = _match_existing_factor_names(remove_intent.factor_names, current_names)
    remove_factor_names.extend(_match_existing_category_names(match_category_terms(left), current_factor_index, current_names))
    add_factor_names = _dedupe_list(add_intent.factor_names)
    target_horizon = add_intent.target.horizon if add_intent.target else None
    unresolved_terms = _dedupe_list(list(remove_intent.unresolved_terms) + list(add_intent.unresolved_terms))
    if not add_factor_names and not unresolved_terms:
        unresolved_terms = [text]
    return PlanMutation(
        mutation_type=MutationType.REPLACE_FACTORS,
        add_factor_names=add_factor_names,
        remove_factor_names=_dedupe_list(remove_factor_names),
        target_horizon=target_horizon,
        target_metric=add_intent.target.metric if add_intent.target else None,
        unresolved_terms=unresolved_terms,
        raw_query=text,
    )


def _split_replace_segments(text: str) -> tuple[str, str]:
    for keyword in REPLACE_KEYWORDS:
        if keyword in text:
            left, right = text.split(keyword, 1)
            return left.strip("，, 。;； "), right.strip("，, 。;； ")
    return "", text


def _extract_committed_plan(context: dict[str, Any]) -> ResearchPlan:
    committed = context.get("committed_plan")
    if isinstance(committed, dict):
        return ResearchPlan(
            selected_factor_names=list(committed.get("selected_factor_names", [])),
            target=_coerce_target(committed.get("target")),
            selection_source=str(committed.get("selection_source") or "semantic_match"),
        )

    selected_factors = _extract_selected_factors(context)
    return ResearchPlan(
        selected_factor_names=[factor["name"] for factor in selected_factors if factor.get("name")],
        target=_coerce_target(context.get("target")),
        selection_source=str(context.get("selection_source") or "semantic_match"),
    )


def _extract_selected_factors(context: dict[str, Any]) -> list[dict]:
    selected = context.get("selected_factors") or context.get("factors") or []
    if isinstance(selected, list):
        return [item for item in selected if isinstance(item, dict)]
    return []


def _coerce_target(target: Any | None) -> ResearchTarget:
    if isinstance(target, ResearchTarget):
        return target
    if isinstance(target, dict):
        return ResearchTarget(metric=target.get("metric", "future_return"), horizon=int(target.get("horizon", 5)))
    return ResearchTarget()


def _match_existing_factor_names(candidate_names: list[str], current_names: set[str]) -> list[str]:
    return [name for name in candidate_names if name in current_names]


def _match_existing_category_names(
    categories: list[str],
    current_factor_index: dict[str, dict],
    current_names: set[str],
) -> list[str]:
    matched = []
    for name, factor in current_factor_index.items():
        if name not in current_names:
            continue
        if factor.get("category") in categories:
            matched.append(name)
    return matched


def _dedupe_list(items: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
