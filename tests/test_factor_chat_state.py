from copy import deepcopy

from src.chat_state import create_chat_state, update_chat_state
from src.research_plan import SelectionStatus


def _seed_committed_state():
    state = create_chat_state()
    state["current_context"].update(
        {
            "committed_plan": {
                "selected_factor_names": ["momentum_20d"],
                "target": {"metric": "future_return", "horizon": 5},
                "selection_source": "explicit",
                "selection_status": "ready",
            },
            "committed_selection": {
                "status": "ready",
                "can_execute": True,
                "selected_factors": [{"name": "momentum_20d", "category": "momentum"}],
                "selection_reasons": {"momentum_20d": "explicit"},
                "unresolved_terms": [],
                "unavailable_factors": [],
                "ambiguous_terms": [],
                "target": {"metric": "future_return", "horizon": 5},
                "selection_source": "explicit",
            },
            "selected_factors": [{"name": "momentum_20d", "category": "momentum"}],
            "selected_factor_names": ["momentum_20d"],
            "factors": [{"name": "momentum_20d", "category": "momentum"}],
            "target": {"metric": "future_return", "horizon": 5},
            "selection_status": "ready",
        }
    )
    return state


def test_failed_mutation_does_not_modify_committed_plan():
    state = _seed_committed_state()
    old_state = deepcopy(state)
    next_state = update_chat_state(
        state,
        "再加 BBI",
        tool_result={
            "selection_result": {
                "status": "recognized_not_implemented",
                "can_execute": False,
                "selected_factors": [{"name": "momentum_20d", "category": "momentum"}],
                "selection_reasons": {"momentum_20d": "explicit"},
                "unresolved_terms": [],
                "unavailable_factors": [{"name": "BBI", "reason": "knowledge_known_but_not_implemented"}],
                "ambiguous_terms": [],
                "target": {"metric": "future_return", "horizon": 5},
                "selection_source": "explicit",
            }
        },
        assistant_reply="pending",
    )

    context = next_state["current_context"]
    assert context["committed_plan"] == old_state["current_context"]["committed_plan"]
    assert context["selected_factor_names"] == ["momentum_20d"]
    assert context["pending_mutation"]["mutation_type"] == "add_factors"
    assert context["pending_selection"]["status"] == "recognized_not_implemented"
    assert context["last_error"] == "recognized_not_implemented"


def test_target_update_does_not_reselect_factors():
    state = _seed_committed_state()
    next_state = update_chat_state(
        state,
        "改成未来10日",
        tool_result={
            "selection_result": {
                "status": "ready",
                "can_execute": True,
                "selected_factors": [{"name": "momentum_20d", "category": "momentum"}],
                "selection_reasons": {"momentum_20d": "explicit"},
                "unresolved_terms": [],
                "unavailable_factors": [],
                "ambiguous_terms": [],
                "target": {"metric": "future_return", "horizon": 10},
                "selection_source": "explicit",
            }
        },
        assistant_reply="pending",
    )

    context = next_state["current_context"]
    assert context["selected_factor_names"] == ["momentum_20d"]
    assert context["target"]["horizon"] == 10
    assert context["committed_plan"]["selected_factor_names"] == ["momentum_20d"]
    assert context["committed_plan"]["target"]["horizon"] == 10


def test_add_existing_factor_is_idempotent():
    state = _seed_committed_state()
    next_state = update_chat_state(
        state,
        "再加上20日动量",
        tool_result={
            "selection_result": {
                "status": "ready",
                "can_execute": True,
                "selected_factors": [{"name": "momentum_20d", "category": "momentum"}],
                "selection_reasons": {"momentum_20d": "explicit"},
                "unresolved_terms": [],
                "unavailable_factors": [],
                "ambiguous_terms": [],
                "target": {"metric": "future_return", "horizon": 5},
                "selection_source": "explicit",
            }
        },
        assistant_reply="pending",
    )

    context = next_state["current_context"]
    assert context["selected_factor_names"] == ["momentum_20d"]
    assert len(context["selected_factors"]) == 1
    assert context["selection_status"] == SelectionStatus.READY.value

