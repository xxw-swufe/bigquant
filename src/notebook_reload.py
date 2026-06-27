"""Reload AutoETF runtime modules in notebook sessions."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


RUNTIME_MODULES = (
    "src.research_plan",
    "src.plan_mutation",
    "src.factor_resolver",
    "src.factor_generator",
    "src.factor_availability",
    "src.agent_tools",
    "src.chat_state",
    "src.chat_agent",
    "src.notebook_chat_ui",
)


def reload_autoetf_runtime() -> list[str]:
    """Reload chat/runtime modules so notebook cell re-runs pick up code changes."""
    reloaded: list[str] = []
    for module_name in RUNTIME_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        else:
            module = importlib.reload(module)
        reloaded.append(module.__name__)
    return reloaded
