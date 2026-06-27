"""Generate a markdown audit matrix for ETF factors."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.etf_factor_library import get_factor_library
from src.factor_availability import build_factor_implementation_state


def _format_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(values)


def build_matrix() -> str:
    factors = get_factor_library()
    lines = [
        "# ETF 因子实现审计矩阵",
        "",
        "说明：",
        "- 运行时事实来源于 `src/etf_factor_library.py`。",
        "- `local_status` / `bigquant_status` 是保守审计结果，不等于回测时的最终可执行结果。",
        "",
        "| 因子 | 知识状态 | 实现状态 | 本地状态 | BigQuant 状态 | 依赖字段 | 额外上下文 | 支持后端 | 选择组 | 选择优先级 | 阻塞原因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for factor in factors:
        local_state = build_factor_implementation_state(
            factor,
            data_backend="local",
            available_fields=factor.get("required_fields", factor.get("required_columns", [])),
        )
        bigquant_state = build_factor_implementation_state(
            factor,
            data_backend="bigquant",
            available_fields=factor.get("required_fields", factor.get("required_columns", [])),
        )
        blocking_reason = local_state["reason"] if not local_state["available"] else "none"
        if not local_state["available"] and bigquant_state["available"]:
            blocking_reason = bigquant_state["reason"]
        lines.append(
            "| {name} | {knowledge} | {impl} | {local} | {bigquant} | {fields} | {context} | {backends} | {group} | {priority} | {reason} |".format(
                name=factor["name"],
                knowledge=factor.get("knowledge_status", "known"),
                impl=factor.get("implementation_status", "unknown"),
                local=local_state["status"] if not local_state["available"] else "available",
                bigquant=bigquant_state["status"] if not bigquant_state["available"] else "available",
                fields=_format_list(list(factor.get("required_fields", factor.get("required_columns", [])))),
                context=_format_list(list(factor.get("required_context", []))),
                backends=_format_list(list(factor.get("supported_backends", []))),
                group=factor.get("selection_group", "-"),
                priority=factor.get("selection_priority", 0),
                reason=blocking_reason,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    output_path = ROOT / "docs" / "factor_implementation_matrix.md"
    output_path.write_text(build_matrix(), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
