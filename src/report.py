"""Markdown report generation."""

from pathlib import Path


def generate_report(
    hypothesis: dict,
    factors: list[dict],
    factor_result: dict,
    backtest_result: dict,
    diagnosis: dict,
    weights: dict[str, float],
) -> str:
    """Generate a Markdown research report."""
    # [AI-CORE]
    lines = [
        "# AutoETF Research Agent 研究报告",
        "",
        "## 1. 用户策略想法",
        "",
        hypothesis.get("user_idea", ""),
        "",
        "## 2. 结构化研究假设",
        "",
        f"- 研究目标：{hypothesis.get('research_goal')}",
        f"- 研究对象：{hypothesis.get('asset_universe')}",
        f"- 预测目标：{', '.join(hypothesis.get('prediction_targets', []))}",
        f"- 组合规则：{hypothesis.get('portfolio_rule')}",
        "",
    ]
    selection_result = hypothesis.get("selection_result") or {}
    if selection_result:
        lines.extend(
            [
                "## 2. 研究计划",
                "",
                f"- 研究状态：{selection_result.get('status')}",
                f"- 是否可执行：{selection_result.get('can_execute')}",
                f"- 目标周期：未来 {selection_result.get('target', {}).get('horizon')} 日",
                "",
            ]
        )
        selected_factors = selection_result.get("selected_factors", [])
        if selected_factors:
            lines.append("### 选择的因子")
            lines.append("")
            for factor in selected_factors:
                lines.append(f"- {factor['name']}：{factor.get('description', '')}")
            lines.append("")
        reasons = selection_result.get("selection_reasons", {})
        if reasons:
            lines.append("### 选择原因")
            lines.append("")
            for name, reason in reasons.items():
                lines.append(f"- {name}: {reason}")
            lines.append("")

    lines.extend(
        [
        "## 3. 候选因子定义",
        "",
        ]
    )
    for factor in factors:
        lines.extend(
            [
                f"### {factor['name']}",
                "",
                f"- 含义：{factor['description']}",
                f"- 公式：`{factor['formula']}`",
                f"- 方向：{factor['direction']}",
                f"- 是否使用未来数据：{factor['uses_future_data']}",
                "",
            ]
        )

    lines.extend(["## 4. 因子权重", ""])
    for factor, weight in weights.items():
        lines.append(f"- {factor}: {weight:.2%}")
    lines.append("")

    lines.extend(["## 5. 单因子分析结果", ""])
    for factor, target_results in factor_result.items():
        lines.append(f"### {factor}")
        lines.append("")
        for target, result in target_results.items():
            summary = result.get("ic_summary", {})
            lines.append(
                f"- {target}: IC均值={summary.get('ic_mean'):.4f}, "
                f"ICIR={summary.get('icir'):.4f}, "
                f"IC正值比例={summary.get('ic_positive_ratio'):.2%}, "
                f"样本数={summary.get('ic_count')}"
            )
        lines.append("")

    performance = backtest_result.get("performance", {})
    lines.extend(
        [
            "## 6. Top 10% 简化回测结果",
            "",
            f"- 总收益：{performance.get('total_return', 0):.2%}",
            f"- 年化收益：{performance.get('annual_return', 0):.2%}",
            f"- 最大回撤：{performance.get('max_drawdown', 0):.2%}",
            f"- 夏普比率：{performance.get('sharpe', 0):.4f}",
            f"- 胜率：{performance.get('win_rate', 0):.2%}",
            f"- 平均换手率：{performance.get('average_turnover', 0):.2%}",
            f"- 交易成本合计：{performance.get('total_cost', 0):.2%}",
            "",
            "## 7. 权重合理性与风险诊断",
            "",
            diagnosis.get("summary", ""),
            "",
            "### 优势",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in diagnosis.get("strengths", []))
    lines.extend(["", "### 风险", ""])
    lines.extend(f"- {item}" for item in diagnosis.get("risks", []))
    lines.extend(["", "### 改进建议", ""])
    lines.extend(f"- {item}" for item in diagnosis.get("improvement_suggestions", []))
    lines.extend(
        [
            "",
            "## 8. 是否建议继续研究",
            "",
            diagnosis.get("research_decision", "谨慎继续"),
            "",
            "## 9. 免责声明",
            "",
            "本系统仅用于量化研究与历史回测分析，不构成任何投资建议。历史回测结果不代表未来收益。",
        ]
    )
    return "\n".join(lines)


def save_report(report: str, path: str = "outputs/research_report.md") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(report, encoding="utf-8")

