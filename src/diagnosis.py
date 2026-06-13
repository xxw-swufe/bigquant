"""Strategy and weight diagnosis."""

import numpy as np


def diagnose_strategy(
    factor_result: dict,
    backtest_result: dict,
    weights: dict[str, float],
) -> dict:
    """Diagnose factor validity, performance stability, and weight reasonableness."""
    # [AI-CORE]
    strengths = []
    risks = []
    suggestions = []

    for factor, weight in weights.items():
        target_results = factor_result.get(factor, {})
        ic_means = [
            item.get("ic_summary", {}).get("ic_mean", np.nan)
            for item in target_results.values()
        ]
        valid_ic = [value for value in ic_means if not np.isnan(value)]
        avg_ic = float(np.mean(valid_ic)) if valid_ic else np.nan
        if np.isnan(avg_ic):
            risks.append(f"{factor} 有效样本不足，暂不评价。")
        elif avg_ic > 0 and weight >= 0.2:
            strengths.append(f"{factor} 平均 IC 为正且权重较高，当前配置具有一定合理性。")
        elif avg_ic <= 0 and weight >= 0.2:
            risks.append(f"{factor} 平均 IC 不理想但权重较高，可能拖累综合得分。")
            suggestions.append(f"建议降低 {factor} 权重，或仅作为过滤条件使用。")

    performance = backtest_result.get("performance", {})
    max_drawdown = performance.get("max_drawdown", np.nan)
    sharpe = performance.get("sharpe", np.nan)
    average_turnover = performance.get("average_turnover", np.nan)

    if not np.isnan(max_drawdown) and max_drawdown < -0.2:
        risks.append("组合最大回撤较大，说明权重方案可能对风险因子考虑不足。")
        suggestions.append("建议提高波动率、回撤等风险控制因子的权重。")
    if not np.isnan(sharpe) and sharpe < 0.5:
        risks.append("组合夏普比率偏低，收益补偿风险的能力不足。")
    if not np.isnan(average_turnover) and average_turnover > 1.0:
        risks.append("组合换手率偏高，交易成本可能显著侵蚀收益。")
        suggestions.append("建议降低调仓频率，或提高入选 ETF 的稳定性约束。")

    if not strengths:
        strengths.append("系统已完成从单因子检验到综合评分回测的完整研究闭环。")
    if not suggestions:
        suggestions.append("建议继续比较等权、研究假设权重和 ICIR 权重的样本外表现。")

    decision = "谨慎继续"
    if risks and len(risks) >= 3:
        decision = "暂不建议继续"
    elif strengths and len(risks) <= 1:
        decision = "继续研究"

    return {
        "summary": "当前诊断基于单因子 IC、组合回测表现、换手率和权重配置合理性。",
        "strengths": strengths,
        "risks": risks,
        "improvement_suggestions": suggestions,
        "research_decision": decision,
        "not_investment_advice": True,
    }

