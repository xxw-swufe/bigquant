"""Markdown report for conditional probability event studies."""

from pathlib import Path

import pandas as pd


def generate_condition_report(
    hypothesis: dict,
    result: dict,
    diagnosis: dict,
) -> str:
    """Generate a Markdown report for the first condition-study MVP."""
    # [AI-CORE]
    lines = [
        "# AutoETF Research Agent 条件事件研究报告",
        "",
        "## 1. 用户问题",
        "",
        hypothesis.get("user_idea", ""),
        "",
        "## 2. 结构化研究假设",
        "",
        f"- 研究对象：{hypothesis.get('asset_universe')}",
        f"- 研究目标：{hypothesis.get('research_goal')}",
        "- 条件规则：",
    ]
    for condition in hypothesis.get("conditions", []):
        lines.append(
            f"  - {condition['description']}：`{condition['field']} {condition['operator']} {condition['value']}`"
        )
    target = hypothesis.get("target", {})
    lines.extend(
        [
            f"- 评价目标：{target.get('description')}：`{target.get('field')} {target.get('operator')} {target.get('value')}`",
            "",
            "## 3. 样本与事件统计",
            "",
            f"- 全样本数量：{result['total_count']}",
            f"- 满足条件样本数量：{result['event_count']}",
            f"- 条件样本占比：{_fmt_pct(result['event_ratio'])}",
            f"- 最小样本阈值：{result['min_event_count']}",
            f"- 样本是否充足：{'是' if result['is_sample_sufficient'] else '否'}",
            "",
            "## 4. 次日上涨概率检验",
            "",
            f"- 条件样本次日上涨概率：{_fmt_pct(result['event_up_probability'])}",
            f"- 全样本次日上涨概率：{_fmt_pct(result['baseline_up_probability'])}",
            f"- 概率提升：{_fmt_pct(result['probability_lift'])}",
            "",
            "## 5. 次日平均收益检验",
            "",
            f"- 条件样本次日平均收益：{_fmt_pct(result['event_mean_return'])}",
            f"- 全样本次日平均收益：{_fmt_pct(result['baseline_mean_return'])}",
            f"- 平均收益提升：{_fmt_pct(result['mean_return_lift'])}",
            "",
            "## 6. 分年份稳定性",
            "",
        ]
    )
    yearly = result.get("yearly_stats")
    if isinstance(yearly, pd.DataFrame) and not yearly.empty:
        lines.append(_dataframe_to_markdown(yearly))
    else:
        lines.append("分年份样本不足，暂不展示。")

    lines.extend(
        [
            "",
            "## 7. 诊断结论",
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


def save_condition_report(report: str, path: str = "outputs/condition_research_report.md") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(report, encoding="utf-8")


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    display_cols = [
        "group",
        "event_count",
        "event_up_probability",
        "baseline_up_probability",
        "probability_lift",
        "event_mean_return",
        "baseline_mean_return",
        "mean_return_lift",
    ]
    data = df[[col for col in display_cols if col in df.columns]].copy()
    for col in data.columns:
        if col not in {"group", "event_count"}:
            data[col] = data[col].map(_fmt_pct)
    headers = list(data.columns)
    rows = data.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
