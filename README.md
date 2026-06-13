# AutoETF Research Agent

面向行业 ETF 轮动的条件事件研究与多因子扩展系统。

本项目用于 BigQuant 比赛场景。第一版 MVP 先聚焦一个最小可验证问题：当 ETF 单日满足“量比 > 1、换手率 < 5、今日上涨”时，下一交易日上涨概率是否高于基准概率。代码结构会保留多因子研究扩展入口，后续可升级为单因子检验、因子标准化、多因子综合评分、Top 10% ETF 回测和权重诊断。

## 研究主线

```text
用户自然语言问题
→ 条件规则结构化
→ BigQuant ETF 数据获取
→ 计算量比、换手率、当日收益、次日收益标签
→ 筛选满足条件的 ETF-日期事件样本
→ 统计次日上涨概率与全样本基准概率
→ 分年份稳定性检验
→ 条件有效性诊断
→ 生成 Markdown 研究报告
```

后续多因子扩展主线：

```text
用户自然语言想法
→ 多个 ETF 候选因子
→ 单因子 IC / 分层收益检验
→ 因子标准化
→ 综合得分
→ Top 10% ETF 回测
→ 权重合理性诊断
```

## 项目结构

```text
autoetf-research-agent/
├── notebooks/
│   └── AutoETF_Research_Agent_MVP.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── agent_parser.py
│   ├── condition_parser.py
│   ├── condition_analysis.py
│   ├── condition_report.py
│   ├── factor_generator.py
│   ├── data_probe_bigquant.py
│   ├── data_loader_bigquant.py
│   ├── factor_analysis.py
│   ├── scoring.py
│   ├── backtest.py
│   ├── diagnosis.py
│   ├── report.py
│   └── plotting.py
├── docs/
│   ├── 方案说明.md
│   └── AI应用说明表.md
├── outputs/
├── tests/
├── run_condition_mvp_demo.py
├── README.md
└── .gitignore
```

## BigQuant 说明

BigQuant 特有代码集中在：

- `src/data_probe_bigquant.py`
- `src/data_loader_bigquant.py`

其他模块保持普通 Python / pandas 风格，方便本地维护和测试。

## 本地验证

本地 synthetic demo 只用于验证流程，不代表真实研究结论：

```bash
python3 run_condition_mvp_demo.py
```

正式研究必须在 BigQuant 中使用平台 ETF 行情数据。

## 免责声明

本系统仅用于量化研究与历史回测分析，不构成任何投资建议。历史回测结果不代表未来收益。
