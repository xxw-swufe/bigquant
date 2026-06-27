"""ETF factor reference library adapted from open-source quant ideas.

The factors in this file are not direct copies of Qlib, WorldQuant 101 Alphas,
or Alphalens. They are simplified ETF-friendly factor templates inspired by:

- Qlib Alpha158 / Alpha360: rolling return, moving average, volatility, volume,
  price-volume relationship, and window features.
- WorldQuant 101 Alphas: formulaic alpha style using rank, lag, rolling mean,
  rolling std, rolling correlation, and price-volume relations.
- Alphalens: factor validation concepts such as IC, quantile return, long-short
  spread, factor turnover, and group analysis.
- ETF / Sector Rotation: momentum, relative strength, volatility control,
  Top-K / Top-percent selection, periodic rebalancing, and equal weighting.

All formulas are written as BigQuant-oriented expressions or pandas hints.
Future returns are labels only and are intentionally excluded from this library.
"""

from __future__ import annotations

import re


LOCAL_SUPPORTED_FACTOR_NAMES = {
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "ma_gap_20d",
    "ma_gap_60d",
    "trend_strength",
    "relative_strength_20d",
    "relative_strength_60d",
    "amount_ratio_20d",
    "volume_ratio_20d",
    "liquidity_20d",
    "volatility_20d",
    "volatility_60d",
    "risk_adjusted_momentum_20d",
    "drawdown_20d",
    "price_volume_confirm_20d",
    "intraday_strength",
    "volume_price_divergence_20d",
    "volume_breakout_20d",
    "obv_20d",
    "obv_trend_20d",
    "reversal_5d",
    "ma_deviation_reversal_20d",
    "rsi_14d",
    "new_high_distance_60d",
    "breakout_60d",
    "ma_alignment_score",
    "downside_volatility_20d",
    "crowding_amount_ratio_5d",
    "short_term_overheat_5d",
    "beta_60d",
}

BIGQUANT_SUPPORTED_FACTOR_NAMES = {
    "momentum_20d",
    "trend_strength",
    "relative_strength_20d",
    "amount_ratio_20d",
    "volatility_20d",
    "volume_ratio_20d",
    "volume_price_divergence_20d",
    "volume_breakout_20d",
    "reversal_5d",
    "ma_deviation_reversal_20d",
    "new_high_distance_60d",
    "breakout_60d",
    "ma_alignment_score",
    "crowding_amount_ratio_5d",
    "short_term_overheat_5d",
}

BENCHMARK_REQUIRED_FACTOR_NAMES = {
    "relative_strength_20d",
    "relative_strength_60d",
    "beta_60d",
}

SELECTION_GROUP_MAP = {
    "momentum_5d": "absolute_momentum",
    "momentum_20d": "absolute_momentum",
    "momentum_60d": "absolute_momentum",
    "ma_gap_20d": "trend",
    "ma_gap_60d": "trend",
    "trend_strength": "trend",
    "ma_alignment_score": "trend",
    "breakout_60d": "trend",
    "new_high_distance_60d": "trend",
    "relative_strength_20d": "relative_momentum",
    "relative_strength_60d": "relative_momentum",
    "amount_ratio_20d": "volume_price",
    "volume_ratio_20d": "volume_price",
    "liquidity_20d": "volume_price",
    "volume_price_divergence_20d": "volume_price",
    "volume_breakout_20d": "volume_price",
    "price_volume_confirm_20d": "volume_price",
    "obv_20d": "volume_pressure",
    "obv_trend_20d": "volume_pressure",
    "volatility_20d": "risk",
    "volatility_60d": "risk",
    "risk_adjusted_momentum_20d": "risk_adjusted",
    "drawdown_20d": "risk",
    "downside_volatility_20d": "risk",
    "crowding_amount_ratio_5d": "crowding",
    "short_term_overheat_5d": "crowding",
    "reversal_5d": "reversal",
    "ma_deviation_reversal_20d": "reversal",
    "rsi_14d": "reversal",
    "beta_60d": "risk",
    "intraday_strength": "kbar",
}

SELECTION_PRIORITY_MAP = {
    "momentum_20d": 100,
    "trend_strength": 95,
    "relative_strength_20d": 90,
    "amount_ratio_20d": 90,
    "volatility_20d": 90,
    "momentum_5d": 70,
    "momentum_60d": 70,
    "ma_gap_20d": 85,
    "ma_gap_60d": 75,
    "volume_ratio_20d": 80,
    "price_volume_confirm_20d": 80,
    "breakout_60d": 80,
    "new_high_distance_60d": 75,
    "risk_adjusted_momentum_20d": 85,
    "drawdown_20d": 70,
    "rsi_14d": 70,
    "obv_20d": 70,
    "obv_trend_20d": 70,
}


BASE_FACTOR_METADATA = {
    "knowledge_status": "known",
    "implementation_status": None,
    "implementation_key": None,
    "asset_scope": ["ETF"],
    "default_period": None,
    "optional_periods": [],
    "data_dependency": "ohlcv",
    "required_fields": [],
    "required_context": [],
    "supported_backends": [],
    "correlated_with": [],
    "is_redundant_with": [],
    "enter_final_score": True,
    "default_weight": None,
    "selection_group": None,
    "selection_priority": 50,
    "missing_value_handling": "drop_or_fill",
    "outlier_handling": "clip_rank",
    "standardization": "cross_section_rank",
    "research_usage": ["condition", "factor_score", "report"],
    "score_shape": "linear",
    "optimal_range": None,
    "layer": "derived_factor",
    "ths_name": None,
    "aliases": [],
    "is_raw_indicator": False,
    "is_derived_factor": True,
    "is_composite_score": False,
}


_CORE_REQUIRED_COLUMNS = {
    "ohlcv": ["date", "instrument", "close", "open", "high", "low", "volume", "amount"],
    "benchmark": ["date", "instrument", "close"],
    "fundamental": ["date", "instrument"],
    "platform_field": ["date", "instrument"],
    "etf_specific": ["date", "instrument"],
}


_DEFAULT_CORRELATED_GROUPS = {
    "momentum_5d": ["roc_5d", "ma_gap_20d", "trend_strength"],
    "momentum_20d": ["roc_20d", "ma_gap_20d", "trend_strength", "momentum_rank_20d"],
    "momentum_60d": ["roc_60d", "ma_gap_60d", "trend_strength"],
    "amount_ratio_20d": ["volume_ratio_20d", "price_volume_confirm_20d", "crowding_amount_ratio_5d"],
    "volatility_20d": ["volatility_60d", "risk_adjusted_momentum_20d", "drawdown_20d"],
}


_SYNONYM_HINTS = {
    "momentum": ["涨幅", "动量", "ROC", "阶段涨幅"],
    "relative_strength": ["相对强弱", "RS", "跑赢", "超额"],
    "amount_ratio": ["成交额倍数", "AMO", "放量"],
    "volume_ratio": ["成交量倍数", "VOL", "量比"],
    "ma_gap": ["均线", "乖离率", "MA"],
    "volatility": ["波动率", "STD", "振幅"],
    "reversal": ["反转", "超跌", "超卖", "修复"],
    "trend": ["趋势", "多头", "强势"],
    "risk": ["风险", "过热", "回撤"],
    "valuation": ["估值", "PE", "PB"],
    "quality": ["质量", "ROE"],
    "growth": ["成长", "利润增长"],
    "crowding": ["拥挤", "过热", "资金拥挤"],
    "breakout": ["突破", "新高", "平台"],
}

_RAW_ETF_FACTOR_LIBRARY: list[dict] = [
    {
        "name": "momentum_5d",
        "description": "ETF 过去 5 个交易日短期价格动量",
        "formula": "close / m_lag(close, 5) - 1",
        "direction": "higher_better",
        "category": "momentum",
        "factor_type": "动量因子",
        "source_ideas": ["Qlib Alpha158 ROC", "Sector Rotation momentum"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "momentum_20d",
        "description": "ETF 过去 20 个交易日中短期价格动量",
        "formula": "close / m_lag(close, 20) - 1",
        "direction": "higher_better",
        "category": "momentum",
        "factor_type": "动量因子",
        "source_ideas": ["Qlib Alpha158 ROC", "Sector Rotation momentum"],
        "bigquant_ready": True,
        "mvp_default": True,
        "uses_future_data": False,
    },
    {
        "name": "momentum_60d",
        "description": "ETF 过去 60 个交易日中期价格动量",
        "formula": "close / m_lag(close, 60) - 1",
        "direction": "higher_better",
        "category": "momentum",
        "factor_type": "动量因子",
        "source_ideas": ["Qlib Alpha158 ROC", "Sector Rotation momentum"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "ma_gap_20d",
        "description": "收盘价相对 20 日均线的位置",
        "formula": "close / m_avg(close, 20) - 1",
        "direction": "higher_better",
        "category": "trend",
        "factor_type": "趋势因子",
        "source_ideas": ["Qlib Alpha158 MA"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "ma_gap_60d",
        "description": "收盘价相对 60 日均线的位置",
        "formula": "close / m_avg(close, 60) - 1",
        "direction": "higher_better",
        "category": "trend",
        "factor_type": "趋势因子",
        "source_ideas": ["Qlib Alpha158 MA"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "trend_strength",
        "description": "5 日均线相对 20 日均线的趋势强度",
        "formula": "m_avg(close, 5) / m_avg(close, 20) - 1",
        "direction": "higher_better",
        "category": "trend",
        "factor_type": "趋势因子",
        "source_ideas": ["Qlib moving average features", "ETF Rotation trend filter"],
        "bigquant_ready": True,
        "mvp_default": True,
        "uses_future_data": False,
    },
    {
        "name": "relative_strength_20d",
        "description": "ETF 过去 20 日收益相对基准指数的超额强度",
        "formula": "etf_return_20d - benchmark_return_20d",
        "direction": "higher_better",
        "category": "relative_strength",
        "factor_type": "动量因子",
        "source_ideas": ["Sector Rotation relative strength"],
        "bigquant_ready": False,
        "pandas_hint": "需要先合并基准指数 20 日收益",
        "mvp_default": True,
        "uses_future_data": False,
    },
    {
        "name": "relative_strength_60d",
        "description": "ETF 过去 60 日收益相对基准指数的中期超额强度",
        "formula": "etf_return_60d - benchmark_return_60d",
        "direction": "higher_better",
        "category": "relative_strength",
        "factor_type": "动量因子",
        "source_ideas": ["Sector Rotation relative strength"],
        "bigquant_ready": False,
        "pandas_hint": "需要先合并基准指数 60 日收益",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "amount_ratio_20d",
        "description": "当日成交额相对过去 20 日均值的放大程度",
        "formula": "amount / m_avg(amount, 20)",
        "direction": "higher_better",
        "category": "liquidity",
        "factor_type": "量价因子",
        "source_ideas": ["Qlib Alpha158 volume features", "WorldQuant price-volume relationship"],
        "bigquant_ready": True,
        "mvp_default": True,
        "uses_future_data": False,
    },
    {
        "name": "volume_ratio_20d",
        "description": "当日成交量相对过去 20 日均值的放大程度",
        "formula": "volume / m_avg(volume, 20)",
        "direction": "higher_better",
        "category": "liquidity",
        "factor_type": "量价因子",
        "source_ideas": ["Qlib Alpha158 VMA", "WorldQuant volume relationship"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "liquidity_20d",
        "description": "过去 20 日平均成交额，用于流动性过滤或评分",
        "formula": "m_avg(amount, 20)",
        "direction": "higher_better",
        "category": "liquidity",
        "factor_type": "风险因子",
        "source_ideas": ["ETF Rotation liquidity filter"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "volatility_20d",
        "description": "过去 20 日收益率波动率，作为风险惩罚因子",
        "formula": "m_std(close / m_lag(close, 1) - 1, 20)",
        "direction": "lower_better",
        "category": "risk",
        "factor_type": "风险因子",
        "source_ideas": ["Qlib Alpha158 STD", "Sector Rotation volatility control"],
        "bigquant_ready": True,
        "mvp_default": True,
        "uses_future_data": False,
    },
    {
        "name": "volatility_60d",
        "description": "过去 60 日收益率波动率，衡量中期风险",
        "formula": "m_std(close / m_lag(close, 1) - 1, 60)",
        "direction": "lower_better",
        "category": "risk",
        "factor_type": "风险因子",
        "source_ideas": ["Qlib Alpha158 STD", "Sector Rotation volatility control"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "risk_adjusted_momentum_20d",
        "description": "20 日动量除以 20 日波动率，衡量单位风险趋势收益",
        "formula": "(close / m_lag(close, 20) - 1) / (m_std(close / m_lag(close, 1) - 1, 20) + 0.001)",
        "direction": "higher_better",
        "category": "risk_adjusted",
        "factor_type": "风险因子",
        "source_ideas": ["Sector Rotation momentum-volatility", "WorldQuant formulaic composition"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "drawdown_20d",
        "description": "收盘价相对过去 20 日最高价的回撤幅度",
        "formula": "close / m_max(close, 20) - 1",
        "direction": "higher_better",
        "category": "risk",
        "factor_type": "风险因子",
        "source_ideas": ["Qlib rolling MAX", "ETF Rotation drawdown control"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "price_volume_corr_20d",
        "description": "过去 20 日价格收益与成交额变化的滚动相关性",
        "formula": "m_corr(close / m_lag(close, 1) - 1, amount / m_lag(amount, 1) - 1, 20)",
        "direction": "higher_better",
        "category": "price_volume",
        "factor_type": "量价因子",
        "source_ideas": ["Qlib Alpha158 CORR/CORD", "WorldQuant rolling correlation"],
        "bigquant_ready": False,
        "pandas_hint": "若 DAI 不支持 m_corr，可在 pandas 中 groupby instrument 后 rolling corr",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "price_volume_confirm_20d",
        "description": "20 日动量与成交额放大共同确认的价量因子",
        "formula": "(close / m_lag(close, 20) - 1) * (amount / m_avg(amount, 20))",
        "direction": "higher_better",
        "category": "price_volume",
        "factor_type": "量价因子",
        "source_ideas": ["WorldQuant price-volume relationship", "ETF Rotation confirmation"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "intraday_strength",
        "description": "收盘价在当日高低价区间中的位置，衡量日内强弱",
        "formula": "(close - low) / (high - low + 0.001)",
        "direction": "higher_better",
        "category": "kbar",
        "factor_type": "量价因子",
        "source_ideas": ["Qlib K-bar features", "WorldQuant formulaic OHLC features"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "volume_price_divergence_20d",
        "description": "成交额明显放大但 20 日价格涨幅不高，捕捉量价背离或潜在补涨",
        "formula": "(amount / m_avg(amount, 20)) / (abs(close / m_lag(close, 20) - 1) + 0.001)",
        "direction": "higher_better",
        "category": "price_volume",
        "factor_type": "量价因子",
        "source_ideas": ["WorldQuant price-volume relationship", "量价背离"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "volume_breakout_20d",
        "description": "成交额放大且价格站上 20 日均线，衡量放量突破",
        "formula": "(amount / m_avg(amount, 20)) * (close / m_avg(close, 20) - 1)",
        "direction": "higher_better",
        "category": "breakout",
        "factor_type": "量价因子",
        "source_ideas": ["量价突破", "ETF Rotation trend confirmation"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "obv_20d",
        "description": "20 日成交量能量（OBV）变化，用于衡量量价同向累积",
        "formula": "m_sum(sign(close - m_lag(close, 1)) * volume, 20)",
        "direction": "higher_better",
        "category": "volume_pressure",
        "factor_type": "量价因子",
        "source_ideas": ["OBV", "量能确认", "volume pressure"],
        "bigquant_ready": False,
        "pandas_hint": "若 DAI 不支持 sign，可在 pandas 中先构造日收益方向后滚动求和",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "obv_trend_20d",
        "description": "20 日 OBV 趋势强度，衡量成交量能量的持续抬升或下行",
        "formula": "m_avg(m_sum(sign(close - m_lag(close, 1)) * volume, 5), 20)",
        "direction": "higher_better",
        "category": "volume_pressure",
        "factor_type": "趋势因子",
        "source_ideas": ["OBV trend", "量能趋势"],
        "bigquant_ready": False,
        "pandas_hint": "建议先计算 OBV 序列，再做 20 日均线",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "reversal_5d",
        "description": "过去 5 日收益的反向值，用于检验短期超跌反弹",
        "formula": "-1 * (close / m_lag(close, 5) - 1)",
        "direction": "higher_better",
        "category": "reversal",
        "factor_type": "反转因子",
        "source_ideas": ["短期反转", "WorldQuant delay/rank style"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "ma_deviation_reversal_20d",
        "description": "价格低于 20 日均线越多，反转得分越高，用于检验均值回归",
        "formula": "-1 * (close / m_avg(close, 20) - 1)",
        "direction": "higher_better",
        "category": "reversal",
        "factor_type": "反转因子",
        "source_ideas": ["均线偏离回归", "Qlib MA"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "rsi_14d",
        "description": "14 日 RSI 强弱指标，用于识别超买超卖；ETF 中建议作为辅助诊断",
        "formula": "100 * m_avg(greatest(close / m_lag(close, 1) - 1, 0), 14) / (m_avg(abs(close / m_lag(close, 1) - 1), 14) + 0.001)",
        "direction": "neutral",
        "category": "reversal",
        "factor_type": "反转因子",
        "source_ideas": ["RSI", "technical reversal"],
        "bigquant_ready": False,
        "pandas_hint": "若 DAI 不支持 greatest，可在 pandas 中计算上涨收益均值 / 绝对收益均值",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "new_high_distance_60d",
        "description": "当前价格距离 60 日新高的距离，越接近新高代表趋势越强",
        "formula": "close / m_max(close, 60) - 1",
        "direction": "higher_better",
        "category": "trend",
        "factor_type": "动量因子",
        "source_ideas": ["新高距离", "Qlib rolling MAX", "breakout momentum"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "breakout_60d",
        "description": "价格相对过去 60 日最高价的位置，用于识别通道突破",
        "formula": "close / m_max(high, 60) - 1",
        "direction": "higher_better",
        "category": "breakout",
        "factor_type": "趋势因子",
        "source_ideas": ["Donchian breakout", "Qlib rolling MAX"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "ma_alignment_score",
        "description": "均线多头排列强度：5 日均线高于 20 日和 60 日均线时得分更高",
        "formula": "(m_avg(close, 5) / m_avg(close, 20) - 1) + (m_avg(close, 20) / m_avg(close, 60) - 1)",
        "direction": "higher_better",
        "category": "trend",
        "factor_type": "趋势因子",
        "source_ideas": ["均线多头排列", "Qlib MA"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "downside_volatility_20d",
        "description": "过去 20 日下跌收益的波动率，用于衡量下行风险",
        "formula": "m_std(least(close / m_lag(close, 1) - 1, 0), 20)",
        "direction": "lower_better",
        "category": "risk",
        "factor_type": "风险因子",
        "source_ideas": ["downside volatility", "risk control"],
        "bigquant_ready": False,
        "pandas_hint": "若 DAI 不支持 least，可在 pandas 中将正收益截断为 0 后 rolling std",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "beta_60d",
        "description": "ETF 过去 60 日相对基准收益的 Beta，衡量对大盘波动的敏感度",
        "formula": "cov(etf_return_1d, benchmark_return_1d, 60) / var(benchmark_return_1d, 60)",
        "direction": "lower_better",
        "category": "risk",
        "factor_type": "风险因子",
        "source_ideas": ["Beta risk", "Alphalens group/risk diagnostics"],
        "bigquant_ready": False,
        "pandas_hint": "需要先合并基准指数收益，再按 instrument 计算 rolling cov / rolling var",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "crowding_amount_ratio_5d",
        "description": "短期成交额暴增程度，用于识别资金拥挤和过热风险",
        "formula": "m_avg(amount, 5) / m_avg(amount, 20)",
        "direction": "lower_better",
        "category": "crowding",
        "factor_type": "风险因子",
        "source_ideas": ["拥挤度", "volume crowding"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "short_term_overheat_5d",
        "description": "短期涨幅过大程度，用于识别追高拥挤风险",
        "formula": "close / m_lag(close, 5) - 1",
        "direction": "lower_better",
        "category": "crowding",
        "factor_type": "风险因子",
        "source_ideas": ["短期过热", "crowding control"],
        "bigquant_ready": True,
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "industry_pe_percentile",
        "description": "ETF 跟踪行业指数 PE 的历史分位数，分位越低估值越便宜",
        "formula": "industry_pe_percentile",
        "direction": "lower_better",
        "category": "valuation",
        "factor_type": "估值因子",
        "source_ideas": ["value factor", "industry valuation"],
        "bigquant_ready": False,
        "pandas_hint": "需要行业指数估值或 ETF 成分股加权估值数据",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "industry_pb_percentile",
        "description": "ETF 跟踪行业指数 PB 的历史分位数，分位越低估值越便宜",
        "formula": "industry_pb_percentile",
        "direction": "lower_better",
        "category": "valuation",
        "factor_type": "估值因子",
        "source_ideas": ["value factor", "industry valuation"],
        "bigquant_ready": False,
        "pandas_hint": "需要行业指数估值或 ETF 成分股加权估值数据",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "weighted_roe",
        "description": "ETF 成分股加权 ROE，用于衡量行业 ETF 的盈利质量",
        "formula": "sum(component_weight * component_roe)",
        "direction": "higher_better",
        "category": "quality",
        "factor_type": "质量因子",
        "source_ideas": ["quality factor", "component weighted fundamentals"],
        "bigquant_ready": False,
        "pandas_hint": "需要 ETF 成分股权重和个股 ROE 数据",
        "mvp_default": False,
        "uses_future_data": False,
    },
    {
        "name": "weighted_profit_growth",
        "description": "ETF 成分股加权净利润增长率，用于衡量行业成长性",
        "formula": "sum(component_weight * component_profit_growth)",
        "direction": "higher_better",
        "category": "growth",
        "factor_type": "成长因子",
        "source_ideas": ["growth factor", "component weighted fundamentals"],
        "bigquant_ready": False,
        "pandas_hint": "需要 ETF 成分股权重和个股利润增长数据",
        "mvp_default": False,
        "uses_future_data": False,
    },
]


def _infer_default_period(name: str) -> int | None:
    match = re.search(r"_(\d+)d$", name)
    return int(match.group(1)) if match else None


def _infer_aliases(factor: dict) -> list[str]:
    aliases = []
    name = factor.get("name", "")
    description = factor.get("description", "")
    category = factor.get("category", "")
    ths_name = factor.get("ths_name")
    if ths_name:
        aliases.append(ths_name)
    if name.startswith("momentum_"):
        aliases.extend(["涨幅", "阶段涨幅", "ROC"])
    elif name.startswith("relative_strength_"):
        aliases.extend(["相对强弱", "RS", "超额收益"])
    elif name.startswith("amount_ratio_"):
        aliases.extend(["成交额倍数", "AMO", "放量"])
    elif name.startswith("volume_ratio_"):
        aliases.extend(["成交量倍数", "VOL", "量比"])
    elif name.startswith("ma_gap_"):
        aliases.extend(["均线", "乖离率", "MA"])
    elif name.startswith("obv_"):
        aliases.extend(["OBV", "能量潮", "量能"])
    elif name.startswith("volatility_"):
        aliases.extend(["波动率", "STD"])
    elif category == "reversal":
        aliases.extend(["反转", "超跌", "超卖"])
    elif category == "trend":
        aliases.extend(["趋势", "多头", "强势"])
    elif category == "risk":
        aliases.extend(["风险", "过热", "回撤"])
    elif category == "valuation":
        aliases.extend(["估值", "PE", "PB"])
    elif category == "quality":
        aliases.extend(["质量", "ROE"])
    elif category == "growth":
        aliases.extend(["成长", "利润增长"])
    elif category == "crowding":
        aliases.extend(["拥挤", "过热"])
    elif category == "breakout":
        aliases.extend(["突破", "新高"])

    aliases.extend(_SYNONYM_HINTS.get(category, []))
    aliases.extend([token for token in re.split(r"[_\s]+", name) if token])
    aliases.extend([token for token in re.split(r"[\s，,。；;（）()]+", description) if token])
    cleaned = []
    seen = set()
    for alias in aliases:
        alias = alias.strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        cleaned.append(alias)
    return cleaned


def _infer_correlated_with(name: str, category: str) -> list[str]:
    correlated = list(_DEFAULT_CORRELATED_GROUPS.get(name, []))
    if category == "momentum":
        correlated.extend(["trend_strength", "ma_gap_20d", "ma_gap_60d"])
    elif category == "liquidity":
        correlated.extend(["amount_ratio_20d", "volume_ratio_20d"])
    elif category == "risk":
        correlated.extend(["volatility_20d", "drawdown_20d"])
    elif category == "reversal":
        correlated.extend(["rsi_14d", "ma_deviation_reversal_20d"])
    elif category == "trend":
        correlated.extend(["ma_alignment_score", "breakout_60d", "momentum_20d"])
    elif category == "crowding":
        correlated.extend(["short_term_overheat_5d", "crowding_amount_ratio_5d"])
    return list(dict.fromkeys([item for item in correlated if item != name]))


def _infer_score_shape(factor: dict) -> str:
    direction = factor.get("direction", "higher_better")
    category = factor.get("category", "")
    name = factor.get("name", "")
    if name in {"turnover_rate", "volume_ratio_intraday", "rsi_14d"}:
        return "range_better"
    if category in {"crowding", "risk"} and direction == "lower_better":
        return "inverted_u"
    if category == "reversal":
        return "u_shape"
    return "linear"


def _infer_bigquant_ready(factor: dict) -> bool | str:
    if factor.get("pandas_hint"):
        return "Depends"
    if factor.get("data_dependency") in {"platform_field", "external_data", "fundamental", "etf_specific", "constituent_data"}:
        return "Depends"
    return bool(factor.get("bigquant_ready", False))


def _infer_data_dependency(factor: dict) -> str:
    if factor.get("category") in {"valuation", "quality", "growth", "fundamental"}:
        return "fundamental"
    if factor.get("category") in {"crowding", "fund_flow", "market_environment"}:
        return "platform_field"
    if factor.get("name") in {"etf_premium_discount", "iopv_deviation", "fund_size", "tracking_error", "bid_ask_spread", "turnover_amount_20d", "liquidity_score", "top10_weight_concentration", "industry_concentration", "constituent_momentum", "constituent_volatility", "constituent_valuation_pe", "constituent_valuation_pb"}:
        return "etf_specific"
    if factor.get("name") in {"main_money_inflow", "main_buy_sell_strength", "large_order_net_volume", "large_order_net_amount", "bbd_net_inflow", "institution_activity", "institution_research_heat", "main_crowding_score"}:
        return "platform_field"
    return factor.get("data_dependency", "ohlcv")


def _infer_required_context(factor: dict) -> list[str]:
    name = factor.get("name", "")
    context = list(factor.get("required_context", []))
    if name in BENCHMARK_REQUIRED_FACTOR_NAMES:
        context.append("benchmark_series")
    if factor.get("category") in {"quality", "growth"}:
        context.append("constituent_data")
    return list(dict.fromkeys(context))


def _infer_supported_backends(factor: dict) -> list[str]:
    name = factor.get("name", "")
    supported = []
    if name in LOCAL_SUPPORTED_FACTOR_NAMES:
        supported.append("local")
    if name in BIGQUANT_SUPPORTED_FACTOR_NAMES:
        supported.append("bigquant")
    return supported


def _infer_implementation_status(factor: dict, supported_backends: list[str]) -> str:
    if supported_backends:
        return "implemented"
    if factor.get("knowledge_status") == "known":
        return "not_implemented"
    return "unknown"


def _infer_selection_group(factor: dict) -> str:
    name = factor.get("name", "")
    if name in SELECTION_GROUP_MAP:
        return SELECTION_GROUP_MAP[name]
    category = factor.get("category", "")
    if category in {"momentum", "trend", "liquidity", "risk", "reversal", "crowding", "breakout", "price_volume", "volume_pressure"}:
        return category
    if category in {"relative_strength"}:
        return "relative_momentum"
    return "misc"


def _infer_selection_priority(factor: dict) -> int:
    name = factor.get("name", "")
    if name in SELECTION_PRIORITY_MAP:
        return SELECTION_PRIORITY_MAP[name]
    if factor.get("mvp_default"):
        return 80
    if factor.get("category") in {"trend", "momentum", "liquidity", "risk"}:
        return 60
    return 40


def _infer_layer(factor: dict) -> str:
    if factor.get("is_raw_indicator"):
        return "raw_data"
    if factor.get("data_dependency") in {"fundamental", "platform_field", "etf_specific", "constituent_data", "external_data"}:
        return "derived_factor"
    return factor.get("layer", "derived_factor")


def _infer_required_columns(factor: dict) -> list[str]:
    name = factor.get("name", "")
    dependency = factor.get("data_dependency", "ohlcv")
    required = list(_CORE_REQUIRED_COLUMNS.get(dependency, ["date", "instrument"]))
    formula = factor.get("formula", "")

    if dependency == "benchmark":
        required = ["date", "instrument", "close"]
    if "amount" in formula:
        required.append("amount")
    if "volume" in formula:
        required.append("volume")
    if "high" in formula:
        required.append("high")
    if "low" in formula:
        required.append("low")
    if "benchmark" in formula or name.startswith("relative_strength_") or name in {"beta_60d"}:
        required.extend(["benchmark_close", "benchmark_date"])
    if name in {"rsi_14d", "downside_volatility_20d", "beta_60d"}:
        required.append("close")

    cleaned = []
    seen = set()
    for column in required:
        if column not in seen:
            cleaned.append(column)
            seen.add(column)
    return cleaned


def _enrich_factor(factor: dict) -> dict:
    enriched = BASE_FACTOR_METADATA.copy()
    enriched.update(factor)
    enriched["implementation_key"] = enriched.get("implementation_key") or enriched["name"]
    enriched["default_period"] = enriched.get("default_period") or _infer_default_period(enriched["name"])
    enriched["aliases"] = enriched.get("aliases") or _infer_aliases(enriched)
    enriched["correlated_with"] = enriched.get("correlated_with") or _infer_correlated_with(
        enriched["name"], enriched.get("category", "")
    )
    enriched["is_redundant_with"] = enriched.get("is_redundant_with") or list(enriched["correlated_with"])
    enriched["score_shape"] = enriched.get("score_shape") or _infer_score_shape(enriched)
    enriched["data_dependency"] = _infer_data_dependency(enriched)
    enriched["required_columns"] = enriched.get("required_columns") or _infer_required_columns(enriched)
    enriched["required_fields"] = enriched.get("required_fields") or list(enriched["required_columns"])
    enriched["required_context"] = _infer_required_context(enriched)
    enriched["supported_backends"] = enriched.get("supported_backends") or _infer_supported_backends(enriched)
    enriched["implementation_status"] = enriched.get("implementation_status") or _infer_implementation_status(
        enriched, enriched["supported_backends"]
    )
    enriched["selection_group"] = enriched.get("selection_group") or _infer_selection_group(enriched)
    enriched["selection_priority"] = int(enriched.get("selection_priority") or _infer_selection_priority(enriched))
    enriched["bigquant_ready"] = _infer_bigquant_ready(enriched)
    enriched["layer"] = _infer_layer(enriched)
    if enriched.get("mvp_default") and enriched.get("default_weight") is None:
        enriched["default_weight"] = 1.0
    return enriched


def _build_factor_library() -> list[dict]:
    return [_enrich_factor(factor) for factor in _RAW_ETF_FACTOR_LIBRARY]


ETF_FACTOR_LIBRARY: list[dict] = _build_factor_library()


def _get_library() -> list[dict]:
    return [factor.copy() for factor in ETF_FACTOR_LIBRARY]


def get_default_factor_candidates() -> list[dict]:
    """Return the MVP default ETF factor candidates."""
    return [factor.copy() for factor in _get_library() if factor["mvp_default"]]


def get_factor_library() -> list[dict]:
    """Return all reference ETF factors."""
    return _get_library()


def find_factors_by_categories(categories: set[str]) -> list[dict]:
    """Return factors that belong to the requested categories."""
    return [
        factor.copy()
        for factor in _get_library()
        if factor["category"] in categories
    ]


def find_factors_by_keyword(keyword: str) -> list[dict]:
    """Return factors whose name, aliases, or description match a keyword."""
    lowered = keyword.strip().lower()
    if not lowered:
        return []
    results = []
    for factor in _get_library():
        haystacks = [
            factor.get("name", ""),
            factor.get("description", ""),
            " ".join(factor.get("aliases", [])),
            factor.get("ths_name", "") or "",
        ]
        if any(lowered in str(haystack).lower() for haystack in haystacks):
            results.append(factor)
    return results


def search_factors(query: str, limit: int = 10) -> list[dict]:
    """Search factors by token and alias matching."""
    normalized = query.strip()
    tokens = [token for token in re.split(r"[\s，,。；;]+", normalized) if token]
    if normalized:
        tokens = [normalized] + tokens
    if not tokens:
        return []
    candidates = []
    seen = set()
    normalized_variants = _build_query_variants(normalized)
    for token in tokens:
        for factor in find_factors_by_keyword(token):
            name = factor["name"]
            if name in seen:
                continue
            seen.add(name)
            candidates.append(factor)
            if len(candidates) >= limit:
                return candidates
    for variant in normalized_variants:
        for factor in find_factors_by_keyword(variant):
            name = factor["name"]
            if name in seen:
                continue
            seen.add(name)
            candidates.append(factor)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _build_query_variants(query: str) -> list[str]:
    variants = []
    if "OBV" in query.upper():
        variants.extend(["OBV", "能量潮", "量能"])
    if "缩量" in query:
        variants.append("缩量")
    if "放量" in query:
        variants.append("放量")
    if "上涨" in query:
        variants.append("上涨")
    if "下跌" in query:
        variants.append("下跌")
    if "趋势" in query:
        variants.append("趋势")
    if "动量" in query:
        variants.append("动量")
    if "相对强弱" in query or "强弱" in query:
        variants.append("相对强弱")
    return list(dict.fromkeys(variants))


def get_factor_templates(periods: list[int] | None = None) -> list[dict]:
    """Generate a small parameterized factor template set."""
    periods = periods or [5, 10, 20, 60, 120]
    templates = []
    for period in periods:
        templates.append(
            {
                "name": f"momentum_{period}d",
                "ths_name": f"{period}日涨幅",
                "aliases": [f"ROC{period}", f"阶段涨幅{period}日"],
                "layer": "derived_factor",
                "category": "momentum",
                "default_period": period,
                "optional_periods": periods,
                "formula": f"close / m_lag(close, {period}) - 1",
                "required_columns": ["close"],
                "data_dependency": "ohlcv",
                "direction": "higher_better",
                "score_shape": "linear",
                "optimal_range": None,
                "correlated_with": [f"roc_{period}d", f"ma_gap_{period}d"],
                "asset_scope": ["ETF"],
                "bigquant_ready": True,
                "mvp_default": period in {20, 60},
                "uses_future_data": False,
                "research_usage": ["condition", "factor_score", "report"],
            }
        )
    return templates
