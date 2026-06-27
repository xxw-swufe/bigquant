# ETF 因子实现审计矩阵

说明：
- 运行时事实来源于 `src/etf_factor_library.py`。
- `local_status` / `bigquant_status` 是保守审计结果，不等于回测时的最终可执行结果。

| 因子 | 知识状态 | 实现状态 | 本地状态 | BigQuant 状态 | 依赖字段 | 额外上下文 | 支持后端 | 选择组 | 选择优先级 | 阻塞原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum_5d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | absolute_momentum | 50 | none |
| momentum_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | absolute_momentum | 50 | none |
| momentum_60d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | absolute_momentum | 50 | none |
| ma_gap_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | trend | 50 | none |
| ma_gap_60d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | trend | 50 | none |
| trend_strength | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | trend | 50 | none |
| relative_strength_20d | known | implemented | missing_context | missing_context | date, instrument, close, open, high, low, volume, amount, benchmark_close, benchmark_date | benchmark_series | local, bigquant | relative_momentum | 50 | missing_context |
| relative_strength_60d | known | implemented | missing_context | unsupported_backend | date, instrument, close, open, high, low, volume, amount, benchmark_close, benchmark_date | benchmark_series | local | relative_momentum | 50 | missing_context |
| amount_ratio_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | volume_price | 50 | none |
| volume_ratio_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | volume_price | 50 | none |
| liquidity_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | volume_price | 50 | none |
| volatility_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | risk | 50 | none |
| volatility_60d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | risk | 50 | none |
| risk_adjusted_momentum_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | risk_adjusted | 50 | none |
| drawdown_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | risk | 50 | none |
| price_volume_corr_20d | known | not_implemented | not_implemented | not_implemented | date, instrument, close, open, high, low, volume, amount | - | - | price_volume | 50 | recognized_not_implemented |
| price_volume_confirm_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | volume_price | 50 | none |
| intraday_strength | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | kbar | 50 | none |
| volume_price_divergence_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | volume_price | 50 | none |
| volume_breakout_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | volume_price | 50 | none |
| obv_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | volume_pressure | 50 | none |
| obv_trend_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | volume_pressure | 50 | none |
| reversal_5d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | reversal | 50 | none |
| ma_deviation_reversal_20d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | reversal | 50 | none |
| rsi_14d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | reversal | 50 | none |
| new_high_distance_60d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | trend | 50 | none |
| breakout_60d | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | trend | 50 | none |
| ma_alignment_score | known | implemented | available | available | date, instrument, close, open, high, low, volume, amount | - | local, bigquant | trend | 50 | none |
| downside_volatility_20d | known | implemented | available | unsupported_backend | date, instrument, close, open, high, low, volume, amount | - | local | risk | 50 | none |
| beta_60d | known | implemented | missing_context | unsupported_backend | date, instrument, close, open, high, low, volume, amount, benchmark_close, benchmark_date | benchmark_series | local | risk | 50 | missing_context |
| crowding_amount_ratio_5d | known | implemented | available | available | date, instrument, amount | - | local, bigquant | crowding | 50 | none |
| short_term_overheat_5d | known | implemented | available | available | date, instrument | - | local, bigquant | crowding | 50 | none |
| industry_pe_percentile | known | not_implemented | not_implemented | not_implemented | date, instrument | - | - | misc | 50 | recognized_not_implemented |
| industry_pb_percentile | known | not_implemented | not_implemented | not_implemented | date, instrument | - | - | misc | 50 | recognized_not_implemented |
| weighted_roe | known | not_implemented | not_implemented | not_implemented | date, instrument | constituent_data | - | misc | 50 | recognized_not_implemented |
| weighted_profit_growth | known | not_implemented | not_implemented | not_implemented | date, instrument | constituent_data | - | misc | 50 | recognized_not_implemented |
