"""Project configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    fund_table: str = "cn_fund_bar1d"
    benchmark_instrument: str = "000300.SH"
    rebalance_frequency: str = "W"
    top_pct: float = 0.10
    min_holdings: int = 3
    cost_bps: float = 10.0
    factor_cols: list[str] = field(
        default_factory=lambda: [
            "relative_strength_20d",
            "momentum_20d",
            "amount_ratio_20d",
            "trend_strength",
            "volatility_20d",
        ]
    )
    target_cols: list[str] = field(
        default_factory=lambda: ["future_return_5d", "future_return_20d"]
    )


DEFAULT_CONFIG = ResearchConfig()

