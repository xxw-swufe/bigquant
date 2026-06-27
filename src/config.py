"""Project configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ResearchConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    data_backend: str = "bigquant"
    akshare_data_source: str = "auto"
    akshare_proxy_patch_token: Optional[str] = None
    akshare_proxy_patch_gateway: str = "101.201.173.125"
    akshare_proxy_patch_hook_domains: list[str] = field(
        default_factory=lambda: [
            "fund.eastmoney.com",
            "push2.eastmoney.com",
            "push2his.eastmoney.com",
            "emweb.securities.eastmoney.com",
            "searchapi.eastmoney.com/api/suggest/get",
        ]
    )
    akshare_proxy_patch_retry: int = 30
    akshare_proxy_patch_fast: bool = True
    fund_table: str = "cn_fund_bar1d"
    local_etf_parquet: str = "data/parquet/local_etf_daily.parquet"
    local_benchmark_parquet: Optional[str] = None
    local_data_root: str = "data/parquet"
    benchmark_instrument: str = "000300.SH"
    rebalance_frequency: str = "W"
    top_pct: float = 0.10
    min_holdings: int = 3
    cost_bps: float = 10.0
    factor_cols: list[str] = field(
        default_factory=lambda: [
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
