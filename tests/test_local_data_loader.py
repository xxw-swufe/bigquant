from pathlib import Path

import pandas as pd

from src.data_access import load_condition_research_data, load_etf_data
from src.data_loader_local import load_etf_data_local


def _make_local_parquet(tmp_path: Path) -> Path:
    rows = []
    dates = pd.date_range("2024-01-01", periods=35, freq="B")
    for i, date in enumerate(dates):
        close = 100 + i
        rows.append(
            {
                "date": date,
                "code": "510300",
                "name": "ETF",
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + i,
                "amount": (1000 + i) * close,
                "turn": 1.0,
            }
        )
    parquet_path = tmp_path / "local_etf_daily.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    return parquet_path


def test_local_loader_computes_shared_features(tmp_path):
    parquet_path = _make_local_parquet(tmp_path)
    df = load_etf_data_local(parquet_path=str(parquet_path))
    assert {"code", "instrument", "return_1d", "amount_ratio_20d", "future_return_5d"}.issubset(df.columns)
    assert df["code"].nunique() == 1
    assert df["future_return_5d"].notna().any()


def test_backend_dispatch_loads_local_parquet(tmp_path):
    parquet_path = _make_local_parquet(tmp_path)
    df = load_etf_data(
        data_backend="local",
        parquet_path=str(parquet_path),
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    cond = load_condition_research_data(
        data_backend="local",
        parquet_path=str(parquet_path),
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    assert df.shape == cond.shape
    assert "future_return_1d" in df.columns
    assert "volume_ratio_20d" in cond.columns
