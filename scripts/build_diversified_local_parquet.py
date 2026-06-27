#!/usr/bin/env python3
"""Build a diversified local ETF parquet snapshot via AKShare."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetch_akshare import build_local_etf_parquet
from src.etf_universe import diversified_etf_metadata, diversified_etf_symbols, symbols_by_sector


def _default_date_range(years: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=365 * years)
    return start.isoformat(), end.isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch diversified ETF parquet from AKShare.")
    parser.add_argument("--years", type=int, default=5, help="Lookback years (default: 5).")
    parser.add_argument("--start-date", default=None, help="Override start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Override end date YYYY-MM-DD.")
    parser.add_argument(
        "--output",
        default="data/parquet/local_etf_daily.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--metadata",
        default="data/parquet/local_etf_metadata.json",
        help="Output metadata json path.",
    )
    parser.add_argument(
        "--data-source",
        default="auto",
        choices=["auto", "eastmoney", "em", "sina", "proxy_patch"],
        help="AKShare upstream source.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.35, help="Pause between symbols.")
    args = parser.parse_args()

    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        start_date, end_date = _default_date_range(args.years)

    symbols = diversified_etf_symbols()
    print(f"Fetching {len(symbols)} diversified ETFs from {start_date} to {end_date} ...")
    print("Sectors:", ", ".join(f"{k}({len(v)})" for k, v in symbols_by_sector().items()))

    data = build_local_etf_parquet(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_path=args.output,
        metadata_path=args.metadata,
        data_source=args.data_source,
        sleep_seconds=args.sleep_seconds,
        universe_metadata=diversified_etf_metadata(),
    )

    print(
        f"Done. rows={len(data)} symbols={data['code'].nunique()} "
        f"dates={data['date'].min().date()}..{data['date'].max().date()} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
