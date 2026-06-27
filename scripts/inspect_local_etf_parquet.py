#!/usr/bin/env python3
"""Inspect local ETF parquet snapshot and print a human-readable QA report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def inspect_local_etf_parquet(
    parquet_path: str | Path,
    metadata_path: str | Path | None = None,
    *,
    sample_rows: int = 5,
) -> dict:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    code_col = "code" if "code" in df.columns else "instrument"

    per_symbol = (
        df.groupby(code_col)
        .agg(
            rows=("date", "size"),
            start=("date", "min"),
            end=("date", "max"),
            close_na=("close", lambda s: int(pd.isna(s).sum())),
            volume_zero=("volume", lambda s: int((s.fillna(0) <= 0).sum())),
        )
        .reset_index()
        .sort_values(code_col)
    )

    dup_count = int(df.duplicated(subset=[code_col, "date"]).sum())
    required = ["date", code_col, "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required if col not in df.columns]

    report = {
        "parquet_path": str(parquet_path),
        "file_size_mb": round(parquet_path.stat().st_size / 1024 / 1024, 2),
        "row_count": int(len(df)),
        "symbol_count": int(df[code_col].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "duplicate_code_date_rows": dup_count,
        "missing_required_columns": missing_cols,
        "columns": list(df.columns),
        "symbols_with_issues": per_symbol[
            (per_symbol["close_na"] > 0) | (per_symbol["rows"] < 200)
        ].to_dict("records"),
    }

    if metadata_path is not None:
        meta_file = Path(metadata_path)
        if meta_file.exists():
            report["metadata"] = json.loads(meta_file.read_text(encoding="utf-8"))
            failed = report["metadata"].get("failed_symbols") or []
            report["failed_symbol_count"] = len(failed)

    print("=" * 60)
    print("AutoETF 本地 ETF 数据检查报告")
    print("=" * 60)
    print(f"文件: {report['parquet_path']}")
    print(f"大小: {report['file_size_mb']} MB")
    print(f"行数: {report['row_count']:,}")
    print(f"标的数: {report['symbol_count']}")
    print(f"日期: {report['date_min']} -> {report['date_max']}")
    print(f"列: {', '.join(report['columns'][:10])}{'...' if len(report['columns']) > 10 else ''}")
    print(f"重复 (code,date): {report['duplicate_code_date_rows']}")
    if report["missing_required_columns"]:
        print(f"缺少必要列: {report['missing_required_columns']}")
    else:
        print("必要 OHLCV 列: OK")

    if "metadata" in report:
        meta = report["metadata"]
        print("-" * 60)
        print("元数据 (local_etf_metadata.json)")
        print(f"  拉取时间: {meta.get('created_at')}")
        print(f"  配置区间: {meta.get('start_date')} -> {meta.get('end_date')}")
        print(f"  成功/计划: {meta.get('succeeded_symbol_count')}/{meta.get('symbol_count')}")
        print(f"  失败数: {len(meta.get('failed_symbols') or [])}")

    print("-" * 60)
    print("每只 ETF 行数（前 10 / 后 5）")
    preview = per_symbol[[code_col, "rows", "start", "end"]].copy()
    preview["start"] = preview["start"].dt.date.astype(str)
    preview["end"] = preview["end"].dt.date.astype(str)
    print(preview.head(10).to_string(index=False))
    if len(preview) > 15:
        print("...")
        print(preview.tail(5).to_string(index=False))

    rows_stats = per_symbol["rows"]
    print("-" * 60)
    print(f"每只行数: min={int(rows_stats.min())}, median={int(rows_stats.median())}, max={int(rows_stats.max())}")

    if report["symbols_with_issues"]:
        print("-" * 60)
        print("需留意的标的（收盘价缺失较多或历史较短）:")
        for item in report["symbols_with_issues"][:10]:
            print(
                f"  {item[code_col]}: rows={item['rows']}, close_na={item['close_na']}, "
                f"{item['start'].date()}..{item['end'].date()}"
            )
    else:
        print("-" * 60)
        print("未发现明显异常标的。")

    print("-" * 60)
    print(f"样例数据（最近 {sample_rows} 行）:")
    show_cols = [c for c in ["date", code_col, "name", "close", "volume", "amount"] if c in df.columns]
    print(df.sort_values(["date", code_col]).tail(sample_rows)[show_cols].to_string(index=False))
    print("=" * 60)

    ok = (
        not report["missing_required_columns"]
        and report["duplicate_code_date_rows"] == 0
        and report["symbol_count"] >= 30
        and report["row_count"] >= 10000
    )
    print("总体结论:", "通过基本质检" if ok else "请检查上述告警项")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local ETF parquet snapshot.")
    parser.add_argument(
        "--parquet",
        default="data/parquet/local_etf_daily.parquet",
        help="Parquet file path.",
    )
    parser.add_argument(
        "--metadata",
        default="data/parquet/local_etf_metadata.json",
        help="Metadata json path.",
    )
    parser.add_argument("--sample-rows", type=int, default=5)
    args = parser.parse_args()
    inspect_local_etf_parquet(args.parquet, args.metadata, sample_rows=args.sample_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
