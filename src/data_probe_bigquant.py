"""BigQuant table probing helpers."""

from pathlib import Path

import pandas as pd


def probe_table(
    table_name: str,
    limit: int = 5,
    start_date: str = "2023-01-01",
    end_date: str = "2023-02-01",
) -> pd.DataFrame:
    """Probe a BigQuant table and return a sample DataFrame."""
    from bigquant import dai

    sql = f"""
    SELECT *
    FROM {table_name}
    WHERE date >= '{start_date}'
      AND date <= '{end_date}'
    ORDER BY date, instrument
    LIMIT {limit}
    """
    df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
    print(f"\n===== {table_name} =====")
    print("shape:", df.shape)
    print("columns:")
    print(list(df.columns))
    return df


def run_data_probe(
    output_path: str = "outputs/table_columns.md",
    start_date: str = "2023-01-01",
    end_date: str = "2023-02-01",
) -> dict[str, pd.DataFrame]:
    """Probe candidate ETF tables. Failures are reported without stopping."""
    tables = [
        "cn_fund_bar1d",
        "cn_fund_real_bar1d",
        "cn_fund_nav",
        "cn_fund_static_data",
        "cn_stock_index_bar1d",
        "cn_index_bar1d",
    ]
    results = {}
    lines = ["# BigQuant Table Probe", ""]
    for table in tables:
        try:
            df = probe_table(table, start_date=start_date, end_date=end_date)
            results[table] = df
            lines.extend([f"## {table}", "", f"- shape: {df.shape}", ""])
            lines.append("```text")
            lines.extend(map(str, df.columns))
            lines.append("```")
            lines.append("")
        except Exception as exc:
            print(f"[WARN] failed to probe {table}: {exc}")
            lines.extend([f"## {table}", "", f"- failed: {exc}", ""])

    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return results
