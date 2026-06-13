"""BigQuant table probing helpers."""

from pathlib import Path

import pandas as pd


def probe_table(table_name: str, limit: int = 5) -> pd.DataFrame:
    """Probe a BigQuant table and return a sample DataFrame."""
    from bigquant import dai

    sql = f"""
    SELECT *
    FROM {table_name}
    ORDER BY date, instrument
    LIMIT {limit}
    """
    df = dai.query(sql).df()
    print(f"\n===== {table_name} =====")
    print("shape:", df.shape)
    print("columns:")
    print(list(df.columns))
    return df


def run_data_probe(output_path: str = "outputs/table_columns.md") -> dict[str, pd.DataFrame]:
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
            df = probe_table(table)
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

