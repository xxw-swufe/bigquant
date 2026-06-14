"""BigQuant ETF data loader."""

import pandas as pd


def load_etf_data_bigquant(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    table_name: str = "cn_fund_bar1d",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Load ETF factor data from BigQuant.

    Run `run_data_probe()` first and adjust table/field names according to the
    actual BigQuant environment.
    """
    from bigquant import dai

    sql = f"""
    SELECT
        date,
        instrument,
        open,
        high,
        low,
        close,
        {volume_col} AS volume,
        amount,
        close / m_lag(close, 20) - 1 AS momentum_20d,
        amount / m_avg(amount, 20) AS amount_ratio_20d,
        m_avg(close, 5) / m_avg(close, 20) - 1 AS trend_strength,
        m_stddev(close / m_lag(close, 1) - 1, 20) AS volatility_20d,
        m_lead(close, 5) / close - 1 AS future_return_5d,
        m_lead(close, 20) / close - 1 AS future_return_20d
    FROM {table_name}
    WHERE
        date >= '{start_date}'
        AND date <= '{end_date}'
    ORDER BY date, instrument
    """
    df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
    if df.empty:
        raise ValueError("Loaded ETF data is empty. Please check table name, date range, and permissions.")
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna()


def load_condition_research_data_bigquant(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    table_name: str = "cn_fund_bar1d",
    volume_col: str = "volume",
    turnover_col: str = "turn",
) -> pd.DataFrame:
    """Load the minimal ETF fields for the first condition-study MVP.

    The first MVP evaluates:
    volume_ratio_5d > 1, turnover < 5, return_1d > 0, and future_return_1d > 0.

    Run `run_data_probe()` first. If the BigQuant ETF table uses names such as
    `vol` or `turn`, pass them through `volume_col` and `turnover_col`.
    """
    from bigquant import dai

    sql = f"""
    SELECT
        date,
        instrument,
        close,
        {volume_col} AS volume,
        {turnover_col} AS turnover,
        {volume_col} / m_avg({volume_col}, 5) AS volume_ratio_5d,
        close / m_lag(close, 1) - 1 AS return_1d,
        m_lead(close, 1) / close - 1 AS future_return_1d
    FROM {table_name}
    WHERE
        date >= '{start_date}'
        AND date <= '{end_date}'
    ORDER BY date, instrument
    """
    df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
    if df.empty:
        raise ValueError("Loaded ETF condition-study data is empty. Please check table, date range, and permissions.")
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna()


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
