"""Fetch ETF daily data from AKShare and persist it as local parquet."""

from __future__ import annotations

import importlib
import os
import json
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.data_features import ensure_etf_feature_columns, standardize_etf_ohlcv_frame


DEFAULT_PROXY_PATCH_GATEWAY = "101.201.173.125"
DEFAULT_PROXY_PATCH_HOOK_DOMAINS = [
    "fund.eastmoney.com",
    "push2.eastmoney.com",
    "push2his.eastmoney.com",
    "emweb.securities.eastmoney.com",
    "searchapi.eastmoney.com/api/suggest/get",
]


def list_akshare_etf_symbols(
    limit: int | None = None,
    *,
    data_source: str = "auto",
    proxy_patch_token: str | None = None,
    proxy_patch_gateway: str = DEFAULT_PROXY_PATCH_GATEWAY,
    proxy_patch_hook_domains: Sequence[str] | None = None,
    proxy_patch_retry: int = 30,
    proxy_patch_fast: bool = True,
) -> list[str]:
    """Fetch ETF symbols from a supported upstream universe."""
    source = (data_source or "auto").lower().strip()
    if source in {"auto", "eastmoney", "em", "proxy_patch"}:
        try:
            if source == "proxy_patch":
                _maybe_install_proxy_patch(
                    token=proxy_patch_token,
                    gateway=proxy_patch_gateway,
                    hook_domains=proxy_patch_hook_domains,
                    retry=proxy_patch_retry,
                    fast=proxy_patch_fast,
                )
            ak = _import_akshare()
            with _temporary_proxy_disable():
                spot = ak.fund_etf_spot_em()
            code_col = _first_existing_column(spot, ["代码", "基金代码", "symbol", "code"])
            if code_col is not None:
                symbols = [str(value).strip() for value in spot[code_col].dropna().astype(str).tolist()]
                if limit is not None:
                    symbols = symbols[:limit]
                return symbols
        except Exception:
            if source == "proxy_patch":
                raise
            if source != "auto":
                raise
    if source in {"auto", "sina"}:
        ak = _import_akshare()
        with _temporary_proxy_disable():
            spot = ak.fund_etf_category_sina(symbol="ETF基金")
        code_col = _first_existing_column(spot, ["代码", "symbol", "code"])
        if code_col is None:
            raise ValueError("Sina ETF category data does not expose a known code column.")
        symbols = [str(value).strip() for value in spot[code_col].dropna().astype(str).tolist()]
        if limit is not None:
            symbols = symbols[:limit]
        return symbols
    raise ValueError(f"Unsupported data_source for ETF symbol discovery: {data_source}")


def fetch_akshare_etf_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "qfq",
    data_source: str = "auto",
    proxy_patch_token: str | None = None,
    proxy_patch_gateway: str = DEFAULT_PROXY_PATCH_GATEWAY,
    proxy_patch_hook_domains: Sequence[str] | None = None,
    proxy_patch_retry: int = 30,
    proxy_patch_fast: bool = True,
) -> pd.DataFrame:
    """Fetch one ETF's daily history from AKShare."""
    source = (data_source or "auto").lower().strip()
    if source in {"auto", "eastmoney", "em", "proxy_patch"}:
        try:
            data = _fetch_etf_daily_eastmoney(
                symbol,
                start_date,
                end_date,
                adjust=adjust,
                use_proxy_patch=(source == "proxy_patch"),
                proxy_patch_token=proxy_patch_token,
                proxy_patch_gateway=proxy_patch_gateway,
                proxy_patch_hook_domains=proxy_patch_hook_domains,
                proxy_patch_retry=proxy_patch_retry,
                proxy_patch_fast=proxy_patch_fast,
            )
            if not data.empty:
                return data
        except Exception:
            if source == "proxy_patch":
                raise
            if source != "auto":
                raise
    if source in {"auto", "sina"}:
        data = _fetch_etf_daily_sina(symbol, start_date, end_date, adjust=adjust)
        if not data.empty:
            return data
    return pd.DataFrame()


def build_local_etf_parquet(
    symbols: Sequence[str] | None = None,
    *,
    start_date: str,
    end_date: str,
    output_path: str = "data/parquet/local_etf_daily.parquet",
    metadata_path: str | None = "data/parquet/local_etf_metadata.json",
    adjust: str = "qfq",
    limit: int | None = None,
    data_source: str = "auto",
    proxy_patch_token: str | None = None,
    proxy_patch_gateway: str = DEFAULT_PROXY_PATCH_GATEWAY,
    proxy_patch_hook_domains: Sequence[str] | None = None,
    proxy_patch_retry: int = 30,
    proxy_patch_fast: bool = True,
    sleep_seconds: float = 0.0,
    universe_metadata: Sequence[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Fetch multiple ETF symbols and write a parquet snapshot."""
    if symbols is None:
        symbols = list_akshare_etf_symbols(
            limit=limit,
            data_source=data_source,
            proxy_patch_token=proxy_patch_token,
            proxy_patch_gateway=proxy_patch_gateway,
            proxy_patch_hook_domains=proxy_patch_hook_domains,
            proxy_patch_retry=proxy_patch_retry,
            proxy_patch_fast=proxy_patch_fast,
        )
    else:
        symbols = list(symbols)
        if limit is not None:
            symbols = symbols[:limit]

    frames: list[pd.DataFrame] = []
    failed: list[dict[str, str]] = []
    succeeded: list[str] = []
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{total}] fetching {symbol} ...", flush=True)
        try:
            frame = fetch_akshare_etf_daily(
                symbol,
                start_date,
                end_date,
                adjust=adjust,
                data_source=data_source,
                proxy_patch_token=proxy_patch_token,
                proxy_patch_gateway=proxy_patch_gateway,
                proxy_patch_hook_domains=proxy_patch_hook_domains,
                proxy_patch_retry=proxy_patch_retry,
                proxy_patch_fast=proxy_patch_fast,
            )
        except Exception as exc:
            failed.append({"symbol": str(symbol), "error": f"{type(exc).__name__}: {exc}"})
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue
        if frame.empty:
            failed.append({"symbol": str(symbol), "error": "empty_frame"})
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue
        frames.append(frame)
        succeeded.append(str(symbol))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not frames:
        if data_source == "proxy_patch" and failed:
            first_error = failed[0]["error"]
            raise ValueError(
                f"Proxy patch mode failed to fetch ETF data. First error: {first_error}"
            )
        raise ValueError("No ETF data was fetched from AKShare.")

    data = pd.concat(frames, ignore_index=True)
    data = standardize_etf_ohlcv_frame(data, source=None, adjust_type=adjust)
    if "amount" not in data.columns and {"close", "volume"}.issubset(data.columns):
        data["amount"] = data["close"] * data["volume"]
    data = ensure_etf_feature_columns(data)
    data = data.sort_values(["code", "date"]).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)

    if metadata_path is not None:
        metadata = {
            "source": "akshare",
            "adjust": adjust,
            "data_source": data_source,
            "proxy_patch_enabled": data_source == "proxy_patch",
            "proxy_patch_gateway": proxy_patch_gateway if data_source == "proxy_patch" else None,
            "start_date": start_date,
            "end_date": end_date,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "symbol_count": len(symbols),
            "succeeded_symbol_count": len(succeeded),
            "row_count": int(len(data)),
            "failed_symbols": failed,
            "succeeded_symbols": succeeded,
            "universe_metadata": list(universe_metadata or []),
            "output_path": str(output),
            "schema_version": "v1",
        }
        meta_path = Path(metadata_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return data


def _normalize_hist_frame(raw: pd.DataFrame, *, symbol: str, adjust_type: str) -> pd.DataFrame:
    data = raw.copy()
    rename_map = {}
    for source, target in [
        ("日期", "date"),
        ("开盘", "open"),
        ("最高", "high"),
        ("最低", "low"),
        ("收盘", "close"),
        ("成交量", "volume"),
        ("成交额", "amount"),
        ("换手率", "turn"),
        ("基金代码", "code"),
        ("代码", "code"),
        ("名称", "name"),
        ("名称", "name"),
    ]:
        if source in data.columns and target not in data.columns:
            rename_map[source] = target
    data = data.rename(columns=rename_map)
    if "date" not in data.columns:
        raise ValueError("AKShare ETF daily data did not include a date column.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = symbol
    data["instrument"] = symbol
    if "name" not in data.columns:
        data["name"] = None
    for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["source"] = "eastmoney"
    data["adjust_type"] = adjust_type
    return data


def _fetch_etf_daily_eastmoney(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str,
    use_proxy_patch: bool = False,
    proxy_patch_token: str | None = None,
    proxy_patch_gateway: str = DEFAULT_PROXY_PATCH_GATEWAY,
    proxy_patch_hook_domains: Sequence[str] | None = None,
    proxy_patch_retry: int = 30,
    proxy_patch_fast: bool = True,
) -> pd.DataFrame:
    if use_proxy_patch:
        _maybe_install_proxy_patch(
            token=proxy_patch_token,
            gateway=proxy_patch_gateway,
            hook_domains=proxy_patch_hook_domains,
            retry=proxy_patch_retry,
            fast=proxy_patch_fast,
        )
    ak = _import_akshare()
    start_fmt = _to_yyyymmdd(start_date)
    end_fmt = _to_yyyymmdd(end_date)
    with _temporary_proxy_disable():
        raw = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_fmt,
            end_date=end_fmt,
            adjust=adjust,
        )
    if raw.empty:
        return raw
    return _normalize_hist_frame(raw, symbol=symbol, adjust_type=adjust)


def _fetch_etf_daily_sina(symbol: str, start_date: str, end_date: str, *, adjust: str) -> pd.DataFrame:
    ak = _import_akshare()
    sina_symbol = _to_sina_symbol(symbol)
    with _temporary_proxy_disable():
        raw = ak.fund_etf_hist_sina(symbol=sina_symbol)
    if raw.empty:
        return raw
    data = raw.copy()
    if "date" not in data.columns:
        return pd.DataFrame()
    data = data.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = symbol
    data["instrument"] = symbol
    data["name"] = data.get("name")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["source"] = "sina"
    data["adjust_type"] = adjust
    data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))]
    if data.empty:
        return data
    return data


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _to_yyyymmdd(date_text: str) -> str:
    value = str(date_text).strip().replace("-", "")
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid date format: {date_text}")
    return value


def _to_sina_symbol(symbol: str) -> str:
    code = str(symbol).strip()
    if code.startswith(("sh", "sz")):
        return code
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "akshare is required for local ETF data fetching. Install it in the local environment first."
        ) from exc
    return ak


def _maybe_install_proxy_patch(
    *,
    token: str | None,
    gateway: str = DEFAULT_PROXY_PATCH_GATEWAY,
    hook_domains: Sequence[str] | None = None,
    retry: int = 30,
    fast: bool = True,
) -> bool:
    """Install akshare-proxy-patch if available and configured."""
    token = token or os.getenv("AKSHARE_PROXY_PATCH_TOKEN")
    if not token:
        raise ValueError(
            "proxy_patch mode requires proxy_patch_token or AKSHARE_PROXY_PATCH_TOKEN."
        )
    try:
        patch = importlib.import_module("akshare_proxy_patch")
    except ImportError as exc:
        raise ImportError(
            "akshare-proxy-patch is not installed. Run `pip install akshare-proxy-patch` first."
        ) from exc

    domains = list(hook_domains) if hook_domains else list(DEFAULT_PROXY_PATCH_HOOK_DOMAINS)
    patch.install_patch(
        gateway,
        auth_token=token,
        retry=retry,
        hook_domains=domains,
        fast=fast,
    )
    return True


@contextmanager
def _temporary_proxy_disable():
    """Temporarily clear common proxy env vars for direct market-data access."""
    proxy_keys = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ]
    saved = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
