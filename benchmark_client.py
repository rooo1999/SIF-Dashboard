"""Fetch Nifty 50 / Nifty 500 daily close levels via yfinance, with the same
incremental disk-cache pattern as upvaly_client.py."""

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "nav_cache"
CACHE_DIR.mkdir(exist_ok=True)

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 500": "^CRSLDX",
}

FLOOR_DATE = dt.date(2020, 1, 1)


def _cache_file(label: str) -> Path:
    safe = label.replace(" ", "_")
    return CACHE_DIR / f"benchmark_{safe}.csv"


def get_benchmark_history(label: str) -> pd.DataFrame:
    """Returns a (date, nav) DataFrame of daily close prices for the given
    benchmark label ('Nifty 50' / 'Nifty 500'), using an incremental cache."""
    ticker = BENCHMARKS[label]
    cache_file = _cache_file(label)

    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        cached = pd.DataFrame(columns=["date", "nav"])

    today = dt.date.today()
    fetch_start = FLOOR_DATE if cached.empty else (cached["date"].max() + pd.Timedelta(days=1)).date()

    if fetch_start <= today:
        try:
            data = yf.download(
                ticker,
                start=fetch_start.isoformat(),
                end=(today + dt.timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=False,
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data["Close"][ticker]
                else:
                    close = data["Close"]
                new_df = pd.DataFrame({"date": close.index, "nav": close.values})
                new_df["date"] = pd.to_datetime(new_df["date"])
                cached = (
                    pd.concat([cached, new_df], ignore_index=True)
                    .drop_duplicates("date")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                cached.to_csv(cache_file, index=False)
        except Exception:  # noqa: BLE001
            pass  # fall back to cached data

    return cached
