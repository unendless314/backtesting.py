"""
Simple Binance OHLCV downloader.

Usage example:
    source .venv/bin/activate
    pip install ccxt pandas
    python scripts/data/fetch_binance.py \
        --symbol BTC/USDT --tf 1d \
        --since 2013-01-01 --out data/raw/BTCUSDT_1d.csv

Supports Binance intervals (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h,
12h, 1d, 3d, 1w, 1M). Paginates automatically until `until` (or now).
Optionally resamples (e.g., 3D, 1W) with pandas before saving.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import pandas as pd

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    sys.exit(
        "Missing dependency 'ccxt'. Activate your venv then run: pip install ccxt\n"
        f"Details: {exc}"
    )


def parse_iso_date(value: str) -> int:
    """Return UTC milliseconds since epoch for an ISO-like date string."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_paginated(
    exchange: "ccxt.binance",
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int],
    limit: int = 1000,
) -> List[Sequence]:
    """Fetch OHLCV rows from Binance, paginating until `until_ms` or no more data."""
    all_rows: List[Sequence] = []
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)

        last_ts = batch[-1][0]
        if until_ms and last_ts >= until_ms:
            break

        # Prepare next page; add 1 ms to avoid duplicate last row
        since = last_ts + 1

        # Respect rate limit
        time.sleep(exchange.rateLimit / 1000)

        if len(batch) < limit:
            break
    return all_rows


def to_dataframe(rows: Iterable[Sequence]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.astype(float)
    return df.sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(rule, label="right", closed="right").agg(agg).dropna()


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, float_format="%.10f")
    print(f"Wrote {len(df):,} rows to {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download OHLCV from Binance via ccxt.")
    parser.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    parser.add_argument("--tf", "--timeframe", dest="timeframe", required=True,
                        help="Binance timeframe, e.g. 1d, 3d, 1w, 1h")
    parser.add_argument("--since", type=parse_iso_date, required=True,
                        help="Start date (ISO, assumed UTC if no tz). Example: 2013-01-01")
    parser.add_argument("--until", type=parse_iso_date, default=None,
                        help="End date (ISO). Defaults to now.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--resample", default=None,
                        help="Optional pandas rule to resample (e.g., 3D, 1W).")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max rows per API call (Binance allows up to 1000).")
    args = parser.parse_args(argv)

    until_ms = args.until or int(time.time() * 1000)

    exchange = ccxt.binance({"enableRateLimit": True})
    print(f"Fetching {args.symbol} {args.timeframe} from {args.since} to {until_ms} ...")
    rows = fetch_ohlcv_paginated(exchange, args.symbol, args.timeframe, args.since, until_ms, args.limit)
    df = to_dataframe(rows)
    if df.empty:
        print("No data returned; check symbol/timeframe/range.")
        return 1

    if args.resample:
        df = resample_ohlcv(df, args.resample)
        print(f"Resampled to {args.resample}, rows now {len(df):,}")

    save_csv(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
