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


def get_candle_close_ms(exchange: "ccxt.binance", timeframe: str, start_ms: int) -> int:
    """Return the exclusive close timestamp (UTC ms) for a candle starting at start_ms."""
    dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    if timeframe == "1M":
        # Calendar month ends at the 1st of next month 00:00:00 UTC
        year = dt.year + (1 if dt.month == 12 else 0)
        month = 1 if dt.month == 12 else dt.month + 1
        next_month_dt = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        return int(next_month_dt.timestamp() * 1000)
    tf_seconds = exchange.parse_timeframe(timeframe)
    return start_ms + int(tf_seconds * 1000)


def is_month_resample_rule(rule: str) -> bool:
    """Return True if rule is a calendar month-end frequency (e.g. 1M, 2ME)."""
    norm = rule.strip()
    norm_lower = norm.lower()
    # Explicitly exclude minutes (e.g. 15min, 15T).
    if "min" in norm_lower or norm.endswith(("T", "t", "m")):
        return False
    try:
        offset = pd.tseries.frequencies.to_offset(norm)
        return isinstance(offset, pd.tseries.offsets.MonthEnd)
    except Exception:
        return norm.upper() in {"M", "1M", "ME", "1ME"}


def get_resampled_bar_close_ms(
    ts: pd.Timestamp,
    rule: str,
    raw_timeframe: str,
    exchange: "ccxt.binance",
) -> int:
    """Return the exclusive closing timestamp (UTC ms) for a resampled bar."""
    if is_month_resample_rule(rule):
        # Month-end rules label the bar on the last day of that calendar month
        # (e.g. 2024-01-31).
        # The month closes at next month 1st 00:00:00 UTC.
        year = ts.year + (1 if ts.month == 12 else 0)
        month = 1 if ts.month == 12 else ts.month + 1
        next_month_dt = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        return int(next_month_dt.timestamp() * 1000)

    # For other rules, the label is the right edge of the bin (start of the last raw candle).
    # Its closing timestamp is when that last raw candle closes.
    try:
        raw_tf_seconds = exchange.parse_timeframe(raw_timeframe)
        close_ts = ts + pd.Timedelta(seconds=raw_tf_seconds)
        return int(close_ts.timestamp() * 1000)
    except Exception:
        offset = pd.tseries.frequencies.to_offset(rule)
        close_ts = ts + offset
        return int(close_ts.timestamp() * 1000)


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
    parser.add_argument(
        "--drop-unclosed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the last candle if it has not closed yet (default: True).",
    )
    args = parser.parse_args(argv)

    until_ms = args.until or int(time.time() * 1000)

    exchange = ccxt.binance({"enableRateLimit": True})
    print(f"Fetching {args.symbol} {args.timeframe} from {args.since} to {until_ms} ...")
    rows = fetch_ohlcv_paginated(
        exchange, args.symbol, args.timeframe, args.since, until_ms, args.limit
    )
    if args.drop_unclosed and rows:
        try:
            close_ms = get_candle_close_ms(exchange, args.timeframe, rows[-1][0])
            now_ms = int(time.time() * 1000)
            if now_ms < close_ms:
                unclosed_dt = pd.to_datetime(rows[-1][0], unit="ms", utc=True)
                print(f"Excluding unclosed candle starting at {unclosed_dt}")
                rows = rows[:-1]
        except Exception as exc:
            print(f"Warning: Could not check unclosed candle: {exc}")

    df = to_dataframe(rows)
    if df.empty:
        print("No data returned; check symbol/timeframe/range.")
        return 1

    if args.resample:
        raw_last_dt = df.index[-1] if not df.empty else None
        raw_last_ms = int(raw_last_dt.timestamp() * 1000) if raw_last_dt is not None else None
        raw_last_close_ms = (
            get_candle_close_ms(exchange, args.timeframe, raw_last_ms)
            if raw_last_ms is not None
            else None
        )

        df = resample_ohlcv(df, args.resample)
        print(f"Resampled to {args.resample}, rows now {len(df):,}")
        if args.drop_unclosed and not df.empty and raw_last_close_ms is not None:
            bar_close_ms = get_resampled_bar_close_ms(
                df.index[-1], args.resample, args.timeframe, exchange
            )
            now_ms = int(time.time() * 1000)
            target_limit_ms = min(args.until, now_ms) if args.until else now_ms

            if bar_close_ms > target_limit_ms or bar_close_ms > raw_last_close_ms:
                unclosed_label = df.index[-1]
                close_dt = pd.to_datetime(bar_close_ms, unit="ms", utc=True)
                print(
                    f"Excluding unclosed resampled candle at {unclosed_label} "
                    f"(closes at {close_dt})"
                )
                df = df.iloc[:-1]
                if df.empty:
                    print("No data remaining after excluding unclosed resampled candle.")
                    return 1

    save_csv(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
