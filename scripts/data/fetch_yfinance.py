"""
Simple Yahoo Finance OHLCV downloader.

Usage example:
    python scripts/data/fetch_yfinance.py --symbol VOO --out data/raw/VOO.csv
    python scripts/data/fetch_yfinance.py --symbol PENDLE-USD --out data/raw/PENDLE.csv
"""

import argparse
import sys
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
except ImportError as exc:
    sys.exit(
        "Missing dependency 'yfinance' or 'pandas'. "
        "Activate your venv then run: pip install yfinance pandas\n"
        f"Details: {exc}"
    )

def main():
    parser = argparse.ArgumentParser(description="Download OHLCV from Yahoo Finance.")
    parser.add_argument("--symbol", required=True, help="Yahoo Finance Symbol (e.g. VOO, PENDLE-USD)")
    parser.add_argument("--period", default="max", help="Data period to download (default: max)")
    parser.add_argument("--interval", default="1d", help="Data interval (default: 1d)")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    
    args = parser.parse_args()
    
    print(f"Fetching {args.symbol} ({args.period}, {args.interval}) from Yahoo Finance...")
    
    # Download data
    df = yf.download(args.symbol, period=args.period, interval=args.interval, progress=False, auto_adjust=False)
    
    if df.empty:
        print(f"No data found for {args.symbol}.")
        return 1
    
    # Format DataFrame to match project convention (lowercase, Date index)
    # yfinance returns: Date (Index), Open, High, Low, Close, Adj Close, Volume
    # We want: Date (Index), open, high, low, close, volume (and maybe adj close)
    
    # Check if MultiIndex columns (common in new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename columns to lowercase
    df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close"
    }, inplace=True)
    
    # Ensure index name is 'Date' or 'datetime'
    df.index.name = 'Date'
    
    # Select columns
    cols = ['open', 'high', 'low', 'close', 'volume']
    if 'adj_close' in df.columns:
        cols.append('adj_close')
        
    df = df[cols]
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_path)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Last date: {df.index[-1]}")

if __name__ == "__main__":
    main()
