"""BTC/USDT daily SMA crossover using local CSV data.

Data
- Expects daily OHLCV at `data/raw/BTCUSDT_1d.csv` (UTC index, lowercase columns).
- You can regenerate via `scripts/data/fetch_binance.py --symbol BTC/USDT --tf 1d --since 2017-09-01 --out data/raw/BTCUSDT_1d.csv`.

Strategy (long only)
- Fast SMA vs slow SMA (defaults 50/200 days).
- Bullish crossover → enter long.
- Bearish crossover → flat (take profit/stop) but DO NOT short.
Exclusive orders keep at most one position.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from backtesting.lib import FractionalBacktest


DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "BTCUSDT_1d.csv"


class SmaCrossDaily(Strategy):
    fast_n = 50
    slow_n = 200

    def init(self):
        price = self.data.Close
        self.fast = self.I(SMA, price, self.fast_n)
        self.slow = self.I(SMA, price, self.slow_n)

    def next(self):
        if crossover(self.fast, self.slow):
            self.buy()
        elif crossover(self.slow, self.fast) and self.position.is_long:
            self.position.close()


def load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tzinfo:
        df.index = df.index.tz_convert(None)
    df.index.name = None
    df = df.rename(columns=str.capitalize)  # open->Open, etc.
    # Ensure column order and presence
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def main():
    df = load_daily_csv(DATA_PATH)
    bt = FractionalBacktest(
        df,
        SmaCrossDaily,
        cash=100_000,
        commission=0.001,  # 0.1%
        exclusive_orders=True,
        # fractional_unit defaults to 1 satoshi (1e-8 BTC). Keep default.
    )
    stats = bt.run()
    print(stats)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / f"sma_1d_{datetime.now():%Y%m%d_%H%M%S}.html"
    bt.plot(filename=str(outfile), open_browser=False)
    print(f"Plot saved to {outfile.resolve()}")


if __name__ == "__main__":
    main()
