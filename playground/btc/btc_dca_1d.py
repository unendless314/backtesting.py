"""DCA vs Lump-sum visualized with Backtesting.py plot (single long position).

- Loads last 2000 daily bars from `data/raw/BTCUSDT_1d.csv`.
- DCA: invest equal cash each day; no shorts. Final day force-close to realize P/L.
- Uses Backtesting plot() so the HTML matches other strategy charts.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from backtesting import Strategy
from backtesting.lib import FractionalBacktest
from backtesting.test import SMA  # unused indicator placeholder

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "BTCUSDT_1d.csv"
WINDOW_DAYS = 2000
TOTAL_BUDGET = 100_000.0
COMMISSION = 0.001  # 0.1%


def load_data(path: Path, window: int) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tzinfo:
        df.index = df.index.tz_convert(None)
    df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]
    return df.tail(window)


class DcaHold(Strategy):
    daily_cash = TOTAL_BUDGET / WINDOW_DAYS

    def init(self):
        # Keep total length for detecting final bar
        self._total_bars = len(self.data.Close)
        # Optional dummy indicator to keep plot structure similar
        self.sma = self.I(SMA, self.data.Close, 50)

    def next(self):
        # On the penultimate bar, close everything so it settles on the final bar
        if len(self.data) == self._total_bars - 1:
            for trade in list(self.trades):
                trade.close()
            if self.position:
                self.position.close()
            return

        # skip if no cash (safety)
        if self.equity <= 0:
            return

        # invest fixed cash portion expressed as fraction of current equity
        fraction = self.daily_cash / self.equity
        fraction = max(0.0, min(0.99, fraction))
        if fraction > 0:
            self.buy(size=fraction)


def main():
    df = load_data(DATA_PATH, WINDOW_DAYS)
    bt = FractionalBacktest(
        df,
        DcaHold,
        cash=TOTAL_BUDGET,
        commission=COMMISSION,
        exclusive_orders=False,  # allow cumulative daily buys
        hedging=False,
        trade_on_close=True,  # allow final-day close to execute
        fractional_unit=1 / 1e6,  # allow μBTC sizing
    )
    stats = bt.run()
    print(stats)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / f"dca_bh_backtest_1d_{datetime.now():%Y%m%d_%H%M%S}.html"
    bt.plot(filename=str(outfile), open_browser=False)
    print(f"Plot saved to {outfile.resolve()}")


if __name__ == "__main__":
    main()
