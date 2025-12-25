"""Quick BTC SMA crossover example on monthly BTC USD data (legacy fixture).

Trading idea:
- Compute two simple moving averages (fast=3 months, slow=6 months) on BTC USD monthly data.
- When the fast MA crosses above the slow MA, go long (buy). When it crosses below, close and go short (sell).
- One position at a time; pay 0.1% commission per trade; exclusive orders avoid overlapping entries.
"""

from datetime import datetime
from pathlib import Path

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import BTCUSD, SMA


class SmaCross(Strategy):
    fast_n = 3
    slow_n = 6

    def init(self):
        price = self.data.Close
        # Precompute indicators once: 3‑month and 6‑month SMAs
        self.fast = self.I(SMA, price, self.fast_n)
        self.slow = self.I(SMA, price, self.slow_n)

    def next(self):
        # If fast MA crosses above slow -> bullish signal: close shorts, go long
        if crossover(self.fast, self.slow):
            self.buy()
        # If fast MA crosses below slow -> bearish signal: close longs, go short
        elif crossover(self.slow, self.fast):
            self.sell()


def main():
    # commission=0.1% round-trip per trade; exclusive_orders enforces single active position
    bt = Backtest(BTCUSD, SmaCross, cash=10_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()
    print(stats)

    # Save interactive chart to playground/results with timestamped filename to avoid overwrites
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / f"sma_1M_{datetime.now():%Y%m%d_%H%M%S}.html"
    bt.plot(filename=str(outfile), open_browser=False)
    print(f"Plot saved to {outfile.resolve()}")


if __name__ == "__main__":
    main()
