"""Quick BTC SMA crossover example for personal experiments."""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import BTCUSD, SMA


class SmaCross(Strategy):
    fast_n = 3
    slow_n = 6

    def init(self):
        price = self.data.Close
        self.fast = self.I(SMA, price, self.fast_n)
        self.slow = self.I(SMA, price, self.slow_n)

    def next(self):
        if crossover(self.fast, self.slow):
            self.buy()
        elif crossover(self.slow, self.fast):
            self.sell()


def main():
    bt = Backtest(BTCUSD, SmaCross, cash=10_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot()


if __name__ == "__main__":
    main()
