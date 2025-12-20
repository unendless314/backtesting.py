from __future__ import annotations

from pathlib import Path

import pandas as pd
from backtesting.lib import FractionalBacktest
from backtesting import Strategy

DATA_PATH = Path('data/raw/BTCUSDT_1d.csv')
RESULTS_DIR = Path('playground/results')
START_CAPITAL = 60_000.0
DCA_SPANS = [1, 100, 200, 300, 400, 500, 600, 800, 1000]


def read_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df.rename(columns=str.capitalize, inplace=True)
    df.set_index('Datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df[df.index >= '2017-09-01']


class DcaStrategy(Strategy):
    dca_span: int = 100
    start_capital: float = START_CAPITAL

    def init(self) -> None:
        self.ath = -float('inf')
        self.dca_active = False
        self.dca_run_days = 0
        self.daily_amount = self.start_capital / max(1, self.dca_span)
        # Precompute total bars so we can force an exit near the end
        self.total_bars = len(self.data.df)
        # Allow multiple DCA cycles, but only after a new ATH occurs
        self.allow_multiple_cycles = True
        self.last_cycle_ath = -float('inf')

    def next(self) -> None:
        price = self.data.Close[-1]
        if price > self.ath:
            self.ath = price
        drawdown = price / self.ath - 1 if self.ath > 0 else 0.0

        cash_available = self._broker._cash
        if self.dca_active:
            if drawdown >= -0.5 or self.dca_run_days >= self.dca_span or cash_available <= 0:
                self.dca_active = False
        elif drawdown <= -0.5 and cash_available > 0:
            # Start a new DCA cycle only after setting a new ATH since the last cycle
            if self.allow_multiple_cycles:
                can_start = self.ath > self.last_cycle_ath
            else:
                can_start = self.dca_run_days == 0
            if can_start:
                self.dca_active = True
                self.dca_run_days = 0
                self.last_cycle_ath = self.ath

        if self.dca_active and self.dca_run_days < self.dca_span and cash_available > 0:
            amount = min(self.daily_amount, cash_available)
            # Convert fixed dollar amount to integer contract units using scaled prices
            # FractionalBacktest scales prices by fractional_unit, so units stay integer-compatible
            units = int(round(amount / price))
            if units <= 0:
                self.dca_active = False
                return
            # Clamp cost to available cash (price is scaled, cash is real dollars)
            est_cost = units * price
            if est_cost > cash_available:
                units = int(cash_available // price)
            if units <= 0:
                self.dca_active = False
                return
            self.buy(size=units, tag='DCA')
            self.dca_run_days += 1
            cash_available = self._broker._cash
            if self.dca_run_days >= self.dca_span or cash_available <= 0:
                self.dca_active = False

        # Close any open position in the final bar (or penultimate, if desired)
        current_bar = len(self.data.Close)
        if current_bar >= self.total_bars - 1 and self.position:
            self.position.close()


def run_backtests() -> None:
    df = read_price_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for span in DCA_SPANS:
        strategy_cls = type(
            f'DcaSpan{span}',
            (DcaStrategy,),
            {'dca_span': span},
        )
        bt = FractionalBacktest(
            df,
            strategy_cls,
            cash=START_CAPITAL,
            fractional_unit=1 / 1e6,
            commission=0.0,
            exclusive_orders=False,
            trade_on_close=True,
            finalize_trades=False,
        )
        stats = bt.run()
        out_path = RESULTS_DIR / f'dca_span_{span}_backtest.html'
        bt.plot(filename=str(out_path))
        print(f'DCA span {span:<4} | Equity {stats["Equity Final [$]"]:.2f} | '
              f'Drawdown {stats["Max. Drawdown [%]"]:.2f} | '
              f'Plots -> {out_path}')


if __name__ == '__main__':
    run_backtests()
