from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_PATH = Path('data/raw/BTCUSDT_1d.csv')
START_CAPITAL = 60_000.0
DCA_DAY_OPTIONS = [1, 100, 200, 300, 400, 500, 600, 800, 1000]


@dataclass
class DcaResult:
    dca_days: int
    activations: int
    invested: float
    equity: float
    cash: float
    shares: float


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df.rename(columns=str.capitalize, inplace=True)
    df.set_index('Datetime', inplace=True)
    return df[df.index >= '2017-09-01']


def simulate_dca(days: int, df: pd.DataFrame) -> DcaResult:
    ath = -float('inf')
    cash = START_CAPITAL
    shares = 0.0
    per_day_amount = START_CAPITAL / days
    dca_active = False
    dca_run_days = 0
    total_dca_days = 0
    activations = 0

    for price in df['Close']:
        if price <= 0:
            continue
        if price > ath:
            ath = price
        drawdown = price / ath - 1

        if dca_active and drawdown >= -0.5:
            dca_active = False
        elif not dca_active and drawdown <= -0.5 and cash > 0:
            dca_active = True
            activations += 1
            dca_run_days = 0

        if dca_active and dca_run_days < days and cash > 0:
            purchase = min(per_day_amount, cash)
            shares += purchase / price
            cash -= purchase
            dca_run_days += 1
            total_dca_days += 1
            if dca_run_days >= days or cash <= 0:
                dca_active = False

    last_price = df['Close'].iloc[-1]
    equity = cash + shares * last_price
    invested = START_CAPITAL - cash
    return DcaResult(
        dca_days=total_dca_days,
        activations=activations,
        invested=invested,
        equity=equity,
        cash=cash,
        shares=shares,
    )


def main() -> None:
    df = read_data()
    print('DCA simulation (start capital = $60,000)')
    print('days | activations | dca_days | invested | equity | cash | shares')
    for days in DCA_DAY_OPTIONS:
        result = simulate_dca(days, df)
        print(
            f'{days:4} | {result.activations:11} | {result.dca_days:8} | '
            f'{result.invested:8,.2f} | {result.equity:9,.2f} | '
            f'{result.cash:9,.2f} | {result.shares:9.4f}'
        )


if __name__ == '__main__':
    main()
