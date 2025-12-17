"""
Quick-and-safe IV band plotter.

Reads a daily OHLC CSV (e.g., data/raw/BTCUSDT_1d.csv), computes simple
historical-volatility–based price levels, and overlays them as scatter
points on the built-in Backtesting.py candlestick plot. The core library
is untouched; everything lives in this helper script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy


def hist_vol_levels(
    close: Sequence[float], lookback: int, horizon_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return +1σ and -1σ price levels based on rolling log-return volatility.
    Works with backtesting._Array (numpy-like) to avoid pandas dependency here.
    """
    close_np = np.asarray(close, dtype=float)
    log_ret = np.diff(np.log(close_np), prepend=np.nan)
    vol_T = (
        pd.Series(log_ret).rolling(lookback).std().to_numpy() * np.sqrt(horizon_days)
    )
    upper = close_np * np.exp(vol_T)
    lower = close_np * np.exp(-vol_T)
    return upper, lower


class IVDotsStrategy(Strategy):
    lookback: int = 30
    horizon_days: int = 7
    use_lines: bool = False

    def init(self) -> None:
        close = self.data.Close

        # σ-based levels (historical vol)
        upper, lower = hist_vol_levels(close, self.lookback, self.horizon_days)

        def sigma_points():
            return upper, lower

        self.I(
            sigma_points,
            overlay=True,
            scatter=not self.use_lines,
            color=("orange", "deepskyblue"),
            name=("IV+1σ", "IV-1σ"),
        )

    def next(self) -> None:  # noqa: D401
        """No trading logic; plotting only."""
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot IV-style scatter bands on Backtesting.py candlesticks."
    )
    p.add_argument(
        "--csv",
        default="data/raw/BTCUSDT_1d.csv",
        type=Path,
        help="Input OHLC CSV with columns datetime, open, high, low, close, volume",
    )
    p.add_argument("--lookback", type=int, default=30, help="Rolling window in days")
    p.add_argument("--horizon", type=int, default=7, help="Days to project (σ·√T)")
    p.add_argument(
        "--tf-days",
        type=int,
        default=1,
        help="Timeframe in days: 1=日K, 3=三日K, 7=週K (resample from daily)",
    )
    p.add_argument(
        "--open-browser",
        action="store_true",
        help="Open plot in browser (default saves only if filename given)",
    )
    p.add_argument(
        "--lines",
        action="store_true",
        help="Draw lines instead of scatter dots (Bollinger-style).",
    )
    p.add_argument(
        "--filename",
        default=None,
        help="Output HTML filename for the Bokeh plot; default auto = playground/results/iv_plot_<tfd>.html",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tf_days = max(1, args.tf_days)
    if args.filename is None:
        args.filename = f"playground/results/iv_plot_{tf_days}d.html"
    Path(args.filename).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["datetime"])
    df = df.rename(
        columns=str.title
    )  # ensure Backtesting expects Open/High/Low/Close/Volume
    df = df.set_index("Datetime")
    if tf_days > 1:
        rule = f"{tf_days}D"
        df = (
            df.resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )

    bt = Backtest(
        df,
        IVDotsStrategy,
        cash=1_000_000,
        commission=0.0,
        exclusive_orders=True,
    )

    stats = bt.run(
        lookback=args.lookback,
        horizon_days=args.horizon,
        use_lines=args.lines,
    )

    # Plot only what we need; disable equity/PL to focus on markers.
    bt.plot(
        filename=args.filename,
        open_browser=args.open_browser,
        plot_equity=False,
        plot_return=False,
        plot_pl=False,
        plot_drawdown=False,
        plot_volume=True,
        plot_trades=False,
        superimpose=False,
    )

    print("Done.")
    print(f"Stats keys: {list(stats.index)}")
    print(f"HTML saved to: {args.filename}")


if __name__ == "__main__":
    main()
