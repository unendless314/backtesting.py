"""Benchmarks: lump-sum Buy & Hold vs. DCA on BTC/USDT daily data (last 2000 days).

Data
- Uses `data/raw/BTCUSDT_1d.csv` (UTC index, lowercase columns); generated via fetch_binance.py.

Setup
- total_budget: USD capital to compare (default 100_000).
- window: last 2000 daily bars.
- Lump-sum: invest total_budget on first day of window and hold.
- DCA: invest total_budget / window_days each day across the window.

Outputs
- Final value, total invested, Return %, CAGR %, Max Drawdown % for both methods.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure


DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "BTCUSDT_1d.csv"
WINDOW_DAYS = 2000
TOTAL_BUDGET = 100_000.0


def load_data(path: Path, window: int) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tzinfo:
        df.index = df.index.tz_convert(None)
    df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]
    return df.tail(window)


def drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    return float(dd)


def lump_sum(df: pd.DataFrame, budget: float) -> Tuple[float, pd.Series]:
    entry_price = df["Close"].iloc[0]
    btc = budget / entry_price
    equity_curve = btc * df["Close"]
    final_value = equity_curve.iloc[-1]
    return final_value, equity_curve


def dca(df: pd.DataFrame, budget: float) -> Tuple[float, pd.Series]:
    daily_invest = budget / len(df)
    btc_held = 0.0
    equity = []
    for price in df["Close"]:
        btc_held += daily_invest / price
        equity.append(btc_held * price)
    equity_curve = pd.Series(equity, index=df.index)
    final_value = equity_curve.iloc[-1]
    return final_value, equity_curve


def cagr(final: float, invested: float, days: int) -> float:
    years = days / 365.25
    if final <= 0 or invested <= 0 or years <= 0:
        return np.nan
    return (final / invested) ** (1 / years) - 1


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def main() -> None:
    df = load_data(DATA_PATH, WINDOW_DAYS)
    days = len(df)
    print(f"Window: last {days} days, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total budget: ${TOTAL_BUDGET:,.0f}")

    # Lump-sum
    ls_final, ls_eq = lump_sum(df, TOTAL_BUDGET)
    ls_ret = ls_final / TOTAL_BUDGET - 1
    ls_cagr = cagr(ls_final, TOTAL_BUDGET, days)
    ls_dd = drawdown(ls_eq)

    # DCA
    dca_final, dca_eq = dca(df, TOTAL_BUDGET)
    dca_ret = dca_final / TOTAL_BUDGET - 1
    dca_cagr = cagr(dca_final, TOTAL_BUDGET, days)
    dca_dd = drawdown(dca_eq)

    print("\nLump-sum Buy & Hold")
    print(f"  Final value: ${ls_final:,.2f}")
    print(f"  Return: {pct(ls_ret)} | CAGR: {pct(ls_cagr)} | Max DD: {pct(ls_dd)}")

    print("\nDCA (budget split evenly across window)")
    print(f"  Final value: ${dca_final:,.2f}")
    print(f"  Return: {pct(dca_ret)} | CAGR: {pct(dca_cagr)} | Max DD: {pct(dca_dd)}")

    # Save text summary
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    txt_out = out_dir / "benchmark_bh_dca_1d.txt"
    with txt_out.open("w") as f:
        f.write(f"Window: last {days} days, {df.index[0].date()} to {df.index[-1].date()}\n")
        f.write(f"Total budget: ${TOTAL_BUDGET:,.0f}\n\n")
        f.write("Lump-sum Buy & Hold\n")
        f.write(f"  Final value: ${ls_final:,.2f}\n")
        f.write(f"  Return: {pct(ls_ret)} | CAGR: {pct(ls_cagr)} | Max DD: {pct(ls_dd)}\n\n")
        f.write("DCA (budget split evenly across window)\n")
        f.write(f"  Final value: ${dca_final:,.2f}\n")
        f.write(f"  Return: {pct(dca_ret)} | CAGR: {pct(dca_cagr)} | Max DD: {pct(dca_dd)}\n")

    # Save HTML summary
    html_out = out_dir / "benchmark_bh_dca_1d.html"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BTC Buy & Hold vs DCA (last {days} days)</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; color: #222; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; margin-top: 16px; min-width: 420px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: right; }}
    th {{ background: #f5f5f5; }}
    td.label {{ text-align: left; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>BTC Buy &amp; Hold vs DCA</h1>
  <p>Window: last {days} days ({df.index[0].date()} → {df.index[-1].date()})<br>
     Total budget: ${TOTAL_BUDGET:,.0f}</p>
  <table>
    <tr><th></th><th>Final Value</th><th>Return</th><th>CAGR</th><th>Max DD</th></tr>
    <tr><td class="label">Lump-sum</td>
        <td>${ls_final:,.2f}</td><td>{pct(ls_ret)}</td><td>{pct(ls_cagr)}</td><td>{pct(ls_dd)}</td></tr>
    <tr><td class="label">DCA</td>
        <td>${dca_final:,.2f}</td><td>{pct(dca_ret)}</td><td>{pct(dca_cagr)}</td><td>{pct(dca_dd)}</td></tr>
  </table>
</body>
</html>
"""
    html_out.write_text(html)
    print(f"\nSaved summary to {txt_out.resolve()}")
    print(f"Saved HTML to {html_out.resolve()}")

    # Interactive equity curve chart (Bokeh)
    chart_out = out_dir / "benchmark_bh_dca_1d_chart.html"
    output_file(chart_out)
    src = ColumnDataSource({
        "date": df.index,
        "bh": ls_eq.values,
        "dca": dca_eq.values,
    })
    p = figure(
        x_axis_type="datetime",
        width=900,
        height=400,
        title=f"BTC Buy & Hold vs DCA (last {days} days)",
    )
    p.line("date", "bh", source=src, color="#2ca02c", legend_label="Lump-sum B&H")
    p.line("date", "dca", source=src, color="#1f77b4", legend_label="DCA")
    p.yaxis.axis_label = "Equity ($)"
    p.legend.location = "top_left"
    p.add_tools(HoverTool(
        tooltips=[("Date", "@date{%F}"), ("B&H", "@bh{$0,0}"), ("DCA", "@dca{$0,0}")],
        formatters={"@date": "datetime"},
        mode="vline",
    ))
    save(p)
    print(f"Saved chart to {chart_out.resolve()}")


if __name__ == "__main__":
    main()
