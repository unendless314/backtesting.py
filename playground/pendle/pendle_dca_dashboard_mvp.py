"""Pendle DCA dashboard that mirrors the BTC logic but ingests `data/raw/PENDLE.csv`. """

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, Range1d, Select, LinearAxis
from bokeh.plotting import figure, output_file, save

DATA_PATH = Path('data/raw/PENDLE.csv')
RESULTS_DIR = Path('playground/results')

START_CAPITAL = 180_000.0
SPANS = [100, 150, 200, 250, 300, 400, 450, 500, 600, 800, 900, 1000, 1200]
THRESHOLDS = [-10, -20, -30, -40, -50, -60, -70]  # percent, negative values


def find_peak_intervals(close: pd.Series, min_len: int = 100, top_n: int = 5) -> List[Tuple[str, int, int]]:
    """Return top_n longest ATH→ATH intervals (label, start_idx, end_idx)."""

    ath = -np.inf
    peak_indices = []
    for i, price in enumerate(close.values):
        if price > ath:
            ath = price
            peak_indices.append(i)

    intervals: List[Tuple[str, int, int, int]] = []  # (len, start, end, label_id)
    for a, b in zip(peak_indices, peak_indices[1:]):
        length = b - a + 1
        if length >= min_len:
            intervals.append((length, a, b, len(intervals)))

    # Pick top_n longest intervals
    intervals = sorted(intervals, key=lambda x: x[0], reverse=True)[:top_n]
    result: List[Tuple[str, int, int]] = []
    for length, a, b, _ in intervals:
        start_date = close.index[a].date()
        end_date = close.index[b].date()
        label = f'{start_date} → {end_date} ({length}d)'
        result.append((label, a, b))
    return result


def read_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    df.rename(columns=str.capitalize, inplace=True)
    df.set_index('Datetime', inplace=True)
    return df[df.index >= '2017-09-01'][['Close']]


def simulate_dca(
    close: pd.Series,
    span: int,
    threshold_pct: float,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Dict[str, float]]:
    """Simulate multi-cycle DCA.

    Buys on every day price drawdown <= threshold_pct (relative to rolling ATH).
    Does not buy when above threshold. Spends fixed daily_amount until cash is
    insufficient or span days reached. No re-start once cash is out.
    Returns equity series, drawdown series, buy records, and summary metrics.
    """

    ath = -np.inf
    dca_run_days = 0
    daily_amount = START_CAPITAL / max(1, span)
    cash = START_CAPITAL
    shares = 0.0

    equity = []
    drawdown = []
    buys: List[Tuple[pd.Timestamp, float, float, float]] = []

    peak_equity = -np.inf

    for ts, price in close.items():
        if price > ath:
            ath = price
        drawdown_pct = price / ath - 1 if ath > 0 else 0.0

        should_buy = (drawdown_pct <= threshold_pct) and cash > 0 and dca_run_days < span
        if should_buy:
            amount = min(daily_amount, cash)
            bought_shares = amount / price
            shares += bought_shares
            cash -= amount
            dca_run_days += 1
            buys.append((ts, price, bought_shares, amount))

        eq = cash + shares * price
        peak_equity = max(peak_equity, eq)
        equity.append(eq)
        dd = (eq / peak_equity - 1) if peak_equity > 0 else 0.0
        drawdown.append(dd)

    equity_series = pd.Series(equity, index=close.index)
    drawdown_series = pd.Series(drawdown, index=close.index)

    buy_df = pd.DataFrame(buys, columns=['Datetime', 'Price', 'Shares', 'Amount'])

    invested = START_CAPITAL - cash
    equity_final = equity_series.iloc[-1]
    avg_cost = 0.0
    if not buy_df.empty and buy_df['Shares'].sum() > 0:
        avg_cost = buy_df['Amount'].sum() / buy_df['Shares'].sum()
    final_price = close.iloc[-1]

    returns = equity_series.pct_change().dropna()
    sharpe = 0.0
    sortino = 0.0
    if not returns.empty:
        mean_ret = returns.mean()
        vol = returns.std(ddof=0)
        downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
        if vol > 0:
            sharpe = mean_ret / vol * np.sqrt(252)
        if downside > 0:
            sortino = mean_ret / downside * np.sqrt(252)

    period_days = (close.index[-1] - close.index[0]).days
    if period_days <= 0:
        period_years = len(close) / 252
    else:
        period_years = period_days / 365.0
    if period_years <= 0:
        period_years = 1 / 252

    annualized_return = 0.0
    if equity_series.iloc[0] > 0:
        annualized_return = (equity_final / equity_series.iloc[0]) ** (1 / period_years) - 1

    last_buy_price = close.iloc[0]
    last_buy_return_pct = 0.0
    last_buy_date = 'N/A'
    if not buy_df.empty:
        last_buy_price = buy_df['Price'].iloc[-1]
        last_buy_date = buy_df['Datetime'].iloc[-1].date().isoformat()
        if last_buy_price > 0:
            last_buy_return_pct = (final_price - last_buy_price) / last_buy_price

    max_dd_len = max_drawdown_duration(drawdown_series)

    longest_below_start = longest_below_start_days(equity_series)
    days_below_start = int((equity_series < START_CAPITAL).sum())

    min_equity = equity_series.min()
    min_equity_date = equity_series.idxmin().date().isoformat()
    floor_loss_pct = (min_equity - START_CAPITAL) / START_CAPITAL

    invested_ratio = invested / START_CAPITAL
    total_return = (equity_final - START_CAPITAL) / START_CAPITAL
    expected_return = invested_ratio * total_return if invested_ratio > 0 else 0.0

    metrics = {
        'span': span,
        'threshold': threshold_pct,
        'equity_final': equity_final,
        'max_dd_pct': drawdown_series.min() * 100,
        'buys': len(buy_df),
        'invested': invested,
        'cash_left': cash,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'avg_cost': avg_cost,
        'final_price': final_price,
        'final_vs_avg_cost_pct': (final_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0,
        'sharpe': sharpe,
        'sortino': sortino,
        'cash_ratio': cash / START_CAPITAL,
        'invested_ratio': invested_ratio,
        'longest_loss_streak_days': longest_below_start,
        'days_below_start': days_below_start,
        'last_buy_price': last_buy_price,
        'last_buy_return_pct': last_buy_return_pct,
        'last_buy_date': last_buy_date,
        'min_equity': min_equity,
        'min_equity_date': min_equity_date,
        'floor_loss_pct': floor_loss_pct,
        'expected_return': expected_return,
    }
    metrics = {
        key: float(value) if isinstance(value, (np.generic,)) else value
        for key, value in metrics.items()
    }
    return equity_series, drawdown_series, buy_df, metrics


def max_drawdown_duration(drawdown: pd.Series) -> int:
    max_duration = 0
    current = 0
    for dd in drawdown:
        if dd < 0:
            current += 1
            if current > max_duration:
                max_duration = current
        else:
            current = 0
    return max_duration


def longest_below_start_days(equity: pd.Series) -> int:
    longest = 0
    current = 0
    for value in equity:
        if value < START_CAPITAL:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_static_sources(close: pd.Series, intervals):
    data_map = {}
    metrics_map = {}

    for label, start, end in intervals:
        sub_close = close.iloc[start:end + 1]
        prefix = f'int_{start}_{end}'
        for span in SPANS:
            for thr in THRESHOLDS:
                eq, dd, buys, metrics = simulate_dca(sub_close, span, thr / 100)
                key = f'{prefix}_span{span}_thr{thr}'
                equity_pct = (eq / eq.iloc[0] - 1).values.tolist()
                data_map[key] = {
                    'date': (sub_close.index.view('int64') // 10**6).tolist(),
                    'equity': eq.values.tolist(),
                    'equity_pct': equity_pct,
                    'drawdown': dd.values.tolist(),
                    'price': sub_close.values.tolist(),
                }
                data_map[key + '_buys'] = {
                    'date': (buys['Datetime'].view('int64') // 10**6).tolist() if not buys.empty else [],
                    'price': buys['Price'].tolist() if not buys.empty else [],
                }
                metrics_map[key] = {'label': label, **metrics}
    return data_map, metrics_map


def render_dashboard(close: pd.Series, data_map, metrics_map, intervals, output_html: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file(output_html, title='DCA MVP Dashboard')

    init_label, init_start, init_end = intervals[0]
    initial_key = f'int_{init_start}_{init_end}_span{SPANS[0]}_thr{THRESHOLDS[0]}'

    source_line = ColumnDataSource(data=data_map[initial_key])
    source_buys = ColumnDataSource(data=data_map[initial_key + '_buys'])

    # Price figure
    p_price = figure(x_axis_type='datetime', width=950, height=260,
                     title='Price & Buy fills')
    p_price.line('date', 'price', source=source_line, color='#1f77b4', legend_label='Price')
    p_price.scatter('date', 'price', source=source_buys, size=6, color='orange',
                    legend_label='Buy fills', alpha=0.8)
    p_price.legend.location = 'top_left'

    # Equity % figure (normalized)
    p_eq = figure(x_axis_type='datetime', width=950, height=240,
                  title='Equity % (from interval start)')
    p_eq.line('date', 'equity_pct', source=source_line, color='#1f77b4', legend_label='Equity %')
    p_eq.legend.location = 'top_left'

    # Drawdown figure alone (dynamic y-range set below)
    p_dd = figure(x_axis_type='datetime', width=950, height=220,
                  title='Drawdown')
    p_dd.line('date', 'drawdown', source=source_line, color='#d62728', legend_label='Drawdown')
    p_dd.legend.location = 'top_left'

    span_select = Select(title='DCA span (days)', value=str(SPANS[0]),
                         options=[str(s) for s in SPANS])
    thr_select = Select(title='Trigger drawdown %', value=str(THRESHOLDS[0]),
                        options=[str(t) for t in THRESHOLDS])
    interval_select = Select(title='Interval (ATH→ATH)', value=init_label,
                             options=[lab for lab, _, _ in intervals])

    metrics_div = Div(text='')

    data_json = json.dumps(data_map)
    metrics_json = json.dumps(metrics_map)

    intervals_json = json.dumps([{ 'label': lab, 'start': s, 'end': e } for lab, s, e in intervals])

    # Precompute initial y-ranges for drawdown
    init_dd = data_map[initial_key]['drawdown']
    init_min_dd = min(init_dd)
    init_max_dd = max(init_dd)
    p_dd.y_range = Range1d(start=init_min_dd - abs(init_min_dd) * 0.1 - 0.02,
                           end=min(0, init_max_dd + 0.02))

    callback = CustomJS(
        args=dict(
            source_line=source_line,
            source_buys=source_buys,
            data_map=data_json,
            metrics_map=metrics_json,
            intervals=intervals_json,
            span_select=span_select,
            thr_select=thr_select,
            interval_select=interval_select,
            metrics_div=metrics_div,
            p_price=p_price,
            p_eq=p_eq,
            p_dd=p_dd,
        ),
        code="""
        const fmtPct = (val) => Number.isFinite(val) ? `${(val * 100).toFixed(2)}%` : '0.00%';
        const fmtVal = (val) => Number.isFinite(val) ? val.toFixed(2) : '0.00';
        const fmtDollar = (val) => Number.isFinite(val) ? `$${val.toFixed(2)}` : '$0.00';

        const intervalsData = JSON.parse(intervals);
        const interval = intervalsData.find(i => i.label === interval_select.value);
        const key = `int_${interval.start}_${interval.end}_span${span_select.value}_thr${thr_select.value}`;
        const data = JSON.parse(data_map);
        const m = JSON.parse(metrics_map)[key];

        // update price/equity data
        const d = data[key];
        source_line.data = d;
        source_buys.data = data[`${key}_buys`];
        source_line.change.emit();
        source_buys.change.emit();

        const exitDate = d.date.length
            ? new Date(d.date[d.date.length - 1]).toLocaleDateString()
            : 'N/A';
        const exitPriceLine = `Exit price: ${fmtDollar(m.final_price)}`;
        const lastBuyLine = m.buys > 0
            ? `Last buy: ${fmtDollar(m.last_buy_price)} on ${m.last_buy_date}`
            : 'No buys executed';

        metrics_div.text = `${m.label}<br>` +
            `<strong>Assumes exit on ${exitDate}</strong><br>` +
            `${exitPriceLine}<br>` +
            `Span: ${m.span}<br>` +
            `Trigger: ${(m.threshold * 100).toFixed(0)}%<br>` +
            `Invested ratio: ${fmtPct(m.invested_ratio)}<br>` +
            `Total return: ${fmtPct(m.total_return)}<br>` +
            `Annualized: ${fmtPct(m.annualized_return)}<br>` +
            `Avg cost: ${fmtDollar(m.avg_cost)}<br>` +
            `Sharpe: ${fmtVal(m.sharpe)}<br>` +
            `Sortino: ${fmtVal(m.sortino)}<br>` +
            `Lowest NAV: ${fmtPct(m.floor_loss_pct)} vs start on ${m.min_equity_date}<br>` +
            `Days below start: ${m.days_below_start} (longest streak ${m.longest_loss_streak_days} days)<br>` +
            `Max DD: ${m.max_dd_pct.toFixed(2)}%<br>` +
            `${lastBuyLine}`;

        // auto-scale y for price and equity_pct
        const price = d.price;
        const eqp = d.equity_pct;
        const minPrice = Math.min(...price);
        const maxPrice = Math.max(...price);
        p_price.y_range.start = minPrice * 0.95;
        p_price.y_range.end = maxPrice * 1.05;
        const minEqp = Math.min(...eqp);
        const maxEqp = Math.max(...eqp);
        p_eq.y_range.start = minEqp - Math.abs(minEqp) * 0.1 - 0.02;
        p_eq.y_range.end = maxEqp + Math.abs(maxEqp) * 0.1 + 0.02;
        const minDD = Math.min(...d.drawdown);
        const maxDD = Math.max(...d.drawdown);
        p_dd.y_range.start = minDD - Math.abs(minDD) * 0.1 - 0.02;
        p_dd.y_range.end = Math.min(0, maxDD + 0.02);
        """,
    )

    span_select.js_on_change('value', callback)
    thr_select.js_on_change('value', callback)
    interval_select.js_on_change('value', callback)

    # Initialize metrics text
    init_metrics = metrics_map[initial_key]
    init_dates = data_map[initial_key]['date']
    init_exit_date = (
        pd.to_datetime(init_dates[-1], unit='ms').date().isoformat()
        if init_dates else 'N/A'
    )

    def fmt_pct(value: float) -> str:
        return f'{value * 100:.2f}%'

    def fmt_dollar(value: float) -> str:
        return f'${value:,.2f}'

    last_buy_line = (
        f'Last buy: {fmt_dollar(init_metrics["last_buy_price"])} on {init_metrics["last_buy_date"]}'
        if init_metrics['buys'] > 0
        else 'No buys executed'
    )

    metrics_div.text = (
        f'{init_metrics["label"]}<br>'
        f'<strong>Assumes exit on {init_exit_date}</strong><br>'
        f'Exit price: {fmt_dollar(init_metrics["final_price"])}<br>'
        f'Span: {init_metrics["span"]}<br>'
        f'Trigger: {init_metrics["threshold"] * 100:.0f}%<br>'
        f'Invested ratio: {fmt_pct(init_metrics["invested_ratio"])}<br>'
        f'Total return: {fmt_pct(init_metrics["total_return"])}<br>'
        f'Annualized: {fmt_pct(init_metrics["annualized_return"])}<br>'
        f'Avg cost: {fmt_dollar(init_metrics["avg_cost"])}<br>'
        f'Sharpe: {init_metrics["sharpe"]:.2f}<br>'
        f'Sortino: {init_metrics["sortino"]:.2f}<br>'
        f'Lowest NAV: {fmt_pct(init_metrics["floor_loss_pct"])} vs start on {init_metrics["min_equity_date"]}<br>'
        f'Days below start: {init_metrics["days_below_start"]} (longest streak {init_metrics["longest_loss_streak_days"]} days)<br>'
        f'Max DD: {init_metrics["max_dd_pct"]:.2f}%<br>'
        f'{last_buy_line}'
    )

    layout = column(row(span_select, thr_select, interval_select), p_price, p_eq, p_dd, metrics_div)
    save(layout)


def main() -> None:
    close = read_price_data()['Close']
    intervals = find_peak_intervals(close)
    data_map, metrics_map = build_static_sources(close, intervals)
    output_html = RESULTS_DIR / f'pendle_dca_dashboard_mvp_{datetime.now():%Y%m%d_%H%M%S}.html'
    render_dashboard(close, data_map, metrics_map, intervals, output_html)
    print(f'Wrote {output_html}')


if __name__ == '__main__':
    main()
