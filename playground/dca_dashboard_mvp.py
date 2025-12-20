from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, Range1d, Select, LinearAxis
from bokeh.plotting import figure, output_file, save

DATA_PATH = Path('data/raw/BTCUSDT_1d.csv')
RESULTS_DIR = Path('playground/results')
OUTPUT_HTML = RESULTS_DIR / 'dca_mvp_dashboard.html'

START_CAPITAL = 60_000.0
SPANS = [100, 200, 300, 400, 500, 600, 800, 1000, 1200]
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
    buys: List[Tuple[pd.Timestamp, float, float]] = []

    peak_equity = -np.inf

    for ts, price in close.items():
        if price > ath:
            ath = price
        drawdown_pct = price / ath - 1 if ath > 0 else 0.0

        should_buy = (drawdown_pct <= threshold_pct) and cash > 0 and dca_run_days < span
        if should_buy:
            amount = min(daily_amount, cash)
            shares += amount / price
            cash -= amount
            dca_run_days += 1
            buys.append((ts, price, shares))

        eq = cash + shares * price
        peak_equity = max(peak_equity, eq)
        equity.append(eq)
        dd = (eq / peak_equity - 1) if peak_equity > 0 else 0.0
        drawdown.append(dd)

    equity_series = pd.Series(equity, index=close.index)
    drawdown_series = pd.Series(drawdown, index=close.index)

    buy_df = pd.DataFrame(buys, columns=['Datetime', 'Price', 'Shares'])

    metrics = {
        'span': span,
        'threshold': threshold_pct,
        'equity_final': equity_series.iloc[-1],
        'max_dd_pct': drawdown_series.min() * 100,
        'buys': len(buy_df),
        'invested': START_CAPITAL - cash,
        'cash_left': cash,
    }
    return equity_series, drawdown_series, buy_df, metrics


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


def render_dashboard(close: pd.Series, data_map, metrics_map, intervals) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file(OUTPUT_HTML, title='DCA MVP Dashboard')

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

        metrics_div.text = `${m.label}<br>` +
            `Span: ${m.span} | Trigger: ${m.threshold*100}%` +
            `<br>Final equity: $${m.equity_final.toFixed(2)}` +
            `<br>Max DD: ${m.max_dd_pct.toFixed(2)}%` +
            `<br>Invested: $${m.invested.toFixed(2)} | Cash left: $${m.cash_left.toFixed(2)}` +
            `<br>Buys: ${m.buys}`;

        // auto-scale y for price and equity_pct
        const price = d.price;
        const eqp = d.equity_pct;
        const minPrice = Math.min(...price);
        const maxPrice = Math.max(...price);
        p_price.y_range.start = minPrice * 0.95;
        p_price.y_range.end = maxPrice * 1.05;
        const minEqp = Math.min(...eqp);
        const maxEqp = Math.max(...eqp);
        p_eq.y_range.start = minEqp - Math.abs(minEqp)*0.1 - 0.02;
        p_eq.y_range.end = maxEqp + Math.abs(maxEqp)*0.1 + 0.02;
        const minDD = Math.min(...d.drawdown);
        const maxDD = Math.max(...d.drawdown);
        p_dd.y_range.start = minDD - Math.abs(minDD)*0.1 - 0.02;
        p_dd.y_range.end = Math.min(0, maxDD + 0.02);
        """,
    )

    span_select.js_on_change('value', callback)
    thr_select.js_on_change('value', callback)
    interval_select.js_on_change('value', callback)

    # Initialize metrics text
    init_metrics = metrics_map[initial_key]
    metrics_div.text = (
        f'{init_metrics["label"]}<br>'
        f'Span: {init_metrics["span"]} | Trigger: {init_metrics["threshold"]*100}%<br>'
        f'Final equity: ${init_metrics["equity_final"]:,.2f}<br>'
        f'Max DD: {init_metrics["max_dd_pct"]:.2f}%<br>'
        f'Invested: ${init_metrics["invested"]:,.2f} | '
        f'Cash left: ${init_metrics["cash_left"]:,.2f}<br>'
        f'Buys: {init_metrics["buys"]}'
    )

    layout = column(row(span_select, thr_select, interval_select), p_price, p_eq, p_dd, metrics_div)
    save(layout)


def main() -> None:
    close = read_price_data()['Close']
    intervals = find_peak_intervals(close)
    data_map, metrics_map = build_static_sources(close, intervals)
    render_dashboard(close, data_map, metrics_map, intervals)
    print(f'Wrote {OUTPUT_HTML}')


if __name__ == '__main__':
    main()
