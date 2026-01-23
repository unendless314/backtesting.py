"""AAVE ATH-to-ATH interval + drawdown viewer mirroring the ETH logic.

Run inside the repo's virtualenv:
    source .venv/bin/activate
    python playground/aave/aave_peak_drawdown_ui_1d.py --help
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation,
    Button,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Label,
    NumeralTickFormatter,
    Range1d,
    Select,
    Span,
)
from bokeh.plotting import figure, output_file, save

DEFAULT_DATA = Path('data/raw/AAVEUSDT_1d.csv')
RESULTS_DIR = Path('playground/results')
DD_LEVELS = [-10, -20, -30, -40, -50, -60, -70, -80, -90]
DD_COLORS = ['#f5c6a5', '#f2a65a', '#d65f5f', '#c44a5c', '#a63d5f', '#7f2f5d', '#5b1f5b', '#3b0f3b', '#1f051f']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize AAVE ATH gaps and their drawdown bands.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--csv', type=Path, default=DEFAULT_DATA,
                        help='Path to CSV with columns Date, open, high, low, close, volume')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Start date (inclusive) for the plot window')
    parser.add_argument('--end', type=str, default='2025-12-31',
                        help='End date (inclusive) for the plot window')
    parser.add_argument('--first-peak-cutoff', type=str,
                        help='Anchor the first peak as the last ATH on or before this date')
    parser.add_argument('--use-high', action='store_true',
                        help='Use the High column instead of Close when finding peaks')
    parser.add_argument('--title', type=str, default='AAVE Peak Gap + Drawdown',
                        help='Plot title prefix')
    parser.add_argument('--min-gap-days', type=int, default=0,
                        help='Only show peak intervals whose gap is at least this many days')
    parser.add_argument('--plot-interval-index', type=int,
                        help='1-based index (after filtering) to plot automatically')
    parser.add_argument('--plot-all-intervals', action='store_true',
                        help='Export a plot for every filtered interval (skips incomplete ones)')
    parser.add_argument('--interval-padding-days', type=int, default=30,
                        help='Days of padding on each side when auto-plotting intervals')
    return parser.parse_args()


def read_price_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    peek = pd.read_csv(csv_path, nrows=1)
    date_col = 'datetime' if 'datetime' in peek.columns else 'Date'
    
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df.rename(columns={date_col: 'Datetime'}, inplace=True)
    df.rename(columns=str.capitalize, inplace=True)
    df.set_index('Datetime', inplace=True)
    df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
    
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def compute_ath_intervals(df: pd.DataFrame, *, use_high: bool,
                          anchor_cutoff: str | None) -> pd.DataFrame:
    """Return intervals between consecutive all-time-high breakouts."""
    metric = 'High' if use_high else 'Close'
    series = df[metric].dropna()
    if series.empty:
        return pd.DataFrame(columns=['peak_date', 'peak_price',
                                     'next_peak_date', 'next_peak_price', 'days_between'])

    prev_cummax = series.cummax().shift(fill_value=float('-inf'))
    is_new_high = series > prev_cummax
    peaks = series[is_new_high]
    if peaks.empty:
        return pd.DataFrame(columns=['peak_date', 'peak_price',
                                     'next_peak_date', 'next_peak_price', 'days_between'])

    peaks_df = peaks.to_frame(name='peak_price')
    peaks_df['peak_date'] = peaks_df.index

    if anchor_cutoff:
        cutoff_ts = pd.to_datetime(anchor_cutoff)
        at_or_before = peaks_df[peaks_df['peak_date'] <= cutoff_ts]
        if at_or_before.empty:
            raise ValueError('No ATH found on/before --first-peak-cutoff; expand the date range.')
        anchor_idx = at_or_before.index[-1]
        peaks_df = peaks_df.loc[anchor_idx:]

    peaks_df = peaks_df.reset_index(drop=True)
    peaks_df['next_peak_date'] = peaks_df['peak_date'].shift(-1)
    peaks_df['next_peak_price'] = peaks_df['peak_price'].shift(-1)
    peaks_df['days_between'] = (peaks_df['next_peak_date'] - peaks_df['peak_date']).dt.days
    peaks_df['days_between'] = peaks_df['days_between'].astype('Int64')
    return peaks_df[['peak_date', 'peak_price', 'next_peak_date', 'next_peak_price', 'days_between']]


def _to_ms(ts: pd.Timestamp) -> float:
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts.timestamp() * 1000.0


def _dd_histogram_html(drawdown: pd.Series) -> str:
    """Return a small HTML table summarizing drawdown day counts per band."""
    edges = [-1_000_000, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0]
    labels = [
        '<-90%',
        '-90% to -80%',
        '-80% to -70%',
        '-70% to -60%',
        '-60% to -50%',
        '-50% to -40%',
        '-40% to -30%',
        '-30% to -20%',
        '-20% to -10%',
        '0% to -10%',
    ]
    cat = pd.cut(drawdown, bins=edges, labels=labels, right=True, include_lowest=True)
    counts = cat.value_counts(sort=False)
    rows = ''.join(f'<tr><td>{label}</td><td style="text-align:right;">{counts.get(label, 0)}</td></tr>'
                   for label in labels)
    return ('<table style="width:260px;">'
            '<tr><th>Band</th><th>Days</th></tr>'
            f'{rows}'
            '</table>')


def _find_segments(drawdown: pd.Series, threshold: float) -> List[Dict[str, float]]:
    """Return contiguous segments where drawdown <= threshold."""
    mask = drawdown <= threshold
    if not mask.any():
        return []
    # Find starts and ends of True runs
    change = mask.astype(int).diff().fillna(mask.iloc[0].astype(int))
    starts = list(drawdown.index[change == 1])
    ends = list(drawdown.index[change == -1])
    if mask.iloc[0]:
        starts = [drawdown.index[0]] + starts
    if mask.iloc[-1]:
        ends = ends + [drawdown.index[-1]]
    segments = []
    for s, e in zip(starts, ends):
        window = drawdown.loc[s:e]
        segments.append({
            'left': _to_ms(s),
            'right': _to_ms(e),
            'min_dd': float(window.min()),
        })
    return segments


def build_interval_dashboard(df: pd.DataFrame, intervals: pd.DataFrame, *,
                             use_high: bool, padding_days: int,
                             title: str) -> column:
    if intervals.empty:
        raise ValueError('No completed intervals available for the interactive view.')

    metric = 'High' if use_high else 'Close'
    df_reset = df.reset_index().rename(columns={'Datetime': 'Datetime'})
    df_reset['Date'] = df_reset['Datetime']
    df_reset['Color'] = df_reset.apply(
        lambda row: '#2ca02c' if row['Close'] >= row['Open'] else '#d62728', axis=1)
    price_source = ColumnDataSource(df_reset)

    pad = pd.Timedelta(days=max(padding_days, 0))
    records: List[dict] = []
    dd_x_all: List[List[float]] = []
    dd_y_all: List[List[float]] = []
    seg_all: List[List[List[dict]]] = [[] for _ in DD_LEVELS]
    dd_table_all: List[str] = []

    for _, interval_row in intervals.iterrows():
        peak = interval_row['peak_date']
        nxt = interval_row['next_peak_date']
        if pd.isna(nxt):
            continue
        window_start = max(peak - pad, df.index[0])
        window_end = min(nxt + pad, df.index[-1])
        window_slice = df.loc[window_start:window_end]
        y_min = float(window_slice['Low'].min())
        y_max = float(window_slice['High'].max())
        y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(y_min * 0.05, 1)
        y_start = y_min - y_pad
        y_end = y_max + y_pad
        label_idx = len(records) + 1
        records.append(dict(
            option=f'{label_idx}. {peak.date()} → {nxt.date()} ({interval_row["days_between"]}d)',
            peak_ms=_to_ms(peak),
            peak_price=interval_row['peak_price'],
            break_ms=_to_ms(nxt),
            break_price=interval_row['next_peak_price'],
            days=int(interval_row['days_between']),
            title=f'{title}: {int(interval_row["days_between"])} days between peaks',
            info=(f'<b>Interval {label_idx}</b>: {peak.date()} → {nxt.date()}<br>'
                  f'First peak {interval_row["peak_price"]:,.0f} • '
                  f'Breakout {interval_row["next_peak_price"]:,.0f}'),
            window_left=_to_ms(window_start),
            window_right=_to_ms(window_end),
            y_start=y_start,
            y_end=y_end,
        ))

        window = df.loc[peak:nxt]
        dd = (window[metric] / interval_row['peak_price'] - 1) * 100.0
        dd_x_all.append([_to_ms(ts) for ts in dd.index])
        dd_y_all.append(dd.tolist())
        for i, level in enumerate(DD_LEVELS):
            seg_all[i].append(_find_segments(dd, level))
        dd_table_all.append(_dd_histogram_html(dd))

    if not records:
        raise ValueError('All intervals lack breakout dates in the selected range.')

    records_df = pd.DataFrame.from_records(records)
    interval_source = ColumnDataSource(records_df)
    initial = records[0]

    x_range = Range1d(initial['window_left'], initial['window_right'])
    tools = "xpan,xwheel_zoom,box_zoom,reset,save"
    p = figure(title=initial['title'], x_axis_type='datetime',
               width=1200, height=550, tools=tools,
               active_drag='xpan', active_scroll='xwheel_zoom',
               toolbar_location='right', x_range=x_range,
               y_range=Range1d(initial['y_start'], initial['y_end']))
    seg_renderer = p.segment('Date', 'High', 'Date', 'Low', color='#4c566a', source=price_source)
    vbar_renderer = p.vbar('Date', width=12*60*60*1000, top='Open', bottom='Close',
                           fill_color='Color', line_color='#4c566a', fill_alpha=0.8, source=price_source)
    price_hover = HoverTool(
        renderers=[vbar_renderer, seg_renderer],
        tooltips=[
            ('Date', '@Date{%F}'),
            ('Open', '@Open{0,0.00}'),
            ('High', '@High{0,0.00}'),
            ('Low', '@Low{0,0.00}'),
            ('Close', '@Close{0,0.00}'),
            ('Volume', '@Volume{0,0.}'),
        ],
        formatters={'@Date': 'datetime'},
        mode='vline'
    )
    p.add_tools(price_hover)
    p.yaxis.formatter = NumeralTickFormatter(format='$0,0')
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price (USDT)'

    peak_span = Span(location=initial['peak_ms'], dimension='height',
                     line_dash='dashed', line_color='#d62728', line_width=2)
    break_span = Span(location=initial['break_ms'], dimension='height',
                      line_dash='dashed', line_color='#2ca02c', line_width=2)
    p.add_layout(peak_span)
    p.add_layout(break_span)

    peak_label = Label(x=initial['peak_ms'], y=initial['peak_price'],
                       x_offset=6, y_offset=8,
                       text=f"First peak {initial['peak_price']:,.0f}",
                       text_color='#d62728', text_font_size='10pt')
    break_label = Label(x=initial['break_ms'], y=initial['break_price'],
                        x_offset=6, y_offset=8,
                        text=f"Breakout {initial['break_price']:,.0f}",
                        text_color='#2ca02c', text_font_size='10pt')
    gap_label = Label(
        x=(initial['peak_ms'] + initial['break_ms']) / 2,
        y=min(initial['peak_price'], initial['break_price']),
        y_offset=-30,
        text=f"{initial['days']} days between peaks",
        text_color='#1f77b4',
        text_align='center',
        text_font_size='12pt',
    )
    p.add_layout(peak_label)
    p.add_layout(break_label)
    p.add_layout(gap_label)

    box = BoxAnnotation(left=initial['window_left'], right=initial['window_right'],
                        fill_alpha=0.08, fill_color='#1f77b4')
    p.add_layout(box)

    # Drawdown subplot
    dd_source = ColumnDataSource(dict(x=dd_x_all[0], y=dd_y_all[0]))
    dd_plot = figure(x_axis_type='datetime', height=260, width=1200,
                     tools=tools, toolbar_location='right',
                     x_range=p.x_range, title='Drawdown (%)')
    def _init_seg_source(seg_list: List[dict], level: float) -> ColumnDataSource:
        return ColumnDataSource(dict(
            left=[s['left'] for s in seg_list],
            right=[s['right'] for s in seg_list],
            top=[level for _ in seg_list],
            bottom=[s['min_dd'] for s in seg_list],
        ))

    seg_sources = []
    for i, level in enumerate(DD_LEVELS):
        src = _init_seg_source(seg_all[i][0], level)
        seg_sources.append(src)
        dd_plot.quad(source=src, left='left', right='right',
                     bottom='bottom', top=level,
                     fill_color=DD_COLORS[i % len(DD_COLORS)],
                     fill_alpha=0.18, line_color=None)

    # Draw the drawdown line last so it stays on top of filled bands
    dd_line = dd_plot.line('x', 'y', source=dd_source, line_width=2.4, color='#1f4b99')
    dd_plot.yaxis.axis_label = 'Drawdown (%)'
    dd_plot.xaxis.axis_label = 'Date'

    for level in DD_LEVELS:
        dd_plot.add_layout(Span(location=level, dimension='width',
                                line_dash='dashed', line_color='#d08770', line_width=1.2))

    dd_hover = HoverTool(
        renderers=[dd_line],
        tooltips=[('Date', '@x{%F}'), ('Drawdown', '@y{0.0}%')],
        formatters={'@x': 'datetime'},
        mode='vline'
    )
    dd_plot.add_tools(dd_hover)

    def _y_bounds(vals: List[float]) -> tuple[float, float]:
        if not vals:
            return (-35, 5)
        m = min(vals)
        return (min(m - 5, -35), 5)

    y_start, y_end = _y_bounds(dd_y_all[0])
    dd_plot.y_range = Range1d(start=y_start, end=y_end)

    options = records_df['option'].tolist()
    select = Select(title='Interval',
                    options=options,
                    value=options[0],
                    width=350)
    info_div = Div(text=initial['info'], width=400)
    dd_table_div = Div(text=dd_table_all[0], width=260)

    callback = CustomJS(args=dict(
        select=select,
        intervals=interval_source,
        fig=p,
        peak_span=peak_span,
        break_span=break_span,
        peak_label=peak_label,
        break_label=break_label,
        gap_label=gap_label,
        box=box,
        info_div=info_div,
        dd_source=dd_source,
        seg_sources=seg_sources,
        dd_x_all=dd_x_all,
        dd_y_all=dd_y_all,
        seg_all=seg_all,
        dd_plot=dd_plot,
        dd_levels=DD_LEVELS,
        dd_table_div=dd_table_div,
        dd_table_all=dd_table_all,
    ), code="""
        const opts = select.options;
        const idx = opts.indexOf(select.value);
        if (idx < 0) { return; }
        const data = intervals.data;
        const peak_ms = data['peak_ms'][idx];
        const peak_price = data['peak_price'][idx];
        const break_ms = data['break_ms'][idx];
        const break_price = data['break_price'][idx];
        const left = data['window_left'][idx];
        const right = data['window_right'][idx];
        const days = data['days'][idx];

        peak_span.location = peak_ms;
        break_span.location = break_ms;

        peak_label.x = peak_ms;
        peak_label.y = peak_price;
        peak_label.text = `First peak ${peak_price.toLocaleString('en-US', {maximumFractionDigits:0})}`;

        break_label.x = break_ms;
        break_label.y = break_price;
        break_label.text = `Breakout ${break_price.toLocaleString('en-US', {maximumFractionDigits:0})}`;

        gap_label.x = (peak_ms + break_ms) / 2;
        gap_label.y = Math.min(peak_price, break_price);
        gap_label.text = `${days} days between peaks`;

        box.left = left;
        box.right = right;

        fig.x_range.start = left;
        fig.x_range.end = right;
        fig.y_range.start = data['y_start'][idx];
        fig.y_range.end = data['y_end'][idx];
        fig.title.text = data['title'][idx];

        info_div.text = data['info'][idx];

        // Update drawdown line
        dd_source.data = {x: dd_x_all[idx], y: dd_y_all[idx]};

        // Update drawdown segments
        function updateSeg(src, allSeg, level){
            const segs = allSeg || [];
            src.data = {
                left: segs.map(s => s.left),
                right: segs.map(s => s.right),
                top: segs.map(_ => level),
                bottom: segs.map(s => s.min_dd),
            };
        }
        for (let k = 0; k < seg_sources.length; k++) {
            const src = seg_sources[k];
            const segsForLevel = (seg_all[k] && seg_all[k][idx]) ? seg_all[k][idx] : [];
            updateSeg(src, segsForLevel, dd_levels[k]);
        }

        // Adjust y-range based on current drawdown values
        const ys = dd_y_all[idx] || [];
        let minY = -35;
        if (ys.length > 0){
            const minVal = Math.min(...ys);
            minY = Math.min(minVal - 5, -35);
        }
        dd_plot.y_range.start = minY;
        dd_plot.y_range.end = 5;

        dd_table_div.text = dd_table_all[idx] || '';
    """)
    select.js_on_change('value', callback)

    prev_button = Button(label='◀ Prev', width=80)
    next_button = Button(label='Next ▶', width=80)
    prev_button.js_on_event('button_click', CustomJS(args=dict(select=select), code="""
        const opts = select.options;
        const idx = opts.indexOf(select.value);
        if (idx > 0) { select.value = opts[idx - 1]; }
    """,))
    next_button.js_on_event('button_click', CustomJS(args=dict(select=select), code="""
        const opts = select.options;
        const idx = opts.indexOf(select.value);
        if (idx < opts.length - 1) { select.value = opts[idx + 1]; }
    """,))

    controls = row(prev_button, select, next_button, sizing_mode='stretch_width')
    layout = column(controls, p, dd_plot, row(info_div, dd_table_div))
    return layout


def main() -> None:
    args = parse_args()
    full_df = read_price_data(args.csv)
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    data_df = full_df.loc[start:end]
    if data_df.empty:
        raise ValueError('No rows in the requested date range; adjust --start/--end')

    report = compute_ath_intervals(full_df, use_high=args.use_high,
                                   anchor_cutoff=args.first_peak_cutoff)
    report = report[(report['peak_date'] >= start) & (report['peak_date'] <= end)]
    if args.min_gap_days > 0 and not report.empty:
        mask = report['days_between'].isna() | (report['days_between'] >= args.min_gap_days)
        report = report.loc[mask]

    completed = report.dropna(subset=['next_peak_date'])
    if completed.empty:
        raise ValueError('No completed intervals to visualize in the selected range.')

    interval_targets: list[tuple[int | None, pd.Series]] = []
    if args.plot_interval_index is not None:
        if args.plot_interval_index < 1 or args.plot_interval_index > len(completed):
            raise ValueError(f'plot_interval_index out of range (1-{len(completed)}).')
        row = completed.iloc[args.plot_interval_index - 1]
        interval_targets.append((args.plot_interval_index, row))

    if args.plot_all_intervals:
        for i, (_, row) in enumerate(completed.iterrows(), start=1):
            interval_targets.append((i, row))

    if interval_targets:
        batch_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        print('\nAuto-plotting selected intervals:')
        seen: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
        for label, row in interval_targets:
            key = (row['peak_date'], row['next_peak_date'])
            if key in seen:
                continue
            seen.add(key)
            # Reuse the drawdown-aware candlestick by calling the dashboard builder on a slice
            pad = pd.Timedelta(days=max(args.interval_padding_days, 0))
            start_w = max(row['peak_date'] - pad, data_df.index[0])
            end_w = min(row['next_peak_date'] + pad, data_df.index[-1])
            sub_df = data_df.loc[start_w:end_w]
            layout = build_interval_dashboard(sub_df, pd.DataFrame([row]),
                                              use_high=args.use_high,
                                              padding_days=args.interval_padding_days,
                                              title=args.title)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            suffix = f"{row['peak_date'].strftime('%Y%m%d')}_{row['next_peak_date'].strftime('%Y%m%d')}"
            if isinstance(label, int):
                suffix = f"{label:02d}_{suffix}"
            outfile = RESULTS_DIR / f"aave_peak_gap_dd_{suffix}_{batch_tag}.html"
            output_file(outfile)
            save(layout)
            print(f'  • {row["peak_date"].date()} → {row["next_peak_date"].date()} saved to {outfile}')

    layout = build_interval_dashboard(
        data_df, completed,
        use_high=args.use_high,
        padding_days=args.interval_padding_days,
        title=args.title)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = RESULTS_DIR / f'aave_peak_gap_dd_{timestamp}.html'
    output_file(outfile)
    save(layout)
    print(f'Saved plot to {outfile}')


if __name__ == '__main__':
    main()
