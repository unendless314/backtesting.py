from __future__ import annotations

from pathlib import Path
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, Select
from bokeh.plotting import figure

import btc_dca_dashboard_mvp as dca

RESULTS_DIR = dca.RESULTS_DIR

METRICS_INFO = {
    'total_return': {'label': 'Total return', 'format': 'pct', 'best': 'max'},
    'annualized_return': {'label': 'Annualized return', 'format': 'pct', 'best': 'max'},
    'floor_loss_pct': {'label': 'Lowest NAV vs start', 'format': 'pct', 'best': 'max'},
    'sharpe': {'label': 'Sharpe ratio', 'format': 'float', 'best': 'max'},
    'sortino': {'label': 'Sortino ratio', 'format': 'float', 'best': 'max'},
    'avg_cost': {'label': 'Average cost', 'format': 'usd', 'best': 'min'},
    'invested_ratio': {'label': 'Invested ratio', 'format': 'pct', 'best': 'max'},
    'days_below_start': {'label': 'Days below start', 'format': 'int', 'best': 'min'},
    'longest_loss_streak_days': {
        'label': 'Longest losing streak',
        'format': 'int',
        'best': 'min',
    },
    'expected_return': {'label': 'Expected return', 'format': 'pct', 'best': 'max'},
}


def format_value(value: float, fmt: str) -> str:
    if not math.isfinite(value):
        return 'n/a'
    if fmt == 'pct':
        return f'{value * 100:.2f}%'
    if fmt == 'float':
        return f'{value:.2f}'
    if fmt == 'int':
        return f'{int(value)}'
    if fmt == 'usd':
        return f'${value:,.2f}'
    return str(value)


def build_metrics_dataframe(metrics_map: Dict[str, dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for metrics in metrics_map.values():
        rows.append(
            {
                **metrics,
                'span': int(metrics['span']),
                'threshold_pct': float(metrics['threshold']) * 100,
                'label': metrics['label'],
            }
        )
    return pd.DataFrame(rows)


def build_interval_summary_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    summary: Dict[str, Dict[str, str]] = {}
    for label in sorted(df['label'].unique()):
        subset = df[df['label'] == label]
        summary[label] = {}
        for key, info in METRICS_INFO.items():
            if key not in subset:
                continue
            ascending = info['best'] == 'min'
            top = subset.sort_values(key, ascending=ascending).head(5)
            lines = [
                f'Span {int(row["span"])}d · Trigger {row["threshold_pct"]:.0f}% · {format_value(row[key], info["format"])}'
                for _, row in top.iterrows()
            ]
            summary[label][key] = f'<strong>Top 5 for {info["label"]}</strong><br>' + '<br>'.join(lines)
    return summary


def build_table_texts(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Return HTML tables per interval + metric so UI updates with interval selection."""

    tables: Dict[str, Dict[str, str]] = {}
    spans = sorted(df['span'].unique())
    thresholds = sorted(df['threshold_pct'].unique(), reverse=True)

    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        tables[label] = {}
        for key, info in METRICS_INFO.items():
            if key not in label_df.columns:
                continue
            sorted_df = label_df.sort_values(key, ascending=(info['best'] == 'min'))
            best_row = sorted_df.iloc[0]
            best_span = int(best_row['span'])
            best_threshold = float(best_row['threshold_pct'])

            header = (
                '<tr><th style="border:1px solid #ccc; padding:4px;">Trigger \\ Span</th>'
                + ''.join(f'<th style="border:1px solid #ccc; padding:4px;">{int(span)}</th>' for span in spans)
                + '</tr>'
            )

            rows_html: List[str] = []
            for thr in thresholds:
                cells: List[str] = []
                thr_label = f'{thr:.0f}%'
                for span in spans:
                    row = label_df[(label_df['threshold_pct'] == thr) & (label_df['span'] == span)]
                    value = format_value(row.iloc[0][key], info['format']) if not row.empty else 'n/a'
                    style = 'background:#fff9c4;' if span == best_span and thr == best_threshold else ''
                    cells.append(
                        f'<td style="border:1px solid #ccc; padding:4px; text-align:center; {style}">{value}</td>'
                    )
                rows_html.append(
                    f'<tr><th style="border:1px solid #ccc; padding:4px;">{thr_label}</th>'
                    + ''.join(cells)
                    + '</tr>'
                )

            table = (
                '<div style="max-height:360px; overflow:auto;">'
                '<table style="border-collapse:collapse; font-size:12px; width:100%;">'
                f'{header}'
                f'{"".join(rows_html)}'
                '</table></div>'
            )
            tables[label][key] = table
    return tables


def build_threshold_slices(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, List[Tuple[int, float]]]]]:
    """Return interval-scoped slices so lines don't connect across intervals."""

    thresholds = sorted(df['threshold_pct'].unique(), reverse=True)
    labels = sorted(df['label'].unique())
    data: Dict[str, Dict[str, Dict[str, List[Tuple[int, float]]]]] = {}

    for label in labels:
        label_df = df[df['label'] == label]
        data[label] = {}
        for key in METRICS_INFO:
            if key not in label_df.columns:
                continue
            thr_map: Dict[str, List[Tuple[int, float]]] = {}
            for thr in thresholds:
                rows = label_df[label_df['threshold_pct'] == thr].sort_values('span')
                thr_map[f'{thr:.0f}'] = [
                    (int(row['span']), float(row[key]))
                    for _, row in rows.iterrows()
                ]
            data[label][key] = thr_map

    return data


def create_heatmap(
    df: pd.DataFrame,
    target_metric: str,
    interval_label: str,
    interval_labels: List[str],
    output_html: Path,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file(output_html, title='DCA Parameter Analysis')

    summary_div = Div(text='', width=920)
    table_div = Div(text='', width=920)
    select_options = [(key, info['label']) for key, info in METRICS_INFO.items() if key in df.columns]
    metric_select = Select(title='Target metric', value=target_metric, options=select_options)
    interval_select = Select(title='Interval', value=interval_label, options=[(label, label) for label in interval_labels])

    summary_map = build_interval_summary_map(df)
    table_map = build_table_texts(df)
    slices = build_threshold_slices(df)
    thresholds = sorted(df['threshold_pct'].unique(), reverse=True)
    spans = sorted(df['span'].unique())

    summary_div.text = summary_map.get(interval_label, {}).get(target_metric, '')
    table_div.text = table_map.get(interval_label, {}).get(target_metric, '')

    threshold_sources: List[ColumnDataSource] = []
    threshold_figs = []
    for thr in thresholds:
        source = ColumnDataSource(data=dict(span=[], value=[]), name=f'slice_source_threshold_{int(thr)}')
        threshold_sources.append(source)
        fig = figure(
            width=920,
            height=220,
            title=f'{METRICS_INFO[target_metric]["label"]} (Trigger {int(thr)}%)',
            x_axis_label='Span (days)',
            y_axis_label=METRICS_INFO[target_metric]['label'],
        )
        fig.line('span', 'value', source=source, line_width=2, color='#1f77b4')
        fig.scatter('span', 'value', source=source, size=6, marker='circle', fill_color='white', line_color='#1f77b4')
        fig.xaxis.ticker = spans
        fig.xaxis.major_label_orientation = 1
        threshold_figs.append((thr, fig))

    initial_slices = slices.get(interval_label, {}).get(target_metric, {})
    for idx, thr in enumerate(thresholds):
        thr_key = f'{int(thr)}'
        rows = initial_slices.get(thr_key, [])
        source = threshold_sources[idx]
        source.data = {
            'span': [int(r[0]) for r in rows],
            'value': [float(r[1]) for r in rows],
        }
    select_labels = {key: info['label'] for key, info in METRICS_INFO.items()}

    callback = CustomJS(
        args=dict(
            metric_select=metric_select,
            interval_select=interval_select,
            summary_div=summary_div,
            table_div=table_div,
            summary_texts=json.dumps(summary_map),
            metric_tables=json.dumps(table_map),
            slices_json=json.dumps(slices),
            thresholds=thresholds,
            threshold_sources=threshold_sources,
            threshold_figs=[fig for _, fig in threshold_figs],
            select_labels=select_labels,
        ),
        code="""
        const summaries = JSON.parse(summary_texts);
        const tables = JSON.parse(metric_tables);
        const slicesByInterval = JSON.parse(slices_json);

        const metric = metric_select.value;
        const interval = interval_select.value;
        const summary_interval = summaries[interval] || summaries[Object.keys(summaries)[0]] || {};
        summary_div.text = summary_interval[metric] || '';
        const tblInterval = tables[interval] || tables[Object.keys(tables)[0]] || {};
        table_div.text = tblInterval[metric] || '';
        const data = (slicesByInterval[interval] || {})[metric] || {};
        for (let i = 0; i < thresholds.length; i++) {
            const thr = `${thresholds[i]}`;
            const rows = data[thr] || [];
            const source = threshold_sources[i];
            source.data = {
                span: rows.map(r => r[0]),
                value: rows.map(r => r[1]),
            };
            source.change.emit();
            const fig = threshold_figs[i];
            const label = select_labels[metric] || metric;
            fig.title.text = `${label} (Trigger ${thr}%)`;
            if (fig.yaxis && fig.yaxis.length) {
                fig.yaxis[0].axis_label = label;
            }
            if (fig.y_range && rows.length) {
                const values = rows.map((r) => r[1]);
                const minVal = Math.min(...values);
                const maxVal = Math.max(...values);
                const bottom = Math.min(0, minVal);
                fig.y_range.start = bottom - Math.abs(bottom) * 0.05 - 0.1;
                fig.y_range.end = maxVal + Math.abs(maxVal) * 0.05 + 0.1;
            }
        }
        """,
    )
    metric_select.js_on_change('value', callback)
    interval_select.js_on_change('value', callback)
    callback.args['metric_select'] = metric_select
    callback.args['interval_select'] = interval_select
    metric_select.value = metric_select.value  # trigger initial render

    layout = column(
        row(metric_select, interval_select),
        table_div,
        summary_div,
        *[column(Div(text=f'<a id="chart_thr_{int(thr)}"></a>'), fig) for thr, fig in threshold_figs],
    )
    save(layout)
    print(f'Wrote {output_html}')


def main() -> None:
    close = dca.read_price_data()['Close']
    intervals = dca.find_peak_intervals(close)
    _, metrics_map = dca.build_static_sources(close, intervals)
    df = build_metrics_dataframe(metrics_map)
    default_metric = 'total_return'
    interval_labels = [label for label, _, _ in intervals]
    initial_label = interval_labels[0]
    output_html = RESULTS_DIR / f'btc_dca_param_analysis_{datetime.now():%Y%m%d_%H%M%S}.html'
    create_heatmap(df, default_metric, initial_label, interval_labels, output_html)


if __name__ == '__main__':
    main()
