from __future__ import annotations

import json
import math
from typing import Dict, List

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Div, Select

import dca_dashboard_mvp as dca

ANALYSIS_HTML = dca.RESULTS_DIR / 'dca_param_analysis.html'

METRICS_INFO = {
    'total_return': {'label': 'Total return', 'format': 'pct', 'best': 'max'},
    'annualized_return': {'label': 'Annualized return', 'format': 'pct', 'best': 'max'},
    'floor_loss_pct': {'label': 'Lowest NAV vs start', 'format': 'pct', 'best': 'max'},
    'sharpe': {'label': 'Sharpe ratio', 'format': 'float', 'best': 'max'},
    'sortino': {'label': 'Sortino ratio', 'format': 'float', 'best': 'max'},
    'unrealized_profit_ratio': {
        'label': 'Unrealized profit ratio',
        'format': 'pct',
        'best': 'max',
    },
    'days_below_start': {'label': 'Days below start', 'format': 'int', 'best': 'min'},
    'longest_loss_streak_days': {
        'label': 'Longest losing streak',
        'format': 'int',
        'best': 'min',
    },
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
    return f'{value}'


def build_metrics_dataframe(metrics_map: Dict[str, dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for metrics in metrics_map.values():
        rows.append(
            {
                **metrics,
                'span': int(metrics['span']),
                'threshold': float(metrics['threshold']),
                'threshold_pct': float(metrics['threshold']) * 100,
                'threshold_label': f'{metrics["threshold"]*100:.0f}%',
                'span_label': str(int(metrics['span'])),
            }
        )
    return pd.DataFrame(rows)


def build_top_texts(df: pd.DataFrame, label: str | None = None) -> Dict[str, str]:
    top_texts: Dict[str, str] = {}
    base_df = df if label is None else df[df['label'] == label]
    for key, info in METRICS_INFO.items():
        if key not in base_df:
            continue
        ascending = info['best'] == 'min'
        sorted_df = base_df.sort_values(key, ascending=ascending).head(5)
        lines: List[str] = []
        for idx, row in sorted_df.iterrows():
            lines.append(
                f'Span {row["span"]}d · Trigger {row["threshold_pct"]:.0f}% · '
                f'{row["label"]} → {format_value(row[key], info["format"])}'
            )
        top_texts[key] = (
            f'<strong>Top spans for {info["label"]}</strong><br>' + '<br>'.join(lines)
        )
    return top_texts


def build_table_texts(df: pd.DataFrame, label: str | None = None) -> Dict[str, str]:
    tables: Dict[str, str] = {}
    filtered_df = df if label is None else df[df['label'] == label]
    spans = sorted(filtered_df['span'].unique())
    thresholds = sorted(filtered_df['threshold_pct'].unique(), reverse=True)

    for key, info in METRICS_INFO.items():
        if key not in filtered_df:
            continue
        sorted_df = filtered_df.sort_values(key, ascending=(info['best'] == 'min'))
        best_row = sorted_df.iloc[0]
        best_span = int(best_row['span'])
        best_threshold = float(best_row['threshold_pct'])

        value_map = {
            (row['threshold_pct'], row['span']): format_value(row[key], info['format'])
            for _, row in filtered_df.iterrows()
        }

        header = '<tr><th style="border:1px solid #ccc; padding:4px;">Trigger \\ Span</th>' + ''.join(
            f'<th style="border:1px solid #ccc; padding:4px;">{int(span)}</th>' for span in spans
        ) + '</tr>'

        rows_html = []
        for thr in thresholds:
            cells = []
            thr_label = f'{thr:.0f}%'
            for span in spans:
                val = value_map.get((thr, span), 'n/a')
                style = 'background:#fff9c4;' if span == best_span and thr == best_threshold else ''
                cells.append(
                    f'<td style="border:1px solid #ccc; padding:4px; text-align:center; {style}">{val}</td>'
                )
            rows_html.append(
                f'<tr><th style="border:1px solid #ccc; padding:4px;">{thr_label}</th>' + ''.join(cells) + '</tr>'
            )

        table = (
            f'<div style="max-height:360px; overflow:auto;">'
            f'<table style="border-collapse:collapse; font-size:12px; width:100%;">'
            f'{header}'
            f'{"".join(rows_html)}'
            f'</table></div>'
        )
        tables[key] = table
    return tables


def build_interval_summary_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    return {
        label: build_top_texts(df, label)
        for label in sorted(df['label'].unique())
    }


def build_interval_table_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    return {
        label: build_table_texts(df, label)
        for label in sorted(df['label'].unique())
    }


def create_heatmap(df: pd.DataFrame, target_metric: str, interval_label: str, interval_labels: List[str]) -> None:
    RESULTS_DIR = dca.RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file(ANALYSIS_HTML, title='DCA Parameter Analysis')

    summary_div = Div(text='', width=900)
    table_div = Div(text='', width=900)
    select_options = [(key, info['label']) for key, info in METRICS_INFO.items() if key in df.columns]
    metric_select = Select(title='Target metric', value=target_metric, options=select_options)
    interval_select = Select(title='Interval', value=interval_label, options=[(label, label) for label in interval_labels])

    summary_map = build_interval_summary_map(df)
    table_map = build_interval_table_map(df)
    summary_div.text = summary_map[interval_label].get(target_metric, '')
    table_div.text = table_map[interval_label].get(target_metric, '')

    callback = CustomJS(
        args=dict(
            summary_div=summary_div,
            table_div=table_div,
            metric_select=metric_select,
            interval_select=interval_select,
            summary_texts=json.dumps(summary_map),
            metric_tables=json.dumps(table_map),
        ),
        code="""
        const summaries = JSON.parse(summary_texts);
        const tables = JSON.parse(metric_tables);
        const metric = metric_select.value;
        const interval = interval_select.value;
        const summary_interval = summaries[interval] || summaries[Object.keys(summaries)[0]];
        const table_interval = tables[interval] || tables[Object.keys(tables)[0]];
        summary_div.text = summary_interval[metric];
        table_div.text = table_interval[metric];
        """
    )
    metric_select.js_on_change('value', callback)
    interval_select.js_on_change('value', callback)

    layout = column(row(metric_select, interval_select), summary_div, table_div)
    save(layout)
    print(f'Wrote {ANALYSIS_HTML}')


def main() -> None:
    close = dca.read_price_data()['Close']
    intervals = dca.find_peak_intervals(close)
    _, metrics_map = dca.build_static_sources(close, intervals)
    df = build_metrics_dataframe(metrics_map)
    default_metric = 'total_return'
    interval_labels = [label for label, _, _ in intervals]
    initial_label = interval_labels[0]
    create_heatmap(df, default_metric, initial_label, interval_labels)


if __name__ == '__main__':
    main()
