from __future__ import annotations

import json
import math
from typing import Dict, List

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, Select
from bokeh.plotting import figure

import dca_dashboard_mvp as dca

RESULTS_DIR = dca.RESULTS_DIR
ANALYSIS_HTML = RESULTS_DIR / 'dca_param_analysis_zh.html'

METRICS_INFO_ZH = {
    'total_return': {'label': '總報酬', 'format': 'pct', 'best': 'max'},
    'annualized_return': {'label': '年化報酬', 'format': 'pct', 'best': 'max'},
    'floor_loss_pct': {'label': '最低淨值', 'format': 'pct', 'best': 'max'},
    'sharpe': {'label': 'Sharpe 比率', 'format': 'float', 'best': 'max'},
    'sortino': {'label': 'Sortino 比率', 'format': 'float', 'best': 'max'},
    'avg_cost': {'label': '平均成本', 'format': 'usd', 'best': 'min'},
    'invested_ratio': {'label': '投入比例', 'format': 'pct', 'best': 'max'},
    'days_below_start': {'label': '低於起始資金天數', 'format': 'int', 'best': 'min'},
    'longest_loss_streak_days': {'label': '最長虧損日數', 'format': 'int', 'best': 'min'},
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
    rows = []
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
    summary = {}
    for label in sorted(df['label'].unique()):
        subset = df[df['label'] == label]
        summary[label] = {}
        for key, info in METRICS_INFO_ZH.items():
            if key not in subset:
                continue
            ascending = info['best'] == 'min'
            top = subset.sort_values(key, ascending=ascending).head(5)
            lines = [
                f'定投 {int(row["span"])} 天 · 觸發 {row["threshold_pct"]:.0f}% · {format_value(row[key], info["format"])}'
                for _, row in top.iterrows()
            ]
            summary[label][key] = f'<strong>{info["label"]} 前五</strong><br>' + '<br>'.join(lines)
    return summary


def build_table_texts(df: pd.DataFrame, label: str | None = None) -> Dict[str, str]:
    subset = df if label is None else df[df['label'] == label]
    tables = {}
    spans = sorted(subset['span'].unique())
    thresholds = sorted(subset['threshold_pct'].unique(), reverse=True)
    for key, info in METRICS_INFO_ZH.items():
        if key not in subset:
            continue
        sorted_df = subset.sort_values(key, ascending=(info['best'] == 'min'))
        best = sorted_df.iloc[0]
        best_span = int(best['span'])
        best_threshold = float(best['threshold_pct'])
        header = (
            '<tr><th style="border:1px solid #ccc; padding:4px;">觸發 \\ 天數</th>'
            + ''.join(f'<th style="border:1px solid #ccc; padding:4px;">{int(span)}</th>' for span in spans)
            + '</tr>'
        )
        rows = []
        for thr in thresholds:
            cells = []
            thr_label = f'{thr:.0f}%'
            for span in spans:
                row = subset[(subset['threshold_pct'] == thr) & (subset['span'] == span)]
                value = format_value(row.iloc[0][key], info['format']) if not row.empty else 'n/a'
                style = 'background:#fff9c4;' if span == best_span and thr == best_threshold else ''
                cells.append(
                    f'<td style="border:1px solid #ccc; padding:4px; text-align:center; {style}">{value}</td>'
                )
            rows.append(
                f'<tr><th style="border:1px solid #ccc; padding:4px;"><a href="#chart_thr_{int(thr)}">{thr_label}</a></th>' + ''.join(cells) + '</tr>'
            )
        table = (
            '<div style="max-height:360px; overflow:auto;">'
            '<table style="border-collapse:collapse; font-size:12px; width:100%;">'
            f'{header}'
            f'{"".join(rows)}'
            '</table></div>'
        )
        tables[key] = table
    return tables


def build_interval_table_map(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    return {
        label: build_table_texts(df, label)
        for label in sorted(df['label'].unique())
    }


def build_threshold_slices(df: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    thresholds = sorted(df['threshold_pct'].unique(), reverse=True)
    data = {}
    for key in METRICS_INFO_ZH:
        if key not in df.columns:
            continue
        thr_map = {}
        for thr in thresholds:
            rows = df[df['threshold_pct'] == thr].sort_values('span')
            thr_map[f'{thr:.0f}'] = [(int(row['span']), float(row[key])) for _, row in rows.iterrows()]
        data[key] = thr_map
    return data


def create_heatmap(df: pd.DataFrame, target_metric: str, interval_label: str, interval_labels: List[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file(ANALYSIS_HTML, title='DCA 參數分析（中文版）')

    summary_div = Div(text='', width=920)
    table_div = Div(text='', width=920)
    select_options = [(key, info['label']) for key, info in METRICS_INFO_ZH.items() if key in df.columns]
    metric_select = Select(title='目標指標', value=target_metric, options=select_options, name='metric_select')
    interval_select = Select(
        title='區間', value=interval_label, options=[(label, label) for label in interval_labels]
    )

    summary_map = build_interval_summary_map(df)
    table_map = build_interval_table_map(df)
    slices = build_threshold_slices(df)
    thresholds = sorted(df['threshold_pct'].unique(), reverse=True)
    spans = sorted(df['span'].unique())

    summary_div.text = summary_map[interval_label].get(target_metric, '')
    table_div.text = table_map.get(interval_label, {}).get(target_metric, '')

    threshold_sources: List[ColumnDataSource] = []
    threshold_figs = []
    for thr in thresholds:
        src = ColumnDataSource(data=dict(span=[], value=[]), name=f'slice_source_threshold_{int(thr)}')
        threshold_sources.append(src)
        fig = figure(
            width=920,
            height=220,
            title=f'{METRICS_INFO_ZH[target_metric]["label"]}（觸發 {int(thr)}%）',
            x_axis_label='定投天數',
            y_axis_label=METRICS_INFO_ZH[target_metric]['label'],
        )
        fig.line('span', 'value', source=src, line_width=2, color='#1f77b4')
        fig.scatter('span', 'value', source=src, size=6, marker='circle', fill_color='white', line_color='#1f77b4')
        fig.xaxis.ticker = spans
        fig.xaxis.major_label_orientation = 1
        threshold_figs.append((thr, fig))

    initial_slices = slices.get(target_metric, {})
    for idx, thr in enumerate(thresholds):
        thr_key = f'{int(thr)}'
        rows = initial_slices.get(thr_key, [])
        src = threshold_sources[idx]
        src.data = {
            'span': [int(r[0]) for r in rows],
            'value': [float(r[1]) for r in rows],
        }
    select_labels = {key: label for key, label in select_options}

    callback = CustomJS(
        args=dict(
            metric_select=metric_select,
            interval_select=interval_select,
            slices=slices,
            thresholds=thresholds,
            threshold_sources=threshold_sources,
            threshold_figs=[fig for _, fig in threshold_figs],
            select_labels=select_labels,
            summary_div=summary_div,
            table_div=table_div,
            summary_texts=json.dumps(summary_map),
            metric_tables=json.dumps(table_map),
        ),
        code="""
        const summaries = JSON.parse(summary_texts);
        const tables = JSON.parse(metric_tables);
        const metric = metric_select.value;
        const interval = interval_select.value;
        const summary_interval = summaries[interval] || summaries[Object.keys(summaries)[0]] || {};
        summary_div.text = summary_interval[metric] || '';
        const table_interval = tables[interval] || tables[Object.keys(tables)[0]] || {};
        table_div.text = table_interval[metric] || '';
        const data = slices[metric] || {};
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
            fig.title.text = `${label}（觸發 ${thr}%）`;
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

    script_div = Div(
        text='<div style="display:none;"></div>',
        width=0,
    )

    layout = column(
        row(metric_select, interval_select),
        table_div,
        summary_div,
        Div(text='<strong>點擊表格左側「觸發 (%)」即可跳到下方對應圖表</strong>'),
        *[column(Div(text=f'<a id="chart_thr_{int(thr)}"></a>'), fig) for thr, fig in threshold_figs],
        script_div,
    )
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
