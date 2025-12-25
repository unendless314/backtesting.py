from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, Range1d, Select
from bokeh.plotting import figure, output_file, save

import btc_dca_dashboard_mvp as dca

RESULTS_DIR = dca.RESULTS_DIR


def render_dashboard_cn(close: pd.Series, data_map, metrics_map, intervals, output_html: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file(output_html, title='DCA MVP 儀表板（中文版）')

    init_label, init_start, init_end = intervals[0]
    initial_key = f'int_{init_start}_{init_end}_span{dca.SPANS[0]}_thr{dca.THRESHOLDS[0]}'

    source_line = ColumnDataSource(data=data_map[initial_key])
    source_buys = ColumnDataSource(data=data_map[initial_key + '_buys'])

    p_price = figure(
        x_axis_type='datetime',
        width=950,
        height=260,
        title='價格與買入記錄',
    )
    p_price.line('date', 'price', source=source_line, color='#1f77b4', legend_label='價格')
    p_price.scatter('date', 'price', source=source_buys, size=6, color='orange', legend_label='買入點', alpha=0.8)
    p_price.legend.location = 'top_left'

    p_eq = figure(
        x_axis_type='datetime',
        width=950,
        height=240,
        title='資產淨值變化（起始日為 0）',
    )
    p_eq.line('date', 'equity_pct', source=source_line, color='#1f77b4', legend_label='資產變化')
    p_eq.legend.location = 'top_left'

    p_dd = figure(
        x_axis_type='datetime',
        width=950,
        height=220,
        title='回撤',
    )
    p_dd.line('date', 'drawdown', source=source_line, color='#d62728', legend_label='回撤')
    p_dd.legend.location = 'top_left'

    span_select = Select(
        title='定投天數',
        value=str(dca.SPANS[0]),
        options=[str(s) for s in dca.SPANS],
    )
    thr_select = Select(
        title='觸發回撤 (%)',
        value=str(dca.THRESHOLDS[0]),
        options=[str(t) for t in dca.THRESHOLDS],
    )
    interval_select = Select(
        title='區間 (ATH→ATH)',
        value=init_label,
        options=[lab for lab, _, _ in intervals],
    )

    metrics_div = Div(text='')

    data_json = json.dumps(data_map)
    metrics_json = json.dumps(metrics_map)
    intervals_json = json.dumps(
        [{'label': lab, 'start': s, 'end': e} for lab, s, e in intervals]
    )

    init_dd = data_map[initial_key]['drawdown']
    init_min_dd = min(init_dd)
    init_max_dd = max(init_dd)
    p_dd.y_range = Range1d(
        start=init_min_dd - abs(init_min_dd) * 0.1 - 0.02,
        end=min(0, init_max_dd + 0.02),
    )

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

        const d = data[key];
        source_line.data = d;
        source_buys.data = data[`${key}_buys`];
        source_line.change.emit();
        source_buys.change.emit();

        const exitDate = d.date.length
            ? new Date(d.date[d.date.length - 1]).toLocaleDateString()
            : 'N/A';
        const exitLine = `退出價格：${fmtDollar(m.final_price)}`;
        const lastBuyLine = m.buys > 0
            ? `最後一次買入：${fmtDollar(m.last_buy_price)} 於 ${m.last_buy_date}`
            : '最後一次買入：無';

        metrics_div.text = `${m.label}<br>` +
            `<strong>假設於 ${exitDate} 退出</strong><br>` +
            `${exitLine}<br>` +
            `定投天數：${m.span}<br>` +
            `觸發回撤：${(m.threshold * 100).toFixed(0)}%<br>` +
            `投入比例：${fmtPct(m.invested_ratio)}<br>` +
            `總報酬：${fmtPct(m.total_return)}<br>` +
            `年化報酬：${fmtPct(m.annualized_return)}<br>` +
            `平均成本：${fmtDollar(m.avg_cost)}<br>` +
            `Sharpe 比率：${fmtVal(m.sharpe)}<br>` +
            `Sortino 比率：${fmtVal(m.sortino)}<br>` +
            `最低淨值：${fmtPct(m.floor_loss_pct)}（${m.min_equity_date}）<br>` +
            `低於起始資金天數：${m.days_below_start}（最長 ${m.longest_loss_streak_days} 天）<br>` +
            `最大回撤：${m.max_dd_pct.toFixed(2)}%<br>` +
            `${lastBuyLine}`;

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
        """
    )

    span_select.js_on_change('value', callback)
    thr_select.js_on_change('value', callback)
    interval_select.js_on_change('value', callback)

    init_metrics = metrics_map[initial_key]
    init_dates = data_map[initial_key]['date']
    init_exit_date = (
        pd.to_datetime(init_dates[-1], unit='ms').date().isoformat()
        if init_dates
        else 'N/A'
    )

    def fmt_pct(value: float) -> str:
        return f'{value * 100:.2f}%'

    def fmt_dollar(value: float) -> str:
        return f'${value:,.2f}'

    last_buy_line = (
        f'最後一次買入：{fmt_dollar(init_metrics["last_buy_price"])} 於 {init_metrics["last_buy_date"]}'
        if init_metrics['buys'] > 0
        else '最後一次買入：無'
    )
    metrics_div.text = (
        f'{init_metrics["label"]}<br>'
        f'<strong>假設於 {init_exit_date} 退出</strong><br>'
        f'退出價格：{fmt_dollar(init_metrics["final_price"])}<br>'
        f'定投天數：{init_metrics["span"]}<br>'
        f'觸發回撤：{init_metrics["threshold"] * 100:.0f}%<br>'
        f'投入比例：{fmt_pct(init_metrics["invested_ratio"])}<br>'
        f'期望報酬：{fmt_pct(init_metrics.get("expected_return", 0.0))}<br>'
        f'總報酬：{fmt_pct(init_metrics["total_return"])}<br>'
        f'年化報酬：{fmt_pct(init_metrics["annualized_return"])}<br>'
        f'平均成本：{fmt_dollar(init_metrics["avg_cost"])}<br>'
        f'Sharpe 比例：{init_metrics["sharpe"]:.2f}<br>'
        f'Sortino 比例：{init_metrics["sortino"]:.2f}<br>'
        f'最低淨值：{fmt_pct(init_metrics["floor_loss_pct"])}（{init_metrics["min_equity_date"]}）<br>'
        f'低於起始資金天數：{init_metrics["days_below_start"]}（最長 {init_metrics["longest_loss_streak_days"]} 天）<br>'
        f'最大回撤：{init_metrics["max_dd_pct"]:.2f}%<br>'
        f'{last_buy_line}'
    )

    layout = column(
        row(span_select, thr_select, interval_select),
        p_price,
        p_eq,
        p_dd,
        metrics_div,
    )
    save(layout)


def main() -> None:
    close = dca.read_price_data()['Close']
    intervals = dca.find_peak_intervals(close)
    data_map, metrics_map = dca.build_static_sources(close, intervals)
    output_html = RESULTS_DIR / f'btc_dca_dashboard_mvp_zh_{datetime.now():%Y%m%d_%H%M%S}.html'
    render_dashboard_cn(close, data_map, metrics_map, intervals, output_html)
    print(f'Wrote {output_html}')


if __name__ == '__main__':
    main()
