from __future__ import annotations

import base64
import html
import math
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data' / 'raw'
OUTPUT = HERE / '2026-11-07_轉倉建議_高收益債轉中國ETF.html'

COLORS = {
    'ink': '#14283b',
    'muted': '#5e6c78',
    'teal': '#138f86',
    'blue': '#3978c5',
    'amber': '#c47a16',
    'coral': '#c6534b',
    'purple': '#7962ad',
    'grid': '#dce2e5',
}


def read_prices(filename: str) -> pd.Series:
    frame = pd.read_csv(DATA / filename, parse_dates=['Date'])
    return frame.set_index('Date')['adj_close'].astype(float).sort_index()


def line_chart(
    series: list[tuple[str, pd.Series, str]],
    *,
    normalize: bool = True,
    percent_axis: bool = False,
    height: int = 430,
) -> str:
    width = 1000
    left, right, top, bottom = 72, 24, 24, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    prepared = []
    for name, values, color in series:
        clean = values.dropna()
        if normalize:
            clean = clean / clean.iloc[0] * 100
        prepared.append((name, clean, color))
    start = min(values.index.min() for _, values, _ in prepared)
    end = max(values.index.max() for _, values, _ in prepared)
    all_values = pd.concat([values for _, values, _ in prepared])
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    padding = max((y_max - y_min) * 0.12, 1 if normalize else 0.15)
    y_min, y_max = y_min - padding, y_max + padding
    if normalize:
        y_min = min(y_min, 92)
        y_max = max(y_max, 108)

    def x_pos(date: pd.Timestamp) -> float:
        return left + (date - start).total_seconds() / (end - start).total_seconds() * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="#fbfcfc"/>',
    ]
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = y_pos(value)
        label = f'{value:.0f}' if normalize else f'{value:.1f}%'
        parts.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>',
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="22" '
            f'fill="{COLORS["muted"]}">{label}</text>',
        ])
    years = list(range(start.year + 1, end.year + 1))
    if len(years) > 7:
        years = [year for year in years if year % 2 == 0]
    for year in years:
        date = pd.Timestamp(year=year, month=1, day=1)
        if not start <= date <= end:
            continue
        x = x_pos(date)
        parts.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1" stroke-dasharray="3 6"/>',
            f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-size="22" '
            f'fill="{COLORS["muted"]}">{year}</text>',
        ])
    for name, values, color in prepared:
        points = ' '.join(f'{x_pos(date):.1f},{y_pos(float(value)):.1f}' for date, value in values.items())
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3.2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        last_date, last_value = values.index[-1], float(values.iloc[-1])
        parts.append(
            f'<circle cx="{x_pos(last_date):.1f}" cy="{y_pos(last_value):.1f}" r="5" fill="{color}"/>'
        )
    legend_x = left
    for name, values, color in prepared:
        label = html.escape(name)
        if normalize:
            label += f' {float(values.iloc[-1]):.1f}'
        parts.extend([
            f'<line x1="{legend_x}" y1="12" x2="{legend_x+28}" y2="12" '
            f'stroke="{color}" stroke-width="4"/>',
            f'<text x="{legend_x+38}" y="18" font-size="22" fill="{COLORS["ink"]}">{label}</text>',
        ])
        label_width = sum(22 if ord(char) > 255 else 12 for char in label)
        legend_x += 38 + label_width + 26
    if percent_axis:
        parts.append(
            f'<text x="{left}" y="{height-2}" font-size="16" fill="{COLORS["muted"]}">'
            '單位：百分點；OAS 為殖利率減去基準利率後的利差</text>'
        )
    parts.append('</svg>')
    return ''.join(parts)


def recent_fund_metrics(filename: str) -> dict[str, float]:
    frame = pd.read_csv(DATA / filename, parse_dates=['Date']).sort_values('Date').set_index('Date')
    latest_date = frame.index[-1]
    latest = frame.iloc[-1]
    year_start = frame.loc[:'2025-12-31'].iloc[-1]
    one_year = frame.loc[latest_date - pd.DateOffset(years=1):]
    return {
        'ytd_total': float(latest['adj_close'] / year_start['adj_close'] - 1),
        'price_pr': float((one_year['close'] <= latest['close']).mean()),
        'total_pr': float((one_year['adj_close'] <= latest['adj_close']).mean()),
    }


def return_bar_chart(rows: list[tuple[str, float, str]], deposit_benchmark: float) -> str:
    width, height = 1000, 460
    left, right, top, bottom = 220, 100, 60, 62
    plot_w = width - left - right
    x_min, x_max = -1.5, 4.0

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="#fbfcfc"/>',
    ]
    for tick in [-1, 0, 1, 2, 3, 4]:
        x = x_pos(tick)
        parts.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>',
            f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" font-size="23" '
            f'fill="{COLORS["muted"]}">{tick}%</text>',
        ])
    benchmark_x = x_pos(deposit_benchmark)
    parts.extend([
        f'<line x1="{benchmark_x:.1f}" y1="34" x2="{benchmark_x:.1f}" y2="{height-bottom}" '
        f'stroke="{COLORS["ink"]}" stroke-width="3" stroke-dasharray="9 8"/>',
        f'<text x="{benchmark_x:.1f}" y="25" text-anchor="middle" font-size="23" '
        f'fill="{COLORS["ink"]}">美元定存示意 {deposit_benchmark:.2f}%</text>',
    ])
    zero_x = x_pos(0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{height-bottom}" '
        f'stroke="{COLORS["muted"]}" stroke-width="2"/>'
    )
    for i, (name, value, color) in enumerate(rows):
        y = 88 + i * 78
        value_x = x_pos(value)
        bar_x, bar_width = min(zero_x, value_x), abs(value_x - zero_x)
        label_x = value_x + 12 if value >= 0 else zero_x + 12
        anchor = 'start'
        parts.extend([
            f'<text x="{left-18}" y="{y+25}" text-anchor="end" font-size="26" '
            f'fill="{COLORS["ink"]}">{html.escape(name)}</text>',
            f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_width:.1f}" height="38" rx="7" fill="{color}"/>',
            f'<text x="{label_x:.1f}" y="{y+28}" text-anchor="{anchor}" font-size="27" '
            f'font-weight="700" fill="{color}">{value:+.2f}%</text>',
        ])
    parts.append('</svg>')
    return ''.join(parts)


def quarterly_rebalanced(prices: pd.DataFrame) -> pd.Series:
    returns = prices.pct_change().fillna(0)
    values = [0.5, 0.5]
    totals = []
    dates = list(returns.index)
    for i, date in enumerate(dates):
        values = [value * (1 + float(returns.iloc[i, j])) for j, value in enumerate(values)]
        total = sum(values)
        totals.append(total)
        if i + 1 < len(dates) and date.to_period('Q') != dates[i + 1].to_period('Q'):
            values = [total / 2, total / 2]
    return pd.Series(totals, index=prices.index)


def performance_metrics(values: pd.Series) -> tuple[float, float, float, float]:
    years = (values.index[-1] - values.index[0]).days / 365.2425
    annual_return = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    annual_volatility = values.pct_change().std() * math.sqrt(252)
    drawdown = values / values.cummax() - 1
    return annual_return, annual_volatility, drawdown.min(), drawdown.iloc[-1]


def image_data(filename: str) -> str:
    path = HERE / 'images' / filename
    mime = 'image/jpeg' if path.suffix.lower() in {'.jpg', '.jpeg'} else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def slide(
    ident: str,
    number: str,
    deck: str,
    label: str,
    title: str,
    subtitle: str,
    body: str,
    source: str,
    extra_class: str = '',
) -> str:
    return f'''
<article class="slide {extra_class}" id="{ident}" data-deck="{deck}" aria-label="{html.escape(title)}">
  <div class="slide-inner">
    <header class="slide-head">
      <div class="eyebrow"><span>{number}</span><span class="label">{label}</span></div>
      <h1>{title}</h1>
      <p class="subtitle">{subtitle}</p>
    </header>
    <main class="content">{body}</main>
    <footer class="slide-foot"><span>{source}</span><span>{number}</span></footer>
  </div>
</article>'''


def make_sheets(slides: list[str], deck: str) -> str:
    sheets = []
    for i in range(0, len(slides), 2):
        last = ' last' if i + 2 >= len(slides) else ''
        sheets.append(f'<section class="sheet {deck}-deck{last}">' + ''.join(slides[i:i+2]) + '</section>')
    return ''.join(sheets)


def build() -> str:
    fund_files = {
        '亞洲債': ('0P000088YM_1d.csv', COLORS['teal']),
        '全球債': ('0P000019VL_1d.csv', COLORS['blue']),
        '美元債': ('0P0000PO17_1d.csv', COLORS['amber']),
        '歐洲債': ('0P00001BK3.F_1d.csv', COLORS['purple']),
    }
    fund_series = [(name, read_prices(filename), color) for name, (filename, color) in fund_files.items()]
    common_funds = pd.concat([values for _, values, _ in fund_series], axis=1, join='inner').dropna()
    common_funds = common_funds.loc['2022-03-07':'2026-11-04']
    aligned_funds = [
        (name, common_funds.iloc[:, i], color)
        for i, (name, _, color) in enumerate(fund_series)
    ]
    funds_chart = line_chart(aligned_funds, height=640)
    recent_metrics = {
        name: recent_fund_metrics(filename)
        for name, (filename, _) in fund_files.items()
    }
    fund_labels = {
        '亞洲債': '富達亞洲｜USD',
        '全球債': '聯博全球｜USD',
        '美元債': '富達美元｜USD',
        '歐洲債': '富達歐洲｜EUR',
    }
    deposit_benchmark = 4 * 308 / 365
    ytd_chart = return_bar_chart([
        (fund_labels[name], recent_metrics[name]['ytd_total'] * 100, color)
        for name, (_, color) in fund_files.items()
    ], deposit_benchmark)
    pr_rows = ''.join(
        f'<tr><td><b>{fund_labels[name]}</b></td>'
        f'<td>{recent_metrics[name]["price_pr"]:.1%}</td>'
        f'<td>{recent_metrics[name]["total_pr"]:.1%}</td></tr>'
        for name in fund_files
    )

    mchi = read_prices('MCHI_1d.csv').loc['2013-11-06':'2026-11-05']
    ashr = read_prices('ASHR_1d.csv').loc['2013-11-06':'2026-11-05']
    pair = pd.concat([mchi, ashr], axis=1, join='inner').dropna()
    pair.columns = ['MCHI', 'ASHR']
    balanced = quarterly_rebalanced(pair)
    allocation_chart = line_chart([
        ('MCHI', pair['MCHI'], COLORS['teal']),
        ('ASHR', pair['ASHR'], COLORS['blue']),
        ('50/50 季再平衡', balanced, COLORS['amber']),
    ], height=320)
    for name, values in [('MCHI', pair['MCHI']), ('ASHR', pair['ASHR']), ('50/50', balanced)]:
        metrics = performance_metrics(values)
        print(name, ' | '.join(f'{value:.4%}' for value in metrics))

    oas_frame = pd.read_csv(DATA / 'BAMLH0A0HYM2.csv', parse_dates=['observation_date'])
    oas = pd.to_numeric(oas_frame.set_index('observation_date')['BAMLH0A0HYM2'], errors='coerce').dropna()
    oas_latest = float(oas.iloc[-1])
    oas_year_start = float(oas.loc[:'2025-12-31'].iloc[-1])
    oas_2026 = oas.loc['2026']
    oas_low = float(oas_2026.min())
    oas_ytd_bps = round((oas_latest - oas_year_start) * 100)
    oas_low_bps = round((oas_latest - oas_low) * 100)
    oas_chart = line_chart(
        [('美國高收益債 OAS', oas, COLORS['coral'])],
        normalize=False, percent_axis=True, height=640,
    )

    usdcny = image_data('usdcny_monthly.png')
    ratio = image_data('mchi_taiex_ratio.png')
    kalshi = image_data('kalshi_rate_hike.jpeg')
    polymarket = image_data('polymarket_fed.png')

    main = []
    main.append(slide('slide-1', '01 / 11', 'main', '配置提案',
        '目前持有四檔債基金，提案轉為 100% MCHI',
        '先辨認目前實際持有的標的，再討論為什麼要改變；這是集中於中國股票的主動選擇，不是避險操作。',
        '''<div class="proposal-flow">
          <div class="current-funds"><span class="node-kicker">目前持有｜國泰世華</span><div class="fund-name-grid">
            <div class="fund-name-card"><b>富達 亞洲非投資等級債券基金</b><span>A 股月配息｜美元</span></div>
            <div class="fund-name-card"><b>聯博 全球非投資等級債券基金</b><span>AT 級別｜美元｜配息可能為本金</span></div>
            <div class="fund-name-card"><b>富達 美元非投資等級債券基金</b><span>含 Rule 144A 債券｜美元</span></div>
            <div class="fund-name-card"><b>富達 歐洲非投資等級債券基金</b><span>A 股月配息｜歐元</span></div>
          </div></div>
          <div class="flow-arrow">贖回・換匯・買進<span>→</span></div>
          <div class="flow-node proposed"><span class="node-kicker">第一階段目標</span><strong>100% MCHI</strong><small>股票・中國・估值與政策風險</small></div>
        </div>
        <div class="premise-row">
          <div class="premise"><b>規劃 10 年</b><span>至少 5 年不需動用，以十年為投資尺度</span></div>
          <div class="premise"><b>不依賴</b><span>原有每月配息生活</span></div>
          <div class="premise"><b>能承受</b><span>較高波動與長期未收復前高</span></div>
        </div>
        <div class="decision-banner">先確認三個前提，再討論標的；任何一項不成立，都應保留其他配置選項。</div>''',
        '家庭共同討論版｜行情數據截至 2026-11-05；逢低訊號截至 2026-11-06'))

    main.append(slide('slide-2', '02 / 11', 'main', '基金與行情資料',
        '四檔現有基金：含息成果與成本',
        '配息可能包含本金，應看含息總報酬；各基金以自己的計價幣別呈現，尚未換算成台幣。',
        f'''<div class="fund-detail-layout">
          <div><table class="data-table holdings-table"><thead><tr><th>基金與級別</th><th>年內扣</th><th>近 3 年<br>年化</th><th>近 4.7 年<br>年化</th></tr></thead><tbody>
            <tr><td><b>富達 亞洲非投資等級債券基金</b><small>A 股月配息｜美元</small></td><td>1.40%</td><td>10.05%</td><td>2.63%</td></tr>
            <tr><td><b>聯博 全球非投資等級債券基金</b><small>AT 級別｜美元</small></td><td>1.63%</td><td>6.99%</td><td>4.04%</td></tr>
            <tr><td><b>富達 美元非投資等級債券基金</b><small>含 Rule 144A｜美元</small></td><td>1.39%</td><td>6.74%</td><td>3.84%</td></tr>
            <tr><td><b>富達 歐洲非投資等級債券基金</b><small>A 股月配息｜歐元</small></td><td>1.39%</td><td>5.50%</td><td>3.33%</td></tr>
          </tbody></table><div class="table-note">內扣費用已反映在淨值，不重複扣除；原始淨值也不等於完整含息成果。</div></div>
          <div class="chart-card fund-chart"><div class="chart-title">原幣還原淨值走勢｜共同起點 = 100</div>{funds_chart}</div>
        </div>
        <div class="decision-banner neutral">近 3 年與近 4.7 年答案不同：不能只用近期配息或近期反彈，代表整段持有的成果。</div>''',
        '走勢：Yahoo Finance adj_close，2022-03-07～2026-11-04｜年化報酬與費用沿用原草案，官方基準日待核對'))

    main.append(slide('slide-recent', '03 / 11', 'main', '近期績效與信用環境',
        '今年不只是淨值回落：含息報酬也偏弱',
        '2025 年底～2026-11-04，各基金以原幣呈現；PR 同截至 2026-11-04，淨值與含息回答不同問題。',
        f'''<div class="oas-change-strip"><div><span>美國高收益債 OAS｜2025 年底 → 2026-11-05</span><b>{oas_year_start:.2f}% → {oas_latest:.2f}%</b></div><div><span>今年以來擴大</span><b>+{oas_ytd_bps} bps</b></div><div><span>自 2026-08-28 低點 {oas_low:.2f}%</span><b>+{oas_low_bps} bps</b></div></div>
        <div class="recent-performance-layout">
          <div class="chart-card recent-return-chart"><div class="chart-title">2026 今年含息總報酬｜原幣</div>{ytd_chart}<div class="chart-note">虛線：年利率 4% × 308／365，僅為同期定存示意，非實際銀行報價；歐元基金不直接與美元定存排名。</div></div>
          <div class="fund-pr-panel"><div class="chart-title">近一年價格百分位 PR｜淨值與含息並列</div><table class="data-table fund-pr-table"><thead><tr><th>基金｜幣別</th><th>淨值 PR</th><th>含息 PR</th></tr></thead><tbody>{pr_rows}</tbody></table><div class="table-note">淨值 PR 全部僅 1.6%～8.1%，但除息會使淨值下降；含息 PR 為 13.4%～60.3%，不能寫成全部都在低檔。</div></div>
        </div>
        <div class="decision-banner">月配息仍然入帳，但沒有轉化成理想的今年總報酬；近期信用利差擴大，也使繼續持有的風險報酬更需要重新評估。</div>''',
        'Yahoo Finance close／adj_close 與四份 Price_PR 報告｜OAS：FRED｜美元定存為 4% 假設情境，歐元基金另需匯率分析'))

    main.append(slide('slide-3', '04 / 11', 'main', '配置動機｜2026 年 8／9 月月報',
        '我們實際借錢給誰？主要持債與風險補償',
        '數字為佔基金資產比例，四檔資料月份不同；重點是月配息背後承擔哪些風險，不是斷定整檔基金必然違約。',
        '''<div class="underlying-grid">
          <div class="underlying-card"><span>富達亞洲非投資等級債｜2026/09</span><h2>前十大 14.55%｜主權與博彩 12.10%</h2><p><b>斯里蘭卡 3.74%｜巴基斯坦 4.06%</b><br>主權債務重整、外匯儲備與 IMF 條件。</p><p><b>澳門博彩合計 4.30%</b><br>新濠博亞 1.73%、永利澳門 1.38%、新濠影滙 1.19%；受單一地區與產業景氣影響。</p></div>
          <div class="underlying-card"><span>聯博全球非投資等級債｜2026/08</span><h2>前十大 4.29%｜單一持債較分散</h2><p><b>DaVita 0.53%｜Pacifico Tres 0.46%</b><br>Ally 0.42%、Asurion 0.41%、DISH 0.41%。</p><p><b>觀察重點在整體組合</b><br>信用評等、存續期間與利差補償；不把少數案例包裝成集中曝險。</p></div>
          <div class="underlying-card"><span>富達美元非投資等級債｜2026/09</span><h2>前十大 12.14%｜企業債＋國庫券</h2><p><b>DirecTV 9.25% 票息／0.93% 權重</b><br>Nissan 7.5%／0.80%、WULF Compute 7.75%／0.80%。</p><p><b>兩檔美國國庫券合計 5.41%</b><br>Rule 144A 與企業信用需逐檔判斷，不把整檔基金視為全高風險企業債。</p></div>
          <div class="underlying-card"><span>富達歐洲非投資等級債｜2026/09</span><h2>前十大 16.90%｜企業債＋德國公債</h2><p><b>ZF 1.75%｜DTEK 1.21%</b><br>Metro Bank 1.02%、Bayer 0.99%；涵蓋產業轉型、戰爭、PIK、銀行與次順位風險。</p><p><b>德國政府債 7.12%</b><br>同時呈現公債與企業債，避免只挑高風險案例。</p></div>
        </div>
        <div class="decision-banner">每月配息不是憑空而來：目前取得的利差，是否值得繼續承擔這些信用風險，並放棄股票長期盈利增長的上行空間？</div>
        <div class="verification-note">依提供的月報資料整理；原始月報、ISIN 與其他欄位仍待逐項核對｜完整前十大持債見附錄 A10～A11。</div>''',
        'PIK 不等於當下付不出現金；Rule 144A 不是「地下債」；債券違約後仍可能有回收價值'))

    main.append(slide('slide-4', '05 / 11', 'main', '配置選項與機會成本',
        '為什麼不維持現狀：選擇要承擔的風險',
        '維持現狀不是無風險；本次提案是明確提高股票風險，以更換十年報酬引擎。',
        '''<div class="metric-strip two-metrics"><div><span>美元公債依天期｜2026-11-05</span><b>4.8%～5.3%</b></div><div><span>美國高收益債 OAS｜額外信用補償</span><b>3.48%</b></div></div>
        <div class="option-grid">
          <div class="option-card"><span class="option-num">A｜不是本次首選</span><h2>維持現狀</h2><p>保留月配息，但近 4.7 年原幣年化僅 2.63%～4.04%</p><small>仍承擔信用、利差、利率與費用；不依賴月配息時，長期吸引力下降。</small></div>
          <div class="option-card"><span class="option-num">B</span><h2>部分轉倉</h2><p>降低單一決策與時點風險</p><small>代價：同時保留兩類風險，配置較複雜。</small></div>
          <div class="option-card selected"><span class="option-num">C｜本次提案</span><h2>100% MCHI</h2><p>爭取盈利增長、股息、回購與估值修復的較高長期上行</p><small>不是同風險升級：需承受更高股票波動、中國集中與大幅回撤。</small></div>
          <div class="option-card"><span class="option-num">D｜日後再決策</span><h2>MCHI / ASHR</h2><p>50/50 分散離岸與 A 股</p><small>代價：仍全是中國股票；歷史上不等於較佳終局。</small></div>
        </div>
        <div class="decision-banner">股市若因衰退或信用事件下跌，高收益債通常也不是避險資產；即使公債利率下行，信用利差擴大仍可能使基金淨值明顯回落。</div>''',
        '公債收益率是美元資產機會成本，不是基金未來報酬；歐元基金需另作同幣別比較｜來源：聯準會 H.15、FRED'))

    main.append(slide('slide-5', '06 / 11', 'main', '十年基準情境',
        'MCHI 的長期報酬引擎，比現有債券更有吸引力',
        '不是等待債券出問題，而是主動把有限利差，換成企業盈利、股息、回購與估值修復的機會。',
        '''<div class="evidence-grid thesis-grid">
          <div class="evidence-card fact"><span class="status">為什麼離開</span><h2>債券現金流穩，但上行有限</h2><p>高收益債主要靠利息與利差，仍有信用風險；四檔基金每年另有約 1.39%～1.63% 內扣費用。</p><small>當家庭不需要月配息，這項優勢對長期財富增長的重要性就下降。</small></div>
          <div class="evidence-card hypothesis"><span class="status">為什麼換成股票</span><h2>持有企業，參與長期複利</h2><p>MCHI 讓同一筆資金成為中國大型企業的股東，報酬可來自盈利增長、股息、回購與估值修復。</p><small>企業股權的上行空間，不像債券票息那樣受到固定現金流限制。</small></div>
          <div class="evidence-card opportunity"><span class="status">為什麼是中國</span><h2>低預期與非擁擠，留下修復空間</h2><p>市場對中國資產的悲觀預期，搭配 AI 應用與量產機會，構成逆向配置的風險溢價。</p><small>不是追逐最熱門市場；台股機會成本仍以同幣別含息資料另行列示。</small></div>
        </div>
        <div class="decision-banner">基準情境：以十年為期，MCHI 有機會帶來高於現有高收益債基金的總報酬；AI 是上行增量，不是唯一成立條件。</div>''',
        '這是主動配置判斷；估值、盈利與台股同口徑比較資料仍按附錄清單補齊'))

    main.append(slide('slide-6', '07 / 11', 'main', '已核對數據｜價格與條件報酬',
        '近期回檔明確，價格位置與條件報酬支持分批',
        '252 日跌幅回答「跌多少」；PR 回答「目前排哪裡」；歷史條件報酬回答「同類訊號後結果如何」。',
        '''<div class="pr-strip"><div><span>近 1 年價格 PR</span><b>11.5%</b><small>明顯低位</small></div><div><span>近 3 年</span><b>47.1%</b><small>接近中位</small></div><div><span>近 5 年</span><b>55.4%</b><small>接近中位</small></div><div><span>全歷史</span><b>53.6%</b><small>非極端低位</small></div></div>
        <div class="signal-layout">
          <div class="signal-main">
            <div class="signal-status"><span>截至 2026-11-06｜MCHI 原始收盤價 51.89 美元</span><b>目前 252 日報酬 −18.10%</b><small>還原收盤價口徑 −16.55%；兩者都觸發 −11.91% 門檻</small></div>
            <div class="outcome-bar" aria-label="歷史價格正報酬 59.43%，負報酬 40.57%"><div class="win" style="width:59.43%">正報酬 59.43%</div><div class="loss">40.57%</div></div>
            <div class="bar-caption">同類逢低訊號後 252 日｜875 個逐日重疊樣本</div>
            <div class="return-stats"><div><span>平均價格報酬</span><b>+5.62%</b></div><div><span>中位數</span><b>+5.49%</b></div><div><span>最差四分位界線 Q1</span><b class="negative">&lt; −9.25%</b></div></div>
          </div>
          <div class="signal-caveat"><span class="status">價格位置 ≠ 估值</span><strong>近一年低檔</strong><b>不代表歷史大底</b><p>PR 是「價格不高於目前價格的歷史天數比例」，不含企業盈利或本益比。</p><small>非重疊快速檢查為 8 次、5 勝 3 敗；樣本有限。這裡支持分批時點，不把歷史比例當成未來承諾。</small></div>
        </div>''',
        'PR：calculate_price_pr.py｜逢低：test_strategy_dip_buy.py｜期末報酬 ≠ 持有期最大跌幅；完整 PR 見附錄 A4'))

    main.append(slide('slide-7', '08 / 11', 'main', '標的科普、持股與選擇',
        '三檔 ETF 實際買到什麼？持股結構決定角色',
        'ETF 是在交易所買賣的一籃子股票基金；三檔皆在美國上市、追蹤公開指數，但持股與集中度並不相同。',
        '''<div class="etf-guide-grid">
          <div class="etf-guide-card selected"><div class="ticker-line"><b>MCHI</b><span>本次第一階段</span></div><h2>BlackRock 貝萊德｜MSCI 中國指數</h2><div class="concentration"><span>前十大持股</span><b>42.3%</b><small>前五大 32.9%</small></div><ul class="holding-list"><li><span>騰訊控股</span><b>13.4%</b></li><li><span>阿里巴巴</span><b>9.3%</b></li><li><span>中國建設銀行</span><b>4.7%</b></li><li><span>中國工商銀行</span><b>2.8%</b></li><li><span>小米集團</span><b>2.7%</b></li></ul><small>科技平台＋大型金融＋消費硬體；作為廣泛配置中國大型企業的核心工具。</small></div>
          <div class="etf-guide-card"><div class="ticker-line"><b>ASHR</b><span>後續重新評估</span></div><h2>DWS｜滬深 300 指數</h2><div class="concentration"><span>前十大持股</span><b>21.7%</b><small>前五大 14.1%</small></div><ul class="holding-list"><li><span>貴州茅台</span><b>3.9%</b></li><li><span>寧德時代</span><b>3.0%</b></li><li><span>中國平安</span><b>2.5%</b></li><li><span>中際旭創</span><b>2.4%</b></li><li><span>招商銀行</span><b>2.3%</b></li></ul><small>本土消費＋製造＋金融＋硬體供應鏈；提供不同 A 股曝險，與 MCHI 互補。</small></div>
          <div class="etf-guide-card"><div class="ticker-line"><b>CQQQ</b><span>集中科技主題</span></div><h2>Invesco 景順｜富時中國科技指數</h2><div class="concentration"><span>前五大持股</span><b>46.94%</b><small>前十大 55.19%｜2026-11-05</small></div><ul class="holding-list"><li><span>拼多多</span><b>12.48%</b></li><li><span>美團</span><b>12.37%</b></li><li><span>騰訊控股</span><b>11.40%</b></li><li><span>百度</span><b>6.64%</b></li><li><span>快手</span><b>4.05%</b></li></ul><small>前五大已接近一半；另有華虹、寒武紀、舜宇等硬體持股，但不是全市場寬基或完整 AI 產業鏈。</small></div>
        </div>
        <div class="decision-banner">MCHI 已持有騰訊、拼多多與美團；加買 CQQQ 主要是加碼相近平台，而非充分分散。ASHR 的不同 A 股曝險，留作後續互補選項。</div>''',
        'MCHI／ASHR 為原草案權重；CQQQ 為 Invesco 官方 2026-11-05｜完整持股與代號見附錄 A12～A14'))

    main.append(slide('slide-8', '09 / 11', 'main', '中國 AI 上行邏輯｜產業鏈分工',
        'AI 不是只有 GPU：中國在哪一段受惠？',
        'Token 是模型處理與生成內容的使用量單位；使用量增長，會延伸到算力、記憶體、網路、電力、資料中心與終端。',
        '''<div class="ai-layer-grid">
          <div class="ai-layer"><span>上游｜運算與關鍵技術</span><h2>設計、製造、記憶體、設備</h2><p><b>美國</b>晶片設計／軟體／模型；<b>台灣</b>先進代工；<b>韓國</b>記憶體；<b>日本</b>部分設備與材料。</p><small>中國投入本土替代，但先進算力仍受技術與出口限制；買 MCHI 不等於持有輝達或台積電。</small></div>
          <div class="ai-layer"><span>中游｜實體建設與規模量產</span><h2>把運算需求變成實體設備</h2><p>伺服器、光模組、網路、電力、資料中心與製造整合，都需要持續擴充。</p><small>中國的完整供應鏈、工程能力與基建部署，是可能受惠的位置；中際旭創是原草案光通訊例子。</small></div>
          <div class="ai-layer china"><span>下游｜雲端、終端與應用</span><h2>把 AI 變成收入與生活應用</h2><p>雲端服務、大模型、軟體、手機、汽車與生活應用，最終都要轉成可計費需求。</p><small>阿里、騰訊、百度與小米是原草案公司例子；AI 收入仍須超過資本開支與競爭成本。</small></div>
        </div>
        <div class="ai-transmission three"><div><b>MCHI｜雲端、模型、應用與終端</b><p>本次核心：參與 AI 商業化，同時保留其他大型企業盈利與股東回報；不是純 AI 或上游晶片 ETF。</p></div><div><b>ASHR｜A 股硬體、製造與基建</b><p>後續檢視是否補上不同產業鏈曝險；公司屬於 AI 產業鏈，不代表一定被 ASHR 持有。</p></div><div><b>CQQQ｜集中網路平台與科技應用</b><p>與 MCHI 有重疊；主題更集中，不代表完整涵蓋晶片、電力與製造。</p></div></div>
        <div class="failure-box"><b>用後續數據驗證，而非預設必然重估</b><div class="failure-row"><span>AI 收入是否超過資本開支與競爭成本</span><span>核心企業自由現金流是否改善</span><span>晶片限制、政策與估值是否抵銷盈利增長</span></div></div>
        <div class="decision-banner">本提案買的是中國在雲端應用、規模量產與基礎建設端的潛在機會，不是把 MCHI 當成「中國版輝達」。</div>''',
        '主動產業情境｜公司例子與實際成分股權重須依官方資料核對，詳見附錄 A14'))

    main.append(slide('slide-9', '10 / 11', 'main', '壓力測試',
        '先同意最壞情境，再討論可能報酬',
        '輸入實際投入本金 V，下列剩餘價值會自動換算；50%～60% 是壓力情境，不是基準預測。',
        '''<label class="principal-input">投入本金 V（選填）<input id="principal" type="number" min="0" step="10000" inputmode="numeric" placeholder="例如 1000000"><span>元｜列印前可填寫，金額只保留在本機頁面</span></label>
        <div class="stress-grid">
          <div class="stress-card"><span>帳面下跌</span><h2>−30%</h2><div><small>剩餘價值</small><b class="money" data-factor="0.7">0.70 × V</b></div><p>回到原本金需 <strong>+42.9%</strong></p></div>
          <div class="stress-card severe"><span>帳面下跌</span><h2>−50%</h2><div><small>剩餘價值</small><b class="money" data-factor="0.5">0.50 × V</b></div><p>回到原本金需 <strong>+100.0%</strong></p></div>
          <div class="stress-card severe"><span>帳面下跌</span><h2>−60%</h2><div><small>剩餘價值</small><b class="money" data-factor="0.4">0.40 × V</b></div><p>回到原本金需 <strong>+150.0%</strong></p></div>
        </div>
        <div class="metric-strip five-year"><div><span>MCHI 五年樣本正報酬比例</span><b>53.43%</b></div><div><span>五年累積報酬中位數</span><b>+6.46%</b></div><div><span>五年 Q1 累積報酬</span><b class="negative">−23.34%</b></div></div>
        <div class="decision-banner">即使持有五年也可能負報酬；歷史最大峰谷回撤 −62.8%，與「進場後再跌」是不同口徑。</div>''',
        '五年樣本為逐日進場、持有 1,260 個交易日的重疊觀察，不是獨立事件機率', 'stress-slide'))

    main.append(slide('slide-10', '11 / 11', 'main', '共同決策',
        '執行之前，先把規則寫下來',
        '同意才執行；若仍需月配息或無法承受股票回撤，就回到其他配置選項。',
        '''<div class="execution-flow">
          <div class="execution-step"><span>第 0 階段</span><h2>共同確認</h2><p>5 年資金期限、配息替代來源、可承受損失與家庭支出幣別。</p></div>
          <div class="execution-step"><span>第一階段</span><h2>成本與固定批次</h2><p>確認贖回到帳、換匯、複委託最低收費與稅務；批次按事先同意日期。</p></div>
          <div class="execution-step"><span>每季</span><h2>書面檢視</h2><p>記錄估值、盈利、原理由與失效條件；不因單一 RSI 或匯率圖自動轉倉。</p></div>
          <div class="execution-step"><span>第二階段</span><h2>重新決策</h2><p>比較維持 MCHI、部分轉 ASHR 或其他資產；50/50 不是必然終局。</p></div>
        </div>
        <div class="consent-box"><b>家人需明確同意的四件事</b><div><span>□ 5 年以上不需動用</span><span>□ 不依賴原有月配息</span><span>□ 理解 30%／50%／60% 損失</span><span>□ 理解美元、人民幣與台幣曝險</span></div></div>
        <div class="decision-banner">保存配置理由、費用、成交價與季檢視紀錄，避免事後改寫決策依據。</div>''',
        '大額配置前，依個人身分與持有額確認台灣海外所得及美國遺產稅等專業意見'))

    appendix = []
    appendix.append(slide('appendix-a1', 'A1 / A15', 'appendix', '附錄｜配置回測',
        '三種中國股票配置的完整歷史對照',
        '共同區間、美元還原收盤價、共同有效交易日；50/50 每季再平衡，未計費稅與成交價差。',
        f'''<div class="appendix-chart">{allocation_chart}</div>
        <table class="data-table"><thead><tr><th>組合</th><th>年化報酬</th><th>年化波動</th><th>最大回撤</th><th>期末距高點</th></tr></thead><tbody>
        <tr><td>100% MCHI</td><td>2.73%</td><td>26.6%</td><td>−62.8%</td><td>−39.9%</td></tr>
        <tr><td>100% ASHR</td><td>4.74%</td><td>27.8%</td><td>−51.3%</td><td>−28.7%</td></tr>
        <tr><td>50/50，每季再平衡</td><td>4.07%</td><td>25.4%</td><td>−55.9%</td><td>−33.3%</td></tr>
        </tbody></table>''',
        '2013-11-06～2026-11-05｜日報酬標準差 × √252；年化報酬按實際日曆天數'))

    appendix.append(slide('appendix-a2', 'A2 / A15', 'appendix', '附錄｜長期持有',
        '五年持有仍可能虧損，回撤收復也需要更大漲幅',
        '五年統計採逐日進場、持有 1,260 個交易日；樣本高度重疊，不應解讀為未來獨立事件機率。',
        '''<div class="two-col">
          <div><table class="data-table large"><thead><tr><th>標的</th><th>五年正報酬</th><th>累積中位數</th><th>Q1 累積</th></tr></thead><tbody><tr><td>MCHI</td><td>53.43%</td><td>+6.46%</td><td class="negative">−23.34%</td></tr><tr><td>ASHR</td><td>44.00%</td><td class="negative">−3.07%</td><td class="negative">−12.20%</td></tr></tbody></table><div class="note-box">持有期較長，不等於低風險保證；歷史五年樣本仍有負報酬。</div></div>
          <div class="recovery-card"><span>MCHI 期末距歷史高點</span><strong>−39.9%</strong><div class="recovery-formula">1 − 39.9% = 60.1%<br>回到前高：1 ÷ 0.601 − 1</div><b>需要約 +66.4%</b><small>下跌 39.9%，不是再上漲 39.9% 就能收復。</small></div>
        </div>''',
        '資料口徑同 A1；還原收盤價為含息總報酬近似，未計投資人實際費稅'))

    adverse_bars = ''.join([
        f'<div class="h-bar-row"><span>再跌 ≥ {threshold}%</span><div class="h-bar-track"><div style="width:{max(value, 0.6)}%;background:{color}"></div></div><b>{value:.2f}%</b></div>'
        for threshold, value, color in [(30, 19.54, COLORS['amber']), (40, 8.69, COLORS['coral']), (50, 0, COLORS['coral']), (60, 0, COLORS['coral'])]
    ])
    appendix.append(slide('appendix-a3', 'A3 / A15', 'appendix', '附錄｜逢低回測',
        '期末勝率與持有期最大不利跌幅，是兩個問題',
        '875 個觸發日不是 875 次獨立實驗；同一輪空頭會連續觸發，且進場後仍可能繼續下跌。',
        f'''<div class="two-col dip-appendix"><div><div class="formula-box">進場：close ÷ close<sub>252 日前</sub> − 1 &lt; −11.91%<br>結果：進場後第 252 個交易日的原始價格報酬</div><div class="note-box"><b>非重疊快速檢查</b><br>首次觸發後 252 日內不重複進場：8 次、5 勝 3 敗、62.5%。</div><div class="note-box warning">0% 只表示既有樣本未曾發生，不代表未來機率為零，也不是峰谷最大回撤。</div></div><div class="bar-card"><h2>進場後 252 日內，自進場價再跌的樣本比例</h2>{adverse_bars}<small>以相同 875 個觸發日計算；不含股息、交易費與稅。</small></div></div>''',
        '完整 MCHI 歷史最大峰谷回撤仍為 −62.8%；不可與本頁進場後最大不利跌幅混用'))

    appendix.append(slide('appendix-a4', 'A4 / A15', 'appendix', '附錄｜價格百分位',
        'MCHI 價格 PR：近一年低位，中長期並非極端低檔',
        '截至 2026-11-06，原始收盤價 51.89 美元；PR 是價格位置，不是本益比或估值分位。',
        '''<table class="data-table pr-table"><thead><tr><th>區間</th><th>原始價格 PR</th><th>含息 PR</th><th>最低價</th><th>中位價</th><th>最高價</th><th>距區間高點</th></tr></thead><tbody>
          <tr><td>近 1 年</td><td><b>11.5%</b></td><td>11.5%</td><td>48.96</td><td>56.18</td><td>64.70</td><td>−19.80%</td></tr>
          <tr><td>近 3 年</td><td><b>47.1%</b></td><td>48.9%</td><td>36.04</td><td>52.73</td><td>66.99</td><td>−22.54%</td></tr>
          <tr><td>近 5 年</td><td><b>55.4%</b></td><td>63.0%</td><td>35.58</td><td>50.24</td><td>70.99</td><td>−26.91%</td></tr>
          <tr><td>全歷史</td><td><b>53.6%</b></td><td>68.0%</td><td>33.44</td><td>50.62</td><td>97.19</td><td>−46.61%</td></tr>
        </tbody></table>
        <div class="two-col compact-notes"><div class="note-box"><b>PR 定義</b><br>區間內「價格 ≤ 目前價格」的交易日比例。PR 11.5% 表示目前處於近一年相對低位，但不保證價格已見底。</div><div class="note-box warning"><b>為什麼分成兩種 PR？</b><br>原始價格與逢低訊號口徑一致；含息價格反映配息後的財富累積。兩者回答不同問題，不混用為估值。</div></div>
        <div class="decision-banner neutral">完整資料：2011-03-31～2026-11-06｜主簡報使用原始價格 PR；含息 PR 與區間統計保留於本頁。</div>''',
        '計算：scripts/calculate_price_pr.py｜資料：data/raw/MCHI_1d.csv｜日曆日期切分近 1／3／5 年'))

    appendix.append(slide('appendix-a5', 'A5 / A15', 'appendix', '附錄｜信用利差',
        '目前 OAS 為 3.48%，但本地資料不足以計完整歷史分位',
        '利差偏低代表承擔信用風險的額外補償較少，不代表利差無法再縮小或即將擴大。',
        f'''<div class="oas-layout"><div class="chart-card">{oas_chart}</div><div class="side-points"><div class="metric"><span>2026-11-05</span><strong>3.48%</strong><small>ICE BofA 美國高收益債 OAS</small></div><div class="point"><b>本地序列自 2023-11-07 起</b><span>不能只用這段資料宣稱完整歷史前 5%～10%</span></div><div class="point"><b>尚待補完整 FRED 序列</b><span>本頁刻意不顯示未核對的歷史分位</span></div></div></div>''',
        '來源：本地 FRED BAMLH0A0HYM2.csv｜2023-11-07～2026-11-05｜單位：百分點'))

    appendix.append(slide('appendix-a6', 'A6 / A15', 'appendix', '附錄｜原始截圖',
        'USD/CNY 月線：只支持技術情境，不支持固定貶值預測',
        'USD/CNY 下跌代表人民幣相對美元升值；RSI 接近超賣不等於匯率已見底。',
        f'''<figure class="full-screenshot"><img src="{usdcny}" alt="USD/CNY 月線完整截圖"><figcaption><b>觀察：</b>USD/CNY 約 6.6755、月線 RSI 約 30.01、黃色均線約 35.25。　<b>限制：</b>不單獨推導未來 1～2 年人民幣貶值 8%～10%，也不是 100% MCHI 的充分理由。</figcaption></figure>''',
        '來源：TradingView 截圖，2026-11-07 擷取；原圖未修改'))

    appendix.append(slide('appendix-a7', 'A7 / A15', 'appendix', '附錄｜原始截圖',
        'MCHI／TAIEX 比值：保留觀察，不作相對績效量化證據',
        '分子是美元 ETF 價格，分母是台幣價格指數；幣別與含息口徑皆不一致。',
        f'''<figure class="full-screenshot"><img src="{ratio}" alt="MCHI 與台股加權指數月線比值截圖"><figcaption><b>畫面數值：</b>RSI 約 36.35、黃色均線約 33.25。　<b>正確用途：</b>原始技術觀察。正式比較應改用 MSCI China／Taiwan 美元含息指數，或 MCHI／EWT 美元還原收盤價並註明代理限制。</figcaption></figure>''',
        '來源：TradingView 截圖，2026-11-07 擷取；主簡報不以本圖宣稱底部確立或相對報酬'))

    appendix.append(slide('appendix-a8', 'A8 / A15', 'appendix', '附錄｜歷史截圖',
        '兩張利率截圖都不是「目前年底升息機率」',
        'Kalshi 與 Polymarket 是不同合約與資料來源，不可合併成單一市場機率，也不支持舊稿的 68%～79%。',
        f'''<div class="rate-images"><figure><img src="{kalshi}" alt="Kalshi 利率市場歷史截圖"><figcaption><b>Kalshi：</b>畫面為 2027 年底利率約 4.18% 的市場隱含路徑，不是年底升息機率。</figcaption></figure><figure><img src="{polymarket}" alt="Polymarket 聯準會歷史合約截圖"><figcaption><b>Polymarket：</b>2026-10-28 的 10 月會議歷史合約，畫面 25 bps 升息 65%；不是目前前瞻證據。</figcaption></figure></div>''',
        '僅保存為歷史截圖；日後引用需標明合約全名、會議／到期日、擷取時間與成交量背景'))

    appendix.append(slide('appendix-a9', 'A9 / A15', 'appendix', '附錄｜執行成本',
        '實際報酬還要扣除轉倉、持有與稅務成本',
        '下列金額依銀行、帳戶與個人身分而異；執行前逐項取得實際費率，不用通用假設代替。',
        '''<div class="cost-grid"><div><b>贖回與帳戶</b><p>申購／贖回費、信託管理費、截止時間與預計入帳日。</p></div><div><b>換匯與交易</b><p>歐元／美元／台幣價差、複委託手續費與最低收費、ETF 買賣價差。</p></div><div><b>持有與配息</b><p>ETF 內扣費用、股息預扣稅；一般需考慮美國 30%，依投資人身分確認。</p></div><div><b>申報與傳承</b><p>台灣海外所得申報，以及非美國人士持有美國註冊 ETF 的遺產稅曝險。</p></div></div>
        <div class="note-box warning">分批買進可能提高最低手續費總額；贖回到買進之間也有資金空窗。使用固定批次與限價單，保存實際成交紀錄。</div>
        <div class="decision-banner neutral">本頁是檢查清單，不構成稅務或法律意見；大額配置前依個人情況諮詢專業人士。</div>''',
        '基金內扣費用已反映在淨值；比較歷史含息報酬時，不可再重複扣除同一筆費用'))

    appendix.append(slide('appendix-a10', 'A10 / A15', 'appendix', '附錄｜主要持債一',
        '富達亞洲與歐洲基金：完整前十大持債',
        '依提供的 2026 年 9 月月報資料整理；名稱保留原始格式，權重為佔基金資產比例。',
        '''<div class="bond-detail-grid"><div><h2 class="appendix-subtitle">富達亞洲非投資等級債｜合計 14.55%</h2><table class="data-table bond-detail-table"><thead><tr><th>主要持債</th><th>權重</th></tr></thead><tbody>
          <tr><td>SRI LANKA GOVT STEP 2/38 RGS</td><td>1.87%</td></tr><tr><td>SRI LANKA GOVT STEP 3/33 RGS</td><td>1.87%</td></tr><tr><td>MELCO RES FIN 5.375% 12/29 RGS</td><td>1.73%</td></tr><tr><td>PAKISTAN GOVT 7.375% 4/31 RGS</td><td>1.46%</td></tr><tr><td>WYNN MACAU LIM 6.75% 2/34 RGS</td><td>1.38%</td></tr><tr><td>PAKISTAN GOVT 6.875% 12/27 RGS</td><td>1.36%</td></tr><tr><td>STANDARD C 7.75%/VAR PERP RGS</td><td>1.25%</td></tr><tr><td>PAKISTAN GOVT 7.5% 3/10/32 RGS</td><td>1.24%</td></tr><tr><td>GLP PTE LTD 9.75% 05/20/28 RGS</td><td>1.20%</td></tr><tr><td>STUDIO CITY FIN 5% 1/15/29 RGS</td><td>1.19%</td></tr>
        </tbody></table></div><div><h2 class="appendix-subtitle">富達歐洲非投資等級債｜合計 16.90%</h2><table class="data-table bond-detail-table"><thead><tr><th>主要持債</th><th>權重</th></tr></thead><tbody>
          <tr><td>GERMANY GOVT 2% 12/10/26 RGS</td><td>7.12%</td></tr><tr><td>ZF EUROPE FIN 7% 06/12/30 RGS</td><td>1.75%</td></tr><tr><td>DTEK FIN PL PIK STEP 12/27 RGS</td><td>1.21%</td></tr><tr><td>ENERGO-PRO AS 8% 05/27/30 RGS</td><td>1.20%</td></tr><tr><td>DEUCE FINCO PLC 7% 11/31 RGS</td><td>1.03%</td></tr><tr><td>METRO BK HLD 12%/VAR 4/29 RGS</td><td>1.02%</td></tr><tr><td>BAYER AG 7%/VAR 09/25/83 RGS</td><td>0.99%</td></tr><tr><td>ENERGIA GRP ROI 5.5% 10/32 RGS</td><td>0.87%</td></tr><tr><td>MAXAM PRILL SAR 6% 7/15/30 RGS</td><td>0.87%</td></tr><tr><td>GRUENENTHAL G 4.625% 11/31 RGS</td><td>0.84%</td></tr>
        </tbody></table></div></div>
        <div class="note-box warning"><b>解讀邊界：</b>亞洲基金呈現主權與博彩曝險；歐洲基金同時持有德國政府債。PIK 不等於當下付不出現金，個別持債不代表整檔基金必然違約。原始月報與 ISIN 仍待逐項核對。</div>''',
        '資料月份：2026 年 9 月｜依提供之月報資料整理，尚未逐項連到原始官方文件核對'))

    appendix.append(slide('appendix-a11', 'A11 / A15', 'appendix', '附錄｜主要持債二',
        '富達美元與聯博全球基金：完整前十大持債',
        '資料月份分別為 2026 年 9 月與 8 月；不可假裝為同一基準日。',
        '''<div class="bond-detail-grid"><div><h2 class="appendix-subtitle">富達美元非投資等級債｜2026/09｜合計 12.14%</h2><table class="data-table bond-detail-table"><thead><tr><th>主要持債</th><th>權重</th></tr></thead><tbody>
          <tr><td>UST BILLS 0% 12/17/26</td><td>3.01%</td></tr><tr><td>UST BILLS 0% 01/14/27</td><td>2.40%</td></tr><tr><td>1261229 BC LTD 10% 4/32 144A</td><td>0.94%</td></tr><tr><td>DIRECTV HLDGS 9.25% 6/32 144A</td><td>0.93%</td></tr><tr><td>1011778 BC ULC 4% 10/30 144A</td><td>0.87%</td></tr><tr><td>CARNIVAL CORP 6.125% 2/33 144A</td><td>0.84%</td></tr><tr><td>SWORD PURCH 8.25% 4/15/33 144A</td><td>0.84%</td></tr><tr><td>NISSAN MOTOR 7.5% 7/17/30 144A</td><td>0.80%</td></tr><tr><td>WULF COMPUTE 7.75% 10/30 144A</td><td>0.80%</td></tr><tr><td>CLEAR CHANNEL 7.125% 2/31 144A</td><td>0.71%</td></tr>
        </tbody></table></div><div><h2 class="appendix-subtitle">聯博全球非投資等級債｜2026/08｜合計 4.29%</h2><table class="data-table bond-detail-table"><thead><tr><th>主要持債</th><th>權重</th></tr></thead><tbody>
          <tr><td>DaVita, Inc. 4.625%, 06/01/30</td><td>0.53%</td></tr><tr><td>Sopaipilla Investor LLC 7.534%, 11/30/48</td><td>0.50%</td></tr><tr><td>Fideicomiso PA Pacifico Tres 7.00%, 01/15/35</td><td>0.46%</td></tr><tr><td>New Red Finance, Inc. 4.00%, 10/15/30</td><td>0.43%</td></tr><tr><td>Ally Financial, Inc. 8.00%, 11/01/31</td><td>0.42%</td></tr><tr><td>Asurion/Asurion Co-Issuer 8.00%, 12/31/32</td><td>0.41%</td></tr><tr><td>DISH DBS Corp. 5.75%, 12/01/28</td><td>0.41%</td></tr><tr><td>Ford Motor Co. 3.25%, 02/12/32</td><td>0.39%</td></tr><tr><td>Fortescue Treasury Pty 6.125%, 04/15/32</td><td>0.37%</td></tr><tr><td>NGL Energy Operating/Finance 8.375%, 02/15/32</td><td>0.37%</td></tr>
        </tbody></table></div></div>
        <div class="note-box warning"><b>解讀邊界：</b>富達美元基金含兩檔美國國庫券合計 5.41%；聯博前十大僅 4.29%，單一持債較分散。Rule 144A 是合格機構私募轉售機制，不是「地下債」；高票息不直接等於瀕臨倒閉。</div>''',
        '依提供之月報資料整理｜整體信用評等、存續期間、到期／最差殖利率與 ISIN 仍待補'))

    appendix.append(slide('appendix-a12', 'A12 / A15', 'appendix', '附錄｜ETF 持股一',
        'MCHI 前十大持股：科技平台與大型金融並存',
        '原草案合計約 42.3%；權重與基準日尚待 BlackRock 官方持股資料逐項核對。',
        '''<table class="data-table equity-detail-table"><thead><tr><th>排名</th><th>公司</th><th>股票代號</th><th>權重</th><th>生活化業務說明</th></tr></thead><tbody>
          <tr><td>1</td><td>騰訊控股</td><td>00700.HK</td><td>13.4%</td><td>微信通訊、遊戲、金融科技與雲端</td></tr><tr><td>2</td><td>阿里巴巴</td><td>BABA / 09988.HK</td><td>9.3%</td><td>淘寶天貓、電商與阿里雲</td></tr><tr><td>3</td><td>中國建設銀行</td><td>00939.HK</td><td>4.7%</td><td>中國大型國有銀行</td></tr><tr><td>4</td><td>中國工商銀行</td><td>01398.HK</td><td>2.8%</td><td>中國大型國有銀行</td></tr><tr><td>5</td><td>小米集團</td><td>01810.HK</td><td>2.7%</td><td>手機、物聯網家電與電動車</td></tr><tr><td>6</td><td>中國銀行</td><td>03988.HK</td><td>2.2%</td><td>中國大型國有銀行</td></tr><tr><td>7</td><td>美團</td><td>03690.HK</td><td>2.0%</td><td>外送與在地生活服務</td></tr><tr><td>8</td><td>拼多多</td><td>PDD</td><td>1.8%</td><td>電商平台與 Temu 母公司</td></tr><tr><td>9</td><td>網易</td><td>NTES / 09999.HK</td><td>1.7%</td><td>遊戲與數位內容</td></tr><tr><td>10</td><td>中國平安保險</td><td>02318.HK</td><td>1.7%</td><td>綜合金融與保險</td></tr>
        </tbody></table>
        <div class="note-box"><b>結構解讀：</b>前五大合計約 32.9%；建設、工商與中國銀行三家合計約 9.7%。MCHI 不是單純科技 ETF，也不是純 AI 或上游晶片工具。</div>''',
        '原草案權重，官方基準日待核對｜公司業務為入門說明，不代表 AI 收入占比'))

    appendix.append(slide('appendix-a13', 'A13 / A15', 'appendix', '附錄｜ETF 持股二',
        'ASHR 前十大持股：A 股產業與 MCHI 有互補性',
        '原草案合計約 21.7%；權重與基準日尚待 DWS 官方持股資料逐項核對。',
        '''<table class="data-table equity-detail-table"><thead><tr><th>排名</th><th>公司</th><th>股票代號</th><th>權重</th><th>生活化業務說明</th></tr></thead><tbody>
          <tr><td>1</td><td>貴州茅台</td><td>600519.SH</td><td>3.9%</td><td>高毛利白酒與本土消費</td></tr><tr><td>2</td><td>寧德時代</td><td>300750.SZ</td><td>3.0%</td><td>電動車動力電池</td></tr><tr><td>3</td><td>中國平安</td><td>601318.SH</td><td>2.5%</td><td>綜合金融保險</td></tr><tr><td>4</td><td>中際旭創</td><td>300308.SZ</td><td>2.4%</td><td>AI 資料中心使用的高速光模組</td></tr><tr><td>5</td><td>招商銀行</td><td>600036.SH</td><td>2.3%</td><td>商業銀行與零售金融</td></tr><tr><td>6</td><td>紫金礦業</td><td>601899.SH</td><td>2.1%</td><td>金、銅等資源</td></tr><tr><td>7</td><td>美的集團</td><td>000333.SZ</td><td>1.9%</td><td>家電與工業自動化</td></tr><tr><td>8</td><td>藥明康德</td><td>603259.SH</td><td>1.7%</td><td>醫藥研發外包</td></tr><tr><td>9</td><td>長江電力</td><td>600900.SH</td><td>1.5%</td><td>水力發電</td></tr><tr><td>10</td><td>興業銀行</td><td>601166.SH</td><td>1.4%</td><td>股份制商業銀行</td></tr>
        </tbody></table>
        <div class="note-box"><b>結構解讀：</b>前五大合計約 14.1%，前十大權重較 MCHI 分散；涵蓋本土消費、製造、金融、硬體與公用事業，可補不同 A 股曝險。</div>''',
        '原草案權重，官方基準日待核對｜屬於 AI 供應鏈的公司不等於整檔 ETF 是純 AI 配置'))

    appendix.append(slide('appendix-a14', 'A14 / A15', 'appendix', '附錄｜ETF 持股三與 AI 對照',
        'CQQQ 完整前十大：平台集中，也有晶片與光學',
        'Invesco 官方 Holdings，2026-11-05；前五大合計 46.94%，前十大合計 55.19%。',
        '''<div class="two-col cqqq-appendix"><div><table class="data-table equity-detail-table compact"><thead><tr><th>公司</th><th>股票代號</th><th>權重</th><th>業務</th></tr></thead><tbody>
          <tr><td>拼多多</td><td>PDD</td><td>12.48%</td><td>電商與 Temu</td></tr><tr><td>美團</td><td>03690.HK</td><td>12.37%</td><td>外送與在地服務</td></tr><tr><td>騰訊控股</td><td>00700.HK</td><td>11.40%</td><td>微信、遊戲與雲端</td></tr><tr><td>百度</td><td>BIDU / 09888.HK</td><td>6.64%</td><td>搜尋、AI 與雲端</td></tr><tr><td>快手</td><td>01024.HK</td><td>4.05%</td><td>短影音與直播</td></tr><tr><td>華虹半導體</td><td>01347.HK</td><td>1.84%</td><td>晶圓代工</td></tr><tr><td>寒武紀</td><td>688256.SH</td><td>1.72%</td><td>AI 晶片與運算</td></tr><tr><td>舜宇光學</td><td>02382.HK</td><td>1.71%</td><td>光學鏡頭與光電產品</td></tr><tr><td>騰訊音樂</td><td>01698.HK</td><td>1.53%</td><td>線上音樂與音訊</td></tr><tr><td>商湯</td><td>00020.HK</td><td>1.45%</td><td>AI 大模型與視覺</td></tr>
        </tbody></table><div class="note-box"><b>與 MCHI 的重疊：</b>騰訊、拼多多與美團同為 MCHI 原草案成分股；加買 CQQQ 主要是提高相近平台集中度。實際重疊權重待同基準日官方資料計算。</div></div>
        <div class="ai-check-list"><h2>從平台到硬體，但仍非完整產業鏈</h2><p><b>官方表已確認：</b>CQQQ 持有華虹半導體、寒武紀、舜宇光學與商湯，除平台外也有部分晶片、光學與 AI 應用曝險。</p><p><b>其他產業鏈例子：</b>中芯國際、海光信息、中際旭創、阿里與小米。</p><p>公司屬於 AI 產業鏈，不代表一定被 MCHI、ASHR 或 CQQQ 持有；其他公司仍須逐檔確認上市地、指數成分與權重。</p><div class="note-box"><b>集中度：</b>前五大 46.94%、前十大 55.19%。第六至第十名已補齊，但持股集中與平台重疊仍是選擇時的主要考量。</div></div></div>''',
        '來源：<a href="https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&amp;productId=ETF-CQQQ">Invesco CQQQ 官方 Holdings</a>｜2026-11-05'))

    appendix.append(slide('appendix-a15', 'A15 / A15', 'appendix', '附錄｜資料與限制',
        '哪些已核對、哪些仍待補，一次列清',
        '本簡報刻意區分本地可重現數據、提供之月報資料與尚未完成的官方核對。',
        '''<div class="source-grid"><div><span class="status">已核對</span><h2>本地數據與 CQQQ 官方持股</h2><p>四檔基金、MCHI、ASHR、CQQQ 的 Yahoo Finance CSV；FRED OAS；Invesco CQQQ 官方前十大持股（2026-11-05）。</p><p>配置回測、五年樣本、逢低訊號、非重疊快速檢查，以及四檔基金與 MCHI 的原始價格／含息 PR。</p></div><div><span class="status pending-bg">仍待核對</span><h2>其他官方與同口徑比較</h2><p>其餘基金／ETF 官方持股、權重、費用、ISIN 與基準日；完整 FRED OAS 歷史分位與 MCHI／CQQQ 同基準日重疊權重。</p><p>MCHI／EWT 或 MSCI China／Taiwan 同幣別含息資料、估值與盈利比較；其他 AI 公司實際成分股歸屬。</p></div></div>
        <div class="file-box"><b>可重現來源</b><p><code>data/raw/*.csv</code>　<code>research/reports/MCHI_Strategy_DipBuy_11p91pct_Drop_All_Time.md</code></p><p><code>scripts/test_strategy_dip_buy.py</code>　<code>scripts/calculate_price_pr.py</code>　<code>research/pdf/build_presentation.py</code></p></div>
        <div class="decision-banner neutral">歷史回測、PR 與技術指標都不保證未來；本文件用於家庭討論，不取代個人化投資、稅務或法律意見。</div>''',
        '規格與數據基準日：2026-11-07｜HTML 圖表由本地資料生成，截圖保持原樣'))

    css = r'''
:root{--ink:#14283b;--muted:#5e6c78;--paper:#f7f7f3;--card:#fff;--teal:#138f86;--teal-soft:#e0f1ed;--blue:#3978c5;--amber:#c47a16;--amber-soft:#fff1d9;--coral:#c6534b;--coral-soft:#fbe8e4;--line:#dce2e5}
*{box-sizing:border-box}html,body{margin:0;padding:0}body{font-family:"PingFang TC","Noto Sans CJK TC","Microsoft JhengHei",Arial,sans-serif;background:#081421;color:var(--ink);font-variant-numeric:tabular-nums}.chrome{display:flex;align-items:center;justify-content:space-between;gap:16px;color:#d8e3eb}.topbar{min-height:52px;padding:8px 22px}.brand{font-size:14px;letter-spacing:.08em}.toolbar{display:flex;gap:8px;align-items:center}button{font:inherit;border:1px solid #385065;background:#13283a;color:#eef6f8;border-radius:8px;padding:7px 12px;cursor:pointer}button:hover,button:focus-visible{background:#1d3b50;outline:2px solid #65c9bd;outline-offset:2px}.deck-shell{min-height:100vh;display:grid;grid-template-rows:auto 1fr auto}.stage-wrap{display:grid;place-items:center;min-height:0;padding:10px 20px}.stage{position:relative;width:min(calc(100vw - 40px),calc((100vh - 132px)*16/9));aspect-ratio:16/9;overflow:hidden;background:var(--paper);box-shadow:0 18px 60px #0008;border-radius:10px}.sheet{display:contents}.slide{display:none;position:absolute;inset:0;background:var(--paper);overflow:hidden}.slide.active{display:block}.slide-inner{container-type:size;width:100%;height:100%;padding:2.5cqw 3.1cqw 1.7cqw;display:flex;flex-direction:column;background:linear-gradient(135deg,#fafaf7 0%,#f3f6f4 100%)}.slide-inner:before{content:"";position:absolute;left:0;top:0;width:100%;height:.55cqw;background:linear-gradient(90deg,var(--teal),#65c9bd 65%,var(--amber))}.slide-head{flex:none}.eyebrow{display:flex;align-items:center;gap:.8cqw;font-size:.78cqw;font-weight:700;letter-spacing:.08em;color:var(--teal);text-transform:uppercase}.label{border-left:1px solid #b8c8ca;padding-left:.8cqw}h1{font-size:2.35cqw;line-height:1.18;letter-spacing:-.025em;margin:.65cqh 0 .45cqh;font-weight:750}.subtitle{font-size:1.02cqw;line-height:1.5;color:var(--muted);margin:0;max-width:92%}.content{display:flex;flex-direction:column;gap:1.35cqh;flex:1;min-height:0;margin-top:1.6cqh}.slide-foot{display:flex;justify-content:space-between;gap:2cqw;border-top:1px solid var(--line);padding-top:.7cqh;font-size:.66cqw;line-height:1.35;color:#64737e;flex:none}.slide-foot span:first-child{max-width:92%}.split{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(0,1fr);gap:1.2cqw;flex:1;min-height:0}.chart-card,.bar-card{background:#fbfcfc;border:1px solid var(--line);border-radius:.8cqw;min-height:0;overflow:hidden}.chart-svg{display:block;width:100%;height:100%;font-family:inherit}.side-points{display:flex;flex-direction:column;gap:1cqw;justify-content:center}.metric,.point{background:#fff;border:1px solid var(--line);border-radius:.75cqw;padding:1.15cqw}.metric span,.point span{display:block;color:var(--muted);font-size:.88cqw;line-height:1.45}.metric strong{display:block;font-size:1.9cqw;color:var(--teal);margin:.45cqh 0}.metric small{font-size:.8cqw;color:var(--muted);line-height:1.4}.point b{display:block;font-size:1.02cqw;margin-bottom:.45cqh}.proposal-flow{display:grid;grid-template-columns:1fr .65fr 1fr;align-items:center;gap:1.2cqw;margin-top:1cqh}.flow-node{background:#fff;border:1px solid var(--line);border-radius:1cqw;padding:1.7cqw;min-height:15cqh;display:flex;flex-direction:column;justify-content:center}.flow-node.proposed{background:var(--teal-soft);border:.16cqw solid var(--teal)}.node-kicker{font-size:.82cqw;color:var(--teal);font-weight:700;letter-spacing:.08em}.flow-node strong{font-size:1.7cqw;margin:1cqh 0}.flow-node small{font-size:.95cqw;color:var(--muted)}.flow-arrow{text-align:center;font-size:.85cqw;color:var(--muted)}.flow-arrow span{display:block;font-size:2.4cqw;color:var(--teal)}.premise-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1cqw}.premise{background:#fff;border-left:.3cqw solid var(--amber);padding:1cqw 1.2cqw;display:flex;flex-direction:column}.premise b{font-size:1.25cqw}.premise span{font-size:.88cqw;color:var(--muted);margin-top:.4cqh}.decision-banner{background:var(--ink);color:#fff;border-radius:.65cqw;padding:1.1cqh 1.25cqw;text-align:center;font-size:.95cqw;line-height:1.45;font-weight:600}.decision-banner.neutral{background:#e7edef;color:var(--ink);border:1px solid #d0dade}.risk-switch{display:grid;grid-template-columns:1fr .18fr 1fr;gap:1cqw;align-items:stretch;flex:1}.risk-column{border:1px solid var(--line);background:#fff;border-radius:1cqw;padding:1.6cqw}.risk-column.equity{background:var(--teal-soft);border-color:#b7dcd5}.risk-column h2{font-size:1.3cqw;line-height:1.35;margin:1cqh 0}.risk-column p{font-size:.9cqw;line-height:1.55;color:var(--muted);margin:1.2cqh 0 0}.tag-row{display:flex;flex-wrap:wrap;gap:.55cqw}.tag-row span{background:#eef2f3;border-radius:10cqw;padding:.6cqh .75cqw;font-size:.82cqw}.equity .tag-row span{background:#d5ebe6}.switch-arrow{display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--teal);font-size:2.5cqw}.switch-arrow small{font-size:.68cqw;color:var(--muted);writing-mode:vertical-rl;margin-top:1cqh}.metric-strip{display:grid;grid-template-columns:repeat(3,1fr);background:#fff;border:1px solid var(--line);border-radius:.75cqw}.metric-strip div{padding:1cqh 1.2cqw;display:flex;align-items:center;justify-content:space-between;gap:1cqw}.metric-strip div+div{border-left:1px solid var(--line)}.metric-strip span{font-size:.85cqw;color:var(--muted)}.metric-strip b{font-size:1.35cqw}.negative{color:var(--coral)!important}.option-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1cqw;flex:1}.option-card{position:relative;background:#fff;border:1px solid var(--line);border-radius:.9cqw;padding:1.4cqw;display:flex;flex-direction:column}.option-card.selected{background:var(--teal-soft);border:.16cqw solid var(--teal)}.option-num{font-size:.75cqw;color:var(--teal);font-weight:700}.option-card h2{font-size:1.45cqw;margin:2cqh 0 1cqh}.option-card p{font-size:1cqw;line-height:1.45;margin:0}.option-card small{font-size:.83cqw;line-height:1.55;color:var(--muted);margin-top:auto;padding-top:2cqh}.evidence-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1cqw;flex:1}.evidence-card{background:#fff;border:1px solid var(--line);border-top:.35cqw solid var(--teal);border-radius:.8cqw;padding:1.35cqw}.evidence-card.hypothesis{border-top-color:var(--amber)}.evidence-card.pending{border-top-color:#8596a1}.status{display:inline-block;font-size:.72cqw;font-weight:700;letter-spacing:.05em;color:var(--teal);background:var(--teal-soft);padding:.45cqh .6cqw;border-radius:.3cqw}.hypothesis .status{color:var(--amber);background:var(--amber-soft)}.pending .status,.pending-bg{color:#526773;background:#e7edef}.evidence-card h2{font-size:1.25cqw;line-height:1.35;margin:1.5cqh 0 1cqh}.evidence-card p{font-size:.95cqw;line-height:1.55}.evidence-card small{font-size:.8cqw;line-height:1.55;color:var(--muted)}.signal-layout{display:grid;grid-template-columns:2.1fr 1fr;gap:1.2cqw;flex:1;min-height:0}.signal-main{background:#fff;border:1px solid var(--line);border-radius:.9cqw;padding:1.4cqw;display:flex;flex-direction:column;justify-content:space-around}.signal-status span,.signal-status small{display:block;font-size:.8cqw;color:var(--muted)}.signal-status b{display:block;font-size:1.35cqw;margin:.6cqh 0}.outcome-bar{height:7cqh;min-height:34px;display:flex;border-radius:.55cqw;overflow:hidden;color:#fff;font-size:1.05cqw;font-weight:700}.outcome-bar div{display:flex;align-items:center;justify-content:center}.outcome-bar .win{background:var(--teal)}.outcome-bar .loss{background:#ba6861;flex:1}.bar-caption{font-size:.78cqw;color:var(--muted);text-align:center}.return-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:.7cqw}.return-stats div{background:#f1f5f4;padding:1cqw;border-radius:.5cqw}.return-stats span{display:block;font-size:.75cqw;color:var(--muted)}.return-stats b{display:block;font-size:1.3cqw;margin-top:.5cqh}.signal-caveat{background:var(--amber-soft);border:1px solid #ead4ad;border-radius:.9cqw;padding:1.35cqw;display:flex;flex-direction:column;justify-content:center}.signal-caveat strong{font-size:1.35cqw;margin:1.5cqh 0 .5cqh}.signal-caveat>b{font-size:1.65cqw;color:var(--amber)}.signal-caveat p{font-size:.9cqw;line-height:1.5}.signal-caveat small{font-size:.78cqw;line-height:1.55;color:#725b38}.mchi-layout{display:grid;grid-template-columns:1fr 1.18fr;gap:1.2cqw;flex:1;min-height:0}.etf-stack{display:flex;flex-direction:column;gap:.75cqw}.etf-card{background:#fff;border:1px solid var(--line);border-radius:.65cqw;padding:.85cqw 1cqw}.etf-card.selected{border-left:.35cqw solid var(--teal);background:var(--teal-soft)}.etf-card div{display:flex;align-items:baseline;gap:.8cqw}.etf-card b{font-size:1.25cqw}.etf-card span{font-size:.85cqw;color:var(--muted)}.etf-card p{font-size:.8cqw;line-height:1.45;margin:.5cqh 0 0}.mini-note{font-size:.75cqw;line-height:1.45;color:var(--muted);padding:0 .3cqw}.screenshot-card{margin:0;background:#fff;border:1px solid var(--line);border-radius:.8cqw;padding:.75cqw;display:flex;flex-direction:column;min-height:0}.figure-title{font-size:.85cqw;font-weight:700;margin-bottom:.5cqh}.screenshot-card img{width:100%;height:100%;min-height:0;object-fit:contain;flex:1}.screenshot-card figcaption{font-size:.68cqw;line-height:1.4;color:var(--muted);padding-top:.5cqh}.ai-chain{display:grid;grid-template-columns:1fr .12fr 1fr .12fr 1fr;align-items:center;gap:.6cqw;flex:1}.chain-card{height:80%;background:#fff;border:1px solid var(--line);border-radius:.85cqw;padding:1.3cqw}.chain-card>span{font-size:.8cqw;color:var(--teal);font-weight:700}.chain-card h2{font-size:1.25cqw;margin:1.5cqh 0}.chain-card p{font-size:.9cqw;line-height:1.55;color:var(--muted)}.chain-arrow{text-align:center;color:var(--teal);font-size:2cqw}.failure-box{background:var(--coral-soft);border:1px solid #ecc8c2;border-radius:.7cqw;padding:1.1cqw}.failure-box>b{font-size:.9cqw}.failure-row{display:grid;grid-template-columns:repeat(3,1fr);gap:.8cqw;margin-top:.8cqh}.failure-row span{font-size:.8cqw;background:#fff8;padding:.75cqw;border-radius:.4cqw}.principal-input{display:flex;align-items:center;gap:.7cqw;font-size:.85cqw;font-weight:700}.principal-input input{width:12cqw;font:inherit;font-size:1cqw;padding:.65cqh .7cqw;border:1px solid #a9babd;border-radius:.4cqw;background:#fff;color:var(--ink)}.principal-input span{font-size:.7cqw;color:var(--muted);font-weight:400}.stress-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1cqw;flex:1}.stress-card{background:#fff;border:1px solid var(--line);border-top:.35cqw solid var(--amber);border-radius:.8cqw;padding:1.1cqw 1.3cqw;display:flex;flex-direction:column;justify-content:center}.stress-card.severe{border-top-color:var(--coral)}.stress-card>span{font-size:.8cqw;color:var(--muted)}.stress-card h2{font-size:2.1cqw;margin:.4cqh 0;color:var(--coral)}.stress-card div{display:flex;justify-content:space-between;align-items:center;gap:.5cqw}.stress-card small{font-size:.75cqw;color:var(--muted)}.stress-card .money{font-size:1.05cqw}.stress-card p{font-size:.85cqw;margin:1cqh 0 0}.stress-card p strong{color:var(--teal)}.five-year div{flex-direction:column;align-items:flex-start;gap:.4cqh}.five-year b{font-size:1.15cqw}.execution-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:1cqw;flex:1}.execution-step{position:relative;background:#fff;border:1px solid var(--line);border-radius:.8cqw;padding:1.2cqw}.execution-step:not(:last-child):after{content:"›";position:absolute;right:-.85cqw;top:38%;font-size:2cqw;color:var(--teal);z-index:2}.execution-step>span{font-size:.75cqw;color:var(--teal);font-weight:700}.execution-step h2{font-size:1.15cqw;margin:1.4cqh 0 1cqh}.execution-step p{font-size:.82cqw;line-height:1.55;color:var(--muted)}.consent-box{background:var(--teal-soft);border:1px solid #bfddd7;border-radius:.7cqw;padding:1cqw}.consent-box>b{font-size:.85cqw}.consent-box>div{display:grid;grid-template-columns:repeat(4,1fr);gap:.7cqw;margin-top:.8cqh}.consent-box span{font-size:.74cqw}.appendix-chart{height:52%;min-height:0;background:#fbfcfc;border:1px solid var(--line);border-radius:.8cqw;overflow:hidden}.data-table{border-collapse:collapse;width:100%;background:#fff;font-size:.85cqw;border:1px solid var(--line)}.data-table th{background:#e8efed;color:var(--ink);font-weight:700;text-align:right}.data-table td,.data-table th{padding:.8cqh .8cqw;border-bottom:1px solid var(--line);text-align:right}.data-table th:first-child,.data-table td:first-child{text-align:left}.data-table tr:last-child td{border-bottom:0}.data-table.large{font-size:.9cqw}.two-col{display:grid;grid-template-columns:1.35fr 1fr;gap:1.2cqw;flex:1;min-height:0}.note-box{background:#fff;border-left:.25cqw solid var(--teal);padding:1cqw;margin-top:1cqh;font-size:.85cqw;line-height:1.5}.note-box.warning{background:var(--amber-soft);border-left-color:var(--amber)}.recovery-card{background:var(--teal-soft);border:1px solid #bfddd7;border-radius:.8cqw;padding:1.3cqw;display:flex;flex-direction:column;justify-content:center;text-align:center}.recovery-card>span{font-size:.85cqw;color:var(--muted)}.recovery-card>strong{font-size:2.2cqw;color:var(--coral);margin:1cqh}.recovery-formula{font-size:.9cqw;line-height:1.6}.recovery-card>b{font-size:1.45cqw;color:var(--teal);margin:1cqh}.recovery-card small{font-size:.75cqw;color:var(--muted)}.formula-box{background:var(--ink);color:#fff;border-radius:.7cqw;padding:1.3cqw;font-size:.95cqw;line-height:1.8}.bar-card{padding:1.2cqw}.bar-card h2{font-size:1cqw;line-height:1.4;margin:0 0 1.5cqh}.h-bar-row{display:grid;grid-template-columns:6.5cqw 1fr 4cqw;align-items:center;gap:.6cqw;margin:1.5cqh 0;font-size:.75cqw}.h-bar-track{height:2.2cqh;min-height:8px;background:#edf1f1;border-radius:.3cqw;overflow:hidden}.h-bar-track div{height:100%;border-radius:.3cqw}.h-bar-row b{text-align:right}.bar-card>small{display:block;font-size:.72cqw;color:var(--muted);margin-top:1.5cqh}.oas-layout{display:grid;grid-template-columns:2fr 1fr;gap:1.2cqw;flex:1;min-height:0}.full-screenshot{margin:0;display:flex;flex-direction:column;min-height:0;flex:1;background:#fff;border:1px solid var(--line);border-radius:.7cqw;padding:.65cqw}.full-screenshot img{width:100%;height:100%;min-height:0;object-fit:contain;flex:1}.full-screenshot figcaption{font-size:.75cqw;line-height:1.45;color:var(--muted);padding-top:.6cqh}.rate-images{display:grid;grid-template-columns:.85fr 1.4fr;gap:1.1cqw;flex:1;min-height:0}.rate-images figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:.7cqw;padding:.7cqw;display:flex;flex-direction:column;min-height:0}.rate-images img{width:100%;height:100%;min-height:0;object-fit:contain;flex:1}.rate-images figcaption{font-size:.72cqw;line-height:1.45;color:var(--muted);padding-top:.6cqh}.cost-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1cqw;flex:1}.cost-grid>div{background:#fff;border:1px solid var(--line);border-radius:.7cqw;padding:1.15cqw}.cost-grid b{font-size:1.05cqw;color:var(--teal)}.cost-grid p{font-size:.85cqw;line-height:1.5;color:var(--muted);margin-bottom:0}.source-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.1cqw;flex:1}.source-grid>div{background:#fff;border:1px solid var(--line);border-radius:.8cqw;padding:1.25cqw}.source-grid h2{font-size:1.2cqw;margin:1.2cqh 0 .5cqh}.source-grid p{font-size:.82cqw;line-height:1.5;color:var(--muted)}.file-box{background:#eaf0f1;border-radius:.6cqw;padding:1cqw;font-size:.8cqw}.file-box p{margin:.5cqh 0}code{font-family:Menlo,Consolas,monospace;font-size:.72cqw;word-break:break-all}.bottombar{padding:8px 20px 12px;justify-content:center}.nav-status{font-size:13px;min-width:150px;text-align:center;color:#a9bcc8}.print-hint{font-size:12px;color:#8fa4b2;margin-left:10px}body.clean .chrome{display:none}body.clean .deck-shell{display:block}body.clean .stage-wrap{padding:0}body.clean .stage{width:100vw;height:100vh;aspect-ratio:auto;border-radius:0;box-shadow:none}
#slide-1 .content{justify-content:center}.proposal-flow{grid-template-columns:1.45fr .42fr .8fr}.current-funds{background:#fff;border:1px solid var(--line);border-radius:1cqw;padding:1.1cqw}.fund-name-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65cqw;margin-top:.8cqh}.fund-name-card{background:#f3f6f5;border-radius:.5cqw;padding:.8cqw}.fund-name-card b{display:block;font-size:.86cqw;line-height:1.4}.fund-name-card span{display:block;font-size:.65cqw;color:var(--muted);margin-top:.4cqh}.fund-detail-layout{display:grid;grid-template-columns:1.2fr 1fr;gap:1.1cqw;flex:1;min-height:0}.holdings-table{font-size:.74cqw;table-layout:fixed}.holdings-table th:first-child{width:46%}.holdings-table td,.holdings-table th{padding:1.15cqh .55cqw;line-height:1.3}.holdings-table td{vertical-align:middle}.holdings-table td b{font-size:.78cqw;line-height:1.4}.holdings-table td small{display:block;font-size:.63cqw;color:var(--muted);margin-top:.35cqh}.table-note{font-size:.7cqw;line-height:1.45;color:var(--muted);background:var(--amber-soft);border-left:.25cqw solid var(--amber);padding:.75cqw;margin-top:1cqh}.fund-chart{display:flex;flex-direction:column}.chart-title{font-size:.72cqw;font-weight:700;padding:.7cqw .8cqw 0}.fund-chart .chart-svg{flex:1;min-height:0}.underlying-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.8cqw;flex:1;min-height:0}.underlying-card{background:#fff;border:1px solid var(--line);border-left:.28cqw solid var(--amber);border-radius:.65cqw;padding:.85cqw 1cqw}.underlying-card>span{font-size:.67cqw;color:var(--teal);font-weight:700}.underlying-card h2{font-size:1.02cqw;margin:.45cqh 0 .6cqh}.underlying-card p{font-size:.72cqw;line-height:1.4;color:var(--muted);margin:.45cqh 0}.underlying-card p b{color:var(--ink);font-size:.77cqw}.verification-note{text-align:center;font-size:.66cqw;line-height:1.4;color:#725b38;background:var(--amber-soft);border-radius:.45cqw;padding:.65cqh .8cqw}.bond-cases-table{font-size:.7cqw;table-layout:fixed}.bond-cases-table th:not(:first-child),.bond-cases-table td:not(:first-child){text-align:left}.bond-cases-table th:first-child{width:16%}.bond-cases-table th:nth-child(2){width:25%}.bond-cases-table th:nth-child(4){width:22%}.bond-cases-table td{padding:1cqh .65cqw;line-height:1.45;vertical-align:top}
.fund-name-card b{font-size:.92cqw}.fund-name-card span{font-size:.72cqw}.holdings-table{font-size:.9cqw}.holdings-table td,.holdings-table th{padding:2.05cqh .55cqw}.holdings-table td b{font-size:.92cqw}.holdings-table td small{font-size:.72cqw}.table-note{font-size:.8cqw;padding:.9cqw}.underlying-card{display:flex;flex-direction:column;justify-content:center;padding:1.2cqw}.underlying-card>span{font-size:.8cqw}.underlying-card h2{font-size:1.2cqw;margin:.7cqh 0}.underlying-card p{font-size:.88cqw;line-height:1.5;margin:.65cqh 0}.underlying-card p b{font-size:.93cqw}.verification-note{font-size:.74cqw}.bond-cases-table{font-size:.82cqw}.bond-cases-table td{padding:1.65cqh .65cqw;line-height:1.5}#appendix-a9 .note-box{font-size:.85cqw;margin-top:2cqh}
.evidence-card.opportunity{border-top-color:var(--blue)}.opportunity .status{color:var(--blue);background:#e6eef8}.etf-guide-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1cqw;flex:1;min-height:0}.etf-guide-card{background:#fff;border:1px solid var(--line);border-radius:.85cqw;padding:1.25cqw;display:flex;flex-direction:column}.etf-guide-card.selected{background:var(--teal-soft);border:.16cqw solid var(--teal)}.ticker-line{display:flex;align-items:center;justify-content:space-between;gap:.6cqw}.ticker-line b{font-size:1.8cqw;color:var(--teal)}.ticker-line span{font-size:.67cqw;font-weight:700;color:var(--teal);background:#d9ece8;padding:.4cqh .5cqw;border-radius:.35cqw}.etf-guide-card h2{font-size:1.05cqw;margin:1.4cqh 0 .3cqh}.etf-guide-card h3{font-size:.9cqw;margin:0;color:var(--muted)}.etf-guide-card p{font-size:.85cqw;line-height:1.55;margin:1.4cqh 0}.etf-guide-card>small{font-size:.8cqw;line-height:1.55;color:var(--muted);margin-top:auto}.ai-macro-chain{display:grid;grid-template-columns:1fr .1fr 1fr .1fr 1fr .1fr 1fr;align-items:center;gap:.45cqw;flex:1}.macro-card{background:#fff;border:1px solid var(--line);border-radius:.75cqw;padding:1cqw;height:78%}.macro-card.china{background:var(--teal-soft);border-color:#b7dcd5}.macro-card>span{font-size:.68cqw;color:var(--teal);font-weight:700}.macro-card h2{font-size:1.03cqw;line-height:1.35;margin:1.1cqh 0}.macro-card p{font-size:.76cqw;line-height:1.5;color:var(--muted)}.macro-arrow{text-align:center;color:var(--teal);font-size:1.6cqw}.ai-transmission{display:grid;grid-template-columns:1fr 1fr;gap:1cqw}.ai-transmission>div{background:#edf4f3;border-radius:.6cqw;padding:.8cqw 1cqw}.ai-transmission b{font-size:.85cqw;color:var(--teal)}.ai-transmission p{font-size:.74cqw;line-height:1.45;margin:.45cqh 0 0;color:var(--muted)}
.eyebrow{font-size:.88cqw}h1{font-size:2.5cqw}.subtitle{font-size:1.16cqw;max-width:96%}.content{gap:1.1cqh;margin-top:1.3cqh}.slide-foot{font-size:.77cqw}.decision-banner{font-size:1.08cqw}.status{font-size:.82cqw}.node-kicker{font-size:.94cqw}.flow-node{min-height:20cqh}.flow-node strong{font-size:1.9cqw}.flow-node small{font-size:1.08cqw}.flow-arrow{font-size:.96cqw}.current-funds{padding:1.35cqw}.fund-name-card{padding:1.05cqw}.fund-name-card b{font-size:1.06cqw}.fund-name-card span{font-size:.82cqw}.premise{padding:1.15cqw 1.25cqw}.premise b{font-size:1.42cqw}.premise span{font-size:1.01cqw}.holdings-table{font-size:1.03cqw}.holdings-table td,.holdings-table th{padding:2.75cqh .5cqw}.holdings-table td b{font-size:1.04cqw}.holdings-table td small{font-size:.82cqw}.table-note{font-size:.91cqw}.chart-title{font-size:.85cqw}.underlying-card>span{font-size:.92cqw}.underlying-card h2{font-size:1.36cqw}.underlying-card p{font-size:1.01cqw}.underlying-card p b{font-size:1.08cqw}.verification-note{font-size:.83cqw}.option-card{justify-content:center}.option-num{font-size:.87cqw}.option-card h2{font-size:1.62cqw}.option-card p{font-size:1.14cqw}.option-card small{font-size:.97cqw;margin-top:2cqh}.evidence-card{display:flex;flex-direction:column;justify-content:center}.evidence-card h2{font-size:1.43cqw}.evidence-card p{font-size:1.09cqw}.evidence-card small{font-size:.94cqw}.signal-status span,.signal-status small{font-size:.94cqw}.signal-status b{font-size:1.52cqw}.outcome-bar{font-size:1.2cqw}.bar-caption{font-size:.89cqw}.return-stats span{font-size:.87cqw}.return-stats b{font-size:1.48cqw}.signal-caveat strong{font-size:1.5cqw}.signal-caveat>b{font-size:1.82cqw}.signal-caveat p{font-size:1.03cqw}.signal-caveat small{font-size:.89cqw}.etf-guide-card{justify-content:center}.ticker-line b{font-size:2cqw}.ticker-line span{font-size:.79cqw}.etf-guide-card h2{font-size:1.2cqw}.etf-guide-card h3{font-size:1.04cqw}.etf-guide-card p{font-size:1cqw}.etf-guide-card>small{font-size:.94cqw;margin-top:2cqh}.macro-card{height:88%;display:flex;flex-direction:column;justify-content:center}.macro-card>span{font-size:.8cqw}.macro-card h2{font-size:1.18cqw}.macro-card p{font-size:.89cqw}.ai-transmission b{font-size:.98cqw}.ai-transmission p{font-size:.86cqw}.failure-box>b{font-size:1.02cqw}.failure-row span{font-size:.91cqw}.principal-input{font-size:.98cqw}.principal-input input{font-size:1.1cqw}.principal-input span{font-size:.81cqw}.stress-card>span{font-size:.94cqw}.stress-card h2{font-size:2.38cqw}.stress-card small{font-size:.87cqw}.stress-card .money{font-size:1.2cqw}.stress-card p{font-size:.99cqw}.metric-strip span{font-size:.96cqw}.metric-strip b{font-size:1.5cqw}.five-year b{font-size:1.35cqw}.execution-step>span{font-size:.87cqw}.execution-step h2{font-size:1.31cqw}.execution-step p{font-size:.96cqw}.consent-box>b{font-size:.98cqw}.consent-box span{font-size:.86cqw}.data-table{font-size:.98cqw}.data-table.large{font-size:1.04cqw}.note-box{font-size:.98cqw}.recovery-card>span{font-size:.98cqw}.recovery-card>strong{font-size:2.45cqw}.recovery-formula{font-size:1.03cqw}.recovery-card>b{font-size:1.62cqw}.recovery-card small{font-size:.87cqw}.formula-box{font-size:1.08cqw}.bar-card h2{font-size:1.14cqw}.h-bar-row{font-size:.87cqw}.bar-card>small{font-size:.84cqw}.metric span,.point span{font-size:1.01cqw}.metric strong{font-size:2.08cqw}.metric small{font-size:.92cqw}.point b{font-size:1.15cqw}.full-screenshot figcaption{font-size:.87cqw}.rate-images figcaption{font-size:.84cqw}.cost-grid b{font-size:1.2cqw}.cost-grid p{font-size:.98cqw}.source-grid h2{font-size:1.36cqw}.source-grid p{font-size:.95cqw}.file-box{font-size:.91cqw}code{font-size:.82cqw}.bond-cases-table{font-size:.94cqw}.bond-cases-table td{padding:1.9cqh .6cqw}#appendix-a9 .note-box{font-size:.97cqw}
.two-metrics{grid-template-columns:repeat(2,1fr)}#slide-4 .option-card small{margin-top:1.2cqh}#slide-3 .underlying-card{justify-content:flex-start;padding:1cqw 1.1cqw}#slide-3 .underlying-card h2{margin:.55cqh 0}#slide-3 .underlying-card p{margin:.5cqh 0}.pr-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:.8cqw;flex:none}.pr-strip>div{display:grid;grid-template-columns:1fr auto;align-items:center;background:#fff;border:1px solid var(--line);border-radius:.6cqw;padding:.65cqh .9cqw}.pr-strip span{font-size:.88cqw;color:var(--muted)}.pr-strip b{font-size:1.5cqw;color:var(--teal)}.pr-strip small{grid-column:1/-1;font-size:.75cqw;color:var(--muted)}#slide-6 .content{gap:.85cqh}#slide-6 .signal-main{padding:1cqw}#slide-6 .outcome-bar{height:5.5cqh}.etf-guide-card{justify-content:flex-start;padding:1.05cqw}.etf-guide-card h2{font-size:1.05cqw;line-height:1.35;margin:.8cqh 0}.concentration{display:grid;grid-template-columns:1fr auto;align-items:center;background:#edf4f3;border-radius:.45cqw;padding:.55cqh .7cqw}.selected .concentration{background:#d5ebe6}.concentration span{font-size:.82cqw;color:var(--muted)}.concentration b{font-size:1.35cqw;color:var(--teal)}.concentration small{grid-column:1/-1;font-size:.72cqw;color:var(--muted)}.holding-list{list-style:none;padding:0;margin:.7cqh 0}.holding-list li{display:flex;justify-content:space-between;align-items:center;padding:.31cqh .15cqw;border-bottom:1px solid var(--line);font-size:.9cqw;line-height:1.3}.holding-list li:last-child{border-bottom:0}.holding-list li b{font-size:.94cqw}.etf-guide-card>small{font-size:.87cqw;line-height:1.45;margin-top:.8cqh}.ai-layer-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1cqw;flex:1;min-height:0}.ai-layer{background:#fff;border:1px solid var(--line);border-top:.3cqw solid var(--blue);border-radius:.7cqw;padding:1cqw;display:flex;flex-direction:column;justify-content:center}.ai-layer.china{background:var(--teal-soft);border-color:#b7dcd5;border-top-color:var(--teal)}.ai-layer>span{font-size:.8cqw;color:var(--teal);font-weight:700}.ai-layer h2{font-size:1.18cqw;line-height:1.35;margin:.8cqh 0}.ai-layer p{font-size:.91cqw;line-height:1.48;margin:.5cqh 0}.ai-layer small{font-size:.81cqw;line-height:1.48;color:var(--muted);margin-top:.6cqh}.ai-transmission.three{grid-template-columns:repeat(3,1fr)}.ai-transmission.three>div{padding:.7cqw .85cqw}.ai-transmission.three b{font-size:.9cqw}.ai-transmission.three p{font-size:.79cqw;line-height:1.4}#slide-8 .failure-box{padding:.8cqw}#slide-8 .failure-row{margin-top:.55cqh}
.pr-table{font-size:.88cqw;table-layout:fixed}.pr-table th:first-child{width:12%}.pr-table td,.pr-table th{padding:1.1cqh .45cqw;line-height:1.3}.compact-notes{flex:none;grid-template-columns:1fr 1fr}.compact-notes .note-box{margin-top:0}.bond-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:.9cqw;flex:1;min-height:0}.appendix-subtitle{font-size:1.03cqw;line-height:1.35;margin:0 0 .6cqh}.bond-detail-table{font-size:.79cqw;table-layout:fixed}.bond-detail-table th:first-child{width:84%}.bond-detail-table td,.bond-detail-table th{padding:.64cqh .45cqw;line-height:1.25;vertical-align:top}.bond-detail-table td:first-child{text-align:left;overflow-wrap:anywhere}.equity-detail-table{font-size:.88cqw;table-layout:fixed}.equity-detail-table th:first-child{width:6%}.equity-detail-table th:nth-child(2){width:17%}.equity-detail-table th:nth-child(3){width:19%}.equity-detail-table th:nth-child(4){width:10%}.equity-detail-table td,.equity-detail-table th{padding:.82cqh .55cqw;line-height:1.28;text-align:left;vertical-align:top}.equity-detail-table td:nth-child(4){text-align:right}.equity-detail-table.compact{font-size:.78cqw}.equity-detail-table.compact th:first-child{width:18%}.equity-detail-table.compact th:nth-child(2){width:27%}.equity-detail-table.compact th:nth-child(3){width:14%}.equity-detail-table.compact td:nth-child(3){text-align:right}.equity-detail-table.compact td:nth-child(4){text-align:left}.cqqq-appendix{grid-template-columns:1.25fr 1fr}.ai-check-list{background:#fff;border:1px solid var(--line);border-radius:.7cqw;padding:1cqw}.ai-check-list h2{font-size:1.08cqw;line-height:1.4;margin:0 0 .8cqh}.ai-check-list p{font-size:.85cqw;line-height:1.45;color:var(--muted)}
#slide-3 .underlying-card{justify-content:center}#slide-3 .underlying-card p{font-size:1.08cqw;line-height:1.52}#slide-3 .underlying-card p b{font-size:1.13cqw}#slide-4 .option-card h2{font-size:1.75cqw}#slide-4 .option-card p{font-size:1.22cqw}#slide-4 .option-card small{font-size:1.05cqw}#slide-4 .metric-strip b{font-size:1.65cqw}#slide-7 .etf-guide-card h2{font-size:1.12cqw}#slide-7 .concentration{padding:.75cqh .8cqw}#slide-7 .concentration span{font-size:.92cqw}#slide-7 .concentration b{font-size:1.5cqw}#slide-7 .concentration small{font-size:.82cqw}#slide-7 .holding-list li{padding:.5cqh .15cqw;font-size:1cqw}#slide-7 .holding-list li b{font-size:1.04cqw}#slide-7 .etf-guide-card>small{font-size:.96cqw;line-height:1.5}.pr-table{font-size:1.02cqw}.pr-table td,.pr-table th{padding:1.6cqh .45cqw}.compact-notes .note-box{font-size:1.05cqw;padding:1.2cqw}#appendix-a4 .content{justify-content:center}.bond-detail-table{font-size:.92cqw}.bond-detail-table td,.bond-detail-table th{padding:1cqh .5cqw;line-height:1.3}.appendix-subtitle{font-size:1.12cqw}#appendix-a10 .note-box,#appendix-a11 .note-box{font-size:1.03cqw;padding:1.15cqw}.equity-detail-table{font-size:1.02cqw}.equity-detail-table td,.equity-detail-table th{padding:1.08cqh .6cqw;line-height:1.3}#appendix-a12 .note-box,#appendix-a13 .note-box{font-size:1.05cqw;padding:1.2cqw}.equity-detail-table.compact{font-size:.92cqw}.equity-detail-table.compact td,.equity-detail-table.compact th{padding:1.25cqh .5cqw}.ai-check-list{padding:1.25cqw}.ai-check-list h2{font-size:1.2cqw}.ai-check-list p{font-size:1cqw;line-height:1.5}.ai-check-list .note-box{font-size:.96cqw}
.oas-change-strip{display:grid;grid-template-columns:1.8fr .7fr 1fr;background:#fff;border:1px solid var(--line);border-radius:.7cqw;flex:none}.oas-change-strip>div{padding:.8cqh 1cqw;display:flex;flex-direction:column;justify-content:center;gap:.35cqh}.oas-change-strip>div+div{border-left:1px solid var(--line)}.oas-change-strip span{font-size:.86cqw;line-height:1.35;color:var(--muted)}.oas-change-strip b{font-size:1.42cqw;color:var(--coral)}.recent-performance-layout{display:grid;grid-template-columns:1.28fr 1fr;gap:1.1cqw;flex:1;min-height:0}.recent-return-chart{display:flex;flex-direction:column}.recent-return-chart .chart-svg{flex:1;min-height:0}.chart-note{font-size:.78cqw;line-height:1.45;color:var(--muted);padding:0 .9cqw .7cqh}.fund-pr-panel{display:flex;flex-direction:column;justify-content:center;min-height:0}.fund-pr-table{font-size:1.05cqw;table-layout:fixed}.fund-pr-table th:first-child{width:48%}.fund-pr-table td,.fund-pr-table th{padding:1.65cqh .65cqw;line-height:1.3}.fund-pr-table td b{font-size:1.02cqw}.fund-pr-panel .table-note{margin-top:1.2cqh}#slide-recent .content{gap:1.05cqh}
@page{size:A4 portrait;margin:7mm}
@media print{html,body{background:#fff!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}.chrome{display:none!important}.deck-shell{display:block;min-height:0}.stage-wrap{display:block;padding:0}.stage{display:block;width:auto;height:auto;aspect-ratio:auto;overflow:visible;background:#fff;box-shadow:none;border-radius:0}.sheet{height:280mm;display:flex;flex-direction:column;justify-content:center;gap:5mm;break-after:page;page-break-after:always}.sheet.last{break-after:auto;page-break-after:auto}.slide{display:block!important;position:relative;inset:auto;width:196mm;height:110.25mm;flex:none;overflow:hidden;border:.2mm solid #cbd5d8;break-inside:avoid;page-break-inside:avoid}.slide-inner{padding:2.5cqw 3.1cqw 1.7cqw}body[data-print="main"] .appendix-deck{display:none!important}body[data-print="appendix"] .main-deck{display:none!important}.principal-input input{border:0;padding:0;font-weight:700}.slide-foot{color:#52616b}a{color:inherit;text-decoration:none}}
@media (max-width:700px){.brand{display:none}.topbar{justify-content:center}.print-hint{display:none}}
'''

    js = r'''
(() => {
  const slides = [...document.querySelectorAll('.slide')];
  const params = new URLSearchParams(location.search);
  const printScope = params.get('print');
  document.body.dataset.print = printScope === 'appendix' ? 'appendix' : 'main';
  if (params.get('clean') === '1') document.body.classList.add('clean');
  let index = 0;
  function indexFromHash() {
    const found = slides.findIndex(slide => '#' + slide.id === location.hash);
    if (found >= 0) index = found;
  }
  function show(next, updateHash = true) {
    index = Math.max(0, Math.min(slides.length - 1, next));
    slides.forEach((slide, i) => slide.classList.toggle('active', i === index));
    const slide = slides[index];
    document.getElementById('nav-status').textContent = `${slide.dataset.deck === 'main' ? '主簡報' : '附錄'}｜${index + 1} / ${slides.length}`;
    document.getElementById('prev').disabled = index === 0;
    document.getElementById('next').disabled = index === slides.length - 1;
    if (updateHash) history.replaceState(null, '', '#' + slide.id);
    document.title = `${slide.querySelector('h1').textContent}｜轉倉提案`;
  }
  indexFromHash();
  show(index, false);
  document.getElementById('prev').addEventListener('click', () => show(index - 1));
  document.getElementById('next').addEventListener('click', () => show(index + 1));
  document.addEventListener('keydown', event => {
    if (event.target.matches('input, textarea')) return;
    if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(index + 1); }
    if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); show(index - 1); }
    if (event.key === 'Home') { event.preventDefault(); show(0); }
    if (event.key === 'End') { event.preventDefault(); show(slides.length - 1); }
  });
  window.addEventListener('hashchange', () => { indexFromHash(); show(index, false); });
  function printDeck(scope) {
    document.body.dataset.print = scope;
    window.print();
  }
  document.getElementById('print-main').addEventListener('click', () => printDeck('main'));
  document.getElementById('print-appendix').addEventListener('click', () => printDeck('appendix'));
  const principal = document.getElementById('principal');
  principal.addEventListener('input', () => {
    const value = Number(principal.value);
    document.querySelectorAll('.money').forEach(node => {
      const factor = Number(node.dataset.factor);
      node.textContent = principal.value && Number.isFinite(value)
        ? new Intl.NumberFormat('zh-TW', {maximumFractionDigits: 0}).format(value * factor) + ' 元'
        : factor.toFixed(2) + ' × V';
    });
  });
})();
'''

    document = f'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="高收益債基金轉中國股票 ETF 的家庭共同決策簡報">
<title>轉倉提案｜高收益債基金 → 中國股票 ETF</title>
<style>{css}</style>
</head>
<body data-print="main">
<div class="deck-shell">
  <header class="topbar chrome"><div class="brand">家庭資產配置討論｜2026-11-07</div><div class="toolbar"><button id="print-main" type="button">列印主簡報（6 頁 A4）</button><button id="print-appendix" type="button">列印附錄</button></div></header>
  <div class="stage-wrap"><div class="stage">{make_sheets(main, 'main')}{make_sheets(appendix, 'appendix')}</div></div>
  <footer class="bottombar chrome"><button id="prev" type="button" aria-label="上一頁">← 上一頁</button><span id="nav-status" class="nav-status"></span><button id="next" type="button" aria-label="下一頁">下一頁 →</button><span class="print-hint">左右鍵翻頁｜建議 Chrome 列印：A4、直向、背景圖形開啟、頁首頁尾關閉</span></footer>
</div>
<script>{js}</script>
</body>
</html>'''
    return document


if __name__ == '__main__':
    OUTPUT.write_text(build(), encoding='utf-8')
    print(f'已產生：{OUTPUT}')
    print(f'檔案大小：{OUTPUT.stat().st_size / 1024 / 1024:.2f} MB')
