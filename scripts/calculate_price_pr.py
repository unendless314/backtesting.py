"""價格百分位（Price Percentile Rank, PR）分析報告。

回答的問題：「現在的價格，在過去某段期間的所有價格裡排第幾？」
PR = 期間內「價格 <= 最新價格」的天數比例。PR 90% 代表現在比過去 90% 的日子都貴。

- 區間一律依「日曆日期」切（近 1 / 3 / 5 年 + 全歷史），不依固定 K 棒筆數，
  因此股票（約 252 交易日/年）、加密貨幣（365 天/年）、基金都適用。
- 若 CSV 有 `adj_close`（含配息調整價格），另列「含息總報酬 PR」，與價格 PR 分開呈現。
- 缺失值（NaN）先排除；最新價格取「最後一筆有效價格」。

用法：
    python3 scripts/calculate_price_pr.py --symbol BTC
    python3 scripts/calculate_price_pr.py --symbol 0P000088YM --name 富達亞洲非投資等級債
"""

# 中文報告必須使用全形標點（，：（）），因此本檔停用「易混淆字元」檢查
# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# (顯示名稱, 年數)；None 代表全歷史
WINDOWS: list[tuple[str, int | None]] = [
    ('近 1 年', 1),
    ('近 3 年', 3),
    ('近 5 年', 5),
    ('全歷史', None),
]
# 資料起點與區間起點的容許落差（週末、假日、基金非每日報價）
WINDOW_TOLERANCE = pd.Timedelta(days=7)
# 最新資料距今超過此天數即警告「資料可能過期」
STALE_DAYS = 7
# 檔名允許的字元：文字（含中文）、數字與 . _ ^ = + -；其餘（含 / \\ 等路徑符號）換成 _
_UNSAFE_FILENAME_CHARS = re.compile(r'[^\w.^=+-]')
DATA_DIR = Path('data/raw')
REPORT_DIR = Path('research/reports')


def percentile_rank(series: pd.Series, value: float) -> float:
    """回傳 series 中 <= value 的比例（0–100）。NaN 不列入分母。"""
    clean = series.dropna()
    if clean.empty:
        raise ValueError('沒有有效價格可計算 PR')
    return float((clean <= value).mean() * 100)


def safe_file_stem(symbol: str) -> str:
    """把代號轉成安全的檔名片段，確保檔案只會落在指定資料夾內。

    報告內仍顯示原始代號；只有組檔名時才用這個結果。
    例：^GSPC → ^GSPC、EURUSD=X → EURUSD=X、BTC/USDT → BTC_USDT、../x → _x
    轉換後為空（例如代號全是 .）時 raise ValueError。
    """
    stem = _UNSAFE_FILENAME_CHARS.sub('_', symbol).lstrip('.')
    if not stem:
        raise ValueError(f'代號無法轉成有效檔名（收到：{symbol!r}）')
    return stem


def data_file_candidates(symbol: str) -> list[str]:
    """資料檔候選檔名（與報告檔名的轉換分開）。

    先去掉交易對的 /（BTC/USDT → BTCUSDT，與 fetch_binance 存檔命名一致），
    再做安全轉換，確保只會在 data/raw/ 內找檔。
    """
    key = safe_file_stem(symbol.replace('/', ''))
    return [f'{key}USDT_1d.csv', f'{key}_1d.csv']


def resolve_csv_path(symbol: str, data_dir: Path = DATA_DIR) -> Path | None:
    """依序嘗試 data_file_candidates() 的檔名，找不到回傳 None。"""
    for name in data_file_candidates(symbol):
        path = data_dir / name
        if path.exists():
            return path
    return None


def load_price_csv(csv_path: Path) -> pd.DataFrame:
    """讀取 CSV，統一日期欄為 `datetime` 並依日期排序。資料有問題時 raise ValueError。"""
    if not csv_path.exists():
        raise ValueError(f'找不到檔案 {csv_path}')
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f'檔案是空的：{csv_path}') from None

    date_col = next((c for c in ('datetime', 'Date') if c in df.columns), None)
    if date_col is None:
        raise ValueError(f'{csv_path} 缺少日期欄位（需要 datetime 或 Date）')
    if 'close' not in df.columns:
        raise ValueError(f'{csv_path} 缺少 close 欄位')
    if df.empty:
        raise ValueError(f'{csv_path} 只有欄位名稱，沒有任何資料列')

    df = df.rename(columns={date_col: 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 價格欄的文字、空白、無限大一律視為缺失值，後續會被排除並計入「已排除」筆數
    for col in ('close', 'adj_close'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where(df[col].abs() != float('inf'))
    return df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)


def analyze_series(dates: pd.Series, values: pd.Series) -> dict:
    """對單一價格序列（close 或 adj_close）計算各區間 PR 與統計。"""
    valid = values.notna()
    if not valid.any():
        raise ValueError('沒有有效價格可計算 PR')
    dates, clean = dates[valid], values[valid]
    latest_date, latest_value = dates.iloc[-1], float(clean.iloc[-1])
    first_date = dates.iloc[0]

    windows = []
    for label, years in WINDOWS:
        if years is None:
            cutoff = first_date
        else:
            cutoff = latest_date - pd.DateOffset(years=years)
            if first_date - cutoff > WINDOW_TOLERANCE:
                windows.append({'label': label, 'insufficient': True})
                continue
        sub = clean[dates >= cutoff]
        windows.append({
            'label': label,
            'insufficient': False,
            'start': dates[dates >= cutoff].iloc[0],
            'samples': len(sub),
            'pr': percentile_rank(sub, latest_value),
            'min': float(sub.min()),
            'median': float(sub.median()),
            'max': float(sub.max()),
            # 區間最高價 <= 0 時「距最高點」沒有意義，以 None 表示
            'from_max': (latest_value / float(sub.max()) - 1) * 100 if sub.max() > 0 else None,
        })
    return {
        'latest_date': latest_date,
        'latest_value': latest_value,
        'first_date': first_date,
        'dropped': int((~valid).sum()),
        'windows': windows,
    }


def _d(ts: pd.Timestamp) -> str:
    return ts.strftime('%Y-%m-%d')


def _p(v: float) -> str:
    """價格格式：大數字用千分位 2 位小數，小數字（基金淨值）用 4 位小數。"""
    return f'{v:,.2f}' if abs(v) >= 1000 else f'{v:.4f}'


def build_report(
    symbol: str,
    csv_path: Path,
    df: pd.DataFrame,
    name: str | None = None,
    today: pd.Timestamp | None = None,
) -> str:
    price = analyze_series(df['datetime'], df['close'])
    adj = None
    if 'adj_close' in df.columns and df['adj_close'].notna().any():
        adj = analyze_series(df['datetime'], df['adj_close'])

    title = f'{symbol}（{name}）' if name else symbol
    latest = price['latest_date']
    today = (today or pd.Timestamp.now()).normalize()
    stale_days = (today.date() - latest.date()).days

    lines = [
        f'# {title} 價格百分位分析 (Price PR)',
        '',
        f'**數據來源**: `{csv_path.name}`',
        f'**完整數據範圍**: {_d(df["datetime"].iloc[0])} 至 {_d(df["datetime"].iloc[-1])}',
        f'**最新有效價格**: {_p(price["latest_value"])}（{_d(latest)}）',
        f'**含息調整價格 (adj_close)**: {"有" if adj else "無"}',
        f'**報告產生日**: {_d(today)}',
    ]
    if price['dropped']:
        lines.append(f'**已排除缺失或無效價格**: {price["dropped"]} 筆')
    if stale_days > STALE_DAYS:
        lines.append(
            f'\n> ⚠️ 最新資料距今已 {stale_days} 天，資料可能過期，建議先更新資料再判讀。'
        )
    if adj and adj['latest_date'] != latest:
        lines.append(
            f'\n> ⚠️ 含息價格的最新有效日期為 {_d(adj["latest_date"])}，'
            f'與價格（{_d(latest)}）不同。'
        )

    lines += ['', '## PR 總覽', '']
    if adj:
        lines += ['| 區間 | 價格／淨值 PR | 含息總報酬 PR |', '| :--- | :--- | :--- |']
        # Python 3.9 的 zip 沒有 strict 參數；兩者都由 WINDOWS 產生，長度恆等
        for pw, aw in zip(price['windows'], adj['windows']):  # noqa: B905
            pr_txt = '資料不足' if pw['insufficient'] else f'**{pw["pr"]:.1f}%**'
            adj_txt = '資料不足' if aw['insufficient'] else f'{aw["pr"]:.1f}%'
            lines.append(f'| {pw["label"]} | {pr_txt} | {adj_txt} |')
    else:
        lines += ['| 區間 | 價格 PR |', '| :--- | :--- |']
        for pw in price['windows']:
            pr_txt = '資料不足' if pw['insufficient'] else f'**{pw["pr"]:.1f}%**'
            lines.append(f'| {pw["label"]} | {pr_txt} |')

    lines += [
        '',
        '## 價格區間明細',
        '',
        '| 區間 | 起始日 | 樣本數 | PR | 最低 | 中位數 | 最高 | 距區間最高點 |',
        '| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |',
    ]
    for w in price['windows']:
        if w['insufficient']:
            lines.append(
                f'| {w["label"]} | 資料不足（資料起始 {_d(price["first_date"])}） '
                '| - | - | - | - | - | - |'
            )
            continue
        from_max = '-' if w['from_max'] is None else f'{w["from_max"]:.2f}%'
        lines.append(
            f'| {w["label"]} | {_d(w["start"])} | {w["samples"]} | {w["pr"]:.1f}% '
            f'| {_p(w["min"])} | {_p(w["median"])} | {_p(w["max"])} | {from_max} |'
        )

    lines += [
        '',
        '## 解讀說明',
        '1. **PR 定義**: 區間內「價格 <= 最新價格」的天數比例。'
        'PR 90% 代表現在比該區間 90% 的日子都貴；PR 10% 代表比 90% 的日子都便宜。',
        '2. **區間長度**: 近 1 年看短期相對位置；近 3、5 年大致涵蓋一個景氣／利率循環，'
        '且不同資產可用同一把尺比較；全歷史的長度因資產而異，跨資產比較時要小心。',
    ]
    if adj:
        lines.append(
            '3. **價格 PR vs 含息 PR**: 配息型商品每次配息淨值都會下降，價格 PR 天生偏低；'
            '含息 PR 把配息加回，長期上漲的商品天生偏高。兩者回答不同問題，請勿混用。'
        )
    lines += [
        '',
        '> PR 是「溫度計」而非買賣訊號：長期上漲的資產，PR 本來就常處於高檔。',
        '',
        '**風險警示**: 歷史數據不代表未來表現。本報告僅供教育與研究用途。',
    ]
    return '\n'.join(lines)


def main(
    argv: list[str] | None = None,
    report_dir: Path = REPORT_DIR,
    data_dir: Path = DATA_DIR,
) -> int:
    """CLI 入口。report_dir / data_dir 參數僅供測試改寫位置，避免碰到真實檔案。"""
    parser = argparse.ArgumentParser(description='計算價格百分位 (PR) 並輸出 Markdown 報告')
    parser.add_argument(
        '--symbol', required=True, help='代號，例如 BTC、VOO、0P000088YM、^GSPC、EURUSD=X'
    )
    parser.add_argument('--file', help='指定 CSV 路徑（預設自動在 data/raw/ 尋找）')
    parser.add_argument('--name', help='顯示用名稱，例如 富達亞洲非投資等級債（可省略）')
    args = parser.parse_args(argv)

    try:
        stem = safe_file_stem(args.symbol)
    except ValueError as e:
        sys.stderr.write(f'錯誤：{e}\n')
        return 1

    if args.file:
        csv_path: Path | None = Path(args.file)
    else:
        try:
            csv_path = resolve_csv_path(args.symbol, data_dir)
        except ValueError as e:
            sys.stderr.write(f'錯誤：{e}\n')
            return 1
        if csv_path is None:
            tried = ' 或 '.join(data_file_candidates(args.symbol))
            sys.stderr.write(f'錯誤：在 {data_dir}/ 找不到 {tried}，請用 --file 指定路徑\n')
            return 1

    try:
        df = load_price_csv(csv_path)
        report = build_report(args.symbol, csv_path, df, args.name)
    except ValueError as e:
        sys.stderr.write(f'錯誤：{e}\n')
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f'{stem}_Price_PR.md'
    out_path.write_text(report, encoding='utf-8')
    sys.stdout.write(f'{report}\n\nReport saved to: {out_path}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
