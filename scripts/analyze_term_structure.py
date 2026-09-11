import argparse
import sys
import pandas as pd
from pathlib import Path

try:
    from ._range_utils import entry_window_mask, parse_date_range, range_suffix
except ImportError:  # 直接執行 python scripts/xxx.py 時走這裡
    from _range_utils import entry_window_mask, parse_date_range, range_suffix


def analyze_term_structure(symbol, csv_path, start_date=None, end_date=None):
    if not csv_path.exists():
        print(f"錯誤：找不到檔案 {csv_path}")
        return None

    try:
        start_ts, end_ts = parse_date_range(start_date, end_date)
    except ValueError as e:
        print(f"錯誤：{e}")
        return None

    # Load Data
    peek = pd.read_csv(csv_path, nrows=1)
    date_col = 'datetime' if 'datetime' in peek.columns else 'Date'
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    if date_col != 'datetime':
        df.rename(columns={date_col: 'datetime'}, inplace=True)
    df.sort_values('datetime', inplace=True)

    full_start = df['datetime'].iloc[0].date()
    full_end = df['datetime'].iloc[-1].date()

    # 1. Auto-detect asset type (使用全量資料前 50 筆，不受進場篩選影響)
    subset = df['datetime'].iloc[:50]
    if len(subset) > 1:
        avg_diff = (subset.iloc[-1] - subset.iloc[0]).days / (len(subset) - 1)
        if avg_diff < 1.1:
            base_year = 365
            asset_label = "Crypto (365 days/year)"
        else:
            base_year = 252
            asset_label = "Stock (252 days/year)"
    else:
        base_year = 365
        asset_label = "Unknown (Default 365)"

    periods = []
    # Generate Quarterly periods from 1Q to 20Q (5 Years)
    for q in range(1, 21):
        years = q * 0.25
        bars = int(base_year * years)
        label = f"{q} Q ({years:.2f} yr)"
        periods.append((label, bars))

    # 進場日篩選（Entry-Window Filtering）：
    # 先算好 ROI，最後才篩進場日；持有結果允許動用 end 之後的資料
    mask = entry_window_mask(df, start_ts, end_ts)

    results = []
    eff_first = None
    eff_last = None

    for label, bars in periods:
        if len(df) <= bars:
            continue

        # Calculate Future ROI for this period
        temp_df = df.copy()
        temp_df['future_close'] = temp_df['close'].shift(-bars)
        temp_df['roi'] = (temp_df['future_close'] - temp_df['close']) / temp_df['close']

        valid_df = temp_df.loc[mask].dropna(subset=['roi'])
        if valid_df.empty:
            continue  # 短區間時長天期自動跳過，不崩潰

        win_rate = (valid_df['roi'] > 0).mean()
        median_roi = valid_df['roi'].median()
        q1_roi = valid_df['roi'].quantile(0.25)
        q3_roi = valid_df['roi'].quantile(0.75)

        results.append({
            "label": label,
            "bars": bars,
            "win_rate": win_rate,
            "median": median_roi,
            "q1": q1_roi,
            "q3": q3_roi,
            "samples": len(valid_df)
        })

        # 以最短的可用天期作為報表開頭的「實際有效進場範圍」
        if eff_first is None:
            eff_first = valid_df['datetime'].iloc[0].date()
            eff_last = valid_df['datetime'].iloc[-1].date()

    if not results:
        print("錯誤：指定區間內沒有任何有效進場樣本（進場日 + 持有天數可能超過資料末端）。")
        return None

    # 指定進場區間顯示文字（未給的邊界以資料實際範圍呈現）
    if start_ts is None and end_ts is None:
        specified_range = "全歷史"
    else:
        s = str(start_ts.date()) if start_ts is not None else str(full_start)
        e = str(end_ts.date()) if end_ts is not None else str(full_end)
        specified_range = f"{s} 至 {e}"

    # --- Generate Markdown ---
    lines = []
    lines.append(f"# {symbol} 投資回報期限結構分析 (Term Structure Analysis)")
    lines.append(f"")
    lines.append(f"**數據來源**: `{csv_path.name}`")
    lines.append(f"**資產類型**: {asset_label}")
    lines.append(f"**完整數據範圍**: {full_start} 至 {full_end}")
    lines.append(f"**指定進場區間**: {specified_range}")
    lines.append(
        f"**實際有效進場範圍**: {eff_first} 至 {eff_last}"
        f"（以最短持有期為準；較長天期因需完整未來資料，進場末端會提早）"
    )
    lines.append(f"**分析方法**: 模擬在此區間內每天買入，並持有不同時長後的「整體回報率」分佈。")
    lines.append(f"")
    lines.append(f"## 回報率期限結構表")
    lines.append(
        f"| 持有期間 | K棒數 (Bars) | 樣本數 | 勝率 (Win Rate) |"
        f" 悲觀情景 (Q1) | **中性預期 (Median)** | 樂觀情景 (Q3) |"
    )
    lines.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        lines.append(
            f"| **{r['label']}** | {r['bars']} | {r['samples']} | {r['win_rate']:.2%} |"
            f" {r['q1']:.2%} | **{r['median']:.2%}** | {r['q3']:.2%} |"
        )

    lines.append(f"")
    lines.append(
        f"> 註：較長天期的實際進場末端會提早，"
        f"因為需要完整的未來資料才能計算結果（樣本數欄可看出差異）。"
    )
    lines.append(f"")
    lines.append(f"## 分析解讀")
    lines.append(f"1.  **勝率收斂**: 觀察勝率是否隨著持有時間拉長而趨近 100%。")
    lines.append(f"2.  **地板效應 (Downside Floor)**: 觀察「悲觀情景 (Q1)」在哪個時間點由負轉正。這代表了該資產的「安全持有期」。")
    lines.append(f"3.  **複利威力**: 隨著時間增加，中位數回報通常會呈現指數級增長，但相對應的風險波動（Q1 與 Q3 的間距）也會放大。")
    lines.append(f"")
    lines.append(
        f"**分析說明**: 本報告的進場日皆在指定區間內，"
        f"但持有結果可能動用指定結束日之後的資料。"
    )
    lines.append(f"**風險警示**: 歷史數據不代表未來表現。本報告僅供教育與研究用途。")

    report_content = "\n".join(lines)
    print(report_content)

    # Save to file
    out_path = (
        Path("research/reports")
        / f"{symbol}_Term_Structure_{range_suffix(start_date, end_date)}.md"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport saved to: {out_path}")
    return report_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--file", help="Specific CSV file path")
    parser.add_argument("--start", help="Start Date (YYYY-MM-DD)，最早可以進場的日期")
    parser.add_argument("--end", help="End Date (YYYY-MM-DD)，最晚可以進場的日期")
    args = parser.parse_args()

    if args.file:
        csv_path = Path(args.file)
    else:
        csv_path = Path(f"data/raw/{args.symbol}USDT_1d.csv")
        if not csv_path.exists():
            csv_path = Path(f"data/raw/{args.symbol}_1d.csv")

    result = analyze_term_structure(args.symbol, csv_path, args.start, args.end)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
