import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_term_structure(symbol, csv_path):
    if not csv_path.exists():
        print(f"Error: File {csv_path} not found.")
        return

    # Load Data
    peek = pd.read_csv(csv_path, nrows=1)
    date_col = 'datetime' if 'datetime' in peek.columns else 'Date'
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    if date_col != 'datetime':
        df.rename(columns={date_col: 'datetime'}, inplace=True)
    df.sort_values('datetime', inplace=True)

    # 1. Auto-detect asset type
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

    results = []
    
    for label, bars in periods:
        if len(df) <= bars:
            continue
            
        # Calculate Future ROI for this period
        temp_df = df.copy()
        temp_df['future_close'] = temp_df['close'].shift(-bars)
        temp_df['roi'] = (temp_df['future_close'] - temp_df['close']) / temp_df['close']
        
        valid_df = temp_df.dropna(subset=['roi'])
        if valid_df.empty:
            continue

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

    # --- Generate Markdown ---
    lines = []
    lines.append(f"# {symbol} 投資回報期限結構分析 (Term Structure Analysis)")
    lines.append(f"")
    lines.append(f"**數據來源**: `{csv_path.name}`")
    lines.append(f"**資產類型**: {asset_label}")
    lines.append(f"**分析方法**: 模擬在此區間內每天買入，並持有不同時長後的「整體回報率」分佈。")
    lines.append(f"")
    lines.append(f"## 回報率期限結構表")
    lines.append(f"| 持有期間 | K棒數 (Bars) | 勝率 (Win Rate) | 悲觀情景 (Q1) | **中性預期 (Median)** | 樂觀情景 (Q3) |")
    lines.append(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for r in results:
        lines.append(f"| **{r['label']}** | {r['bars']} | {r['win_rate']:.2%} | {r['q1']:.2%} | **{r['median']:.2%}** | {r['q3']:.2%} |")

    lines.append(f"")
    lines.append(f"## 分析解讀")
    lines.append(f"1.  **勝率收斂**: 觀察勝率是否隨著持有時間拉長而趨近 100%。")
    lines.append(f"2.  **地板效應 (Downside Floor)**: 觀察「悲觀情景 (Q1)」在哪個時間點由負轉正。這代表了該資產的「安全持有期」。")
    lines.append(f"3.  **複利威力**: 隨著時間增加，中位數回報通常會呈現指數級增長，但相對應的風險波動（Q1 與 Q3 的間距）也會放大。")
    lines.append(f"")
    lines.append(f"**風險警示**: 歷史數據不代表未來表現。本報告僅供教育與研究用途。")

    report_content = "\n".join(lines)
    print(report_content)

    # Save to file
    out_path = Path("research/reports") / f"{symbol}_Term_Structure_Analysis.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--file", help="Specific CSV file path")
    args = parser.parse_args()
    
    if args.file:
        csv_path = Path(args.file)
    else:
        csv_path = Path(f"data/raw/{args.symbol}USDT_1d.csv")
        if not csv_path.exists():
            csv_path = Path(f"data/raw/{args.symbol}_1d.csv")
            
    analyze_term_structure(args.symbol, csv_path)

if __name__ == "__main__":
    main()
