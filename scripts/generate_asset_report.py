import argparse
import pandas as pd
from pathlib import Path
import sys

def analyze_crypto(symbol, csv_path, start_date=None, end_date=None, hold_days=365):
    # Load Data
    if not csv_path.exists():
        return f"Error: File {csv_path} not found."

    # First, peek at the columns to decide how to parse dates
    peek_df = pd.read_csv(csv_path, nrows=1)
    date_col = 'datetime' if 'datetime' in peek_df.columns else 'Date'
    
    if date_col not in peek_df.columns:
         # Fallback: maybe it's the index or first column? 
         # But for now let's just try to read it and let pandas fail if not found
         pass

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    
    # Standardize column name to 'datetime'
    if date_col != 'datetime':
        df.rename(columns={date_col: 'datetime'}, inplace=True)
        
    df.sort_values('datetime', inplace=True)
    
    # Pre-calculation checks
    full_start = df['datetime'].iloc[0].date()
    full_end = df['datetime'].iloc[-1].date()

    # Auto-detect asset type and set default hold_days if not specified by user
    # (Note: argparse default is None now to allow detection, handled in main or here)
    if hold_days is None:
        # Check average time difference in the first 50 records
        subset = df['datetime'].iloc[:50]
        if len(subset) > 1:
            avg_diff = (subset.iloc[-1] - subset.iloc[0]).days / (len(subset) - 1)
            if avg_diff < 1.1:
                hold_days = 365
                asset_type = "Crypto (365 days/year)"
            else:
                hold_days = 252
                asset_type = "Stock (252 days/year)"
        else:
             # Fallback if too few data
             hold_days = 365
             asset_type = "Unknown (Default 365)"
        print(f"Detected Asset Type: {asset_type} -> Setting hold_days to {hold_days}")
    
    # Calculate future price
    df['future_close'] = df['close'].shift(-hold_days)
    df['roi'] = (df['future_close'] - df['close']) / df['close']

    # Filter by Date Range
    mask = pd.Series([True] * len(df), index=df.index)
    if start_date:
        mask &= (df['datetime'] >= start_date)
    if end_date:
        mask &= (df['datetime'] <= end_date)
    
    # We must also drop rows where future_close is NaN (the last `hold_days`)
    # However, if the user specified an end_date that is long ago, the future might be known.
    # So we filter by mask first, THEN check for validity. 
    
    analysis_df = df.loc[mask].copy()
    valid_df = analysis_df.dropna(subset=['future_close'])
    
    total_samples = len(valid_df)
    if total_samples == 0:
        return "Not enough data for the selected range."

    # Stats Calculation
    winners = valid_df[valid_df['roi'] > 0]
    losers = valid_df[valid_df['roi'] < 0]
    
    count_win = len(winners)
    count_loss = len(losers)
    prob_win = count_win / total_samples
    prob_loss = count_loss / total_samples
    
    # Mean Stats
    avg_win = winners['roi'].mean() if not winners.empty else 0
    avg_loss = abs(losers['roi'].mean()) if not losers.empty else 0
    
    # Median Stats (Robust)
    median_return = valid_df['roi'].median()
    median_win = winners['roi'].median() if not winners.empty else 0
    median_loss = abs(losers['roi'].median()) if not losers.empty else 0

    # Quartiles (Scenario Analysis)
    q1_return = valid_df['roi'].quantile(0.25)
    q3_return = valid_df['roi'].quantile(0.75)

    # Conservative Kelly Inputs
    # Win: 25th percentile of winners (Conservative profit expectation)
    conservative_win = winners['roi'].quantile(0.25) if not winners.empty else 0
    # Loss: 75th percentile of absolute losses (Conservative loss expectation - assumes larger losses)
    conservative_loss = abs(losers['roi']).quantile(0.75) if not losers.empty else 0

    # Aggressive Kelly Inputs (Risk Ceiling)
    # Win: 75th percentile of winners (Optimistic profit expectation)
    aggressive_win = winners['roi'].quantile(0.75) if not winners.empty else 0
    # Loss: 25th percentile of absolute losses (Optimistic loss expectation - assumes smaller losses)
    aggressive_loss = abs(losers['roi']).quantile(0.25) if not losers.empty else 0

    # Skewness
    skewness = valid_df['roi'].skew()
    
    # Odds Calculation
    b_odds_mean = avg_win / avg_loss if avg_loss > 0 else 0
    b_odds_median = median_win / median_loss if median_loss > 0 else 0
    b_odds_conservative = conservative_win / conservative_loss if conservative_loss > 0 else 0
    b_odds_aggressive = aggressive_win / aggressive_loss if aggressive_loss > 0 else 0

    # Kelly Criterion
    # Standard (Mean-based)
    kelly_mean = prob_win - (prob_loss / b_odds_mean) if b_odds_mean > 0 else 0
    
    # Robust (Median-based)
    kelly_median = prob_win - (prob_loss / b_odds_median) if b_odds_median > 0 else 0

    # Conservative (Q1 Win / Q3 Loss)
    kelly_conservative = prob_win - (prob_loss / b_odds_conservative) if b_odds_conservative > 0 else 0

    # Aggressive (Q3 Win / Q1 Loss)
    kelly_aggressive = prob_win - (prob_loss / b_odds_aggressive) if b_odds_aggressive > 0 else 0
    
    # Generate Markdown Content
    lines = []
    lines.append(f"# {symbol} {hold_days}天 持倉回測分析")
    lines.append(f"")
    lines.append(f"**數據來源**: `{csv_path.name}`")
    lines.append(f"**完整數據範圍**: {full_start} 至 {full_end}")
    lines.append(f"**分析進場區間**: {valid_df['datetime'].iloc[0].date()} 至 {valid_df['datetime'].iloc[-1].date()}")
    lines.append(f"")
    lines.append(f"## 統計概覽 (Statistics Overview)")
    lines.append(f"| 指標 | 數值 | 說明 |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **總交易樣本數** | {total_samples} 天 | 模擬在此區間內每天買入 |")
    lines.append(f"| **勝率 (Win Rate)** | **{prob_win:.2%}** ({count_win} 天) | 持有 {hold_days} 天後獲利的機率 |")
    lines.append(f"| **賠率 (Loss Rate)** | {prob_loss:.2%} ({count_loss} 天) | 持有 {hold_days} 天後虧損的機率 |")
    lines.append(f"| **偏度 (Skewness)** | {skewness:.2f} | >1 代表具有「暴漲長尾」特性 |")
    lines.append(f"")
    lines.append(f"## 回報率分析 (Returns Analysis)")
    lines.append(f"| 指標 | 平均值 (Mean) | 中位數 (Robust) |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **整體回報率** | {valid_df['roi'].mean():.2%} | **{median_return:.2%}** |")
    lines.append(f"| **獲利交易平均回報** | +{avg_win:.2%} | +{median_win:.2%} |")
    lines.append(f"| **虧損交易平均回報** | -{avg_loss:.2%} | -{median_loss:.2%} |")
    lines.append(f"| **盈虧比 (Reward/Risk)** | **{b_odds_mean:.2f}** | **{b_odds_median:.2f}** |")
    lines.append(f"")
    lines.append(f"## 情景分析 (Scenario Analysis - Based on Quartiles)")
    lines.append(f"| 情景 (Scenario) | 機率分界 (Percentile) | 預期回報率 (ROI) |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **樂觀情景 (Optimistic)** | Top 25% (Q3) | > **{q3_return:.2%}** |")
    lines.append(f"| **基準情景 (Baseline)** | Median (Q2) | **{median_return:.2%}** |")
    lines.append(f"| **悲觀情景 (Pessimistic)** | Bottom 25% (Q1) | < **{q1_return:.2%}** |")
    lines.append(f"")
    lines.append(f"## 凱利公式資金管理建議 (Kelly Criterion)")
    lines.append(r"> 公式: $f^* = p - \frac{q}{b}$")
    lines.append(f"")
    lines.append(f"| 策略基準 | 建議倉位 (f*) | 半凱利 (保守配置) |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **標準策略 (基於平均值)** | `{kelly_mean:.2%}` | `{kelly_mean*0.5:.2%}` |")
    lines.append(f"| **穩健策略 (基於中位數)** | `{kelly_median:.2%}` | `{kelly_median*0.5:.2%}` |")
    lines.append(f"| **保守策略 (基於保守預期)** | `{kelly_conservative:.2%}` | `{kelly_conservative*0.5:.2%}` |")
    lines.append(f"| **激進策略 (基於樂觀預期)** | `{kelly_aggressive:.2%}` | `(不建議)` |")
    lines.append(f"")
    lines.append(f"## 分析說明與風險提示")
    lines.append(f"本報告基於歷史數據進行回測，模擬在「分析進場區間」內的**每一天**都買入該資產，並嚴格持有 **{hold_days} 天**後的結果。")
    lines.append(f"")
    lines.append(f"### 1. 凱利公式策略定義")
    lines.append(f"本報告提供四種不同風險偏好的資金配置建議，請根據個人風險承受能力參考：")
    lines.append(f"*   **標準策略 (Standard):** 採用歷史平均獲利與虧損。適合相信長期機率回歸平均的投資人。")
    lines.append(f"*   **穩健策略 (Robust):** 採用歷史中位數 (Median)。過濾了極端暴漲暴跌，比標準策略更貼近一般市場常態。")
    lines.append(f"*   **保守策略 (Conservative):** 假設獲利只有 Q1 水準 (賺得少)，且虧損高達 Q3 水準 (賠得多)。這是「壓力測試」下的建議倉位。如果您擔心未來行情平庸或運氣不佳，請勿超過此比例。")
    lines.append(f"*   **激進策略 (Aggressive / Risk Ceiling):** 假設獲利能達 Q3 水準 (大賺)，且虧損僅有 Q1 水準 (小賠)。這是基於「最樂觀劇本」算出的理論上限。**若您的實際倉位超過此數值，代表您正在進行非理性的過度槓桿，風險極高。**")
    lines.append(f"")
    lines.append(f"### 2. 其他說明")
    lines.append(f"*   **為什麼要看中位數 (Robust)?** 加密貨幣常有極端暴漲行情，這會拉高「平均值」並誤導投資人。中位數能更真實地反映「一般情況下」的投資回報。")
    lines.append(f"*   **風險警示:** 歷史數據不代表未來表現。本報告僅供教育與研究用途，不構成任何財務建議。")
    
    return "\n".join(lines), hold_days

def main():
    parser = argparse.ArgumentParser(description="Generate Crypto Markdown Report")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., BTC, AAVE)")
    parser.add_argument("--file", help="Specific CSV file path (optional)")
    parser.add_argument("--start", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=None, help="Holding period days (Auto-detect if omitted)")
    
    args = parser.parse_args()
    
    # Infer path or use provided one
    if args.file:
        csv_path = Path(args.file)
    else:
        # Try Crypto pattern first
        csv_path = Path(f"data/raw/{args.symbol}USDT_1d.csv")
        if not csv_path.exists():
            # Try Generic/Stock pattern
            csv_path = Path(f"data/raw/{args.symbol}_1d.csv")
    
    report_content, hold_days = analyze_crypto(args.symbol, csv_path, args.start, args.end, args.days)
    
    # Output file
    range_str = "All_Time"
    if args.start:
        range_str = f"{args.start}_to_{args.end if args.end else 'Now'}"
        
    filename = f"{args.symbol}_{hold_days}d_Hold_{range_str}.md"
    out_path = Path("research/reports") / filename
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report generated: {out_path}")

if __name__ == "__main__":
    main()
