import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def run_dip_buy_test(symbol, csv_path, hold_days=365, threshold_pct=-0.1115):
    # Load Data
    if not csv_path.exists():
        print(f"Error: File {csv_path} not found.")
        return

    # Peek to check column names
    peek = pd.read_csv(csv_path, nrows=1)
    date_col = 'datetime' if 'datetime' in peek.columns else 'Date'

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    
    # Normalize date column
    if date_col != 'datetime':
        df.rename(columns={date_col: 'datetime'}, inplace=True)
        
    df.sort_values('datetime', inplace=True)

    # Auto-detect asset type and set default hold_days if not specified
    if hold_days is None:
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
             hold_days = 365
             asset_type = "Unknown (Default 365)"
        print(f"Detected Asset Type: {asset_type} -> Setting hold_days to {hold_days}")
    
    # 1. Calculate Past 1-Year Return (Momentum)
    df['past_return'] = df['close'].pct_change(periods=hold_days)
    
    # 2. Calculate Future Return (The Outcome)
    df['future_close'] = df['close'].shift(-hold_days)
    df['future_roi'] = (df['future_close'] - df['close']) / df['close']

    # Remove rows where we don't have enough data
    valid_df = df.dropna(subset=['past_return', 'future_roi']).copy()
    
    if valid_df.empty:
        print("Not enough data to run the analysis.")
        return
    
    # --- Strategy Group: Buy Only when Past Return < Threshold ---
    signal_mask = valid_df['past_return'] < threshold_pct
    strat_df = valid_df[signal_mask].copy()
    
    strat_count = len(strat_df)
    total_count = len(valid_df)

    if strat_count == 0:
        print(f"No trading days found matching the criteria (Dip < {threshold_pct:.2%}).")
        return

    # Stats Calculation for Strategy Group
    winners = strat_df[strat_df['future_roi'] > 0]
    losers = strat_df[strat_df['future_roi'] < 0]

    count_win = len(winners)
    count_loss = len(losers)
    prob_win = count_win / strat_count
    prob_loss = count_loss / strat_count

    # Mean Stats
    avg_win = winners['future_roi'].mean() if not winners.empty else 0
    avg_loss = abs(losers['future_roi'].mean()) if not losers.empty else 0
    
    # Median Stats
    median_return = strat_df['future_roi'].median()
    median_win = winners['future_roi'].median() if not winners.empty else 0
    median_loss = abs(losers['future_roi'].median()) if not losers.empty else 0

    # Quartiles
    q1_return = strat_df['future_roi'].quantile(0.25)
    q3_return = strat_df['future_roi'].quantile(0.75)

    # Conservative Kelly Inputs
    # Win: 25th percentile of winners (Conservative profit expectation)
    conservative_win = winners['future_roi'].quantile(0.25) if not winners.empty else 0
    # Loss: 75th percentile of absolute losses (Conservative loss expectation - assumes larger losses)
    conservative_loss = abs(losers['future_roi']).quantile(0.75) if not losers.empty else 0

    # Aggressive Kelly Inputs (Risk Ceiling)
    # Win: 75th percentile of winners (Optimistic profit expectation)
    aggressive_win = winners['future_roi'].quantile(0.75) if not winners.empty else 0
    # Loss: 25th percentile of absolute losses (Optimistic loss expectation - assumes smaller losses)
    aggressive_loss = abs(losers['future_roi']).quantile(0.25) if not losers.empty else 0

    # Skewness
    skewness = strat_df['future_roi'].skew()

    # Odds
    b_odds_mean = avg_win / avg_loss if avg_loss > 0 else 0
    b_odds_median = median_win / median_loss if median_loss > 0 else 0
    b_odds_conservative = conservative_win / conservative_loss if conservative_loss > 0 else 0
    b_odds_aggressive = aggressive_win / aggressive_loss if aggressive_loss > 0 else 0

    # Kelly Criterion
    kelly_mean = prob_win - (prob_loss / b_odds_mean) if b_odds_mean > 0 else 0
    kelly_median = prob_win - (prob_loss / b_odds_median) if b_odds_median > 0 else 0
    kelly_conservative = prob_win - (prob_loss / b_odds_conservative) if b_odds_conservative > 0 else 0
    kelly_aggressive = prob_win - (prob_loss / b_odds_aggressive) if b_odds_aggressive > 0 else 0

    # --- Generate Markdown Content ---
    lines = []
    # Title indicating Strategy
    lines.append(f"# {symbol} 逢低買入策略 (Dip Buy Strategy)")
    lines.append(f"")
    lines.append(f"**數據來源**: `{csv_path.name}`")
    lines.append(f"**策略條件**: 當「過去一年報酬率」低於 `{threshold_pct:.2%}` 時買入")
    lines.append(f"**持有期間**: {hold_days} 天")
    lines.append(f"**分析進場區間**: {valid_df['datetime'].iloc[0].date()} 至 {valid_df['datetime'].iloc[-1].date()}")
    lines.append(f"")
    lines.append(f"## 統計概覽 (Statistics Overview)")
    lines.append(f"| 指標 | 數值 | 說明 |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **觸發交易天數** | {strat_count} 天 | 佔總樣本數的 {strat_count/total_count:.1%} |")
    lines.append(f"| **勝率 (Win Rate)** | **{prob_win:.2%}** ({count_win} 天) | 買入持有 {hold_days} 天後獲利的機率 |")
    lines.append(f"| **賠率 (Loss Rate)** | {prob_loss:.2%} ({count_loss} 天) | 買入持有 {hold_days} 天後虧損的機率 |")
    lines.append(f"| **偏度 (Skewness)** | {skewness:.2f} | >1 代表具有「暴漲長尾」特性 |")
    lines.append(f"")
    lines.append(f"## 回報率分析 (Returns Analysis)")
    lines.append(f"| 指標 | 平均值 (Mean) | 中位數 (Robust) |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **整體回報率** | {strat_df['future_roi'].mean():.2%} | **{median_return:.2%}** |")
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
    lines.append(f"本報告基於歷史數據進行回測，僅統計符合「策略條件」的進場日表現。")
    lines.append(f"")
    lines.append(f"### 1. 凱利公式策略定義")
    lines.append(f"本報告提供四種不同風險偏好的資金配置建議，請根據個人風險承受能力參考：")
    lines.append(f"*   **標準策略 (Standard):** 採用歷史平均獲利與虧損。適合相信長期機率回歸平均的投資人。")
    lines.append(f"*   **穩健策略 (Robust):** 採用歷史中位數 (Median)。過濾了極端暴漲暴跌，比標準策略更貼近一般市場常態。")
    lines.append(f"*   **保守策略 (Conservative):** 假設獲利只有 Q1 水準 (賺得少)，且虧損高達 Q3 水準 (賠得多)。這是「壓力測試」下的建議倉位。如果您擔心未來行情平庸或運氣不佳，請勿超過此比例。")
    lines.append(f"*   **激進策略 (Aggressive / Risk Ceiling):** 假設獲利能達 Q3 水準 (大賺)，且虧損僅有 Q1 水準 (小賠)。這是基於「最樂觀劇本」算出的理論上限。**若您的實際倉位超過此數值，代表您正在進行非理性的過度槓桿，風險極高。**")
    lines.append(f"")
    lines.append(f"### 2. 其他說明")
    lines.append(f"*   **策略邏輯:** 此策略假設市場具有「均值回歸」特性，試圖在價格低於過去一年特定幅度時進場，以期獲得較好的風險回報比。")
    lines.append(f"*   **風險警示:** 歷史數據不代表未來表現。本報告僅供教育與研究用途，不構成任何財務建議。")

    report_content = "\n".join(lines)
    
    # Print to console
    print(report_content)

    # Save to file
    threshold_str = f"{abs(threshold_pct)*100:.2f}pct_Drop".replace(".", "p") 
    filename = f"{symbol}_Strategy_DipBuy_{threshold_str}.md"
    out_path = Path("research/reports") / filename
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC", help="Symbol (e.g., BTC, VOO)")
    parser.add_argument("--file", help="Specific CSV file path")
    parser.add_argument("--days", type=int, default=None, help="Holding period days (Auto-detect if omitted)")
    parser.add_argument("--threshold", type=float, default=-0.1115, help="Drop threshold (e.g., -0.1115 for -11.15%)")
    
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

    run_dip_buy_test(args.symbol, csv_path, args.days, args.threshold)

if __name__ == "__main__":
    main()