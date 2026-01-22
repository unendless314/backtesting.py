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

    # Skewness
    skewness = valid_df['roi'].skew()
    
    # Odds Calculation
    b_odds_mean = avg_win / avg_loss if avg_loss > 0 else 0
    b_odds_median = median_win / median_loss if median_loss > 0 else 0

    # Kelly Criterion
    # Standard (Mean-based)
    kelly_mean = prob_win - (prob_loss / b_odds_mean) if b_odds_mean > 0 else 0
    
    # Robust (Median-based)
    kelly_median = prob_win - (prob_loss / b_odds_median) if b_odds_median > 0 else 0
    
    # Generate Markdown Content
    lines = []
    lines.append(f"# {symbol} {hold_days}-Day Hold Analysis")
    lines.append(f"")
    lines.append(f"**Data Source**: `{csv_path.name}`")
    lines.append(f"**Full Data Range**: {full_start} to {full_end}")
    lines.append(f"**Analyzed Buy Window**: {valid_df['datetime'].iloc[0].date()} to {valid_df['datetime'].iloc[-1].date()}")
    lines.append(f"")
    lines.append(f"## Statistics Overview")
    lines.append(f"| Metric | Value |")
    lines.append(f"| :--- | :--- |")
    lines.append(f"| **Total Trading Days** | {total_samples} days |")
    lines.append(f"| **Win Rate** | **{prob_win:.2%}** ({count_win} days) |")
    lines.append(f"| **Loss Rate** | {prob_loss:.2%} ({count_loss} days) |")
    lines.append(f"| **Skewness** | {skewness:.2f} (>1 implies fat tail) |")
    lines.append(f"")
    lines.append(f"## Returns Analysis")
    lines.append(f"| Metric | Mean (Avg) | Median (Robust) |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **General Return** | {valid_df['roi'].mean():.2%} | **{median_return:.2%}** |")
    lines.append(f"| **Win Magnitude** | +{avg_win:.2%} | +{median_win:.2%} |")
    lines.append(f"| **Loss Magnitude** | -{avg_loss:.2%} | -{median_loss:.2%} |")
    lines.append(f"| **Reward/Risk Ratio** | **{b_odds_mean:.2f}** | **{b_odds_median:.2f}** |")
    lines.append(f"")
    lines.append(f"## Kelly Criterion Analysis")
    lines.append(f"> Formula: $f^* = p - \\frac{{q}}{{b}}$")
    lines.append(f"")
    lines.append(f"| Strategy | Allocation (f*) | Half-Kelly |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| **Standard (Based on Mean)** | `{kelly_mean:.2%}` | `{kelly_mean*0.5:.2%}` |")
    lines.append(f"| **Robust (Based on Median)** | `{kelly_median:.2%}` | `{kelly_median*0.5:.2%}` |")
    lines.append(f"")
    lines.append(f"## Notes")
    lines.append(f"Analysis generated based on buying everyday within the window and holding for exactly {hold_days} days.")
    lines.append(f"* **Robust Kelly** uses Median Reward/Risk, which is less sensitive to extreme outliers (e.g., 100x pumps) and provides a safer baseline for position sizing.")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate Crypto Markdown Report")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., BTC, AAVE)")
    parser.add_argument("--file", help="Specific CSV file path (optional)")
    parser.add_argument("--start", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=365, help="Holding period days")
    
    args = parser.parse_args()
    
    # Infer path or use provided one
    if args.file:
        csv_path = Path(args.file)
    else:
        csv_path = Path(f"data/raw/{args.symbol}USDT_1d.csv")
    
    report_content = analyze_crypto(args.symbol, csv_path, args.start, args.end, args.days)
    
    # Output file
    range_str = "All_Time"
    if args.start:
        range_str = f"{args.start}_to_{args.end if args.end else 'Now'}"
        
    filename = f"{args.symbol}_{args.days}d_Hold_{range_str}.md"
    out_path = Path("research/reports") / filename
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report generated: {out_path}")

if __name__ == "__main__":
    main()
