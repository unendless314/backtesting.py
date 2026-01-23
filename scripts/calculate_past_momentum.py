import pandas as pd
from pathlib import Path

def get_momentum_q1(symbol, csv_path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    # Read first row to check columns
    peek = pd.read_csv(csv_path, nrows=1)
    date_col = 'datetime' if 'datetime' in peek.columns else 'Date'
    
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    # Normalize column name
    df.rename(columns={date_col: 'datetime'}, inplace=True)
    df.sort_values('datetime', inplace=True)
    
    # Calculate Past 1-Year Return (Momentum)
    # Using 365 days shift if data is daily including weekends (Crypto), 
    # but for Stocks (VOO) trading days are ~252.
    # However, 'backtesting.py' logic usually implies row-based shift or time-based?
    # Let's stick to the logic used in 'test_strategy_dip_buy.py': pct_change(periods=hold_days)
    # For VOO, 365 rows is more than a year (approx 1.5 years). 
    # WE SHOULD FIX THIS: For stocks, 1 year is ~252 trading days.
    
    # Simple Heuristic: Check average delta between rows
    avg_delta = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days / len(df)
    
    if avg_delta < 1.1: # Daily data including weekends (Crypto)
        lookback = 365
    else: # Trading days only (Stocks ~ 1.4 days avg gap due to weekends)
        lookback = 252

    df['past_return'] = df['close'].pct_change(periods=lookback)
    
    valid_df = df.dropna(subset=['past_return'])
    q1 = valid_df['past_return'].quantile(0.25)
    
    print(f"[{symbol}] (Lookback={lookback} bars)")
    print(f"  Past 1-Year Return Q1: {q1:.4f} ({q1:.2%})")
    print(f"  Interpretation: 25% of the time, the 1-year return is worse than {q1:.2%}")

if __name__ == "__main__":
    get_momentum_q1("BTC", Path("data/raw/BTCUSDT_1d.csv"))
    get_momentum_q1("ETH", Path("data/raw/ETHUSDT_1d.csv"))
    get_momentum_q1("VOO", Path("data/raw/VOO_1d.csv"))