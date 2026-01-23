scripts/
├── data/          # Data-fetch & update utilities (e.g., download OHLCV, resample, save CSV)

Purpose
- Keep data acquisition and preparation logic isolated from the backtesting core.
- Avoid committing bulky CSVs to git; add data outputs to .gitignore.

Suggested usage
- Put fetchers like `fetch_binance.py` in `scripts/data/`.
- Use virtualenv (`source .venv/bin/activate`) then install extras you need, e.g. `pip install ccxt pandas`.
- Store raw downloads under `data/raw/` and derived (resampled) files under `data/resampled/`.
- Example:  
  `python scripts/data/fetch_binance.py --symbol BTC/USDT --tf 1d --since 2013-01-01 --out data/raw/BTCUSDT_1d.csv`  
  Add `--resample 3D` or `--resample 1W` if you want aggregated bars.

Best practices
- Keep credentials (if any) in env vars, not in code.
- Log download ranges and timeframes for reproducibility.
- Handle rate limits/backoff when calling exchanges.
- Make outputs deterministic (fixed start/end, consistent timezone, column order).

---

# Analysis & Strategy Scripts

This directory also contains various Python scripts for generating quantitative analysis reports, backtesting strategies, and exploring term structures.

## 📊 Report Generators

### 1. `generate_asset_report.py`
Generates a comprehensive Markdown report for a specific asset ("Buy & Hold" strategy).

*   **Features:**
    *   **Auto-Detection:** Automatically detects if the asset is Crypto (365 days) or Stock (252 days) based on data density.
    *   **Metrics:** Calculates Win Rate, Median ROI, Skewness, Kelly Criterion (Standard, Robust, Conservative, Aggressive).
    *   **Scenario Analysis:** Optimistic (Q3), Baseline (Median), Pessimistic (Q1).
    *   **Language:** Traditional Chinese (繁體中文).

*   **Usage:**
    ```bash
    # Auto-detect asset type (recommended)
    python3 scripts/generate_asset_report.py --symbol BTC
    python3 scripts/generate_asset_report.py --symbol VOO

    # Manually specify file path and holding period
    python3 scripts/generate_asset_report.py --symbol BTC --file data/raw/BTCUSDT_1d.csv --days 365
    ```

### 2. `analyze_term_structure.py`
Analyzes the "Term Structure of Returns" to visualize how risk and return evolve over time (from 1 Quarter to 5 Years).

*   **Features:**
    *   Generates a quarterly breakdown (1Q to 20Q).
    *   Shows Win Rate convergence and Downside Risk (Q1) evolution.
    *   Helps identify the "Safe Holding Period" for an asset.

*   **Usage:**
    ```bash
    python3 scripts/analyze_term_structure.py --symbol ETH
    ```

---

## 🧪 Strategy Backtesting

### 1. `test_strategy_dip_buy.py`
Backtests a "Mean Reversion" (Dip Buy) strategy: Buy only when the price drops below a certain threshold compared to 1 year ago.

*   **Features:**
    *   Compares the Strategy performance vs. Random Buy (Baseline).
    *   Calculates specialized Kelly Criterion for the strategy.
    *   **Parameters:**
        *   `--threshold`: The drop percentage trigger (e.g., `-0.3358` for -33.58%).
    
*   **Usage:**
    ```bash
    # Test buying BTC when it drops 11.15% from 1 year ago
    python3 scripts/test_strategy_dip_buy.py --symbol BTC --threshold -0.1115

    # Test buying VOO when 1-year return is < 4.29%
    python3 scripts/test_strategy_dip_buy.py --symbol VOO --threshold 0.0429
    ```

---

## 🛠 Helper Scripts

### 1. `calculate_past_momentum.py`
A helper script to calculate the Q1 (25th percentile) of the "Past 1-Year Return" for an asset. This is useful for determining the input `threshold` for `test_strategy_dip_buy.py`.

*   **Usage:**
    ```bash
    python3 scripts/calculate_past_momentum.py
    ```

## 📂 Output Directory
All generated reports are saved to: `research/reports/`