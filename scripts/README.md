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

## 🗓 統一日期與持有期參數（--start / --end / --days）

三隻研報腳本使用相同的「日期區間 + 持有期」參數模型，用來回答：**「不同的進場區間 × 不同的持有期，對投資績效造成什麼影響？」**

| 參數 | 白話意義 | 範例 |
| :--- | :--- | :--- |
| `--start` | 最早可以進場的日期（YYYY-MM-DD） | `--start 2021-09-10` |
| `--end` | 最晚可以進場的日期（YYYY-MM-DD） | `--end 2022-12-31` |
| `--days` | 每筆進場往後持有幾天 | `--days 730` |

> 例外：`analyze_term_structure.py` 不提供 `--days`，它以固定的 1 季到 20 季（5 年）梯子一次呈現所有持有期。

**語義說明（白話）**：
- 日期篩的是**進場日**：在 `--end` 之前進場、但結果發生在 `--end` 之後的交易**算數**。
- 指標先在全量資料上計算，指定 `--start` 不會讓區間第一年沒有訊號。
- 資料最末端不足一個持有期的進場會被**捨棄**；報表開頭會顯示「實際有效進場範圍」供確認。
- 日期格式嚴格限定 `YYYY-MM-DD`；格式錯誤、`start > end`、或區間內無有效樣本時，會印出中文錯誤訊息並結束，不寫出報告檔。

**建議工作流程（兩步驟）**：
```bash
# 第一步：用期限結構梯子表，看這個區間「抱多久」比較安全
python3 scripts/analyze_term_structure.py --symbol BNB --start 2021-09-10

# 第二步：對選定的持有期，用單期報告看細節（凱利倉位、情景分析）
python3 scripts/generate_asset_report.py --symbol BNB --start 2021-09-10 --days 730
```

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

    # 指定進場區間與持有期
    python3 scripts/generate_asset_report.py --symbol BTC --start 2021-11-10 --end 2022-11-21 --days 365
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

    # 指定進場區間（持有期維度固定為 1Q–20Q 梯子，無 --days）
    python3 scripts/analyze_term_structure.py --symbol BNB --start 2021-09-10
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
        *   `--start` / `--end`: 進場日區間（YYYY-MM-DD）。
        *   `--days`: 持有天數（省略時自動偵測資產類型）。

*   **Usage:**
    ```bash
    # Test buying BTC when it drops 11.15% from 1 year ago
    python3 scripts/test_strategy_dip_buy.py --symbol BTC --threshold -0.1115

    # Test buying VOO when 1-year return is < 4.29%
    python3 scripts/test_strategy_dip_buy.py --symbol VOO --threshold 0.0429

    # 指定進場區間
    python3 scripts/test_strategy_dip_buy.py --symbol BNB --threshold -0.1578 --start 2021-09-10
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