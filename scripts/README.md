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
