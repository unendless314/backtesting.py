# Playground

Place personal experiments here so they are kept separate from the library code.

Suggested workflow
- Activate venv: `source ../.venv/bin/activate` (run from this folder) or `source .venv/bin/activate` from repo root.
- Copy/modify example scripts here; avoid committing sensitive data or API keys.
- Run monthly sample: `python btc_sma_1M.py`
- Run daily sample (uses local CSV): `python btc_sma_1d.py`

Naming convention for new strategies
- File name: `<asset>_<strategy>_<tf>.py`, e.g. `btc_sma_1d.py`, `eth_breakout_4h.py`.
- Output plots: include timeframe in the filename, e.g. `results/sma_1d_<timestamp>.html`.
- Data files: keep the timeframe in the filename (`BTCUSDT_1d.csv`, `BTCUSDT_3d.csv`, `BTCUSDT_1w.csv`) and store under `data/`.

Benchmarks / utils
- `btc_bh_dca_1d.py`: compares lump-sum buy & hold vs DCA over the last 2000 daily bars.
- `dca_bh_backtest_1d.py`: illustrative DCA-as-frequent-trading example; shows that high-frequency small buys with 0.1% fees can underperform (see `results/dca_bh_backtest_1d_20251216_105439.html` as a cautionary example).

Notes
- Imports work because the package was installed with `pip install -e .`.
- Keep generated HTML/plots outside version control; they can be large and user-specific.
