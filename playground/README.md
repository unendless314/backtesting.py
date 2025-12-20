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
- Keep a simple script-to-output mapping under **Script → HTML outputs** below so it’s easier to recall which Python file generates each artifact; update it whenever you add a new dashboard.

Script → HTML outputs
| Script | Generated HTML | Description |
| --- | --- | --- |
| `btc_bh_dca_1d.py` | `results/benchmark_bh_dca_1d.html` (static summary) + `results/benchmark_bh_dca_1d_chart.html` (Bokeh equity comparison) | Tables + chart comparing lump-sum vs DCA over the last 2,000 daily bars using `data/raw/BTCUSDT_1d.csv`. |
| `btc_dca_1d.py` | `results/dca_bh_backtest_1d_{timestamp}.html` | Fractional Backtest chart showing the DCA trade schedule + equity vs cash (uses the same BTCUSD CSV). |
| `btc_peak_drawdown_ui_1d.py` | `results/btc_peak_gap_dd_{interval}_{timestamp}.html` | Drawdown-aware interval dashboard; `--plot-interval-index`/`--plot-all-intervals` batch-run option also writes one file per interval with a `batch_tag`. |
| `btc_sma_1d.py` | `results/sma_1d_{YYYYMMDD_HHMMSS}.html` | Daily SMA crossover Backtest chart that resaves with a timestamp to avoid overwriting. |
| `btc_sma_1M.py` | `results/sma_1M_{YYYYMMDD_HHMMSS}.html` | Monthly SMA crossover demo using `backtesting.test.BTCUSD`; chart filename is timestamped for uniqueness. |
| `dca_dashboard_mvp.py` | `results/dca_mvp_dashboard.html` | Interactive DCA MVP dashboard with price, equity %, drawdown, and metric summaries for each ATH→ATH interval (sources `data/raw/BTCUSDT_1d.csv`). |
| `dca_dashboard_mvp_zh.py` | `results/dca_mvp_dashboard_zh.html` | 中文化的 DCA MVP 儀表板，輸出與英文版相同的資料，但 summary、標題與文字都轉為繁體中文。 |
| `dca_param_analysis.py` | `results/dca_param_analysis.html` | Parameter heatmap for span/trigger combos with per-interval summaries and tables (reuses the dashboard’s simulations). |
| `dca_param_analysis_zh.py` | `results/dca_param_analysis_zh.html` | 中文化的參數分析頁面，表格可點擊的 Trigger 連結會跳至下方靜態圖表，摘要與表格皆為中文描述。 |

Please ignore the `iv_plot` HTML files (`iv_plot*.html`); they come from a different workflow and are not produced by the scripts listed above.
