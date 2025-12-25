# Playground

Place personal experiments here so they are kept separate from the library code.

Suggested workflow
- Activate venv: `source ../.venv/bin/activate` (run from this folder) or `source .venv/bin/activate` from repo root.
- Copy/modify example scripts here; avoid committing sensitive data or API keys.
- Run monthly sample: `python legacy/btc/btc_sma_1M.py`
- Run daily sample (uses local CSV): `python legacy/btc/btc_sma_1d.py`

## Naming & archiving
- Keep each experiment filename as `<asset>_<strategy>_<tf>.py` (for example `btc_sma_1d.py` or `eth_breakout_4h.py`) so it's easy to see what market you're targeting.
- Store new-market scripts under `playground/` or a dedicated subfolder such as `playground/pendle/`; reserve `playground/legacy/btc/` for the existing Bitcoin-only experiments to avoid collisions with fresh data.
- Include the asset prefix when saving HTML plots so every output becomes `results/<asset>_<strategy>_<tf>_<timestamp>.html`. That way you always know which market a dashboard refers to, even when files accumulate.

Naming convention for new strategies
- File name: `<asset>_<strategy>_<tf>.py`, e.g. `pendle_dashboard_1d.py` or `nflx_momentum_4h.py`.
- Output plots: include asset, strategy, and timeframe: `results/<asset>_<strategy>_<tf>_<timestamp>.html` so each HTML page clearly names the trading target.
- Data files: keep the timeframe in the filename (`BTCUSDT_1d.csv`, `PENDLEUSDT_1d.csv`, `NFLX_1d.csv`) and store under `data/` (consider `data/pendle/` for new markets).

Benchmarks / utils
- `legacy/btc/btc_bh_dca_1d.py`: compares lump-sum buy & hold vs DCA over the last 2000 daily bars.
- `legacy/btc/btc_dca_1d.py`: illustrative DCA-as-frequent-trading example; shows that high-frequency small buys with 0.1% fees can underperform (see `results/dca_bh_backtest_1d_20251216_105439.html` as a cautionary example).
- `btc_dca_backtest.py`: runs `FractionalBacktest` for spans in `[1,100,…,1000]` and writes `results/btc_dca_span_<span>_backtest_<timestamp>.html` for each span.
- `btc_dca_simulation.py`: prints day/activation stats for the base DCA trigger so you can sanity-check cash usage before making plots.

Notes
- Imports work because the package was installed with `pip install -e .`.
- Keep generated HTML/plots outside version control; they can be large and user-specific.
- Keep a simple script-to-output mapping under **Script → HTML outputs** below so it’s easier to recall which Python file generates each artifact; update it whenever you add a new dashboard.

Script → HTML outputs
| Script | Generated HTML | Description |
| --- | --- | --- |
| `legacy/btc/btc_bh_dca_1d.py` | `results/benchmark_bh_dca_1d.html` (static summary) + `results/benchmark_bh_dca_1d_chart.html` (Bokeh equity comparison) | Tables + chart comparing lump-sum vs DCA over the last 2,000 daily bars using `data/raw/BTCUSDT_1d.csv`. |
| `legacy/btc/btc_dca_1d.py` | `results/dca_bh_backtest_1d_{timestamp}.html` | Fractional Backtest chart showing the DCA trade schedule + equity vs cash (uses the same BTCUSD CSV). |
| `btc_dca_backtest.py` | `results/btc_dca_span_{span}_backtest_{timestamp}.html` | FractionalBacktest output per span, instrumented with custom DCA strategy and asset-prefixed filenames. |
| `legacy/btc/btc_peak_drawdown_ui_1d.py` | `results/btc_peak_gap_dd_{interval}_{timestamp}.html` | Drawdown-aware interval dashboard; `--plot-interval-index`/`--plot-all-intervals` batch-run option also writes one file per interval with a `batch_tag`. |
| `legacy/btc/btc_sma_1d.py` | `results/btc_sma_1d_{YYYYMMDD_HHMMSS}.html` | Daily SMA crossover Backtest chart that resaves with a timestamp to avoid overwriting. |
| `legacy/btc/btc_sma_1M.py` | `results/btc_sma_1M_{YYYYMMDD_HHMMSS}.html` | Monthly SMA crossover demo using `backtesting.test.BTCUSD`; chart filename is timestamped for uniqueness. |
| `btc_dca_dashboard_mvp.py` | `results/btc_dca_dashboard_mvp_{timestamp}.html` | Interactive DCA MVP dashboard with price, equity %, drawdown, and metric summaries for each ATH→ATH interval (sources `data/raw/BTCUSDT_1d.csv`). |
| `btc_dca_dashboard_mvp_zh.py` | `results/btc_dca_dashboard_mvp_zh_{timestamp}.html` | 中文化的 DCA MVP 儀表板，輸出與英文版相同的資料，但 summary、標題與文字皆轉為繁體中文。 |
| `btc_dca_param_analysis.py` | `results/btc_dca_param_analysis_{timestamp}.html` | Parameter heatmap for span/trigger combos with per-interval summaries and tables (reuses the dashboard’s simulations). |
| `btc_dca_param_analysis_zh.py` | `results/btc_dca_param_analysis_zh_{timestamp}.html` | 中文化的參數分析頁面，表格可點擊的 Trigger 連結會跳至下方靜態圖表，摘要與表格皆為中文描述。 |

Please ignore the `iv_plot` HTML files (`iv_plot*.html`); they come from a different workflow and are not produced by the scripts listed above.
