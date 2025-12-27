# Playground

Place personal experiments here so they are kept separate from the library code.

Suggested workflow
- Activate venv: `source ../.venv/bin/activate` (run from this folder) or `source .venv/bin/activate` from repo root.
- Copy/modify example scripts here; avoid committing sensitive data or API keys.
- Run monthly sample: `python btc/btc_sma_1M.py`
- Run daily sample (uses local CSV): `python btc/btc_sma_1d.py`

## Naming & archiving
- Keep each experiment filename as `<asset>_<strategy>_<tf>.py` (for example `btc_sma_1d.py` or `eth_breakout_4h.py`) so it's easy to see what market you're targeting.
- Store new-market scripts under `playground/` or a dedicated subfolder such as `playground/pendle/`; reserve `playground/btc/` for the existing Bitcoin-only experiments to avoid collisions with fresh data.
- Include the asset prefix when saving HTML plots so every output becomes `results/<asset>_<strategy>_<tf>_<timestamp>.html`. That way you always know which market a dashboard refers to, even when files accumulate.

Naming convention for new strategies
- File name: `<asset>_<strategy>_<tf>.py`, e.g. `pendle_dashboard_1d.py` or `nflx_momentum_4h.py`.
- Output plots: include asset, strategy, and timeframe: `results/<asset>_<strategy>_<tf>_<timestamp>.html` so each HTML page clearly names the trading target.
- Data files: keep the timeframe in the filename (`BTCUSDT_1d.csv`, `PENDLEUSDT_1d.csv`, `NFLX_1d.csv`) and store under `data/` (consider `data/pendle/` for new markets).

Benchmarks / utils
- `btc/btc_bh_dca_1d.py`: compares lump-sum buy & hold vs DCA over the last 2000 daily bars.
- `btc/btc_dca_1d.py`: illustrative DCA-as-frequent-trading example; shows that high-frequency small buys with 0.1% fees can underperform (see `results/dca_bh_backtest_1d_20251216_105439.html` as a cautionary example).
- `btc/btc_dca_backtest.py`: runs `FractionalBacktest` for spans in `[1,100,…,1000]` and writes `results/btc_dca_span_<span>_backtest_<timestamp>.html` for each span.
- `btc/btc_dca_simulation.py`: prints day/activation stats for the base DCA trigger so you can sanity-check cash usage before making plots.

Notes
- Imports work because the package was installed with `pip install -e .`.
- Keep generated HTML/plots outside version control; they can be large and user-specific.
- Keep a simple script-to-output mapping under **Script → HTML outputs** below so it’s easier to recall which Python file generates each artifact; update it whenever you add a new dashboard.

Script → HTML outputs
| Script | Generated HTML | Description |
| --- | --- | --- |
| `btc/btc_bh_dca_1d.py` | `results/benchmark_bh_dca_1d.html` (static summary) + `results/benchmark_bh_dca_1d_chart.html` (Bokeh equity comparison) | Tables + chart comparing lump-sum vs DCA over the last 2,000 daily bars using `data/raw/BTCUSDT_1d.csv`. |
| `btc/btc_dca_1d.py` | `results/dca_bh_backtest_1d_{timestamp}.html` | Fractional Backtest chart showing the DCA trade schedule + equity vs cash (uses the same BTCUSD CSV). |
| `btc/btc_dca_backtest.py` | `results/btc_dca_span_{span}_backtest_{timestamp}.html` | FractionalBacktest output per span, instrumented with custom DCA strategy and asset-prefixed filenames. |
| `btc/btc_peak_drawdown_ui_1d.py` | `results/btc_peak_gap_dd_{interval}_{timestamp}.html` | Drawdown-aware interval dashboard; `--plot-interval-index`/`--plot-all-intervals` batch-run option also writes one file per interval with a `batch_tag`. |
| `playground/pendle/pendle_dca_dashboard_mvp.py` | `results/pendle_dca_dashboard_mvp_{timestamp}.html` | Pendle 版本的 DCA MVP 儀表板：原樣從 BTC 量化邏輯複製，只改成讀 `data/raw/PENDLE.csv`，輸出檔名稱帶 `pendle` 前綴。 |
| `playground/voo/voo_dca_dashboard_mvp.py` | `results/voo_dca_dashboard_mvp_{timestamp}.html` | VOO 版本的 DCA MVP 儀表板：直接複製 Pendle 的 DCA dashboard，改用 `data/raw/VOO.csv` 並輸出帶 `voo` 前綴的 HTML。 |
| `btc/btc_sma_1d.py` | `results/btc_sma_1d_{YYYYMMDD_HHMMSS}.html` | Daily SMA crossover Backtest chart that resaves with a timestamp to avoid overwriting. |
| `btc/btc_sma_1M.py` | `results/btc_sma_1M_{YYYYMMDD_HHMMSS}.html` | Monthly SMA crossover demo using `backtesting.test.BTCUSD`; chart filename is timestamped for uniqueness. |
| `btc/btc_dca_dashboard_mvp.py` | `results/btc_dca_dashboard_mvp_{timestamp}.html` | Interactive DCA MVP dashboard with price, equity %, drawdown, and metric summaries for each ATH→ATH interval (sources `data/raw/BTCUSDT_1d.csv`). |
| `btc/btc_dca_dashboard_mvp_zh.py` | `results/btc_dca_dashboard_mvp_zh_{timestamp}.html` | 中文化的 DCA MVP 儀表板，輸出與英文版相同的資料，但 summary、標題與文字皆轉為繁體中文。 |
| `btc/btc_dca_param_analysis.py` | `results/btc_dca_param_analysis_{timestamp}.html` | Parameter heatmap for span/trigger combos with per-interval summaries and tables (reuses the dashboard’s simulations). |
| `btc/btc_dca_param_analysis_zh.py` | `results/btc_dca_param_analysis_zh_{timestamp}.html` | 中文化的參數分析頁面，表格可點擊的 Trigger 連結會跳至下方靜態圖表，摘要與表格皆為中文描述。 |
| `playground/pendle/pendle_peak_drawdown_ui_1d.py` | `results/pendle_peak_gap_dd_{interval}_{timestamp}.html` | 和 BTC 脚本相同的 ATH 間隔 + drawdown 圖，使用 `data/raw/PENDLE.csv`（目前只有一段長熊市，因此可視化主要集中在那一輪）。 |
| `playground/voo/voo_peak_drawdown_ui_1d.py` | `results/voo_peak_gap_dd_{interval}_{timestamp}.html` | 同樣基於 ATH 至 ATH 間隔的 drawdown 儀表板，但資料源換成 `data/raw/VOO.csv`，圖表前綴為 `voo`。 |
| `playground/pendle/pendle_dca_param_analysis.py` | `results/pendle_dca_param_analysis_{timestamp}.html` | Pendle 版 span/threshold 參數熱圖，資料與統計由 `playground/pendle/pendle_dca_dashboard_mvp.py` 提供，因此資料來源仍是 `data/raw/PENDLE.csv`。 |
| `playground/voo/voo_dca_param_analysis.py` | `results/voo_dca_param_analysis_{timestamp}.html` | VOO span/threshold 熱圖：參數與摘要來自 `playground/voo/voo_dca_dashboard_mvp.py`，輸出名稱以 `voo` 為前綴。 |

Please ignore the `iv_plot` HTML files (`iv_plot*.html`); they come from a different workflow and are not produced by the scripts listed above.
