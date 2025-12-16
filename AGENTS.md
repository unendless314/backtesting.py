# Repository Guidelines

## Project Structure & Modules
- Core package lives in `backtesting/`: `backtesting.py` defines `Backtest`, `Strategy`, and public API; `lib.py` holds reusable helpers; `_stats.py`, `_plotting.py`, and `_util.py` contain internal analytics, charts, and utilities; `autoscale_cb.js` supports interactive plots.
- Tests and sample datasets are in `backtesting/test/` (`_test.py`, `BTCUSD.csv`, `GOOG.csv`, `EURUSD.csv`). Run them from the repo root.
- Documentation sources sit in `doc/` (examples in `doc/examples/`). Packaging/config lives in `pyproject.toml`, `setup.cfg`, and `MANIFEST.in`.

## Environment & Installation
- Use a fresh virtualenv. Install dev extras:  
  `pip install -e '.[test,doc,dev]'`
- For quick editing only: `pip install -e .`
- Note for assistants: workspace has helper notes in `AI_NOTES.md`; review it before guiding the user (beginner) and remind to activate `.venv` via `source .venv/bin/activate`.

## Build, Test, and Development Commands
- Run full test suite (unittest-based): `python -m backtesting.test`
- Run a focused test: `python -m unittest backtesting.test._test.TestBacktest.test_run`
- Lint/format check: `ruff check backtesting` (line length 100)  
  Secondary style gate: `flake8 backtesting`
- Type checks: `mypy backtesting`
- Optional coverage run: `coverage run -m backtesting.test && coverage report`

## Coding Style & Naming
- Python 3 style, 4-space indents, keep lines ≤100 (ruff config) to satisfy both ruff and flake8 (120 max). Prefer single quotes for strings, f-strings for interpolation.
- Functions/methods: `snake_case`; classes: `PascalCase`; module-internal helpers: prefix `_`.
- Add type hints on public APIs; keep docstrings pdoc-friendly (Markdown in docstrings is rendered).
- Avoid heavy dependencies; keep utilities pure and vectorized where possible.

## Testing Guidelines
- Add/extend tests in `backtesting/test/_test.py`; reuse the bundled datasets or generate deterministic data (`random_ohlc_data`).
- Mirror existing naming: `TestBacktest`, `Test...`; keep tests fast (suite currently <1s/core).
- New features or bugfixes must ship with at least one unit test demonstrating behavior/regression.

## Commit & Pull Request Guidelines
- Commit messages in this repo commonly use short tags (`BUG:`, `DOC:`, `MNT:`, `BRK:`). Follow that pattern plus a brief imperative summary (e.g., `BUG: Fix commission calculation`).
- PRs should describe motivation, approach, and user-facing impact; link related issues. Include test/lint results in the description. Add screenshots or HTML snippets only if the change affects plotting output.
- Keep PRs focused and small; prefer separate commits for logic, tests, and docs when practical.

## Security & Configuration
- Do not commit credentials or data beyond the provided CSV fixtures. Plotly/JS assets live locally; no external CDN keys are required.
- Local config belongs in env variables or ignored files; confirm `.gitignore` coverage before adding tooling outputs.

## Data Fetching Scripts (Binance via ccxt)
- New folder `scripts/data/` holds data-update utilities. Main tool: `scripts/data/fetch_binance.py`.
- Dependencies: activate `.venv` then `pip install ccxt pandas`.
- Usage example:  
  `python scripts/data/fetch_binance.py --symbol BTC/USDT --tf 1d --since 2013-01-01 --out data/raw/BTCUSDT_1d.csv --resample 3D`  
  `--tf` supports Binance intervals (1m–1M); `--resample` applies pandas rule (e.g., 3D/1W) before saving.
- Outputs should stay local; `.gitignore` ignores `data/**` (csv/parquet/feather, raw/resampled). Keep credentials in env vars.
- Purpose: isolate data acquisition from core backtesting code; safe to modify without touching `backtesting/`.
