# AI Notes for This Workspace

- Primary user is a beginner to Python and quantitative trading. Keep instructions step-by-step, avoid jargon, and include exact commands.
- Always run project commands inside the repo’s virtual environment: `source .venv/bin/activate` (created at `/Users/linchunchiao/Documents/Python/backtesting.py/.venv`).
- To exit the environment, use `deactivate`.
- Minimal install already done: `pip install -e .`. For full tooling (tests/docs/dev), use `pip install -e '.[test,doc,dev]'` inside the venv.
- Quick sanity test for BTC data:  
  `python - <<'PY'\nfrom backtesting.test import BTCUSD; print(BTCUSD.head())\nPY`
- Playground for experiments lives in `playground/`; starter script: `playground/btc_sma.py` (run with `python playground/btc_sma.py`). Keep instructions beginner-friendly and remind to activate `.venv`.
- Preference: when AI writes/modifies strategy code, include short inline comments/docstrings explaining the trading idea, entry/exit rules, key params, and why certain settings (commission, cash, exclusive_orders) are used—user is a beginner.
- Save plots under `playground/results/` with timestamped filenames to avoid overwrite and to keep repo root clean; default pattern used in btc_sma.py: `results/sma_YYYYMMDD_HHMMSS.html`.
