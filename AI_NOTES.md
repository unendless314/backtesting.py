# AI Notes for This Workspace

- Primary user is a beginner to Python and quantitative trading. Keep instructions step-by-step, avoid jargon, and include exact commands.
- Always run project commands inside the repo’s virtual environment: `source .venv/bin/activate` (created at `/Users/linchunchiao/Documents/Python/backtesting.py/.venv`).
- To exit the environment, use `deactivate`.
- Minimal install already done: `pip install -e .`. For full tooling (tests/docs/dev), use `pip install -e '.[test,doc,dev]'` inside the venv.
- Quick sanity test for BTC data:  
  `python - <<'PY'\nfrom backtesting.test import BTCUSD; print(BTCUSD.head())\nPY`
- Playground for experiments lives in `playground/`; starter script: `playground/btc_sma.py` (run with `python playground/btc_sma.py`). Keep instructions beginner-friendly and remind to activate `.venv`.
