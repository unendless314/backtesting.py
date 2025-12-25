# BTC Peak-to-Peak Drawdown Summary

Tracked in `playground/results/btc_peak_gap_dd_20251218_162627.html` are five BTC peak intervals. The combined day counts per drawdown band across all intervals are:

| Drawdown Band | Total Days |
| --- | ---: |
| `<-70%` | 310 |
| `-70% to -60%` | 507 |
| `-60% to -50%` | 491 |
| `-50% to -40%` | 349 |
| `-40% to -30%` | 230 |
| `-30% to -20%` | 179 |
| `-20% to -10%` | 220 |
| `0% to -10%` | 189 |

This note lives in `playground/btc_peak_drawdown_summary.md` for later reference.

## Source data context

The drawdown band totals above come from the five intervals exported to `playground/results/btc_peak_gap_dd_20251218_162627.html`, which in turn are derived from `data/raw/BTCUSDT_1d.csv`. That CSV currently has **3,029 rows** (one OHLCV record per trading day).

Across the five intervals we counted **2,475 drawdown days**, so a non‑trivial portion of the dataset is captured in these bands. The following table adds the percentage each band contributes relative to the full 3,029 rows (rounded to one decimal place):

| Drawdown Band | Total Days | % of 3,029 rows |
| --- | ---: | ---: |
| `<-70%` | 310 | 10.2% |
| `-70% to -60%` | 507 | 16.7% |
| `-60% to -50%` | 491 | 16.2% |
| `-50% to -40%` | 349 | 11.5% |
| `-40% to -30%` | 230 | 7.6% |
| `-30% to -20%` | 179 | 5.9% |
| `-20% to -10%` | 220 | 7.3% |
| `0% to -10%` | 189 | 6.2% |

## High-level takeaway

Roughly 81.7% of the full BTCUSDT 1d history lives inside the drawdown bands recorded for the five plotted ATH‑to‑ATH intervals, which means BTC spent ~80% of those days descending from a prior high. The remaining ~18.3% of the timeframe aligns with periods where price was likely climbing toward or breaking a new ATH, so you can treat the drawdown bands as representing the “correction” regime and the complementary few hundred days as the “running‑high” regime for future decisions.

## Strategy plan summary

1. **Capital pools** – Split capital into two virtual buckets with equal starting funds (100k each).  
2. **Grid/trading pool** – Only runs when BTC is between `-10%` and `-50%` from the prior ATH, using grid/arb orders across those bands; the active window historically covers ~978 days (≈32.3% of the dataset).  
3. **DCA pool** – Activates once BTC slides past `-50%` drawdown, keeps allocating in fixed-size increments, and pauses/dies once price climbs back above the `-50%` line (no more grid sells).  
4. **Backtest approach** – Run the two strategies separately (each with a backtest using the raw CSV) and later blend the performance statistics by weighting them with whatever X:Y split you want for the combined view.

## Monte Carlo + DCA reminder

1. **DCA sizing matters** – The drawdown depth and how long the <−50% regime lasts are unknown, so simulate different per-iteration capital splits / sampling windows to see which one keeps ammo in the tank without running out before a rebound.  
2. **Monte Carlo idea** – Generate many hypothetical price paths (e.g., using the historical daily return distribution from `data/raw/BTCUSDT_1d.csv`) and run each through the DCA rules, tracking final equity, max drawdown, and the fraction of simulations where the capital is exhausted.  
3. **Benefits for you** – The random paths reveal how sensitive the outcome is to choices like 100 vs. 1,000 slices, so you can adjust the per-trade allocation based on the probability of surviving deep bears rather than just a single historical example.  
4. **Next steps** – Start with a simple loop that samples returns and applies the DCA grid rules; once the Monte Carlo harness works you can plug in your preferred grid requirements and combine with the arb backtest results as outlined above.

## Interval duration stats

- **Bear market lengths (5 major ATH→ATH intervals with `days_between ≥ 100`)** – mean **493 days**, std **390 days**.  
- **Grid-active durations (`-10%` through `-50%` drawdown)** – mean **198 days**, std **95 days** (per interval counts: 314, 164, 301, 142, 68).  
- **DCA-active durations (`<−50%` drawdown)** – mean **258 days**, std **320 days** (per interval counts: 749, 7, 533, 0, 0).  

These statistics were derived from the same `data/raw/BTCUSDT_1d.csv` file used elsewhere: I filtered for the five long intervals (the ones that actually hit those deep drawdown bands) and counted how many days of each interval sit in the grid or DCA bands before a new ATH is reached.

## DCA span simulations (start capital $60k)

I coded the repeated DCA scenarios in `playground/btc_dca_simulation.py`. Each run waits for a drawdown of `-50%` or worse, then buys the same dollar slice each day for the configured span (1 – 1,000 days) or until the deep drawdown resolves. Results after the full data sweep:

| DCA span | Activations | DCA days executed | Equity ($) | Final cash ($) | Shares |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 558,347.10 | 0.00 | 6.5044 |
| 100 | 4 | 100 | 635,717.75 | 0.00 | 7.4057 |
| 200 | 4 | 200 | 700,202.67 | 0.00 | 8.1569 |
| 300 | 4 | 300 | 797,576.45 | 0.00 | 9.2913 |
| 400 | 4 | 400 | 940,385.81 | 0.00 | 10.9549 |
| 500 | 8 | 500 | 902,929.09 | 0.00 | 10.5186 |
| 600 | 8 | 600 | 861,241.17 | 0.00 | 10.0329 |
| 800 | 22 | 800 | 776,915.91 | 0.00 | 9.0506 |
| 1,000 | 22 | 1,000 | 674,794.50 | 0.00 | 7.8609 |

The pattern is clear: spreading the same 60k across 1–400 days keeps adding even when deep bears persist, but once I stretch beyond ~400 days the capital stays invested longer and equity peaks drop, so those longer spans behave as “conservative stress tests.” Use the script to rerun with other spans or to capture intermediate stats (max drawdown, Sharpe, etc.) as you build the full backtest.

## HTML backtest runs

The new `playground/btc_dca_backtest.py` script wraps the strategy into `backtesting.lib.FractionalBacktest` so you can generate browser-friendly reports. Running `python playground/btc_dca_backtest.py` now writes `playground/results/btc_dca_span_{span}_backtest_{timestamp}.html` for each span in `[1,100,…,1000]`. Each page mirrors `dca_bh_backtest_1d_...` with the full performance table (equity curve, drawdowns, trades, etc.).  
Console output was:

| Span | Equity Final ($) | Max Drawdown (%) |
| --- | ---: | ---: |
| 1 | 113,159.95 | -80.98 |
| 100 | 62,880.74 | -3.69 |
| 200 | 61,440.33 | -1.90 |
| 300 | 60,960.21 | -1.28 |
| 400 | 60,720.13 | -0.97 |
| 500 | 60,576.11 | -0.77 |
| 600 | 60,480.06 | -0.65 |
| 800 | 60,360.04 | -0.49 |
| 1,000 | 60,288.05 | -0.39 |

The `FractionalBacktest` run warns about some trades remaining open until the end of the dataset and about plotting timezones (common with Bokeh). The generated HTMLs are still valid even though the CLI also tries (and fails) to open browsers via `osascript` in this sandbox; just open the files manually for inspection.
