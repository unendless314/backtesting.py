# BTC 365-Day Hold Analysis

**Data Source**: `BTCUSDT_1d.csv`
**Full Data Range**: 2017-09-01 to 2026-01-21
**Analyzed Buy Window**: 2017-09-01 to 2025-01-21

## Statistics Overview
| Metric | Value |
| :--- | :--- |
| **Total Trading Days** | 2700 days |
| **Win Rate** | **70.19%** (1895 days) |
| **Loss Rate** | 29.81% (805 days) |
| **Skewness** | 2.43 (>1 implies fat tail) |

## Returns Analysis
| Metric | Mean (Avg) | Median (Robust) |
| :--- | :--- | :--- |
| **General Return** | 84.71% | **49.63%** |
| **Win Magnitude** | +137.13% | +95.70% |
| **Loss Magnitude** | -38.67% | -38.52% |
| **Reward/Risk Ratio** | **3.55** | **2.48** |

## Kelly Criterion Analysis
> Formula: $f^* = p - \frac{q}{b}$

| Strategy | Allocation (f*) | Half-Kelly |
| :--- | :--- | :--- |
| **Standard (Based on Mean)** | `61.78%` | `30.89%` |
| **Robust (Based on Median)** | `58.18%` | `29.09%` |

## Notes
Analysis generated based on buying everyday within the window and holding for exactly 365 days.
* **Robust Kelly** uses Median Reward/Risk, which is less sensitive to extreme outliers (e.g., 100x pumps) and provides a safer baseline for position sizing.