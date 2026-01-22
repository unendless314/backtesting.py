# ETH 365-Day Hold Analysis

**Data Source**: `ETHUSDT_1d.csv`
**Full Data Range**: 2017-08-17 to 2026-01-22
**Analyzed Buy Window**: 2017-08-17 to 2025-01-22

## Statistics Overview
| Metric | Value |
| :--- | :--- |
| **Total Trading Days** | 2716 days |
| **Win Rate** | **56.26%** (1528 days) |
| **Loss Rate** | 43.74% (1188 days) |
| **Skewness** | 2.58 (>1 implies fat tail) |

## Returns Analysis
| Metric | Mean (Avg) | Median (Robust) |
| :--- | :--- | :--- |
| **General Return** | 130.48% | **21.23%** |
| **Win Magnitude** | +263.24% | +76.74% |
| **Loss Magnitude** | -40.27% | -38.17% |
| **Reward/Risk Ratio** | **6.54** | **2.01** |

## Kelly Criterion Analysis
> Formula: $f^* = p - \frac{q}{b}$

| Strategy | Allocation (f*) | Half-Kelly |
| :--- | :--- | :--- |
| **Standard (Based on Mean)** | `49.57%` | `24.78%` |
| **Robust (Based on Median)** | `34.50%` | `17.25%` |

## Notes
Analysis generated based on buying everyday within the window and holding for exactly 365 days.
* **Robust Kelly** uses Median Reward/Risk, which is less sensitive to extreme outliers (e.g., 100x pumps) and provides a safer baseline for position sizing.