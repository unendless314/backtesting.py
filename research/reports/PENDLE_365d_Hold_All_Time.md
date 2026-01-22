# PENDLE 365-Day Hold Analysis

**Data Source**: `PENDLE.csv`
**Full Data Range**: 2021-04-29 to 2026-01-22
**Analyzed Buy Window**: 2021-04-29 to 2025-01-21

## Statistics Overview
| Metric | Value |
| :--- | :--- |
| **Total Trading Days** | 1364 days |
| **Win Rate** | **59.68%** (814 days) |
| **Loss Rate** | 40.32% (550 days) |
| **Skewness** | 1.58 (>1 implies fat tail) |

## Returns Analysis
| Metric | Mean (Avg) | Median (Robust) |
| :--- | :--- | :--- |
| **General Return** | 496.85% | **90.44%** |
| **Win Magnitude** | +875.84% | +622.72% |
| **Loss Magnitude** | -64.04% | -69.80% |
| **Reward/Risk Ratio** | **13.68** | **8.92** |

## Kelly Criterion Analysis
> Formula: $f^* = p - \frac{q}{b}$

| Strategy | Allocation (f*) | Half-Kelly |
| :--- | :--- | :--- |
| **Standard (Based on Mean)** | `56.73%` | `28.36%` |
| **Robust (Based on Median)** | `55.16%` | `27.58%` |

## Notes
Analysis generated based on buying everyday within the window and holding for exactly 365 days.
* **Robust Kelly** uses Median Reward/Risk, which is less sensitive to extreme outliers (e.g., 100x pumps) and provides a safer baseline for position sizing.