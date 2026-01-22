# AAVE 365-Day Hold Analysis

**Data Source**: `AAVEUSDT_1d.csv`
**Full Data Range**: 2020-10-15 to 2026-01-22
**Analyzed Buy Window**: 2020-10-15 to 2025-01-22

## Statistics Overview
| Metric | Value |
| :--- | :--- |
| **Total Trading Days** | 1561 days |
| **Win Rate** | **54.84%** (856 days) |
| **Loss Rate** | 45.16% (705 days) |
| **Skewness** | 3.12 (>1 implies fat tail) |

## Returns Analysis
| Metric | Mean (Avg) | Median (Robust) |
| :--- | :--- | :--- |
| **General Return** | 45.74% | **16.00%** |
| **Win Magnitude** | +126.57% | +86.76% |
| **Loss Magnitude** | -52.41% | -56.08% |
| **Reward/Risk Ratio** | **2.42** | **1.55** |

## Kelly Criterion Analysis
> Formula: $f^* = p - \frac{q}{b}$

| Strategy | Allocation (f*) | Half-Kelly |
| :--- | :--- | :--- |
| **Standard (Based on Mean)** | `36.14%` | `18.07%` |
| **Robust (Based on Median)** | `25.65%` | `12.82%` |

## Notes
Analysis generated based on buying everyday within the window and holding for exactly 365 days.
* **Robust Kelly** uses Median Reward/Risk, which is less sensitive to extreme outliers (e.g., 100x pumps) and provides a safer baseline for position sizing.