# VOO 365-Day Hold Analysis

**Data Source**: `VOO.csv`
**Full Data Range**: 2010-09-09 to 2026-01-21
**Analyzed Buy Window**: 2010-09-09 to 2024-08-06

## Statistics Overview
| Metric | Value |
| :--- | :--- |
| **Total Trading Days** | 3500 days |
| **Win Rate** | **90.74%** (3176 days) |
| **Loss Rate** | 9.26% (324 days) |
| **Skewness** | 0.27 (>1 implies fat tail) |

## Returns Analysis
| Metric | Mean (Avg) | Median (Robust) |
| :--- | :--- | :--- |
| **General Return** | 18.89% | **20.61%** |
| **Win Magnitude** | +21.56% | +22.19% |
| **Loss Magnitude** | -7.26% | -7.83% |
| **Reward/Risk Ratio** | **2.97** | **2.83** |

## Kelly Criterion Analysis
> Formula: $f^* = p - \frac{q}{b}$

| Strategy | Allocation (f*) | Half-Kelly |
| :--- | :--- | :--- |
| **Standard (Based on Mean)** | `87.63%` | `43.81%` |
| **Robust (Based on Median)** | `87.48%` | `43.74%` |

## Notes
Analysis generated based on buying everyday within the window and holding for exactly 365 days.
* **Robust Kelly** uses Median Reward/Risk, which is less sensitive to extreme outliers (e.g., 100x pumps) and provides a safer baseline for position sizing.