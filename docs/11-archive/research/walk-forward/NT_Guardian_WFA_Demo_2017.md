# Walk-Forward Analysis Demonstration with backtrader

**Source:** https://ntguardian.wordpress.com/2017/06/19/walk-forward-analysis-demonstration-backtrader/
**Author:** Curtis Miller
**Date:** June 19, 2017
**Archived:** February 5, 2026

---

> DISCLAIMER: Any losses incurred based on the content of this post are the responsibility of the trader, not me. I, the author, neither take responsibility for the conduct of others nor offer any guarantees. None of this should be considered as financial advice; the content of this article is only for educational/entertainment purposes.

## Overview

This article demonstrates how to implement walk-forward analysis with Backtrader, the same library Filos (now Maestro) was built on.

## Key Concepts

### Cross-Validation for Time Series

Data scientists want to fit training models to data that will do a good job of predicting future, out-of-sample data points. This is not done by finding the model that performs the best on the training data. This is called **overfitting**; a model can appear to do well on training data but will not generalize to out-of-sample data.

For trading algorithms:
- We evaluate by profitability, not predictive accuracy
- Time and order matter - cannot reshuffle data
- Must preserve temporal order in folds

### Walk-Forward Analysis

For time-dependent data, we employ walk-forward analysis:

1. Divide data into N periods
2. Fit algorithm on period 1, test on period 2
3. Fit on period 2, test on period 3
4. Continue until end of dataset
5. Use out-of-sample results for evaluation

### TimeSeriesSplitImproved

The article provides an improved version of sklearn's TimeSeriesSplit:

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np

class TimeSeriesSplitImproved(TimeSeriesSplit):
    """Time Series cross-validator with fixed_length option"""
    
    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=1, test_splits=1):
        # ... implementation allows:
        # - fixed_length: training sets of same size
        # - train_splits: number of splits for training
        # - test_splits: number of splits for testing
```

## Key Results

The walk-forward analysis on SMAC (Simple Moving Average Crossover) showed:

| Period | Return | Fast MA | Slow MA |
|--------|--------|---------|---------|
| 1 | 0.70 (30% loss) | 5 | 10 |
| 2 | 1.01 | 50 | 100 |
| 3 | 1.00 | 30 | 100 |
| 4 | 1.01 | 15 | 70 |
| 5 | 0.99 | 50 | 80 |
| 6 | 0.95 | 25 | 60 |
| 7 | 0.94 | 35 | 60 |
| 8 | 1.00 | 50 | 100 |
| 9 | 1.00 | 25 | 100 |

**Conclusion:** Optimization led to overfitting. Final account was 68% of original value.

## Lessons Learned

> "Some traders discuss optimization with an audible smirk, telling a similar story of an unsuspecting novice setting up a SMAC strategy, optimizing the fast and slow moving average windows, seeing good results in a backtest, applying the optimized strategy to future data, and failing to replicate their earlier stellar results."

### Defenses Against Overfitting

1. **Hold-out test set** - Keep recent data truly out-of-sample
2. **Walk-forward analysis** - Test multiple periods
3. **Simple strategies** - Fewer parameters = less overfitting
4. **Multiple assets** - Diversify validation

---

## Relevance to Maestro

This article directly influenced Filos/Maestro's walk-forward implementation:
- `TimeSeriesSplitImproved` concept → Maestro's WFA engine
- Fixed-length windows → Maestro's window modes
- Multi-period optimization → Optuna integration

The original Filos WFA spec (2020) cited this article as a primary reference.
