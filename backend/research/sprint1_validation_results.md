# Sprint 1 Academic Paper Validation Results
**Run date:** 2026-03-01 00:41
**Transaction costs:** 0.06% round-trip
**Walk-forward:** 5 folds, expanding window, 70% train
---

## Signal 1: Matched Filter OFI (Market-Cap Normalization)
**Data:** BTC OHLCV + OI daily, 2024-02-02 to 2026-02-14, 744 rows
**Method:** OI change / price as OFI proxy. Thresholds: 30th/70th percentile from training.
**Note:** Original paper uses L2 order book OFI normalized by market cap. We use OI change as proxy.
**Walk-Forward:** 5 folds, expanding window

| Fold | Train Period | Test Period | OOS Sharpe | OOS Return | Win Rate | MaxDD |
|------|-------------|-------------|------------|------------|----------|-------|
| 1 | 2024-02-02 to 2024-09-11 | 2024-09-12 to 2024-10-25 | -6.25 | -18.1% | 16.3% | -19.7% |
| 2 | 2024-02-02 to 2024-10-25 | 2024-10-26 to 2024-12-08 | -1.01 | -6.3% | 16.3% | -14.9% |
| 3 | 2024-02-02 to 2024-12-08 | 2024-12-09 to 2025-01-21 | -0.55 | -2.5% | 18.6% | -8.6% |
| 4 | 2024-02-02 to 2025-01-21 | 2025-01-22 to 2025-03-06 | -2.73 | -9.1% | 13.9% | -10.7% |
| 5 | 2024-02-02 to 2025-03-06 | 2025-03-07 to 2025-04-19 | 2.03 | 9.1% | 32.6% | -10.9% |

**Aggregate:**
- Mean OOS Sharpe: -1.70
- Median OOS Sharpe: -1.01
- Min OOS Sharpe: -6.25
- % Folds Profitable: 20%
- t-stat (Sharpe > 0): -1.24 (p=0.859)

**VERDICT: NEEDS MORE DATA**
**Reason:** OI proxy may not capture true OFI dynamics. Original requires L2 book data.

---

## Signal 2: DAR Funding Rate Prediction (Inan)
**Data:** BTC OHLCV + funding rate, 2024-02-02 to 2026-02-14, 744 rows
**Method:** DAR(1): funding_t = c + φ*funding_{t-1}. Signal: predicted FR > mean+0.5σ → short; < mean-0.5σ → long.
**Walk-Forward:** 5 folds, expanding window

| Fold | Train Period | Test Period | OOS Sharpe | OOS Return | Win Rate | MaxDD |
|------|-------------|-------------|------------|------------|----------|-------|
| 1 | 2024-02-02 to 2024-09-11 | 2024-09-12 to 2024-10-25 | 1.78 | 5.6% | 13.9% | -6.8% |
| 2 | 2024-02-02 to 2024-10-25 | 2024-10-26 to 2024-12-08 | -3.81 | -10.1% | 7.0% | -12.9% |
| 3 | 2024-02-02 to 2024-12-08 | 2024-12-09 to 2025-01-21 | 1.89 | 4.1% | 4.7% | -3.2% |
| 4 | 2024-02-02 to 2025-01-21 | 2025-01-22 to 2025-03-06 | -2.68 | -14.6% | 16.3% | -17.3% |
| 5 | 2024-02-02 to 2025-03-06 | 2025-03-07 to 2025-04-19 | -3.56 | -13.6% | 20.9% | -14.3% |

**Aggregate:**
- Mean OOS Sharpe: -1.28
- Median OOS Sharpe: -2.68
- Min OOS Sharpe: -3.81
- % Folds Profitable: 40%
- t-stat (Sharpe > 0): -0.99 (p=0.811)

**VERDICT: SKIP**
**Reason:** Funding rate mean-reversion via DAR shows weak/inconsistent signal.

---

## Signal 3: HAR Funding→Vol (Kim)
**Data:** BTC OHLCV + funding rate, 2024-02-02 to 2026-02-14
**Method:** HAR model: RV_t = c + β_d·RV_{t-1} + β_w·RV_{t-5} + β_m·RV_{t-22} [+ γ·FR_{t-1}]
**Walk-Forward:** 5 folds, expanding window

**This is a vol forecast model, not a directional signal.**

### Baseline HAR (no funding)
| Fold | Test Period | MAE | RMSE |
|------|------------|-----|------|
| 1 | 2024-09-28 to 2024-11-09 | 0.013755 | 0.017275 |
| 2 | 2024-11-10 to 2024-12-22 | 0.015642 | 0.020049 |
| 3 | 2024-12-23 to 2025-02-03 | 0.012272 | 0.014169 |
| 4 | 2025-02-04 to 2025-03-18 | 0.015781 | 0.021702 |
| 5 | 2025-03-19 to 2025-04-30 | 0.015678 | 0.019078 |

### Enhanced HAR (with funding rate)
| Fold | Test Period | MAE | RMSE |
|------|------------|-----|------|
| 1 | 2024-09-28 to 2024-11-09 | 0.013622 | 0.017140 |
| 2 | 2024-11-10 to 2024-12-22 | 0.015398 | 0.019801 |
| 3 | 2024-12-23 to 2025-02-03 | 0.012176 | 0.014027 |
| 4 | 2025-02-04 to 2025-03-18 | 0.015743 | 0.022081 |
| 5 | 2025-03-19 to 2025-04-30 | 0.015183 | 0.018870 |

**MAE Improvement:** 1.34% average across folds
**Paired t-test (baseline vs enhanced MAE):** t=2.49, p=0.067

**VERDICT: MARGINAL**
**Reason:** Funding rate improves vol forecasts by 1.3% on average OOS.

---

## Signal 4: LightGBM Vol Enhancement
**Data:** BTC OHLCV + all derivatives, 2024-02-02 to 2026-02-14
**Method:** LightGBM regression predicting next-day RV.
**Base features:** lagged RV (1,2,3), weekly RV, monthly RV, volume
**Enhanced features:** + funding rate, OI change, LSR, liquidations
**Walk-Forward:** 5 folds, expanding window

### Baseline (OHLCV only)
| Fold | Test Period | MAE | RMSE |
|------|------------|-----|------|
| 1 | 2024-09-28 to 2024-11-09 | 0.015337 | 0.019974 |
| 2 | 2024-11-10 to 2024-12-22 | 0.014643 | 0.019118 |
| 3 | 2024-12-23 to 2025-02-03 | 0.012501 | 0.015038 |
| 4 | 2025-02-04 to 2025-03-18 | 0.017083 | 0.022841 |
| 5 | 2025-03-19 to 2025-04-30 | 0.015908 | 0.019742 |

### Enhanced (+ derivatives)
| Fold | Test Period | MAE | RMSE |
|------|------------|-----|------|
| 1 | 2024-09-28 to 2024-11-09 | 0.013601 | 0.017912 |
| 2 | 2024-11-10 to 2024-12-22 | 0.015013 | 0.019270 |
| 3 | 2024-12-23 to 2025-02-03 | 0.011384 | 0.013780 |
| 4 | 2025-02-04 to 2025-03-18 | 0.016083 | 0.022428 |
| 5 | 2025-03-19 to 2025-04-30 | 0.014968 | 0.019108 |

**MAE Improvement:** 5.90% average across folds
**Paired t-test:** t=2.57, p=0.062

**VERDICT: IMPLEMENT**
**Reason:** Derivatives features improve LightGBM vol forecasts by 5.9% on average OOS.

---

## Signal 5: LWI (Liquidity Withdrawal Index)
**VERDICT: CANNOT IMPLEMENT**

**Reason:** LWI = cancellations / (depth + additions) requires L2 order book data with:
- Order cancellation events
- Book depth snapshots
- Order additions/modifications

We have no L2 order book data in our dataset. Would need a live or historical LOB feed (e.g., Tardis.dev, Kaiko, or direct exchange websocket recording).

**Data needed:** L2 order book snapshots or event-level book data at sub-second granularity.
**Estimated cost:** $200-500/mo for historical LOB data provider.

---

## Signal 6: TPE Configuration Audit
**Audit of `walk_forward_optuna.py` against paper recommendations:**

- ✅ Using TPESampler (correct)
- ✅ multivariate=True (recommended by paper)
- ✅ n_startup_trials=15 (paper recommends ≥10)
- ✅ Pruning enabled (Hyperband/Median)
- ⚠️ group not set — paper recommends group=True for structured search

**VERDICT: AUDIT COMPLETE — see findings above**

---

## Summary

| Signal | Type | Verdict | Key Metric |
|--------|------|---------|------------|
| 1. OFI Proxy | Directional | See above | OOS Sharpe |
| 2. DAR Funding | Directional | See above | OOS Sharpe |
| 3. HAR Vol | Vol Forecast | See above | MAE improvement |
| 4. LightGBM Vol | Vol Forecast | See above | MAE improvement |
| 5. LWI | Directional | CANNOT IMPLEMENT | No data |
| 6. TPE Audit | Config | COMPLETE | N/A |
