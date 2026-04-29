# Cross-Asset Relative Momentum — Research Report

> **SUPERSEDED (2026-02-16):** Results in this report were invalidated when the universe was expanded from 10 to 30 tokens — Sharpe dropped from 3.29 to 0.69 (p=1.0). The original 10-token results suffer from survivorship bias and tiny universe artifacts. This report is preserved for research history only. Do not use these results for strategy decisions.

**Date:** 2026-02-14
**Status:** ❌ Invalidated — survivorship bias confirmed by expanded universe test
**Scripts:** `backtest_new_alpha.py`, `backtest_momentum_gauntlet.py`
**Results:** `data/backtest_results/new_alpha_results.json`, `momentum_gauntlet.json`

## Strategy Description

Long top quintile / short bottom quintile of a crypto token universe, ranked by trailing N-day returns. Classic cross-sectional momentum factor applied to crypto perpetuals.

**Best configuration:** lb=14 (14-day lookback), reb=7 (weekly rebalance)

## Results Summary

### Walk-Forward Performance (10 expanding folds)
| Metric | Value |
|--------|-------|
| OOS Sharpe | 3.29 |
| CAGR | 575.6% |
| Max Drawdown | -23.1% |
| p-value (WF t-test) | < 0.001 |
| Data period | 2023-05-18 → 2026-02-10 (2.7 years) |

### Gauntlet Test Results

| Test | Result | Detail |
|------|--------|--------|
| ✅ Realistic commissions | Passed | Corrected Sharpe 3.29 (original was conservative) |
| ✅ Random permutation (500x) | Passed | 0/500 shuffles beat real (p=0.000) |
| ✅ Regime independence | Passed | Bull 3.14, Bear 3.25 — works in both |
| ✅ Low turnover | Passed | 2.3 tokens flip/rebalance, 100-day avg hold |
| ✅ Zero market beta | Passed | Basket correlation -0.052 |
| ✅ Slippage robust | Passed | Sharpe 2.90 even at 100bps costs |

## Critical Caveats ⚠️

### 1. Survivorship Bias (MAJOR)
The 10-token universe was hand-selected: ARB, AVAX, BTC, ETH, FET, INJ, LINK, OP, SOL, SUI. These are all tokens that *still exist and trade* in Feb 2026. Tokens that crashed to zero, delisted, or lost liquidity are excluded.

### 2. Tiny Universe
With only 10 tokens and quintile = 2, you're always long 2 and short 2. This is extremely concentrated. One token dominating (SOL: +312%) can drive the entire result.

### 3. Long Leg Dominance
| Component | Sharpe | CAGR |
|-----------|--------|------|
| Long leg only | 1.99 | 301.8% |
| Short leg only | 0.54 | 9.5% |
| Equal-weight basket | 0.38 | — |

The short side contributes minimally. The strategy is essentially "pick the hottest 2 tokens each week" — momentum works, but the Sharpe is inflated by the spread construction.

### 4. Short Sample
2.7 years captures exactly one crypto cycle (2023 recovery → 2024-25 bull → 2026 volatility). The 2022 bear market is not in sample.

### 5. Execution Reality
- Shorting alts requires borrow availability + funding costs
- 2 positions per side = massive concentration risk
- Slippage on small-cap alts (INJ, FET) likely worse than 20bps

## What's Real vs What's Noise

**Real:** Cross-sectional momentum works in crypto. This is well-documented in academic literature. Winners keep winning, losers keep losing, at least at 1-4 week horizons.

**Noise:** The specific Sharpe of 3.29 is likely a sample artifact of this particular 10-token universe in this particular period. Expected true Sharpe with a proper universe: 1.0-1.5.

## Next Steps to Validate

1. **Expand universe to 30-50 tokens** — include all available Bybit/Binance perps
2. **Include 2021-2022 data** — need bear market performance
3. **Add delisted tokens** (survivorship-free backtest)
4. **Test quintile vs decile** — with larger universe
5. **Combine with V3.1** — momentum factor as additional signal alongside M2 macro

## Comparison to Other Research

| Strategy | OOS Sharpe | p-value | Verdict |
|----------|-----------|---------|---------|
| Mega V3.1 (M2 Macro) | 1.072 | 0.004 | ✅ Validated |
| Cross-Asset Momentum | 3.290 | 0.000 | ⚠️ Promising but biased |
| Price Structure (best) | 0.952 | 0.029 | ❌ Failed deep dive |
| Funding Carry | 1.190 | 0.100 | ❌ Not significant |
| RSI Mean-Reversion | Negative | — | ❌ Destroyed by costs |

## Files
- `backend/backtest_new_alpha.py` — Initial 3-strategy screen
- `backend/backtest_momentum_gauntlet.py` — 5-test stress test suite
- `data/backtest_results/new_alpha_results.json` — All strategy results
- `data/backtest_results/momentum_gauntlet.json` — Gauntlet results
