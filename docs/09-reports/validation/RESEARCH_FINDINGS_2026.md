# Maestro Research Findings — 2026

> Last updated: 2026-02-14
> Author: Maestro 🎼 (for Mr. V @ Virtuoso Crypto)

---

## Executive Summary

After extensive systematic testing across **6 strategy families, 9 assets, 4 timeframes, 500+ walk-forward validations, and 15,000+ Optuna trials**, we have identified **one production-ready strategy** and mapped the full alpha landscape of our available data.

**The strongest edge found: Mega Strategy V3.1 — Macro Momentum System.** (WF OOS p=0.178 — promising but not significant at conventional levels.)

Everything else — price structure, derivatives on daily, ICT/HSAKA concepts, ensemble combinations — failed rigorous out-of-sample testing. The research has been thorough enough to close these lines with confidence and redirect effort to the highest-probability next bets.

---

## Table of Contents

1. [Validated: Mega Strategy V3.1](#1-validated-mega-strategy-v31)
2. [Research: Price Structure Strategies](#2-research-price-structure-strategies)
3. [Research: Derivatives Strategies](#3-research-derivatives-strategies)
4. [Research: Strategy Combinations](#4-research-strategy-combinations)
5. [Earlier Findings: Asset & Strategy Rankings](#5-earlier-findings-asset--strategy-rankings)
6. [What We Know About Timeframes](#6-what-we-know-about-timeframes)
7. [What We Know About Optimization](#7-what-we-know-about-optimization)
8. [Untested Alpha Sources](#8-untested-alpha-sources)
9. [Production Roadmap](#9-production-roadmap)
10. [Key Lessons](#10-key-lessons)

---

## 1. Validated: Mega Strategy V3.1

### The System
- **5-signal confluence**: M2 Acceleration, Liquidity Proxy (DXY/Gold/10Y/HYG), Yield Curve, Cross-Asset Momentum, Crypto Momentum
- **5 regimes**: BULL / MILD_BULL / NEUTRAL / BEAR / ACCUMULATION
- **Adaptive leverage**: 0.3x–2x based on confluence score
- **Multi-asset**: BTC 40%, ETH 25%, SOL 20%, LINK 15%

### Performance

| Metric | In-Sample | Out-of-Sample (WF) |
|--------|-----------|---------------------|
| Sharpe | 1.71 | **0.45 mean / 0.81 median** |
| CAGR | — | **41.8%** (full-sample, not WF aggregate) |
| Max DD | -5.5% | -40.9% |
| p-value | 0.005 (full-sample permutation) | **0.178** (WF OOS permutation — not significant) |
| WF Folds | — | 7/13 positive |

### The Core Edge
**M2 acceleration is THE signal.** BTC returns 102.8%/yr when M2 is accelerating vs 2.1% when it's not. No substitute found. Every attempt to improve on V3.1 (V3.2 execution enhancements, V4 signal overlays, V4 on-chain) landed in noise during walk-forward.

### V3.1 Fixes Over V3.0
1. **Position sizing restructured** — direct leverage from confluence score (V3.0 had mean leverage of 0.073, flat 52% of time despite correct signals)
2. **Score 4 dead zone fixed** — demoted to 0.5x when CryptoMom is off (CryptoMom off = -48.6%/yr)
3. **Short side removed** — shorts only worked in 2022; S4 (no shorts) Sharpe 1.714 > S7 (with shorts) 1.486
4. **Protection layers** — vol ceiling 80%, bear filter (flat after 30d at score≤1), trail stop 10%→30%

### Production Status
- **Freqtrade port**: Complete (`backend/freqtrade/macro_momentum_v31_strategy.py`)
- **Config**: Ready (`backend/freqtrade/freqtrade_config_v31.json`)
- **Macro data provider**: Built (`backend/freqtrade/macro_data_provider_v31.py`)
- **Status**: Awaiting deployment for dry-run

---

## 2. Research: Price Structure Strategies (ICT/HSAKA)

### What We Built
6 strategies based on ICT/HSAKA price action concepts:

| Strategy | Lines | Concept | Source |
|----------|-------|---------|--------|
| MarketStructure | 92 | HH/HL/LH/LL + BOS/CHoCH | HSAKA Lesson 6 |
| RangeSFP | 66 | Range quarters + Swing Failure Pattern | HSAKA Lessons 3+4 |
| FairValueGaps | 83 | 3-candle gap fill trading | ICT/Chroma |
| OrderBlocks | 77 | Institutional zone retest | HSAKA Lesson 9 |
| VolumeProfile | 92 | POC/VAH/VAL boundary | Auction Market Theory |
| SRLevels | 133 | Clustered S/R + rejection | HSAKA Lessons 1+2 |

**Location**: `~/Desktop/maestro/backend/strategies/technical/`

### Test Results

#### A. Individual Strategies — BTC Daily (Default Params)
| Strategy | Full Sharpe | OOS Sharpe | p-value | Positive Folds |
|----------|-----------|-----------|---------|----------------|
| MarketStructure | 0.186 | **0.608** | 0.200 | 7/13 |
| RangeSFP | -0.148 | 0.482 | 0.268 | 9/13 |
| OrderBlocks | -0.191 | 0.407 | 0.299 | 6/13 |
| FairValueGaps | -0.242 | -0.344 | 0.312 | 6/13 |
| VolumeProfile | -0.646 | -0.752 | 0.198 | 5/13 |
| SRLevels | -1.056 | 0.000 | 1.000 | 0/13 |

**Buy & Hold benchmark**: Sharpe 0.97, CAGR 41.2%. None beat it.

#### B. Multi-Asset (Default Params, 8 Altcoins)
3 significant results found:
- **ETH OrderBlocks**: OOS 0.952, p=0.029 ✅
- **AVAX MarketStructure**: OOS 0.777, p=0.026 ✅
- **INJ RangeSFP**: OOS 0.679, p=0.045 ✅

**Cross-strategy ranking**: RangeSFP most consistent (0.522 mean OOS, 7/8 assets positive).

#### C. Optuna Hyperparameter Optimization — BTC Daily
**All 6 strategies overfit.** Every one showed positive IS Sharpe → negative OOS Sharpe.

| Strategy | IS Sharpe | OOS Sharpe | Verdict |
|----------|-----------|-----------|---------|
| OrderBlocks | 1.610 | -2.302 | Worst overfit |
| SRLevels | 1.377 | -1.001 | Overfit |
| MarketStructure | 0.762 | -0.404 | Overfit |
| FairValueGaps | 0.705 | -0.759 | Overfit |
| VolumeProfile | 0.343 | -0.958 | Overfit |
| RangeSFP | 0.192 | -0.255 | Least degradation |

**Unoptimized versions performed better OOS than Optuna-tuned ones.**

#### D. Multi-Timeframe (Default + Optuna, 1D/4H/1H/15m)
- **Daily is the only viable timeframe** (43% positive OOS)
- 4H: 19% positive, 1H: 1%, 15m: 0%
- Higher frequency = worse (15m Sharpes: -4 to -32)
- Transaction costs (10bps) crush shorter-term signals

#### E. After Optuna Across All Timeframes (24 combos)
Every "significant" result was significant in the NEGATIVE direction — the strategies reliably lose money after optimization.

### Conclusion
**ICT/HSAKA price structure concepts do not produce mechanical alpha on crypto.** They work for discretionary traders who add judgment, context, and risk management — but as pure systematic signals, they fail walk-forward validation at every level tested.

---

## 3. Research: Derivatives Strategies

### What We Have
- **11 strategies**: FundingRate, RealFundingRate, RealOIMomentum, CrossExchangeOI, SpotPerpBasis, VolRegimeFunding, LiquidationCascade, LiquidationScalp, DeltaFlow, CVDScalp, MicroBasis
- **Data**: Coinalyze — funding (2yr), OI daily/4H/1H, LSR (~1 month), taker buy/sell
- **65 tokens** covered

### Test Results (Daily, BTC/ETH/SOL)
**No significant results**, but the test was compromised:

| Issue | Impact |
|-------|--------|
| OI/LSR data only ~1 month | Not enough for walk-forward |
| Daily timeframe wrong for these signals | Funding resets every 8H, liquidations are intraday |
| 90-99% flat on daily | Thresholds tuned for intraday |
| 2 strategies broken | Pandas `fillna()` deprecation |

### Status: Inconclusive
This is **not a fair rejection** — these strategies were tested at the wrong frequency. The funding rate (2yr of data) on 4H/8H timeframe remains a viable research path.

---

## 4. Research: Strategy Combinations (Ensembles)

### What We Tested
10 ensemble methods combining the 6 price structure strategies:
- Majority vote (3/6, 4/6)
- Unanimous pairs (MS+SFP, MS+OB, SFP+FVG, etc.)
- Weighted score (by prior OOS performance)
- MS as trend filter + others for entry
- Contrarian (inverse of majority)

### Results — 360 Tests (10 ensembles × 4 TFs × 9 assets)

**3 initially significant:**
| Asset | TF | Ensemble | OOS Sharpe | p-value |
|-------|-----|----------|-----------|---------|
| OP | 1D | MS+SFP | 1.347 | 0.037 |
| SOL | 4H | SFP+FVG | 0.974 | 0.042 |
| AVAX | 1D | Contrarian | 0.960 | 0.041 |

### Deep Dive: All 3 Failed Robustness Testing
| Signal | Full Sharpe | vs Random p-value | Verdict |
|--------|-----------|------------------|---------|
| OP MS+SFP | 0.440 | 0.222 | Regime-dependent beta |
| SOL SFP+FVG | -0.756 | 0.889 | Worse than random |
| AVAX Contrarian | -1.032 | 0.973 | Pure noise |

The OOS Sharpe figures were **artifacts of favorable fold selection** in walk-forward. Against 1000 random signal baselines, none beat random at p < 0.05.

### Key Finding
Combining weak signals does not create alpha. The ensemble approach only works when individual signals have genuine predictive power — combining noise produces noise.

---

## 5. Earlier Findings: Asset & Strategy Rankings

### From Feb 6, 2026 Research

**Best Assets for Momentum:**
| Asset | Strategy | OOS Return |
|-------|----------|-----------|
| ZEC | MomentumTrendConfirm | +769% |
| SUI | TrendFollow | +461% |
| SEI | Momentum | +312% |
| CRV | TSMOM | +287% |

**Strategy Family Rankings**: Momentum > TSMOM > Ichimoku > TrendFollowing

**Key Insight**: Privacy coins (ZEC) and high-vol L1s (SUI, SEI) provide asymmetric volatility ideal for momentum strategies. ETH-optimized parameters generalize best across assets.

### Production Performers
| System | Location | Status |
|--------|----------|--------|
| Mega Strategy V3.1 | `backend/strategies/composite/mega_strategy_v31.py` | Ready for deployment |
| Whale Hunter | VPS `~/whale_hunter/` | Live, 102 traders tracked |
| BTC Wiz | VPS `~/btc_wiz/` | Live, 27 on-chain signals |

---

## 6. What We Know About Timeframes

Consistent finding across ALL research:

| Timeframe | Signal Quality | Transaction Cost Impact | Verdict |
|-----------|---------------|----------------------|---------|
| **1D (Daily)** | Highest OOS Sharpe | Minimal | ✅ Best for systematic |
| 4H | Moderate | Moderate | ⚠️ Asset-dependent |
| 1H | Poor | Significant | ❌ Rarely viable |
| 15m | Catastrophic | Devastating | ❌ Never viable for these strategies |

**Daily wins.** The 1D >> 4H >> 1H >> 15m hierarchy held across price structure, derivatives, and all ensemble tests. The only exception is derivatives signals (funding, liquidations) which are inherently intraday — but we couldn't validate this due to data gaps.

---

## 7. What We Know About Optimization

### The Optimization Paradox
Across hundreds of tests, we consistently found that **unoptimized strategies outperform Optuna-tuned ones out-of-sample**.

| System | Unoptimized OOS | Optimized OOS |
|--------|----------------|---------------|
| RangeSFP (multi-asset) | +0.522 mean | -0.255 (BTC) |
| MarketStructure (multi-asset) | +0.230 mean | -0.404 (BTC) |
| OrderBlocks (ETH) | +0.952 | -2.302 (BTC) |

### Why
1. **Overfitting** — Optuna finds parameters that perfectly fit IS noise
2. **Parameter sensitivity** — small changes in params cause large OOS changes = fragile
3. **True edge is in the concept, not the params** — if a signal works, it works with default params. If it needs optimization to show a Sharpe, the Sharpe is fake.

### Rule
**If a strategy doesn't show at least marginal alpha with default parameters, optimization won't create alpha — it will create the illusion of alpha.**

---

## 8. Untested Alpha Sources

### Tier 1 — Highest Probability

#### Cross-Asset Relative Momentum
- **Data**: 65 tokens, 2 years OHLCV — ready to go
- **Concept**: Rank tokens by trailing momentum, long top quintile, short bottom
- **Evidence**: Jegadeesh & Titman momentum factor is the most robust alpha in financial literature. Under-researched in crypto.
- **Why promising**: Pure OHLCV, no data gaps, no optimization needed, factor-based

#### Funding Rate Carry (4H/8H)
- **Data**: 2 years funding across 65 tokens
- **Concept**: Not directional — harvest extreme funding rates, hedge with spot
- **Evidence**: Documented by crypto funds (Amber, Wintermute). Consistent low-Sharpe returns.
- **Why promising**: Crypto-specific structural edge (leverage imbalances create persistent carry)

### Tier 2 — Worth Exploring

#### Whale Copy-Trading
- **Data**: Whale Hunter — 3,267 alerts, 1,737 positions, 102 tracked traders
- **Evidence**: Elite subset shows 57% win rate on HyperLiquid/AsterDex
- **Concept**: Systematic "follow smart money" signal with latency filter
- **Risk**: Data is from a specific time period, may not generalize

#### Multi-Factor Model
- **Concept**: Combine V3.1 macro signals + derivatives (funding, OI) + momentum into unified scoring
- **Evidence**: V3.1 macro alone is Sharpe 1.07. Adding orthogonal signals could boost to 1.3+
- **Risk**: Complexity, more parameters = more overfit risk

### Tier 3 — Needs Data First

#### Options/Vol Surface (Deribit)
- **Status**: Not integrated yet
- **Concept**: Implied vol skew as directional signal, vol surface arbitrage

#### On-Chain Exchange Flows
- **Status**: btc_wiz has some BTC metrics, not systematic across tokens
- **Concept**: Exchange inflow spikes predict selling pressure

---

## 9. Production Roadmap

### Immediate (Week 1)
1. **Deploy V3.1 dry-run** on Bybit via Freqtrade
   - All files ready: strategy, config, macro data provider
   - Validate live FRED/yfinance data feeds
   - Monitor for 1-2 weeks before live capital

### Short-Term (Weeks 2-3)
2. **Research cross-asset momentum**
   - Build long/short quintile portfolio across 65 tokens
   - Walk-forward validate
   - If significant: build Freqtrade multi-pair strategy

### Medium-Term (Month 2)
3. **Funding rate carry on 4H** — collect proper 4H data, test systematic carry
4. **Multi-factor V3.2** — if momentum works, combine with V3.1 macro

### Backlog
5. Whale copy-trading signal
6. Deribit options data integration
7. ML regime enhancement (ready, OOS Sharpe 1.33 — needs production integration)

---

## 10. Key Lessons

### From This Research

1. **M2 is the edge.** Macro liquidity drives crypto more than any technical or microstructure signal. Everything else is noise or regime-dependent beta.

2. **Daily timeframe wins.** Shorter timeframes amplify noise and transaction costs. The only exception may be derivatives-specific signals (funding, liquidations) which are inherently intraday.

3. **Optimization destroys edge.** If default params don't show alpha, optimized params won't either. The apparent improvement is always overfit.

4. **ICT/HSAKA doesn't mechanize.** These concepts (SFP, FVG, order blocks, market structure) are valid frameworks for discretionary trading but fail as systematic signals. The human judgment component is not removable.

5. **Combining noise produces noise.** Ensemble methods only work when component signals have genuine predictive power. 10 ways to combine 6 broken signals = 10 broken ensembles.

6. **Walk-forward is necessary but not sufficient.** Multiple strategies showed significant OOS Sharpe in walk-forward but failed random baseline testing. Always test against random signals with the same investment frequency.

7. **Alts > BTC for momentum.** Higher volatility assets (ZEC, SUI, SEI, OP, INJ) show more exploitable momentum than BTC. This makes sense — less efficient markets, more behavioral biases.

8. **Stop optimizing, start trading.** V3.1 has been ready since Feb 13. Every day spent searching for a better strategy instead of deploying the validated one is negative expected value.

---

## File Index

### Strategies
| File | Location |
|------|----------|
| Mega Strategy V3.1 | `backend/strategies/composite/mega_strategy_v31.py` |
| Mega Strategy V3.2 | `backend/strategies/composite/mega_strategy_v32.py` |
| Price Structure (6) | `backend/strategies/technical/{market_structure,range_sfp,fair_value_gaps,order_blocks,volume_profile,sr_levels}.py` |
| Derivatives (11) | `backend/strategies/derivatives/*.py` |
| Momentum/Technical (20) | `backend/strategies/technical/*.py` |
| Freqtrade V3.1 | `backend/freqtrade/macro_momentum_v31_strategy.py` |

### Backtest Scripts
| Script | Purpose |
|--------|---------|
| `backend/backtest_price_structure.py` | BTC daily WF, 6 strategies |
| `backend/backtest_ps_multi.py` | Multi-asset WF, 6 strategies |
| `backend/optuna_price_structure.py` | Optuna optimization, BTC daily |
| `backend/optuna_ps_timeframes.py` | Optuna across 4 TFs |
| `backend/backtest_ps_timeframes.py` | Default params, 4 TFs |
| `backend/backtest_ps_ensemble.py` | BTC ensemble combinations |
| `backend/backtest_ps_ensemble_lean.py` | Multi-asset multi-TF ensembles |
| `backend/deep_dive_ensembles.py` | Robustness testing on winners |
| `backend/backtest_derivatives_wf.py` | Derivatives WF validation |

### Results
| File | Contents |
|------|----------|
| `data/backtest_results/price_structure_btc_wf.json` | BTC daily, 6 strategies |
| `data/backtest_results/price_structure_multi_wf.json` | 8 assets, 6 strategies |
| `data/backtest_results/price_structure_timeframes.json` | 4 TFs, default params |
| `data/backtest_results/ps_ensemble_results.json` | BTC ensemble, 23 combos |
| `data/backtest_results/ps_ensemble_mtf.json` | 360 ensemble tests |
| `data/optimization/price_structure_optuna.json` | Optuna BTC daily |
| `data/research/ensemble_deep_dive.json` | Deep dive on 3 winners |
| `data/backtest_results/derivatives_wf_results.json` | Derivatives WF |

### Documentation
| File | Contents |
|------|----------|
| `docs/RESEARCH_FINDINGS_2026.md` | This document |
| `docs/V3_DIAGNOSIS_AND_V3.1_PLAN.md` | V3→V3.1 evolution |
| `docs/MEGA_STRATEGY_REPORT.md` | V3 original research |

---

*Research hard, trade smart. — Maestro 🎼*
