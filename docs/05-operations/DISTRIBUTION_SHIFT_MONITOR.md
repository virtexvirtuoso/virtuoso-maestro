# Distribution Shift Monitor — Detecting the Reality Gap in Live Trading

> **Virtuoso Crypto × Maestro Research Platform**
> February 15, 2026
>
> **Status: DESIGN SPEC — Not yet deployed.** Code samples below are specifications, not running production code. Deploy after Tier 1 live trading is stable.

---

## The Problem: Backtest ≠ Live

A strategy with Sharpe 3.0 in backtests can bleed money live. The cause is **distribution shift** — the statistical properties of the market change between the backtest period and live deployment. Returns, volatility, correlation structure, and tail behavior all drift over time.

This kills more strategies than bad fees ever will.

### Why It Happens

| Cause | Description | V4 Exposure |
|-------|-------------|-------------|
| **Regime change** | Market transitions from trending to mean-reverting, or vice versa | Medium — SMA50 assumes markets trend |
| **Volatility compression** | Vol drops as asset class matures, reducing trend-following profits | Medium — crypto vol declining year-over-year |
| **Correlation breakdown** | Portfolio diversification fails when correlations spike | High — Top 5 are correlated L1s (avg ρ = 0.54, spikes to 0.68 in stress) |
| **Microstructure evolution** | Market maker behavior, liquidity depth, execution quality change | Low — V4 trades daily, not sensitive to microstructure |
| **Crowding** | Strategy becomes popular, edge gets arbitraged away | Low — SMA50 is well-known but hard to crowding-arb a daily signal |
| **Survivorship shift** | Assets in backtest may not represent future tradable universe | High — Top 5 selected from survivors |

---

## Measuring Distribution Shift

### Wasserstein Distance (Earth Mover's Distance)

The minimum "work" required to transform one distribution into another. Measures how far apart backtest and live return distributions are.

```
W_p(μ, ν) = (inf_γ ∫ d(x,y)^p dγ(x,y))^(1/p)
```

- **W ≈ 0**: Distributions are identical. Model assumptions hold.
- **W small**: Minor drift. Monitor but no action needed.
- **W large**: Significant shift. Model assumptions may be breaking down.

**For V4**: Compute 1D Wasserstein distance between:
- Backtest daily returns (full walk-forward OOS sample)
- Rolling 60-day live daily returns

### Sinkhorn Divergence

Regularized optimal transport metric. More stable than raw Wasserstein for multivariate distributions. Useful for measuring joint distribution shift across multiple features simultaneously.

```
L_λ(P) = (C, P) − εH(P)
```

**For V4**: Compute 2D Sinkhorn divergence on (return, volatility) pairs to detect joint drift in both return level and vol regime.

### Kolmogorov-Smirnov Test

Non-parametric test for whether two samples come from the same distribution.

- **p > 0.05**: Cannot reject that distributions are the same. Model OK.
- **p < 0.05**: Distributions have significantly diverged. Alert.

### Additional Metrics

| Metric | What It Catches | Formula |
|--------|----------------|---------|
| **Rolling Sharpe divergence** | Performance degradation | \|Sharpe_backtest − Sharpe_live_60d\| |
| **Volatility ratio** | Vol regime shift | σ_live_30d / σ_backtest |
| **Tail ratio** | Fat tail emergence | P(r < -2σ)_live / P(r < -2σ)_backtest |
| **Autocorrelation shift** | Trend structure change | AC(1)_live_60d − AC(1)_backtest |
| **Correlation drift** | Diversification breakdown | mean(ρ_live_30d) − mean(ρ_backtest) |
| **Max drawdown velocity** | Faster-than-expected losses | DD_live / time vs DD_backtest / time |

---

## V4 Specific Vulnerability Analysis

### What Protects V4

| Property | Why It Reduces Shift Risk |
|----------|--------------------------|
| Signal simplicity (SMA50) | Minimal overfit surface area. One parameter, robust across 30-60 range. |
| Parameter stability | Not tuned to a knife edge — neighboring parameters produce similar results |
| Walk-forward validation | OOS performance estimated honestly, not in-sample |
| Reactive risk management | Trailing stops adapt to live volatility — they're reactive, not predictive |
| Futures execution (Bybit perps) | Funding rate costs monitored; adaptive leverage mitigates |
| Low trade frequency | ~20 trades/yr per asset — less exposed to execution microstructure |

### What Makes V4 Vulnerable

| Vulnerability | Severity | Mitigation |
|---------------|----------|------------|
| "Crypto trends" assumption weakens as market matures | 🟡 Medium | Monitor autocorrelation. If AC(1) drops below 0, trend-following edge is gone. |
| Top 5 selected from 2021-2026 survivors | 🔴 High | Adaptive rotation. Don't lock into static basket. |
| Correlated L1 portfolio (PC1 = 66%) | 🔴 High | Monitor correlation. Add cash/stablecoin allocation when correlation spikes. |
| Never seen 2018-style multi-year bleed | 🟡 Medium | 2022 test (+3.1%) is encouraging but only 1 year of bear. |
| Volatility compression over time | 🟡 Medium | Monitor vol ratio. If crypto vol drops to equity-like levels, SMA50 edge shrinks. |
| Trailing stop calibration may not persist | 🟢 Low | ATR-based stops auto-adapt to vol changes. |

---

## Implementation: Live Distribution Shift Monitor

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   SHIFT MONITOR                          │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Backtest    │    │  Live Data   │    │  Alert     │  │
│  │  Reference   │───▶│  Rolling     │───▶│  Engine    │  │
│  │  Distribution│    │  60d Window  │    │            │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│                                                          │
│  Metrics computed daily:                                 │
│  • Wasserstein distance (returns)                        │
│  • KS test p-value                                       │
│  • Rolling Sharpe divergence                             │
│  • Volatility ratio                                      │
│  • Correlation drift                                     │
│  • Autocorrelation shift                                 │
│  • Tail ratio                                            │
│                                                          │
│  Alert levels:                                           │
│  🟢 GREEN  — All metrics within 1σ of backtest           │
│  🟡 YELLOW — 1-2 metrics in warning zone                 │
│  🟠 ORANGE — 3+ metrics in warning zone                  │
│  🔴 RED    — Wasserstein or KS test critical             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Reference Distribution (from backtest)

Build the reference from V4d's walk-forward OOS returns:

```python
import numpy as np
import json

# Load V4d OOS returns from all 14 walk-forward folds
with open("data/backtest_results/v4_honest_system.json") as f:
    results = json.load(f)

# Extract daily OOS returns across all folds
reference_returns = np.array(results["oos_daily_returns"])
reference_vol = np.std(reference_returns) * np.sqrt(252)
reference_sharpe = np.mean(reference_returns) / np.std(reference_returns) * np.sqrt(252)
reference_autocorr = np.corrcoef(reference_returns[:-1], reference_returns[1:])[0, 1]
reference_tail_ratio = np.mean(reference_returns < -2 * np.std(reference_returns))
```

### Daily Live Check

```python
from scipy.stats import ks_2samp, wasserstein_distance

def check_distribution_shift(live_returns_60d, reference_returns):
    """
    Compare rolling 60-day live returns against backtest reference.
    Returns alert level and metrics.
    """
    metrics = {}

    # 1. Wasserstein distance
    metrics["wasserstein"] = wasserstein_distance(reference_returns, live_returns_60d)

    # 2. KS test
    ks_stat, ks_pvalue = ks_2samp(reference_returns, live_returns_60d)
    metrics["ks_stat"] = ks_stat
    metrics["ks_pvalue"] = ks_pvalue

    # 3. Sharpe divergence
    ref_sharpe = np.mean(reference_returns) / np.std(reference_returns) * np.sqrt(252)
    live_sharpe = np.mean(live_returns_60d) / np.std(live_returns_60d) * np.sqrt(252)
    metrics["sharpe_divergence"] = abs(ref_sharpe - live_sharpe)

    # 4. Volatility ratio
    ref_vol = np.std(reference_returns) * np.sqrt(252)
    live_vol = np.std(live_returns_60d) * np.sqrt(252)
    metrics["vol_ratio"] = live_vol / ref_vol

    # 5. Autocorrelation shift
    ref_ac = np.corrcoef(reference_returns[:-1], reference_returns[1:])[0, 1]
    live_ac = np.corrcoef(live_returns_60d[:-1], live_returns_60d[1:])[0, 1]
    metrics["autocorr_shift"] = live_ac - ref_ac

    # 6. Tail ratio
    ref_tail = np.mean(reference_returns < -2 * np.std(reference_returns))
    live_tail = np.mean(live_returns_60d < -2 * np.std(reference_returns))
    metrics["tail_ratio"] = live_tail / max(ref_tail, 0.001)

    # Alert level
    warnings = 0
    if metrics["ks_pvalue"] < 0.05:
        warnings += 2  # Critical
    if metrics["wasserstein"] > np.std(reference_returns) * 0.5:
        warnings += 2  # Critical
    if metrics["sharpe_divergence"] > 1.0:
        warnings += 1
    if metrics["vol_ratio"] > 1.5 or metrics["vol_ratio"] < 0.5:
        warnings += 1
    if metrics["autocorr_shift"] < -0.1:
        warnings += 1
    if metrics["tail_ratio"] > 2.0:
        warnings += 1

    if warnings >= 4:
        metrics["alert"] = "RED"
    elif warnings >= 3:
        metrics["alert"] = "ORANGE"
    elif warnings >= 1:
        metrics["alert"] = "YELLOW"
    else:
        metrics["alert"] = "GREEN"

    return metrics
```

### Alert Thresholds

| Metric | Green | Yellow | Orange/Red |
|--------|-------|--------|------------|
| Wasserstein distance | < 0.5σ_ref | 0.5–1.0σ_ref | > 1.0σ_ref |
| KS test p-value | > 0.10 | 0.05–0.10 | < 0.05 |
| Sharpe divergence | < 0.5 | 0.5–1.0 | > 1.0 |
| Volatility ratio | 0.7–1.3 | 0.5–0.7 or 1.3–1.5 | < 0.5 or > 1.5 |
| Autocorrelation shift | > -0.05 | -0.05 to -0.10 | < -0.10 |
| Tail ratio | < 1.5 | 1.5–2.0 | > 2.0 |
| Correlation drift | < +0.10 | +0.10 to +0.15 | > +0.15 |

### Response Protocol

| Alert | Action |
|-------|--------|
| 🟢 **GREEN** | Continue trading normally. Log metrics. |
| 🟡 **YELLOW** | Review which metric triggered. No position changes. Increase monitoring frequency to check intraday. |
| 🟠 **ORANGE** | Reduce position sizes by 50%. Tighten trailing stops by 30%. Review metrics daily. Alert Mr. V. |
| 🔴 **RED** | Flatten all positions. Do NOT re-enter until metrics return to Yellow or below. Full system review. Alert Mr. V immediately. |

---

## What Distribution Shift Looks Like in Practice

### Scenario 1: Volatility Compression
- **Signal**: Vol ratio drops below 0.5 (crypto vol halves)
- **Impact**: SMA50 produces fewer signals, trailing stops hit more often on noise
- **V4 response**: System naturally reduces exposure (fewer SMA50 crosses), but returns shrink
- **Monitor response**: YELLOW → watch for sustained compression

### Scenario 2: Correlation Spike (2022-style)
- **Signal**: Mean pairwise correlation jumps from 0.54 to 0.80+
- **Impact**: Portfolio "diversification" disappears, all assets fall together
- **V4 response**: Trailing stops should fire across the board, forcing flat
- **Monitor response**: ORANGE → reduce position sizes, tighten stops

### Scenario 3: Trend Structure Breakdown
- **Signal**: Autocorrelation of daily returns drops from positive to zero/negative
- **Impact**: SMA50 trend-following stops working — whipsaw regime
- **V4 response**: Frequent SMA50 crosses with immediate stop-outs. Death by a thousand cuts.
- **Monitor response**: RED → flatten. This means the fundamental assumption (crypto trends) is broken.

### Scenario 4: Fat Tail Emergence
- **Signal**: Tail ratio > 3.0 (3x more extreme losses than backtest predicted)
- **Impact**: Trailing stops may gap through, drawdowns exceed backtest MaxDD
- **V4 response**: DD breaker should catch at -25% portfolio level
- **Monitor response**: RED → flatten immediately

---

## Backtesting the Monitor Itself

Before deploying, validate the monitor using historical regime transitions:

| Period | Expected Alert | Why |
|--------|---------------|-----|
| 2021 Q1-Q3 (bull) | 🟢 GREEN | Strong trends, high vol, normal crypto |
| 2021 May crash | 🟡 YELLOW | Tail ratio spike, brief |
| 2022 Jan-Jun (bear onset) | 🟠 ORANGE | Correlation spike, vol regime change |
| 2022 Nov FTX | 🔴 RED | Extreme tail, correlation 1.0, trend break |
| 2023 recovery | 🟡→🟢 | Gradual return to trending regime |
| 2024 bull | 🟢 GREEN | Strong trends, normal distribution |

If the monitor correctly flags these known events in backtested data, it's calibrated properly for live deployment.

---

## Integration with V4 Production System

```
V4 Trading System
├── Layer 1: SMA50 Signal Generation
├── Layer 2: Risk Management (trailing stops, vol ceiling, DD breaker)
├── Layer 3: Portfolio Construction (sizing, rebalancing)
├── Layer 4: Regime Position Sizing (derivatives composite score)
└── Layer 5: Distribution Shift Monitor ← NEW
    ├── Daily metric computation
    ├── Alert generation
    ├── Automated position scaling (ORANGE → 50% reduction)
    ├── Automated flatten (RED → all positions closed)
    └── Discord/Telegram alerts to Mr. V
```

The monitor sits ABOVE all other layers. It has override authority — even if SMA50 says long and regime says risk-on, a RED alert forces flat.

---

## Key Insight

> The goal is not to prevent distribution shift — it's inevitable.
> The goal is to **detect it early** and **reduce exposure before it destroys capital**.

V4 is simple enough that its core assumptions are explicit and testable:
1. Crypto markets trend (measurable via autocorrelation)
2. Volatility is high enough for SMA50 to generate edge (measurable via vol ratio)
3. Portfolio assets are sufficiently uncorrelated (measurable via correlation matrix)
4. Tail risk is bounded (measurable via tail ratio)

When any of these assumptions break down, the monitor catches it. That's the honest way to trade a systematic strategy.

---

*Maestro Research Platform — Virtuoso Crypto*
*Distribution Shift Monitor · February 2026*
