# On-Chain Signal Deep Test Battery — 2026-02-17

Six experiments testing on-chain metrics as BTC trading signals using real blockchain data (SOPR, NUPL, NVT, Reserve Risk) and proxy computations over 9+ years of price history.

## Test Parameters

| Parameter | Value |
|-----------|-------|
| BTC data | 2017-01-01 to 2026-02-16 (3,334 days) |
| Real on-chain signals | SOPR, NUPL, NVT, Reserve Risk |
| Commission | 20 bps round-trip |
| Walk-forward folds | 10 (expanding window) |
| Permutation tests | 200 shuffles |
| Annualization | sqrt(365) for crypto |

## Signals That Survived Walk-Forward (p < 0.05)

| Strategy | Test | Full Sharpe | WF Sharpe | p-value | CAGR | MaxDD |
|----------|------|------------|-----------|---------|------|-------|
| NUPL Regime (real) | T1 | 0.888 | **0.953** | **0.031** | 31.0% | -60.8% |
| NVT Spike Events | T6 | 1.274 | **0.934** | **0.048** | 68.8% | -63.0% |
| Exchange Flow Proxy | T4 | 1.011 | **0.862** | **0.045** | 35.5% | -55.5% |

## Marginal Signals (0.05 < p < 0.10)

| Strategy | Test | Full Sharpe | WF Sharpe | p-value | CAGR |
|----------|------|------------|-----------|---------|------|
| Multi-Extreme Overlay | T6 | 0.952 | 0.831 | 0.060 | 45.3% |
| NUPL Extreme Events | T6 | 1.017 | 0.797 | 0.059 | 57.6% |
| SOPR Extreme Events | T6 | 0.997 | 0.784 | 0.061 | 56.3% |
| Puell Multiple | T4 | 0.749 | 0.705 | 0.082 | 21.6% |
| 5-Signal Confluence | T3 | 1.095 | 0.677 | 0.099 | 58.1% |
| Reserve Risk (real) | T1 | 0.728 | 0.649 | 0.096 | 21.1% |
| Continuous Composite | T3 | 0.695 | 0.644 | 0.090 | 19.4% |

---

## T1: Longer History (Full BTC Cycle, 2017+)

Tests real on-chain JSON data over 9 years of BTC spot history.

| Strategy | Full Sharpe | WF Sharpe | p-value | CAGR | MaxDD | Exposure |
|----------|------------|-----------|---------|------|-------|----------|
| Buy & Hold | 1.017 | - | - | 58.7% | -83.4% | 100% |
| SOPR Regime (real) | 1.058 | 0.640 | 0.126 | 56.3% | -77.5% | 98.2% |
| **NUPL Regime (real)** | 0.888 | **0.953** | **0.031** | 31.0% | -60.8% | 71.3% |
| Reserve Risk (real) | 0.728 | 0.649 | 0.096 | 21.1% | -61.0% | - |
| NVT Signal (real) | 0.624 | 0.408 | 0.180 | 18.6% | -64.2% | - |

**Finding:** NUPL regime is the only signal that survives WF at 5%. It maps NUPL values to cycle phases: capitulation (<0) = full long, euphoria (>0.75) = flat. Cuts MaxDD by 23 percentage points vs B&H while giving up roughly half the CAGR. SOPR shows promise but fails to reach significance.

## T2: Regime-Conditional (Exit-Only Filters on 200-SMA Momentum)

Tests whether on-chain signals improve a simple momentum strategy when used only as exit overrides.

| Strategy | Full Sharpe | WF Sharpe | p-value | vs Baseline |
|----------|------------|-----------|---------|-------------|
| Baseline (200 SMA) | 1.060 | - | - | - |
| Pi Cycle Exit | 1.396 | 0.633 | 0.139 | +0.336 |
| SOPR Exit | 1.061 | 0.632 | 0.139 | +0.001 |
| NVT Exit | 0.935 | 0.617 | 0.119 | -0.125 |
| NUPL Exit | 0.932 | 0.572 | 0.184 | -0.128 |

**Finding:** Exit-only filters do not significantly improve 200-SMA momentum. Pi Cycle exit shows the best full-sample improvement (+0.336 Sharpe, cuts MaxDD from -70% to -51%) but does not survive WF validation. On-chain signals are better used as standalone regime detectors than as overlays on existing momentum.

## T3: Composite Scoring (V3-Style 0-5 Confluence)

Combines multiple on-chain signals into a single score, mirroring the V3 confluence pattern.

| Strategy | Full Sharpe | WF Sharpe | p-value | CAGR | MaxDD |
|----------|------------|-----------|---------|------|-------|
| 5-Signal Confluence (0-5) | 1.095 | 0.677 | 0.099 | 58.1% | -73.5% |
| Continuous z-Score Composite | 0.695 | 0.644 | 0.090 | 19.4% | -59.0% |

Score distribution (5-signal): `{1: 36, 2: 358, 3: 1170, 4: 1538, 5: 232}`

**Finding:** Composite scoring reaches marginal significance (p~0.09-0.10) but neither variant breaks through at 5%. The binary confluence is better than continuous z-score for CAGR, but the continuous composite has better drawdown control. Compositing does not add alpha over the best individual signal (NUPL regime at p=0.031).

## T4: Alternative Signals (RHODL, Puell, Exchange Flows, STH/LTH)

Tests less popular on-chain metrics using proxy computations.

| Strategy | Full Sharpe | WF Sharpe | p-value | CAGR | MaxDD |
|----------|------------|-----------|---------|------|-------|
| **Exchange Flow Proxy** | 1.011 | **0.862** | **0.045** | 35.5% | -55.5% |
| Puell Multiple | 0.749 | 0.705 | 0.082 | 21.6% | -55.6% |
| STH/LTH Realized | 1.196 | 0.647 | 0.142 | 63.2% | -64.1% |
| RHODL Ratio | 0.315 | 0.277 | 0.224 | 4.8% | -65.2% |

**Finding:** Exchange Flow proxy is the standout (WF 0.862, p=0.045) — volume spikes directionally aligned with price trend predict continuation. Puell Multiple shows marginal edge. STH/LTH Realized looks excellent full-sample but doesn't survive WF — classic overfitting pattern. RHODL is noise. Exchange Flow needs validation with real exchange inflow/outflow data (currently volume-based proxy).

## T5: Weekly Timeframe

Resamples all on-chain signals to weekly frequency with 4-week smoothing.

| Strategy | Full Sharpe | WF Sharpe | p-value | CAGR | MaxDD |
|----------|------------|-----------|---------|------|-------|
| B&H (weekly) | 2.706 | - | - | 2627% | -81.7% |
| Weekly SOPR | 2.609 | 1.069 | 0.212 | 1288% | -61.2% |
| Weekly NUPL | 2.324 | 1.757 | 0.100 | 472% | -54.2% |
| Weekly Composite | 1.990 | 1.284 | 0.127 | 331% | -57.6% |

**Finding:** Weekly signals show high absolute Sharpes but inflated by fewer observations. WF p-values are weaker (0.10-0.21) due to only 8 folds on weekly data. Weekly NUPL is the most promising (WF Sharpe 1.757) but needs more history for statistical confidence. The weekly smoothing does improve drawdown control across all variants.

## T6: Event-Based (Extreme Readings as Risk-Off Overlays)

Uses always-in (100% long) as baseline and triggers risk-off periods only on extreme on-chain readings.

| Strategy | Full Sharpe | WF Sharpe | p-value | vs B&H | Risk-Off Days | MaxDD |
|----------|------------|-----------|---------|--------|---------------|-------|
| Always In (baseline) | 1.017 | - | - | - | 0% | -83.4% |
| **NVT Spike Events** | 1.274 | **0.934** | **0.048** | +0.257 | 57.2% | -63.0% |
| Multi-Extreme | 0.952 | 0.831 | 0.060 | -0.065 | 24.5% | -76.6% |
| NUPL Extreme | 1.017 | 0.797 | 0.059 | +0.000 | 1.0% | -78.8% |
| SOPR Extreme | 0.997 | 0.784 | 0.061 | -0.020 | 2.1% | -83.0% |

**Finding:** NVT spike events is the cleanest actionable signal. When NVT z-score > 2.0, reduce exposure to 30% for 30 days. This improves Sharpe by +0.257 over always-in while cutting MaxDD from -83% to -63%. The 57% risk-off time means this is more of a "selective long" strategy than a pure overlay. Multi-extreme (2+ signals at extremes simultaneously) also promising at p=0.06 with a more traditional overlay profile (24.5% risk-off).

---

## Conclusions

### What Works

1. **NUPL regime detection** is the strongest standalone on-chain signal (WF Sharpe 0.953, p=0.031). Use real NUPL data from CryptoQuant/Glassnode, not proxies.
2. **NVT spike events** provide the best risk-adjusted improvement over buy-and-hold (WF 0.934, p=0.048). Simple implementation: go defensive when NVT z-score spikes.
3. **Exchange Flow proxy** survives WF (p=0.045) but needs real exchange flow data to validate. Volume-directional alignment is the mechanism.

### What Doesn't Work

1. **Exit-only filters** (T2) do not improve momentum strategies. On-chain is better standalone.
2. **Composite scoring** (T3) does not beat the best individual signal. Adding signals adds noise, not diversification — consistent with the optimization paradox observed in other Maestro research.
3. **RHODL ratio** is noise (WF Sharpe 0.277, p=0.22).
4. **Pi Cycle and MVRV** (well-known, likely arbitraged) show full-sample promise but fail WF.

### Recommended Next Steps

1. **Integrate NUPL regime** into V4 as a 6th module with 10% risk budget — the only on-chain signal with p < 0.05 on real data
2. **Add NVT spike detector** as a risk-off overlay (like the SOL LSR extreme that survived in derivatives research)
3. **Source real exchange flow data** (CryptoQuant, Glassnode) to validate the volume-proxy finding
4. **Skip weekly for now** — promising direction but insufficient statistical power with current data length
5. **Do not composite** — use NUPL and NVT as independent, orthogonal signals rather than combining them

---

## Reproduction

```bash
cd /Users/ffv_macmini/Desktop/maestro
python3 backend/research/onchain_deep_tests.py
```

Source: `backend/research/onchain_deep_tests.py`
Results: `data/research/onchain_deep_tests.json`
Real on-chain data: `data/onchain/{sopr,nupl,nvt,reserve_risk}.json`
