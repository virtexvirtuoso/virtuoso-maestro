# Ultrathink Review — V4 System Gaps, Open Questions, and Next Steps

> **Virtuoso Crypto × Maestro Research Platform**
> February 15, 2026

---

## Purpose

After 3,500+ walk-forward tests, 65 signal variants, and 48 tokens tested across a 48-hour research sprint, this document examines what's missing, what's questionable, and what deserves further investigation before deploying V4 to production.

---

## Critical Concerns

### 1. The V4 Top 5 Result Needs Scrutiny 🚨

**Sharpe 3.19 is suspiciously high.** Several structural concerns:

- **Selection bias**: We screened 44 tokens → filtered to 31 survivors → picked Top 5 by OOS Sharpe. The "OOS" screening step and the "OOS" performance measurement may overlap. If the same walk-forward folds were used to both select AND measure, the Sharpe is inflated.
- **Survivorship bias**: SOL, FTM, AVAX, BNB, SUI are all tokens that survived and thrived 2021–2026. We tested with perps data starting 2021 — we literally selected the biggest winners.
- **SMA50 rides winners**: Our own research proved SMA50 Sharpe correlates 0.604 with buy-and-hold returns. Picking the top SMA50 performers IS picking the top B&H performers. That's not alpha — it's leveraged beta selection.
- **No true holdout**: Every token was used for both selection and measurement. No separation between "selection set" and "validation set."

**Required test**: A TRUE out-of-sample validation — pick the Top 5 using data through 2023 only, then measure performance on 2024–2025 unseen data. If Sharpe drops from 3.19 to ~1.5, the selection bias is confirmed.

**Estimated effort**: 30 minutes.

### 2. BTC Wiz On-Chain Battery Never Reported ⏳

The sub-agent testing 27 on-chain signals appears stuck or dead. This is the **single largest untested signal class** and the most promising potential Layer 4 enhancement because:

- On-chain data is fundamentally different from derivatives — it measures actual holder behavior, not trader positioning
- MVRV correctly detected 5/5 cycle tops and bottoms — terrible at timing but real information
- Reserve Risk showed Sharpe 1.035 as M2 complement
- RHODL, NVT, SOPR, Puell Multiple, etc. were never properly tested with our rigorous methodology (14-fold WF + permutation + Bonferroni)

**Required action**: Re-run the BTC Wiz full battery with proper methodology.

**Estimated effort**: 10–15 minutes (sub-agent).

### 3. Adaptive Asset Selection Never Tested 🔄

The Top 5 is **static** (SOL, FTM, AVAX, BNB, SUI). But asset performance rotates:
- FTM was exceptional in 2021, mediocre in 2024
- SUI didn't exist before 2023
- Static selection assumes the future looks like the past

We tested cross-sectional momentum as long/short (failed — shorts kill it) but never tested it as **long-only asset rotation within V4**:

1. Each month: rank all tokens by 3-month momentum
2. Hold Top 5 that also pass SMA50 filter
3. Apply trailing stops + vol ceiling per V4d architecture
4. This adapts to regime changes, avoids dead tokens, and captures emerging leaders

**Required test**: Rolling Top 5 selection (quarterly re-ranking) vs static Top 5.

**Estimated effort**: 15 minutes (sub-agent).

---

## Methodological Gaps

### 4. Trailing Stop Parameters Not Optimized for V4

BTC 12%, ETH 15%, SOL 8% originated from V3.2 research — a different system architecture. Questions:

- Were these values optimal for V4 specifically?
- What about the other 41 tokens? We used a heuristic (2x ATR(20), floored at 5%, capped at 20%)
- Our own research showed optimization hurts trend-following — so aggressive optimization here could backfire

**Recommended approach**: Test 3 fixed universal stops (8%, 12%, 15%) vs ATR-adaptive across the full portfolio. Keep it simple. Avoid Optuna here — the parameter space is small enough for grid search.

### 5. Regime Detector Only Tested Simplistically

The weighted regime score (p=0.012 on BTC) is our only statistically significant derivatives signal. But we tested it as a simple position-scaling overlay. Untested applications:

| Application | Description | Why It Matters |
|-------------|-------------|----------------|
| Regime-dependent asset mix | Score 3–4 → full Top 5. Score 0–1 → BTC/ETH only (flight to quality). Score 2 → equal weight all. | Adapts portfolio risk to market conditions |
| Regime-dependent stops | Tighten stops during score 0–1 (risk-off), loosen during 3–4 (risk-on) | Dynamic risk management |
| Historical regime mapping | What does the score look like during COVID crash, May 2021, FTX collapse, 2023 recovery? | Validates the regime detector on known events |

### 6. Spot Data Limited to 3 Tokens

V4d (Sharpe 1.52) was tested on **spot data** for BTC, ETH, SOL — properly validated. But V4 expanded (Sharpe 3.19) was tested on **perps price data** from `data/ohlcv/` (44 tokens). We're claiming a "spot-only strategy" but validating on perps prices.

Perps prices differ from spot due to:
- Contango/backwardation effects
- Funding rate influence on price
- Occasional premiums/discounts during volatility

**Required action**: Download spot data for the Top 10–15 assets via yfinance and re-validate V4 expanded on actual spot prices.

### 7. Permutation Test Invalid for Trend-Following

V4d's permutation p-value was 0.85 — which looks like a failure. But this is a **known limitation**: standard permutation tests shuffle daily returns, destroying the temporal autocorrelation that trend-following exploits. The shuffled series can produce similar Sharpes by chance because the return distribution is preserved.

Our bootstrap CIs (which preserve serial dependence) DO confirm significance: [0.67, 2.31] excludes zero.

**Required action**: Implement **block permutation** (shuffle blocks of 20–60 days, not individual days) to get a valid p-value for trend-following strategies. This is the methodologically correct test and would resolve the apparent contradiction.

---

## Untested Areas

| Gap | Why It Matters | Difficulty | Priority |
|-----|---------------|------------|----------|
| **Selection bias holdout test** | Top 5 Sharpe 3.19 may be inflated by selection bias | Easy | 🔴 Critical |
| **On-chain signals (BTC Wiz 27)** | Only major signal class not properly tested | Medium | 🔴 Critical |
| **Asset correlation analysis** | Are Top 5 actually diversifying, or just 5 correlated high-beta L1s? | Easy | 🟡 High |
| **Adaptive asset rotation** | Fixed Top 5 may not persist through regime changes | Medium | 🟡 High |
| **V4 on true spot data (15+ tokens)** | Strategy claims spot, validated on perps | Easy | 🟡 High |
| **Block permutation test** | Standard permutation invalid for trend-following | Medium | 🟡 High |
| **Monte Carlo forward simulation** | What does V4 look like in 1,000 random futures? | Medium | 🟢 Medium |
| **Slippage and execution modeling** | Real spot trading has spread + market impact | Easy | 🟢 Medium |
| **Regime-dependent portfolio** | Use the one working overlay intelligently | Medium | 🟢 Medium |
| **Trail stop optimization for V4** | Current values from V3.2, not V4-specific | Easy | 🟢 Medium |

---

## Research Lines Definitively Closed ✅

These categories have been thoroughly tested and should NOT be revisited:

| Category | Tests Run | Data Used | Verdict |
|----------|-----------|-----------|---------|
| M2 acceleration as trading signal | 10+ tests, 7yr spot data | yfinance BTC/ETH/SOL + FRED M2 | Dead — short-window artifact, permutation 0/4 significant |
| MVRV Z-Score as timing signal | 8+ tests, full-cycle | BTC Wiz + yfinance | Harmful — inverse-filters good trades |
| V3 5-signal macro confluence | V3, V3.1, V3.2, V3.5 HYG | Macro + crypto data | Architecture fundamentally flawed |
| Price structure (ICT/HSAKA) | 500+ WF tests, 6 strategies, 9 assets, 4 TFs | Binance perps | Doesn't mechanize at any timeframe |
| Derivatives as standalone signals | 816+ tests, 48 tokens, 17 variants | Coinalyze + CoinGlass | No daily TF alpha (0 survive Bonferroni) |
| Shorts in crypto | 10+ tests across multiple strategies | Multiple | Destroys returns 5/6 times |
| Multi-timeframe (4H on daily) | 4 variants | 4H + daily OHLCV | All worse than daily-only |
| Cross-asset derivatives divergence | 5 variants, 5 assets | CoinGlass LSR/FR/Liq/Taker | Dead — taker rotation lost 96% |
| Whale copy-trading | 4 variants | Whale Hunter DB (1,746 trades) | All negative returns |
| Fear & Greed contrarian | 4 variants, 8yr data | CoinGlass F&G | Backwards for crypto (greed > fear) |
| Funding rate carry / standalone | 7 variants, 18 tokens, 5yr | CoinGlass funding | Noise — mean-reversion destroys capital |
| Taker volume standalone | 7 variants, 25 tokens | CoinGlass taker | Noise — 0 survive Bonferroni |
| Liquidation standalone | 6 variants, 6 assets | CoinGlass liquidations | Noise — cascade overlay has marginal filter value |
| Cross-sectional momentum (L/S) | 81+ combos, 30 tokens | Binance perps 2021–2026 | Short side destroys all value (p=1.0) |
| Session/time-based signals | 5 strategies | Binance perps multi-TF | Dead — time of day has no edge in crypto |
| Larry Williams nested swings | 4 strategies | Binance perps | No better than simple HH/HL detection |
| Cycle indices (CPS/TPI/BPI) | 3 variants | Proxy metrics | Underperform B&H without real on-chain |
| Vol regime strategies | 4 strategies | Coinalyze derivatives | All failed permutation |

---

## The Strategic Question

> **Are we polishing a production system, or are we still looking for alpha?**

These are different workstreams:

### Path A: Validate & Deploy
1. Run the selection bias holdout test (30 min)
2. Run asset correlation analysis (10 min)
3. If Top 5 survives holdout → paper trade for 2–4 weeks
4. If it doesn't → deploy V4d BTC/ETH/SOL (Sharpe 1.52, already validated)
5. Research continues in background but doesn't block deployment

### Path B: One More Research Push
1. BTC Wiz on-chain battery (biggest untested signal class)
2. Adaptive asset rotation
3. Block permutation for proper p-values
4. Regime-dependent portfolio construction
5. Then deploy whatever system emerges

### Path C: Deploy V4d Now, Research in Parallel
1. V4d BTC/ETH/SOL is fully validated (Sharpe 1.52, bootstrap CI excludes zero)
2. Deploy as paper trade immediately
3. Run expanded portfolio + on-chain + rotation research alongside
4. Upgrade to expanded portfolio once validated
5. Real market feedback > another 1,000 permutation tests

---

## Recommended Priority Actions

| # | Action | Time | Impact |
|---|--------|------|--------|
| 1 | **Selection bias holdout test** — pick Top 5 using 2021–2023 data, measure on 2024–2025 | 30 min | Validates or invalidates the 3.19 Sharpe |
| 2 | **Re-run BTC Wiz on-chain battery** — 27 signals with proper WF + permutation | 15 min | Biggest untested alpha source |
| 3 | **Asset correlation analysis** — pairwise correlation of Top 5 during SMA50-long periods | 10 min | Tests if diversification is real |
| 4 | **Download spot data for Top 15** — yfinance for SOL, FTM, AVAX, BNB, SUI, etc. | 10 min | Validates spot vs perps price assumption |
| 5 | **Adaptive rotation** — rolling quarterly Top 5 vs static | 15 min | Tests persistence of asset selection |
| 6 | **Regime-dependent portfolio** — use weighted regime for dynamic asset mix + stop sizing | 15 min | Exploits only surviving derivatives signal |

**Total estimated time: ~1.5 hours for all 6 actions.**

---

## Key Lessons from the Sprint

1. **Simple beats complex**: V4 (SMA50 + trailing stops) outperforms V3's 5-signal macro confluence
2. **Risk management IS the alpha**: Trailing stops contributed more Sharpe than any signal we tested
3. **Diversification is free Sharpe**: Portfolio construction jumps risk-adjusted returns dramatically
4. **Crypto is a momentum market**: Every contrarian signal failed. Trend-following survives.
5. **Derivatives data is noise at daily TF**: 200+ tests, 0 standalone survivors after Bonferroni
6. **Optimization often hurts**: Especially for trend-following strategies
7. **Look-ahead bias is real**: 124% Sharpe inflation from same-bar signals — verify everything
8. **Funding drag matters**: 8–15%/yr hidden tax makes spot strictly better for long-only
9. **More signals ≠ better**: The best system uses ONE indicator (SMA50) plus good risk management

---

*Maestro Research Platform — Virtuoso Crypto*
*Ultrathink Review · February 15, 2026*
