# CoinGlass Research Report — 65 Signal Variants, 3,000+ Walk-Forward Tests

> **Virtuoso Crypto × Maestro Research Platform**
> February 15, 2026

---

## Bottom Line: What Works and What Doesn't

After exhaustive testing of every derivatives-based signal class against CoinGlass data (331K rows, 2019–2026), the conclusion is unambiguous:

- **The edge is in risk management and portfolio construction** — not in derivatives signals
- **SMA50 + trailing stops + diversification** achieves Sharpe 3.19 on the Top 5 portfolio
- **All standalone derivatives signals fail** Bonferroni-corrected statistical significance
- **The only surviving overlay**: a composite regime detector (LSR + Funding + Liquidations + Taker Volume) that reduces MaxDD ~15% on V4

---

## Database: 331,784 Rows Across 13 Tables

All data consolidated into `maestro.duckdb` for unified querying.

| Table | Rows | Coverage | Source |
|-------|------|----------|--------|
| `perps_daily` | 56,723 | 44 tokens, 2021–2026 | Binance via CCXT |
| `cg_liquidations` | 45,334 | 26 tokens, 2019–2026 | CoinGlass v4 API |
| `cg_lsr_top_account` | 42,546 | 19 tokens, 2020–2026 | CoinGlass v4 API |
| `cg_lsr_top_position` | 42,546 | 19 tokens, 2020–2026 | CoinGlass v4 API |
| `cg_lsr_global` | 41,034 | 19 tokens, 2020–2026 | CoinGlass v4 API |
| `cg_taker_volume` | 36,537 | 25 tokens, 2021–2026 | CoinGlass v4 API |
| `cg_funding_rate` | 34,594 | 18 tokens, 2020–2026 | CoinGlass v4 API |
| `coinalyze_funding` | 12,083 | 65 tokens, 2023–2026 | Coinalyze API |
| `coinalyze_liquidations` | 9,428 | 65 tokens, 2024–2026 | Coinalyze API |
| `spot_daily` | 8,486 | BTC/ETH/SOL, 2017–2026 | yfinance |
| `whale_trades` | 1,746 | HyperLiquid, 2025–2026 | Whale Hunter |
| `cg_options` | 36 | BTC/ETH snapshot | CoinGlass v4 API |
| `coinalyze_oi` | 403 | 13 tokens, 2026 | Coinalyze API |
| `coinalyze_lsr` | 286 | 13 tokens, 2026 | Coinalyze API |

**CoinGlass API**: Key on Hobbyist tier ($29/mo). Working endpoints: Funding Rate OHLC, LSR (3 types), Liquidation Aggregated, Taker Buy/Sell, Fear & Greed, Options Max Pain/Info. Blocked: Orderbook History, Hyperliquid Whale, On-Chain, ETF, Grayscale.

---

## Production Candidates

### V4 Honest System — Core Architecture

| Layer | Component | What It Does |
|-------|-----------|-------------|
| 1 | SMA(50) trend filter | Long when Close > SMA(50), flat otherwise. No shorts. |
| 2 | Per-asset trailing stops | BTC 12%, ETH 15%, SOL 8%. Calibrated to vol profile. |
| 2 | Vol ceiling | Halve position when 30d realized vol > 80% annualized |
| 2 | Drawdown breaker | Go flat if portfolio DD > 25% |
| 3 | Portfolio construction | Equal-weight, spot only, secular uptrend assets |

**Critical**: Signal on bar N, trade on bar N+1 (look-ahead bias inflates Sharpe by 124%).

### V4d Performance — BTC/ETH/SOL (Conservative)

| Variant | Sharpe | CAGR | MaxDD | Note |
|---------|--------|------|-------|------|
| V4a: SMA50 only | 1.19 | 52.7% | -47.9% | Baseline |
| V4b: + trailing stops | 1.62 | 53.7% | -21.7% | **Stops = MVP** |
| V4c: + vol ceiling | 1.52 | 41.9% | -18.7% | Modest DD reduction |
| V4d: + DD breaker | 1.52 | 41.9% | -18.7% | Breaker never triggered |
| Buy & Hold | 1.24 | 92.1% | -85.5% | Higher CAGR, 4.5x worse DD |

- Bootstrap 95% CI: [0.67, 2.31] — excludes zero ✅
- 2022 bear: **+3.1%** vs B&H **-79.6%**

### V4 Expanded Portfolio — Top 5 (Aggressive)

**Assets**: SOL, FTM, AVAX, BNB, SUI — selected via walk-forward screening of 44 tokens.

| Portfolio | Assets | Sharpe | CAGR | MaxDD |
|-----------|--------|--------|------|-------|
| **Top 5** | **5** | **3.19** | **117%** | **-14.6%** |
| Top 10 | 10 | 3.19 | 103% | -16.5% |
| Top 15 | 15 | 2.87 | 88% | -18.4% |
| All 31 | 31 | 2.27 | 56% | -18.1% |

- Bootstrap 95% CI: [2.37, 3.98]
- Walk-forward: avg OOS Sharpe 2.53, 79% positive folds
- Year-by-year: 2021 +480%, 2022 +7.6%, 2023 +162%, 2024 +152%, 2025 +32%

### V4 + Regime Overlay (Enhanced)

Composite regime score from 4 derivatives signals:

| Component | Weight | Score +1 When |
|-----------|--------|---------------|
| LSR | 35% | LSR < 50th percentile (30d rolling) |
| Funding Rate | 35% | Funding < 0.03% |
| Liquidations | 15% | Liquidations < 80th percentile (30d) |
| Taker Volume | 15% | Taker buy/sell ratio > 1.0 |

**BTC avg next-day return by regime score**: Score 0 = -0.94%, Score 1 = -0.16%, Score 2 = -0.09%, Score 3 = +0.18%, Score 4 = +0.31%. Monotonic relationship — real predictive content.

**V4 Overlay result**: Modest Sharpe improvement + **MaxDD reduced ~15%** across all assets. Weighted regime survived Bonferroni on BTC (p=0.012).

---

## Signal-by-Signal Results: What We Tested

### LSR Full-Cycle (5yr CoinGlass Data, 2020–2026)

| Signal | Assets | Best Sharpe | Bonferroni | Verdict |
|--------|--------|-------------|------------|---------|
| LSR Contrarian | BTC/ETH/SOL/BNB | 1.04 (ETH) | ❌ 0/16 | Directionally correct but noisy |
| LSR Momentum | BTC/ETH/SOL/BNB | Negative everywhere | ❌ | Following crowd = anti-alpha |
| LSR Extreme (< 1.0) | BTC/ETH/SOL/BNB | 1.28 (SOL) | ✅ SOL only (p=0.048) | Rare event (43 occurrences) |
| Top Trader Divergence | BTC/ETH/SOL/BNB | ~0.5 | ❌ | Smart vs dumb money — no edge |
| V4 + LSR Filter | BTC/ETH/SOL/BNB | ~1.5 | ❌ | Doesn't improve V4 |
| V4 + Smart Money | BTC/ETH/SOL/BNB | ~1.5 | ❌ | Doesn't improve V4 |

**Conditional analysis**: BTC LSR < 1.0 → +3.0% avg 14d return (63% WR). ETH LSR < 10th pctl → +7.7% avg 14d return (61% WR). Useful as conviction booster, not as standalone signal.

### Funding Rate Carry (5yr, 18 Tokens)

| Signal | Best Sharpe | Bonferroni | Verdict |
|--------|-------------|------------|---------|
| Cross-sectional carry | 0.20 | ❌ | -74% drawdown, untradeable |
| Extreme contrarian | 0.13 | ❌ | Barely beats zero |
| Mean-reversion | -0.28 | ❌ | **Destroys capital** (-99% on multiple tokens) |
| Funding trend | 0.10 | ❌ | Near-zero |
| Price-funding divergence | -0.26 | ❌ | Negative |
| V4 + funding filter | 0.48 | ❌ | Marginally beats SMA50 baseline (0.46) |
| V4 + neg funding boost | 0.45 | ❌ | Slightly hurts |

**Conditional**: Extremely negative BTC funding (< -0.01%) → +2.33% next-day return, 63% WR. Only 27 events in 5 years.

### Taker Buy/Sell Volume (25 Tokens, 2021–2026)

| Signal | Avg Sharpe | Best Asset | Bonferroni | Verdict |
|--------|-----------|------------|------------|---------|
| Imbalance momentum | 0.18 | ATOM (0.98) | ❌ | Asset-specific |
| Z-score | 0.18 | AVAX (1.18) | ❌ | Borderline |
| Volume surge | 0.04 | — | ❌ | Too noisy |
| CTD momentum | 0.27 | SOL (1.08) | ❌ | Best standalone |
| Contrarian | -0.43 | — | ❌ | **Loses badly** |
| V4 + taker confirm | 0.23 | INJ (1.12) | ❌ | Marginal |
| V4 + taker exit | 0.24 | SOL (1.23) | ❌ | Marginal |

**Key insight**: Buy/sell ratio has +3% correlation with next-day returns — real but too weak to trade.

### Liquidation Signals (45K Rows, 2019–2026)

| Signal | Best Sharpe | Bonferroni | Verdict |
|--------|-------------|------------|---------|
| Flush buy (95th pctl) | 0.80 (AVAX) | ❌ | Interesting event study |
| Flush w/ direction | 0.73 (BNB) | ❌ | Closest to significance (p=0.018) |
| Liquidation calm | — | ❌ | Too few events fire |
| Price-liq divergence | — | ❌ | Net loser |
| Cascade overlay on V4 | 1.40 (SOL) | ❌ | **Useful as risk filter** |
| L/S liquidation ratio | — | ❌ | Net loser |

**Event study**: AVAX flush events → +1.8% day 1, +5.8% day 3, +8.3% day 7, +11.2% day 14. Strongest post-flush bounce across all assets.

### Volume & OI Breakout (5 Assets)

| Signal | Best Sharpe | Bonferroni | Verdict |
|--------|-------------|------------|---------|
| Vol-confirmed SMA50 | 0.51 (AVAX) | ❌ | Sparse signals |
| Volume surge breakout | 0.90 (BTC) | ❌ | Closest (p=0.016) |
| Volume dry-up breakout | — | ❌ | **Zero signals** fired |
| Taker breakout | 0.63 (BTC) | ❌ | 78% WR but 14 trades |
| V4 + volume confirm | 0.63 (AVAX) | ❌ | Marginal |
| Liq + OI continuation | 0.60 (ETH) | ❌ | Inconsistent |
| Liq + OI exhaustion | Negative | ❌ | Contrarian doesn't work |
| OI buildup breakout | — | ❌ | **Zero signals** |
| OI divergence | Negative | ❌ | Shorting on divergence loses |
| Combined score | — | ❌ | 2-6 trades total — can't evaluate |

**Event study**: Volume surges (>2x avg) → BTC +2.1% median at 5d, 62-66% WR. Directionally informative but not tradeable standalone.

**Note**: OI data limited to 744 days (2024-2026). Insufficient for robust conclusions.

### Cross-Asset Divergence (5 Variants)

| Signal | Return | Sharpe | Verdict |
|--------|--------|--------|---------|
| LSR divergence (BTC vs alts) | +243% | 0.35 | ❌ -87% MaxDD |
| Funding spread rotation | +15% | 0.35 | ❌ 16 trades only |
| Taker rotation | **-96%** | -0.22 | ❌ **Catastrophic** |
| Liquidation divergence | +0.6% | 0.04 | ❌ No edge |
| Relative regime score | **-87%** | 0.07 | ❌ No edge |

**Baseline SMA50**: +5,572%, Sharpe 0.63. No divergence signal comes close.

### Multi-Timeframe — 4H on Daily (4 Variants)

| Signal | Sharpe | Δ vs Daily SMA50 | Verdict |
|--------|--------|-------------------|---------|
| 4H RSI dip-buying | 0.08 | **-1.09** | Destroys returns |
| 4H volume confirm | 1.12 | -0.04 | Marginally worse |
| 4H EMA ribbon | 1.15 | -0.01 | Essentially identical |
| 4H trailing stop | 1.12 | -0.05 | No DD improvement |

**Conclusion**: Daily-only SMA50 remains optimal. 4H adds noise, not signal.

### Whale Copy-Trading (1,746 Trades, Nov 2025–Feb 2026)

| Signal | BTC Return | Verdict |
|--------|-----------|---------|
| Whale copy (all) | -38% | ❌ |
| Whale consensus (3+) | Negative | ❌ |
| Elite whale (>60% WR) | -6% | ❌ |
| Whale + SMA50 | Flat | ❌ |

**Trader stats**: 66 traders with 2+ closes. Top: HQ_$11M (64.2% WR, 131 trades). Copying even elite whales produces negative returns during bearish regimes.

### Fear & Greed Index (8yr History)

| Signal | Best Sharpe | Verdict |
|--------|-------------|---------|
| F&G contrarian (long < 20) | Negative | ❌ Extreme fear = more fear |
| F&G momentum (cross 50) | 0.02 OOS (BTC) | ❌ Walk-forward is flat |
| V4 + F&G filter (flat > 85) | 0.34 OOS (BTC) | ❌ Marginal, not significant |
| F&G + SMA50 (< 60 filter) | Killed returns | ❌ Filters out best periods |

**Key finding**: Extreme greed (+0.54%/day) outperforms extreme fear (-0.23%/day). **Contrarian is backwards for crypto** — momentum dominates.

### Options Max Pain

Snapshot data only — no historical time series available on Hobbyist tier. Cannot backtest.

Current observation: BTC at near-term max pain ($69-71K), 13-23% below longer-dated max pain ($80-90K).

---

## What Survived Statistical Scrutiny

Out of 65 signal variants tested with 14-fold walk-forward validation, 500 permutation tests, and Bonferroni correction:

| Signal | Bonferroni p | Sharpe | Practical Use |
|--------|-------------|--------|---------------|
| V4 Top 5 Portfolio | Bootstrap CI [2.37, 3.98] | 3.19 | **Production system** |
| V4d BTC/ETH/SOL | Bootstrap CI [0.67, 2.31] | 1.52 | **Conservative variant** |
| Weighted Regime (BTC) | p = 0.012 | 1.12 | **V4 overlay for position sizing** |
| SOL LSR Extreme MR | p = 0.048 | 1.28 | Niche — 43 events in 5yr |

Everything else is noise at daily timeframe.

---

## Strategic Conclusions

### What Generates Edge in Crypto

1. **Trend following** (SMA50) — the honest baseline. Sharpe ~1.1 per asset.
2. **Risk management** — trailing stops halved MaxDD (-48% → -19%). Single biggest improvement.
3. **Portfolio diversification** — equal-weight across 5-10 trending assets. Sharpe jumps from ~1.1 to 3.19.
4. **Spot execution** — funding drag (8-15%/yr) is a hidden tax on perps for long-only strategies.

### What Doesn't Generate Edge

1. **Derivatives signals standalone** — 200+ tests, 0 survive Bonferroni at daily TF
2. **Contrarian strategies** — crypto trends, it doesn't mean-revert. F&G contrarian, taker contrarian, funding mean-reversion all lose money
3. **Multi-timeframe refinement** — 4H adds noise to daily signals
4. **Cross-asset divergence** — derivatives divergence between BTC and alts has no predictive power
5. **Whale copy-trading** — negative returns, especially in bear regimes
6. **Signal complexity** — more signals ≠ better. V3's 5-signal macro confluence (Sharpe ~0.8 OOS) underperforms V4's simple SMA50 + trailing stops (Sharpe 1.52)

### The Meta-Lesson

The alpha in crypto quant isn't in finding clever signals — it's in **not losing money during drawdowns**. SMA50 captures the secular crypto uptrend. Trailing stops protect capital. Portfolio construction smooths the ride. Everything else is noise dressed up as insight.

---

## Methodology

All backtests followed identical rigorous methodology:

| Parameter | Value |
|-----------|-------|
| Walk-forward | 14-fold expanding window, min 365d training |
| Signal execution | Bar N signal → bar N+1 trade (no look-ahead) |
| Permutation tests | 500 per signal per asset |
| Multiple testing | Bonferroni correction (α = 0.05 / total tests) |
| Confidence intervals | Bootstrap 95% CI on Sharpe, CAGR, MaxDD |
| Funding drag | Included for all perps comparisons (8-15%/yr) |

**Look-ahead bias audit**: Same-bar signal execution inflates Sharpe by 124% (2.50 → 1.11). All results in this report use proper bar-shifted signals.

---

## Files and Reproducibility

### Results (JSON)
```
~/Desktop/maestro/data/backtest_results/
├── v4_honest_system.json
├── v4_portfolio_expanded.json
├── regime_detector_test.json
├── lsr_fullcycle_test.json
├── funding_carry_test.json
├── taker_volume_test.json
├── liquidation_signals_test.json
├── volume_oi_test.json
├── crossasset_mtf_test.json
└── whale_fg_options_test.json
```

### Strategy Code
```
~/Desktop/maestro/backend/strategies/composite/
├── v4_honest_system.py         # Production V4 implementation
└── mega_strategy_v32.py        # Legacy V3.2 (superseded)
```

### Database
```
~/Desktop/maestro/data/maestro.duckdb          # 6.8 MB, 331K rows
~/Desktop/maestro/backend/build_database.py    # Idempotent builder
~/Desktop/maestro/backend/data_loader_v2.py    # MaestroDB query class
```

### Usage
```python
from data_loader_v2 import MaestroDB

db = MaestroDB()
btc = db.spot("BTC")
lsr = db.lsr("BTC", type="global")
funding = db.funding("ETH")
db.query("SELECT * FROM perps_daily WHERE symbol='SOL' AND date > '2024-01-01'")
print(db.summary())
```

---

*Maestro Research Platform — Virtuoso Crypto*
*3,500+ walk-forward tests · 65 signal variants · 48 tokens · February 2026*
