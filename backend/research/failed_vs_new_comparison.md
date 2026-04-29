# Failed Strategies vs New Strategies — Side-by-Side Comparison

**Context**: 12 standard scalping strategies tested on 5m/15m timeframes across 1,449 tests. **ALL FAILED** (0 Bonferroni survivors). Daily trend-following showed weak edge (105 survivors).

**Goal**: Understand WHY these 29 new strategies have better odds of success.

---

## ❌ The 12 Failed Strategies

| Strategy | Type | Timeframe | Why It Failed |
|----------|------|-----------|---------------|
| VWAP Reversion | Mean Reversion | 5m/15m | **Too short** (transaction costs 10bps destroy edge), **single factor** (just VWAP cross) |
| StochRSI | Mean Reversion | 5m/15m | **Noisy indicator** (whipsaw), **no regime filter**, **everyone uses it** (crowded) |
| EMA Ribbon | Trend | 5m/15m | **Too short** (lag + transaction costs), **single factor** (just EMA cross) |
| Bollinger MR | Mean Reversion | Daily | **No reversal confirmation**, **blind MR in trends**, **tested on daily where MR fails** |
| RSI Extreme | Mean Reversion | 5m/15m | **Single threshold** (RSI 30/70), **no trend filter**, **crowded trade** |
| Momentum Burst | Momentum | 5m/15m | **Too short**, **no volume confirmation**, **false signals common** |
| Range Breakout | Breakout | 5m/15m | **No volatility filter** (false breakouts in low vol), **no trend context** |
| MACD Scalp | Trend | 5m/15m | **Lagging indicator**, **too short timeframe**, **transaction costs kill edge** |
| Keltner MR | Mean Reversion | 5m/15m | **Similar to Bollinger** (same failure modes), **no confirmation** |
| Dual EMA+ADX | Trend | 5m/15m | **Too short**, **ADX lags**, **transaction costs** |
| Volume Spike BO | Breakout | 5m/15m | **No price confirmation** (volume spike ≠ directional edge), **too short** |
| Stochastic Cross | Momentum | 5m/15m | **Noisy**, **no trend filter**, **single factor** |

### Common Failure Patterns
1. ⛔ **Too Short Holding Period** (5m/15m) → 10bps transaction costs eat all profit
2. ⛔ **Single-Factor Indicators** (RSI, MACD, Bollinger alone) → no structural edge
3. ⛔ **No Regime Awareness** (mean reversion in trends, trend-following in ranges)
4. ⛔ **Crowded Trades** (everyone knows RSI 30/70, no alpha)
5. ⛔ **No Confirmation** (entry on indicator alone, no price/volume/volatility filter)

---

## ✅ The 29 New Strategies — What's Different?

### Category 1: Crypto-Specific Structural Edge
*These exploit market microstructure that doesn't exist in traditional markets*

| Strategy | Edge | Why It's Different |
|----------|------|-------------------|
| **Funding Rate Carry** (Carver) | Harvest funding payments from crowded trades | ✅ **Crypto-only** (no equivalent in stocks/futures), ✅ **Structural** (market microstructure, not TA), ✅ **Measurable edge** (funding rate = direct P&L) |
| **Spot-Perp Spread MR** (OU Process) | Mean reversion on basis (spot vs perpetual) | ✅ **Arbitrage-like** (cointegration), ✅ **Half-life tells you speed** (adaptive, not blind), ✅ **Not a price pattern** |

**Key Difference**: These aren't technical patterns—they're **market structure inefficiencies**. Even if everyone knows them, the edge persists (like how funding arbitrage still works despite being public knowledge).

---

### Category 2: Multi-Factor Conditional Entry
*Not just "RSI < 30"—requires 3-4 conditions to align*

| Strategy | Factors | Failed Equivalent | Key Improvement |
|----------|---------|-------------------|-----------------|
| **NR4/NR7 Trend Continuation** (Crabel) | 1. NR4/7 (vol compression)<br>2. Trend (50 EMA)<br>3. Close bias (top/bottom 1/3)<br>4. Volume expansion | Range Breakout (single factor) | ✅ **4 factors** vs 1, ✅ **Conditional** (only in trend), ✅ **Confirmation** (close bias + volume) |
| **Stretch Reversal** (Crabel) | 1. 2.5 SD extreme<br>2. Reversal bar (close in top/bottom 50%)<br>3. Volume confirmation | Bollinger MR (just 2 SD touch) | ✅ **Reversal bar confirmation** (not blind entry), ✅ **2-step filter** (statistical + behavioral) |
| **Failure Test** (Grimes) | 1. Breakout attempt<br>2. Failure (close back in range)<br>3. Volume on failure<br>4. Not in strong trend | Range Breakout (opposite logic) | ✅ **Fading failures** (anti-pattern), ✅ **Range-aware** (skip in trends), ✅ **Volume confirmation** |
| **Trend Pullback Entry** (Grimes) | 1. Trend (50 EMA)<br>2. Pullback (20 EMA)<br>3. Rejection bar<br>4. ADX > 20 | EMA Ribbon (just EMA cross) | ✅ **Context + Trigger** (trend + pullback), ✅ **Better R/R** (buy low in uptrend, not chase) |

**Key Difference**: **3-4 confirmations** reduce false signals exponentially. Failed strategies had 1 condition (RSI < 30). New strategies have 3-4 (NR4 + trend + close bias + volume).

---

### Category 3: Adaptive/Dynamic Parameters
*Not fixed thresholds (RSI 30/70)—parameters adapt to market conditions*

| Strategy | Adaptive Element | Failed Equivalent | Key Improvement |
|----------|------------------|-------------------|-----------------|
| **KAMA** (Kaufman) | Efficiency Ratio adjusts smoothing (fast in trends, slow in chop) | EMA Ribbon (fixed periods) | ✅ **Regime-aware** (adapts to market state), ✅ **Low lag in trends** (fast), ✅ **Noise filter in chop** (slow) |
| **VIDYA** (Chande) | CMO adjusts smoothing (high momentum = fast, low momentum = slow) | EMA (fixed smoothing) | ✅ **Momentum-adaptive**, ✅ **Faster than KAMA** (uses CMO not ER) |
| **Kalman Filter Pairs** (Chan) | Hedge ratio β changes with market (not static OLS) | Static pairs trading (fixed β) | ✅ **Dynamic β** (adapts to regime shifts), ✅ **Responsive** (Kalman tracks changes) |
| **OU Process Half-Life** (Chan) | Half-life tells you HOW FAST MR occurs (adaptive holding period) | Bollinger MR (blind entry, fixed exit) | ✅ **Adaptive holding** (exit after half-life * 3), ✅ **Speed detection** (skip if too slow) |
| **EWMAC** (Carver) | Multiple EWMA pairs (8/32, 16/64, 32/128, 64/256) capture trends at different speeds | MACD (fixed 12/26) | ✅ **Multi-speed** (4-6 EWMA pairs), ✅ **Volatility-scaled** (position sizing adapts) |

**Key Difference**: **Parameters change with market conditions**. Failed strategies used fixed thresholds (RSI 30/70 forever). New strategies adapt (KAMA fast in trends, slow in chop).

---

### Category 4: Right Holding Period (4h-2w Sweet Spot)
*Not 5m/15m (transaction costs) or >1 month (regime risk)*

| Strategy | Holding Period | Why This Works |
|----------|----------------|----------------|
| **NR4/NR7** | 2-7 days | ✅ **4h-2w window** (per research), ✅ **Transaction costs manageable** (10bps on 2% move = 0.5% of profit, not 5%) |
| **OU Process** | 2-20 days (half-life) | ✅ **Adaptive** (fast when MR is fast, slow when MR is slow), ✅ **4h-2w window** |
| **Funding Carry** | Days to weeks | ✅ **Funding resets 8h** (3x daily income), ✅ **Not directional** (low turnover) |
| **Dual Momentum** | Weeks to months | ✅ **Monthly rebalance** (very low transaction costs), ✅ **Trend capture** (not scalping) |
| **TSMOM** | 1 month (rebalance monthly) | ✅ **Academic proof** (12-month lookback works), ✅ **Low turnover** |

**Key Difference**: **Holding period matches transaction cost reality**. 5m/15m strategies need 50-100 trades to profit $1, but pay $0.50 in fees. 4h-2w strategies need 5-10 trades, pay $0.05 in fees.

---

### Category 5: Regime-Aware (Not Blind Application)
*Mean reversion ONLY in mean-reverting regimes, trend-following ONLY in trends*

| Strategy | Regime Filter | Failed Equivalent | Key Improvement |
|----------|---------------|-------------------|-----------------|
| **NR4/NR7** | Only trade in direction of 50 EMA trend | Range Breakout (no trend filter) | ✅ **Continuation** (with trend), not reversal (against trend) |
| **Williams %R** | Only buy pullbacks in uptrend (>50 MA), only sell rallies in downtrend (<50 MA) | RSI Extreme (blind entries) | ✅ **With-trend MR**, not counter-trend |
| **OU Process** | Only trade when half-life <20 days (fast MR); skip when half-life >20 (slow/no MR) | Bollinger MR (always trades) | ✅ **Speed filter** (only trade fast MR), ✅ **Adaptive** (recalculates half-life) |
| **Bollinger Momentum BO** (Radge) | Only buy BB breakouts ABOVE 100-day MA (uptrend) | Bollinger MR (opposite logic, no trend filter) | ✅ **WITH trend**, not against, ✅ **Breakout** (not MR) |
| **Failure Test** (Grimes) | Only fade breakouts in RANGES; skip if strong trend | Range Breakout (no range detection) | ✅ **Range-aware** (fading works in ranges, not trends) |

**Key Difference**: **Trade the right strategy in the right regime**. Failed strategies applied mean reversion in trends (loss) or trend-following in ranges (whipsaw). New strategies filter for regime FIRST, then trade.

---

## 📊 Comparison Table: Failed vs New (Representative Examples)

| Dimension | Failed: RSI Extreme | New: NR4/NR7 Trend Continuation | Why NR4 Wins |
|-----------|---------------------|--------------------------------|--------------|
| **Timeframe** | 5m/15m | 4h/daily | 10bps fee = 0.5% of 2% move (not 5% of 0.4% move) |
| **Entry conditions** | 1 (RSI < 30) | 4 (NR4 + trend + close bias + volume) | 4 factors >> 1 factor (false signals drop exponentially) |
| **Regime filter** | None (blind entry) | 50 EMA trend (only trade with trend) | Avoids counter-trend whipsaw |
| **Confirmation** | None | Volume expansion + close in top/bottom 1/3 | Confirms commitment, not just indicator noise |
| **Exit logic** | Fixed (RSI 50) | Adaptive (3x NR range or trend break) | Risk/reward based on entry setup, not arbitrary |
| **Edge source** | Overbought/oversold (crowded) | Volatility compression → expansion (structural) | Less crowded, more mechanical |

| Dimension | Failed: Bollinger MR | New: OU Process MR (Half-Life) | Why OU Wins |
|-----------|----------------------|-------------------------------|--------------|
| **Entry signal** | Price touches ±2 SD band | Z-score <-1.5 AND half-life <20 days | Half-life = speed detection (skip if too slow) |
| **Regime awareness** | None (trades in trends too) | Half-life filter (only fast MR) | Avoids slow MR (trends) |
| **Holding period** | Fixed (exit at MA) | Adaptive (3x half-life) | Matches natural reversion time |
| **Exit logic** | MA cross (fixed) | Z-score cross zero OR half-life >20 (adaptive) | Adapts to regime shift |
| **Math foundation** | Heuristic (2 SD = overbought) | Stochastic process (AR(1), half-life) | Grounded in math, not just TA |

| Dimension | Failed: MACD Scalp | New: EWMAC (Carver) | Why EWMAC Wins |
|-----------|-------------------|---------------------|----------------|
| **Timeframe** | 5m/15m | Daily (can adapt to 4h) | Transaction costs manageable |
| **EWMA pairs** | 1 (12/26) | 4-6 (8/32, 16/64, 32/128, 64/256) | Captures trends at multiple speeds |
| **Position sizing** | Fixed (e.g., 1 contract) | Volatility-scaled (target 10% vol) | Risk-adjusted (same risk across market states) |
| **Forecast combination** | Single MACD | Average of 4-6 EWMA pairs | Smooths noise, diversifies across speeds |
| **Exit logic** | MACD cross zero | Forecast < 2 (weak signal) | Adaptive exit, not fixed cross |

---

## 🔍 Key Takeaways

### Why Failed Strategies Failed
1. ⛔ **Transaction costs** (10bps) destroy edges on 5m/15m (need 50-100 trades for $1 profit, pay $0.50 in fees)
2. ⛔ **Single-factor indicators** (RSI, MACD alone) = no structural edge, just noise
3. ⛔ **No regime awareness** (mean reversion in trends = losses, trend-following in ranges = whipsaw)
4. ⛔ **Fixed thresholds** (RSI 30/70 forever) = not adaptive to changing markets
5. ⛔ **Crowded trades** (everyone knows RSI 30 = buy) = no alpha left

### Why New Strategies Might Work
1. ✅ **Right holding period** (4h-2w) = transaction costs manageable (10bps = 0.5% of profit, not 5%)
2. ✅ **Multi-factor conditional** (3-4 confirmations) = exponentially fewer false signals
3. ✅ **Regime-aware** (MR in MR regimes, trend in trend regimes) = avoids worst losses
4. ✅ **Adaptive parameters** (KAMA, Kalman, half-life) = changes with market
5. ✅ **Structural edge** (funding rate, cointegration, volatility cycles) = not just TA patterns

### Probability of Success (Subjective Estimates)

| Strategy | Success Probability | Rationale |
|----------|---------------------|-----------|
| **Funding Rate Carry** | 70% | Crypto-specific, structural, data ready, proven concept |
| **NR4/NR7 Trend Continuation** | 60% | Multi-factor, proven pattern, simple rules |
| **Kalman Filter Pairs** | 50% | Math-grounded, but transaction costs critical |
| **EWMAC Trend Following** | 55% | Proven across asset classes, volatility-scaled |
| **OU Process MR** | 50% | Half-life is powerful, but MR is hard in crypto trends |
| **Dual Momentum** | 65% | Academic backing, low transaction costs |
| **TSMOM** | 60% | Robust across markets, simple implementation |

**Compare to**: Failed strategies had 0% success (0/1,449 tests passed).

---

## 🎯 Action Plan

1. **Start with Funding Rate Carry** (70% success odds, data ready)
2. **Validate rigorously** (walk-forward, OOS Sharpe > 0.5, Bonferroni correction)
3. **Move to NR4/NR7 and Dual Momentum** (60-65% odds, well-documented)
4. **Test EWMAC and TSMOM** (55-60% odds, robust across markets)
5. **Save advanced strategies** (Kalman, OU, Lopez de Prado) for later

**Goal**: Find 3-5 strategies with OOS Sharpe > 0.5 and max DD < 30%. Build portfolio with correlation < 0.3 between strategies.

---

**Bottom Line**: These 29 strategies are NOT just "more indicators." They're **structurally different** (multi-factor, regime-aware, adaptive, right holding period). If they fail, we'll learn WHY (and that's valuable). But odds of finding 3-5 winners are HIGH (vs 0% with previous approach).
