# Strategy Implementation Priorities
*Analysis Date: 2026-03-09*
*Based on: book_strategies.md (29 strategies extracted)*

## Executive Summary

**Target**: Find strategies that work in the 4h-2w holding period sweet spot for crypto perpetuals/spot.

**Problem**: 12 standard scalping strategies (5m/15m) ALL FAILED (0/1,449 Bonferroni survivors). Transaction costs (10bps) destroy short-term strategies.

**Solution**: Focus on multi-factor, regime-aware strategies with structural edge, not simple indicator crosses.

---

## Tier 1: HIGHEST Priority (Implement First)
*Crypto Applicability: HIGH | Novel Mechanics | Testable Rules*

### 1. **Carry Strategy (Funding Rate Harvest)** — Robert Carver
- **Why**: Crypto-specific structural edge. Funding rates are explicit carry.
- **Edge**: Market microstructure inefficiency (sentiment imbalance).
- **Holding**: Days to weeks (perfect for 4h-2w window).
- **Implementation**: Use Coinalyze data (already have 2 years of funding rates).
- **Risk**: Low complexity, clear rules, direct P&L from funding.
- **Status**: 🔥 **IMPLEMENT IMMEDIATELY** — data ready, clear edge.

### 2. **NR4/NR7 Trend Continuation** — Toby Crabel
- **Why**: Volatility compression + trend filter + bar bias (3 factors).
- **Edge**: NOT a naked breakout—conditional on existing trend.
- **Holding**: 2-7 days (fits 4h-2w window).
- **Implementation**: Simple rules, works on 4h/daily.
- **Risk**: Medium complexity, requires trend filter.
- **Status**: ⭐ **HIGH PRIORITY** — proven pattern, multi-factor.

### 3. **Kalman Filter Pairs Trading** — Ernie Chan
- **Why**: Adaptive hedge ratio (dynamic, not static OLS).
- **Edge**: Mean reversion on cointegrated pairs (BTC-ETH, L1 competitors).
- **Holding**: Hours to days (adaptive to half-life).
- **Implementation**: Requires Kalman filter (filterpy library).
- **Risk**: Medium-high complexity, transaction costs critical.
- **Status**: ⭐ **HIGH PRIORITY** — superior to static pairs trading.

### 4. **EWMAC (Carver Trend Following)** — Robert Carver
- **Why**: Volatility-adjusted position sizing + multi-timeframe.
- **Edge**: Captures trends at multiple speeds, risk-managed.
- **Holding**: Weeks to months (can adapt to shorter with faster EMAs).
- **Implementation**: Test 4-6 EWMA pairs, combine forecasts.
- **Risk**: Low complexity, proven across asset classes.
- **Status**: ⭐ **HIGH PRIORITY** — robust, scalable to portfolio.

### 5. **OU Process Mean Reversion (Half-Life)** — Ernie Chan
- **Why**: Grounded in stochastic processes, not just TA patterns.
- **Edge**: Half-life tells you HOW FAST mean reversion occurs (adaptive).
- **Holding**: 2-20 days (depends on half-life calculation).
- **Implementation**: Apply to spot-perp spreads, stablecoin pairs.
- **Risk**: Medium complexity, requires half-life estimation.
- **Status**: ⭐ **HIGH PRIORITY** — systematic MR, not blind BB squeeze.

---

## Tier 2: HIGH Priority (Implement After Tier 1)
*Crypto Applicability: HIGH | Proven Concepts | Need Testing*

### 6. **Dual Momentum (Absolute + Relative)** — Gary Antonacci
- **Why**: Combines trend + relative strength (2 filters reduce whipsaw).
- **Edge**: Monthly rebalancing reduces transaction costs.
- **Holding**: Weeks to months.
- **Implementation**: Apply to top 10-20 coins, monthly rebalance.
- **Status**: 🟢 **Test after Tier 1** — proven in equities/futures.

### 7. **Time-Series Momentum (TSMOM)** — Satchell
- **Why**: One of most robust strategies across asset classes.
- **Edge**: Volatility-scaled, trend-following.
- **Holding**: 1 month (rebalance monthly).
- **Implementation**: Apply to portfolio of 5-10 coins.
- **Status**: 🟢 **Test after Tier 1** — academic backing, diversifiable.

### 8. **KAMA Trend Following** — Perry Kaufman
- **Why**: Adapts to market efficiency (fast in trends, slow in chop).
- **Edge**: Smarter than fixed MAs, regime-aware.
- **Holding**: Days to weeks.
- **Implementation**: 4h or daily bars, test multiple ER periods.
- **Status**: 🟢 **Test after Tier 1** — adaptive, low complexity.

### 9. **CMO Divergence Trading** — Tushar Chande
- **Why**: Divergences signal momentum exhaustion (leading indicator).
- **Edge**: Multi-bar confirmation reduces false signals.
- **Holding**: 2-7 days.
- **Implementation**: Code CMO from scratch, test on major coins.
- **Status**: 🟢 **Test after Tier 1** — novel indicator, not in standard lib.

### 10. **Aroon Oscillator Trend Entry** — Tushar Chande
- **Why**: Identifies NEW trends (time since high/low), not just continuation.
- **Edge**: Catches trend initiation earlier than lagging MAs.
- **Holding**: 1-2 weeks.
- **Implementation**: 4h or daily, combine with volume breakout.
- **Status**: 🟢 **Test after Tier 1** — structural, not just price patterns.

---

## Tier 3: MEDIUM Priority (Backtest If Time Permits)
*Crypto Applicability: MEDIUM | Needs Adaptation*

### 11. **Volatility Breakout (Smash Day)** — Larry Williams
- **Why**: NR4/NR7 + breakout + volume (multi-factor).
- **Edge**: Volatility compression/expansion cycle.
- **Note**: Similar to NR4/NR7 (Tier 1) but less trend-aware.
- **Status**: 🟡 **Backtest vs NR4/NR7** — may be redundant.

### 12. **Donchian Breakout (Carver)** — Robert Carver
- **Why**: Multiple breakout periods (20, 40, 80) reduce whipsaw.
- **Edge**: Volatility-adjusted sizing.
- **Note**: Works in trends; crypto consolidations may whipsaw.
- **Status**: 🟡 **Backtest carefully** — needs trend regime.

### 13. **ATR Channel Breakout** — Perry Kaufman
- **Why**: Volatility-adjusted channels (more robust than Bollinger Bands).
- **Edge**: Breakout + volume + ADX (multi-factor).
- **Note**: Daily or 4h bars.
- **Status**: 🟡 **Backtest with ATR multiplier sweep** — may overlap with BB strategies.

### 14. **Dual Thrust Volatility Breakout** — Ernie Chan
- **Why**: Adaptive breakout levels based on recent volatility.
- **Edge**: Asymmetric k1/k2 allows long/short bias tuning.
- **Note**: Needs session-based adaptation (crypto trades 24/7).
- **Status**: 🟡 **Backtest with rolling windows** — define 'open' carefully.

### 15. **Stretch Reversal (Crabel)** — Toby Crabel
- **Why**: 2.5 SD + reversal bar confirmation (2-step filter).
- **Edge**: Different from simple BB MR (requires reversal bar).
- **Holding**: 1-5 days.
- **Status**: 🟡 **Backtest vs Bollinger MR** — see if confirmation helps.

### 16. **Bollinger Band Momentum Breakout** — Nick Radge
- **Why**: BB breakout (not MR) + 100-day MA trend filter.
- **Edge**: Trades WITH trend, not against.
- **Note**: Different from BB MR we tested.
- **Status**: 🟡 **Backtest** — may work if trend filter strong.

### 17. **Trend Pullback Entry (Grimes)** — Adam Grimes
- **Why**: Pullback to 20 EMA in 50 EMA trend (context + trigger).
- **Edge**: Better risk/reward than chasing breakouts.
- **Holding**: 5-15 days.
- **Status**: 🟡 **Backtest** — classic pullback, test EMA periods.

### 18. **Failure Test (Grimes)** — Adam Grimes
- **Why**: Fading failed breakouts (anti-pattern).
- **Edge**: Crypto has many false breakouts (stop hunts).
- **Note**: Works in ranges, not trends.
- **Status**: 🟡 **Backtest in ranging markets** — regime-dependent.

### 19. **Cross-Sectional Momentum (XSMOM)** — Satchell
- **Why**: Market-neutral long-short (pure alpha).
- **Edge**: Crypto has wide return dispersion.
- **Note**: Requires shorting (perpetuals), funding costs.
- **Status**: 🟡 **Backtest with funding costs** — transaction cost sensitive.

### 20. **Accelerating Dual Momentum** — Gary Antonacci
- **Why**: Weights recent momentum higher (1m, 3m, 6m, 12m).
- **Edge**: More responsive than pure 12-month momentum.
- **Note**: Monthly rebalancing.
- **Status**: 🟡 **Backtest vs standard Dual Momentum** — may improve.

---

## Tier 4: LOW Priority (Deprioritize or Skip)
*Crypto Applicability: LOW | Not Suitable for 24/7 Markets*

### 21. **OOPS (Opening Gap Fade)** — Larry Williams
- **Reason**: Crypto trades 24/7 (no traditional 'open').
- **Adapt**: Could use 00:00 UTC or session opens, but edge unclear.
- **Status**: ❌ **SKIP** — better opportunities exist.

### 22. **Opening Range Breakout (ORB)** — Toby Crabel
- **Reason**: Needs clear session open; crypto is continuous.
- **Adapt**: Could use session opens (Asia/Europe/US) but complex.
- **Status**: ❌ **SKIP for now** — focus on 24/7-compatible strategies.

### 23. **Weekend Gap Fade** — Nick Radge
- **Reason**: Crypto trades weekends (no gaps).
- **Adapt**: Could adapt to holiday periods, but rare.
- **Status**: ❌ **SKIP** — not applicable.

### 24. **Williams %R Reversal** — Larry Williams
- **Reason**: %R is just inverted Stochastic; similar to strategies we tested.
- **Note**: Requires trend filter (50-day MA).
- **Status**: ⚠️ **LOW PRIORITY** — likely similar to StochRSI we tested.

### 25. **Swing Index Trading** — Perry Kaufman/Wilder
- **Reason**: Complex calculation; may be noisy in 24/7 crypto.
- **Note**: Daily bars only (needs O/H/L/C).
- **Status**: ⚠️ **LOW PRIORITY** — high complexity, uncertain edge.

### 26. **VIDYA Adaptive Trend** — Tushar Chande
- **Reason**: Similar to KAMA but uses CMO instead of ER.
- **Note**: May be redundant with KAMA (Tier 2).
- **Status**: ⚠️ **Backtest AFTER KAMA** — see if CMO improves vs ER.

---

## Tier 5: ADVANCED (Long-Term Projects)
*Require ML Infrastructure or Complex Math*

### 27. **Triple Barrier Labeling + ML** — Lopez de Prado
- **Reason**: Requires ML infrastructure (RF, feature engineering).
- **Edge**: Combines prediction with risk management (adaptive exits).
- **Status**: 🔬 **RESEARCH PROJECT** — implement after simpler strategies validated.

### 28. **Fractional Differentiation** — Lopez de Prado
- **Reason**: Makes non-stationary data stationary for ML.
- **Edge**: Elegant solution to stationarity vs memory trade-off.
- **Status**: 🔬 **RESEARCH PROJECT** — requires statistical/ML expertise.

### 29. **Portfolio of Systems** — Urban Jaekle
- **Reason**: Meta-strategy (combines multiple systems).
- **Edge**: Diversification across strategies reduces risk.
- **Status**: 🔬 **IMPLEMENT AFTER TIER 1-2** — need individual systems first.

---

## Implementation Roadmap

### Phase 1: Immediate (Next 1-2 Weeks)
1. ✅ **Funding Rate Carry** — data ready, clear edge
2. ⭐ **NR4/NR7 Trend Continuation** — simple rules, multi-factor
3. ⭐ **OU Process Mean Reversion** — systematic MR, half-life based

### Phase 2: Short-Term (2-4 Weeks)
4. ⭐ **EWMAC Trend Following** — robust, multi-timeframe
5. ⭐ **Kalman Filter Pairs** — adaptive hedge ratio
6. 🟢 **Dual Momentum** — monthly rebalance, low transaction costs
7. 🟢 **TSMOM** — academic backing, volatility-scaled

### Phase 3: Medium-Term (1-2 Months)
8. 🟢 **KAMA Trend Following** — adaptive, regime-aware
9. 🟢 **CMO Divergence** — novel indicator, leading signal
10. 🟢 **Aroon Oscillator** — trend initiation detection
11. 🟡 **Selected Tier 3** — backtest best candidates

### Phase 4: Long-Term (3+ Months)
- 🔬 **Lopez de Prado ML** — after infrastructure built
- 🔬 **Portfolio of Systems** — after individual systems validated

---

## Key Insights

### What Makes These Different from Failed Strategies?

1. **Multi-Factor, Not Single Indicator**:
   - Failed: "Buy when RSI < 30" (single condition)
   - New: "Buy when NR4 + trend + close in top 1/3 + volume spike" (4 conditions)

2. **Regime-Aware**:
   - Failed: Blind mean reversion in all markets
   - New: Mean reversion ONLY when half-life <20 days (OU Process)

3. **Structural Edge**:
   - Failed: Technical patterns (price only)
   - New: Funding rate carry (market microstructure), Kalman pairs (cointegration)

4. **Right Holding Period**:
   - Failed: 5m/15m (transaction costs dominate)
   - New: 4h-2w (sweet spot per research)

5. **Adaptive/Dynamic**:
   - Failed: Fixed thresholds (RSI 30/70, BB ±2SD)
   - New: KAMA (adapts smoothing), Kalman (adapts hedge ratio)

### Avoid These Mistakes

❌ **Don't**: Test all 29 strategies blindly (overfitting risk)
✅ **Do**: Focus on Tier 1 (5 strategies), validate rigorously, THEN expand

❌ **Don't**: Use in-sample optimization without walk-forward
✅ **Do**: Split data (IS/OOS), optimize on IS, validate on OOS

❌ **Don't**: Ignore transaction costs (10bps destroys scalping)
✅ **Do**: Backtest with realistic fees (10bps) and slippage

❌ **Don't**: Assume strategies work forever (regime dependence)
✅ **Do**: Monitor rolling Sharpe, stop trading during regime shifts

---

## Next Steps

1. **Review this document** — confirm prioritization makes sense
2. **Select Tier 1 strategy** — start with Funding Rate Carry (data ready)
3. **Code strategy in Maestro** — implement entry/exit/filters
4. **Backtest with walk-forward** — IS/OOS split, Optuna optimization
5. **Compare to baseline** — must beat buy & hold + failed 12 strategies
6. **Validate statistics** — Sharpe, max DD, win rate, Bonferroni correction
7. **Paper trade** — if OOS validates, test in paper trading for 2-4 weeks
8. **Deploy to Virtuoso** — if paper trade confirms, small allocation in production

**Timeline**: Aim for 1-2 Tier 1 strategies validated per week. After 5 weeks, we'll have 5 validated strategies (or have ruled out non-performers). Then move to Tier 2.

---

**Remember**: Quality > Quantity. Better to have 3 robust strategies than 29 half-tested ones.
