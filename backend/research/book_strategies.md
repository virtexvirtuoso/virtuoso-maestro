# Trading Strategies from Classic Literature
*Extracted: 2026-03-09*
*Target: Crypto perpetuals/spot, 4h-2w holding periods*

## Overview
This document contains concrete, backtestable trading strategies extracted from classic trading literature.
Focus: Multi-factor, regime-aware, structural strategies (not simple indicator crosses).

---

# Trading Strategies from Classic Literature
*Extracted: 2026-03-09*
*Target: Crypto perpetuals/spot, 4h-2w holding periods*

## Overview
This document contains concrete, backtestable trading strategies extracted from classic trading literature.
Focus: Multi-factor, regime-aware, structural strategies (not simple indicator crosses).

---

## Strategy 1: Volatility Breakout (Smash Day) (Source: Long-Term Secrets to Short-Term Trading, Larry Williams)

**Type:** breakout
**Timeframe:** Daily
**Holding Period:** 1-5 days
**Crypto Applicability:** HIGH — Crypto has high volatility and clear breakout patterns; 24/7 market reduces gap risk

### Entry Rules
1. Identify a narrow range day (NR4 or NR7: narrowest range in 4 or 7 days)
2. Buy on a breakout above the high of the narrow range day
3. Or sell short on a breakdown below the low of the narrow range day
4. Confirm with increased volume on breakout (>1.5x average)

### Exit Rules
1. Initial stop: opposite side of the narrow range bar
2. Profit target: 2-3x the narrow range bar's range
3. Trail stop to breakeven after 1x range profit
4. Exit after 5 days if target not hit

### Filters
1. Avoid during consolidation: require ADX > 20 for trending market
2. Minimum range contraction: NR day range should be <50% of 10-day average range

### Parameters
- lookback_nr: 4 or 7 days (optimize)
- volume_multiplier: 1.5x
- profit_target: 2-3x NR range
- max_holding: 5 days

### Why It Might Work Where Others Failed
Exploits volatility compression/expansion cycle. Narrow ranges build pressure; breakout releases it. NOT just a price pattern—structural market behavior.

### Implementation Notes
Crypto-adapt: Use 4-hour or daily bars. Volume may be less reliable (use OI changes or funding rate spikes as confirmation). Works best on high-cap pairs with clean trend structure.

---


## Strategy 2: Williams %R Reversal (Source: Long-Term Secrets to Short-Term Trading, Larry Williams)

**Type:** mean_reversion
**Timeframe:** 4h-Daily
**Holding Period:** 2-7 days
**Crypto Applicability:** MEDIUM — Needs trending context to avoid whipsaw; crypto trends can persist longer than traditional markets

### Entry Rules
1. Calculate Williams %R (14 periods): %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
2. **BUY SETUP**: %R crosses above -95 (extreme oversold) AND price is above 50-day MA (uptrend)
3. **SELL SETUP**: %R crosses below -5 (extreme overbought) AND price is below 50-day MA (downtrend)
4. Entry: next bar open after cross

### Exit Rules
1. Exit long when %R reaches -20 (mean reversion complete)
2. Exit short when %R reaches -80
3. Stop loss: 2x ATR(14) from entry
4. Time stop: 7 days

### Filters
1. Only trade in direction of larger trend (50-day MA)
2. Avoid when ATR < 30% of price (low volatility = poor risk/reward)
3. Skip if prior 3 bars are inside bars (consolidation)

### Parameters
- r_period: 14
- r_buy_threshold: -95
- r_sell_threshold: -5
- r_exit_long: -20
- r_exit_short: -80
- trend_ma: 50
- stop_atr_mult: 2.0

### Why It Might Work Where Others Failed
Combines mean reversion with trend filter. Most MR strategies fail because they fight the trend. This trades WITH the trend after pullbacks—buying weakness in uptrends, selling strength in downtrends.

### Implementation Notes
Crypto-adapt: Use 4h bars for medium-term swing, or daily for position. Watch for funding rate extremes as additional confirmation (extreme negative funding = overcrowded shorts, good for bounce).

---


## Strategy 3: OOPS (Opening Gap Fade) (Source: Long-Term Secrets to Short-Term Trading, Larry Williams)

**Type:** mean_reversion
**Timeframe:** Daily
**Holding Period:** 1-3 days
**Crypto Applicability:** LOW — Crypto trades 24/7 (no traditional 'open'); adapt using 00:00 UTC or session-based logic

### Entry Rules
1. Today's open gaps above/below yesterday's high/low
2. Price trades back through yesterday's high/low within first 2 hours
3. **SHORT**: Gap up above yesterday's high, then trade back below yesterday's high
4. **LONG**: Gap down below yesterday's low, then trade back above yesterday's low

### Exit Rules
1. Target: Prior day's close or midpoint
2. Stop: 1.5x gap size beyond entry
3. Exit by close if target not hit

### Filters
1. Gap must be >0.5% of price
2. Volume on gap day should be above average (confirms exhaustion)
3. Avoid earnings/major news days (crypto: avoid CPI, FOMC, major exchange listings)

### Parameters
- min_gap_pct: 0.5%
- stop_mult: 1.5x gap size
- entry_window: 2 hours from open

### Why It Might Work Where Others Failed
Gaps often represent emotional extremes. When price immediately reverses, it signals false breakout/exhaustion. Mean reversion to prior equilibrium.

### Implementation Notes
Crypto-adapt: Define 'open' as 00:00 UTC or use 4h bars with first candle = open. OR adapt to weekend gaps (Friday close to Monday open). Better: use funding rate resets as 'open' proxy.

---


## Strategy 4: Opening Range Breakout (ORB) (Source: Day Trading with Short Term Price Patterns, Toby Crabel)

**Type:** breakout
**Timeframe:** Intraday (5m-1h bars after OR established)
**Holding Period:** Few hours to 1 day
**Crypto Applicability:** MEDIUM — Adaptable to session opens (Asia/Europe/US) or rolling windows; needs volatility to work

### Entry Rules
1. Define Opening Range (OR): first 30-60 minutes of trading session (e.g., 00:00-01:00 UTC)
2. Calculate OR high and OR low
3. **BUY**: Price breaks above OR high by >0.2% with volume >1.3x OR average
4. **SELL**: Price breaks below OR low by >0.2% with volume >1.3x OR average
5. Enter immediately on breakout confirmation

### Exit Rules
1. Profit target: OR range * 2 (if OR = 1%, target = 2% move)
2. Stop loss: Opposite side of OR (e.g., long stop = OR low)
3. If not hit by end of session, exit at session close (or 12h later)
4. Move stop to breakeven after 1x OR range captured

### Filters
1. Only trade if OR range is 0.3-1.5% of price (too narrow = noise, too wide = no compression)
2. Avoid if prior day's range <50% of 10-day average (low volatility regime)
3. Require price to be within 5% of 20-period high/low (trending, not ranging market)

### Parameters
- or_period_min: 30-60
- breakout_threshold: 0.2%
- volume_mult: 1.3x
- profit_mult: 2x OR range
- or_min_range: 0.3%
- or_max_range: 1.5%

### Why It Might Work Where Others Failed
Opening ranges establish support/resistance for the session. Breakout indicates directional conviction. Multi-factor: range compression + breakout + volume = not just price action.

### Implementation Notes
Crypto-adapt: Use 00:00-01:00 UTC as 'open', or track Asia/Europe/US session opens separately. Backtest multiple OR window sizes (30/60/90 min). Consider OR on 4h timeframe for positional edge.

---


## Strategy 5: NR4/NR7 Trend Continuation (Source: Day Trading with Short Term Price Patterns, Toby Crabel)

**Type:** breakout/momentum
**Timeframe:** 4h-Daily
**Holding Period:** 2-7 days
**Crypto Applicability:** HIGH — Volatility compression is a universal pattern; crypto's high vol makes expansions dramatic

### Entry Rules
1. Identify NR4 or NR7 day: narrowest range in 4 or 7 bars
2. **Trend Context**: Price must be above/below 50-period EMA (directional bias)
3. **Entry Long**: NR bar occurs in uptrend + close in top 1/3 of NR bar → buy break of NR high
4. **Entry Short**: NR bar occurs in downtrend + close in bottom 1/3 of NR bar → sell break of NR low
5. Confirm with volume expansion (>1.5x avg) on breakout bar

### Exit Rules
1. Stop: Opposite extreme of NR bar
2. Target: 3x NR range or recent swing high/low
3. Trail stop: move to entry +1 ATR after 2x NR range profit
4. Time exit: 7 bars if no momentum

### Filters
1. NR range must be <50% of 10-bar average range (true compression)
2. ADX > 20 (trending market, not consolidation)
3. Avoid if NR bar close is in middle third (indecision)

### Parameters
- nr_lookback: 4 or 7
- trend_ema: 50
- close_threshold: top/bottom 1/3 of range
- volume_mult: 1.5x
- profit_target: 3x NR range
- max_holding: 7 bars

### Why It Might Work Where Others Failed
Combines volatility compression (NR) with trend filter (EMA) and bar bias (close position). This is NOT a naked breakout—it's conditional on existing trend, making it a CONTINUATION pattern, not reversal.

### Implementation Notes
Crypto-adapt: Works on 4h, daily, even 1h for active trading. Backtest both NR4 and NR7 separately. Consider adding funding rate filter (extreme funding = overextension, skip entry).

---


## Strategy 6: Stretch Reversal (Crabel Stretch) (Source: Day Trading with Short Term Price Patterns, Toby Crabel)

**Type:** mean_reversion
**Timeframe:** Daily
**Holding Period:** 1-5 days
**Crypto Applicability:** HIGH — Crypto often overextends on momentum; mean reversion at medium timeframes can be profitable

### Entry Rules
1. **Stretch defined**: Price closes >2.5 standard deviations from 20-day MA
2. **Buy setup**: Downside stretch (close < MA - 2.5*SD) + bullish reversal bar (close > open, in top 50% of range)
3. **Sell setup**: Upside stretch (close > MA + 2.5*SD) + bearish reversal bar (close < open, in bottom 50% of range)
4. Entry: next bar open after reversal bar confirmation

### Exit Rules
1. Target: 20-day MA (return to mean)
2. Stop: Beyond recent extreme by 1.5 ATR
3. Partial exit (50%) at 50% retracement to MA
4. Time stop: 5 days

### Filters
1. No trend filter—this is a pure mean reversion play
2. Avoid if reversal bar's range is <0.5 ATR (weak signal)
3. Skip if RSI is between 40-60 (not extreme enough)

### Parameters
- ma_period: 20
- stretch_threshold: 2.5 SD
- reversal_bar_close: top/bottom 50% of range
- stop_atr_mult: 1.5
- max_holding: 5 days

### Why It Might Work Where Others Failed
Statistical extremes tend to revert. Unlike simple Bollinger Band mean reversion (which we proved fails), this requires REVERSAL BAR confirmation, reducing false entries. It's a 2-step filter: statistical + behavioral.

### Implementation Notes
Crypto-adapt: Daily timeframe or 4h. Backtest with rolling Sharpe to identify regime shift (stop trading in persistent trending regime). Add funding rate extreme as confirmation (e.g., funding >0.1% suggests exhaustion).

---


## Strategy 7: OU Process Mean Reversion (Half-Life Based) (Source: Algorithmic Trading, Ernie Chan)

**Type:** mean_reversion
**Timeframe:** 4h-Daily
**Holding Period:** Half-life period (typically 2-20 days)
**Crypto Applicability:** HIGH — Crypto spot-perp spreads, CEX-DEX arbitrage, and stablecoin depegs exhibit mean-reverting behavior

### Entry Rules
1. Calculate z-score: z = (price - MA) / SD (rolling 20-period)
2. Estimate half-life: fit AR(1) model log(price_t) = λ*log(price_{t-1}) + ε, half-life = -log(2)/log(λ)
3. **Entry threshold**: |z| > 1.5 AND half-life < 20 periods (fast mean reversion)
4. **Long**: z < -1.5 (oversold)
5. **Short**: z > 1.5 (overbought)
6. Position size proportional to |z| (larger deviation = larger position)

### Exit Rules
1. Exit when z crosses zero (mean reversion complete)
2. Stop loss: z exceeds ±3.0 (trend forming, not mean-reverting)
3. Time stop: 3x half-life (if mean reversion hasn't occurred, regime changed)

### Filters
1. Half-life must be <20 periods (if >20, too slow to be tradable)
2. Half-life must be >2 periods (if <2, too noisy)
3. Skip if rolling Hurst exponent >0.6 (trending, not mean-reverting)

### Parameters
- z_threshold: 1.5
- z_stop: 3.0
- ma_period: 20
- max_half_life: 20
- min_half_life: 2
- hurst_threshold: 0.6

### Why It Might Work Where Others Failed
Grounded in stochastic processes, not just technical patterns. Half-life tells you HOW FAST mean reversion occurs—no need to guess holding period. Adaptive to changing market regimes (half-life recalculates).

### Implementation Notes
Crypto-adapt: Apply to spot-perp spreads (BTC spot vs BTC-PERP), CEX-DEX spreads, or stablecoin pairs. Backtest on 4h/daily. Use Kalman filter to estimate MA dynamically (more responsive). Can also trade BTC-ETH spread (cross-asset MR).

---


## Strategy 8: Dual Thrust Volatility Breakout (Source: Algorithmic Trading, Ernie Chan)

**Type:** breakout
**Timeframe:** Daily (calculate ranges), trade on 1h-4h
**Holding Period:** Intraday to 1 day
**Crypto Applicability:** MEDIUM — Requires volatility to work; crypto's 24/7 trading needs session-based adaptation

### Entry Rules
1. Calculate daily range components: HH = highest high of past N days, LL = lowest low of past N days, HC = highest close, LC = lowest close
2. Range = max(HH - LC, HC - LL)
3. Upper breakout threshold: Open + k1 * Range
4. Lower breakout threshold: Open - k2 * Range
5. **Long**: Price breaks above upper threshold
6. **Short**: Price breaks below lower threshold

### Exit Rules
1. Exit at end of day (e.g., 23:59 UTC) or after 12h
2. Stop loss: opposite threshold (e.g., long stop = lower threshold)
3. Profit target: 1.5x Range or trailing stop at 1 ATR

### Filters
1. Only trade if today's ATR > 1.2x average ATR (volatility expansion)
2. Avoid if prior day was NR4 or NR7 AND no breakout yet (waiting for compression to resolve)
3. Skip if market is range-bound: require ADX > 15

### Parameters
- n_days: 4
- k1_upper: 0.7
- k2_lower: 0.7
- min_atr_mult: 1.2
- min_adx: 15

### Why It Might Work Where Others Failed
Adaptive breakout levels based on recent volatility, not static levels. Asymmetric k1/k2 allows tuning for long/short bias. Works in trending AND ranging markets (depending on calibration).

### Implementation Notes
Crypto-adapt: Define 'open' as 00:00 UTC or use rolling 4h windows. Backtest k1/k2 separately (crypto may have long bias). Consider funding rate as breakout confirmation (positive funding on upside breakout = crowded longs, may fade).

---


## Strategy 9: Kalman Filter Dynamic Hedge Ratio (Source: Algorithmic Trading, Ernie Chan)

**Type:** mean_reversion
**Timeframe:** 1h-4h
**Holding Period:** Hours to days (adaptive)
**Crypto Applicability:** HIGH — Many cointegrated crypto pairs (BTC-ETH, L1 competitors, stablecoins); Kalman filter adapts to regime changes

### Entry Rules
1. Select pair (e.g., BTC and ETH) and test for cointegration (ADF test, p < 0.05)
2. Use Kalman filter to estimate dynamic hedge ratio β_t (instead of static regression)
3. Calculate spread: S_t = price_A - β_t * price_B
4. Calculate z-score: z = (S_t - MA(S_t)) / SD(S_t)
5. **Entry long spread**: z < -1.5 (buy A, sell B in ratio β_t)
6. **Entry short spread**: z > 1.5 (sell A, buy B in ratio β_t)

### Exit Rules
1. Exit when z crosses zero (mean reversion complete)
2. Stop loss: z exceeds ±3.0 (cointegration broke down)
3. Time stop: 3 days (if mean reversion doesn't occur, regime change)

### Filters
1. Cointegration must hold: re-test ADF every 30 days, stop trading if p > 0.05
2. Half-life of spread must be <20 periods (fast mean reversion)
3. Skip if spread volatility is <0.5% (insufficient edge vs transaction costs)

### Parameters
- z_threshold: 1.5
- z_stop: 3.0
- ma_period: 20
- adf_retest_days: 30
- max_half_life: 20
- min_spread_vol: 0.5%

### Why It Might Work Where Others Failed
Kalman filter is adaptive—hedge ratio β changes with market conditions. Traditional pairs trading uses static β (OLS regression), which fails when regimes shift. This is DYNAMIC mean reversion, not static.

### Implementation Notes
Crypto-adapt: Trade BTC-ETH, SOL-AVAX, stablecoin pairs (USDT-USDC). Use 1h or 4h bars. Implement Kalman filter in Python (filterpy library). Transaction costs are CRITICAL—backtest with realistic fees (10bps).

---


## Strategy 10: EWMAC (Carver Trend Following) (Source: Systematic Trading, Robert Carver)

**Type:** trend
**Timeframe:** Daily (multi-timeframe variants)
**Holding Period:** Weeks to months (depends on EWMA periods)
**Crypto Applicability:** HIGH — Trend following works across timeframes; crypto has strong, persistent trends

### Entry Rules
1. Calculate two EWMAs: fast (e.g., 16-day) and slow (e.g., 64-day)
2. Calculate raw forecast: (EWMA_fast - EWMA_slow) / instrument_price_volatility
3. Normalize forecast to [-20, +20] scale (cap extremes)
4. Position = forecast * volatility_scalar / instrument_volatility
5. **Long**: forecast > 0
6. **Short**: forecast < 0

### Exit Rules
1. Exit when forecast crosses zero (trend reversal)
2. OR exit when forecast magnitude drops below 2 (weak signal)
3. No fixed stop loss—position sizing handles risk

### Filters
1. Minimum forecast magnitude: |forecast| > 2 (avoid noise)
2. Skip if instrument volatility > 2x recent average (regime break)
3. No trading during extreme volatility spikes (>3 SD moves)

### Parameters
- fast_ewma: 8, 16, 32, 64 (test multiple)
- slow_ewma: 32, 64, 128, 256
- forecast_cap: 20
- min_forecast: 2
- vol_scalar: adjust to target 10% vol

### Why It Might Work Where Others Failed
Volatility-adjusted position sizing prevents blow-ups. Multiple EWMA pairs capture trends at different speeds. Forecast scaling ensures consistent risk across instruments—not just price patterns, but risk-managed systematically.

### Implementation Notes
Crypto-adapt: Run on daily bars. Test 4-6 EWMA pairs (fast/slow combos) and combine forecasts (average or weighted). Apply to BTC, ETH, SOL, BNB. Backtest with portfolio approach (diversification across coins and EWMA speeds).

---


## Strategy 11: Carry Strategy (Funding Rate Harvest) (Source: Systematic Trading, Robert Carver)

**Type:** carry/arbitrage
**Timeframe:** Daily (funding resets 8h)
**Holding Period:** Days to weeks
**Crypto Applicability:** HIGH — Crypto perpetuals have explicit funding rates—direct carry measurement

### Entry Rules
1. Calculate 7-day average funding rate for each coin
2. Rank coins by funding rate (most positive to most negative)
3. **Long**: Coins with negative funding rate < -0.05% (getting paid to hold long)
4. **Short**: Coins with positive funding rate > 0.05% (getting paid to hold short)
5. Position size: proportional to |funding rate| * forecast_confidence

### Exit Rules
1. Exit when funding rate crosses zero (carry disappears)
2. Exit when funding rate reverses >50% (e.g., -0.1% → -0.05%)
3. Time stop: 14 days

### Filters
1. Minimum |funding rate| > 0.05% (insufficient edge below this)
2. Skip if coin's 7-day realized vol > 100% annualized (carry edge destroyed by volatility)
3. Avoid coins with <$50M open interest (liquidity risk)

### Parameters
- funding_avg_days: 7
- min_funding_threshold: 0.05%
- max_realized_vol: 100%
- min_open_interest: 50M
- max_holding: 14 days

### Why It Might Work Where Others Failed
Funding rates represent market sentiment imbalance. Persistent positive funding = crowded longs (short opportunity). Persistent negative funding = crowded shorts (long opportunity). This is NOT directional—it's a structural edge from market microstructure.

### Implementation Notes
Crypto-specific strategy. Collect funding rate data from Coinalyze or exchange APIs. Combine with trend filter (e.g., only short high-funding coins if they're in downtrend). Backtest with 10bps fees + funding paid/received.

---


## Strategy 12: Donchian Breakout (Carver Variant) (Source: Systematic Trading, Robert Carver)

**Type:** breakout
**Timeframe:** Daily
**Holding Period:** Weeks
**Crypto Applicability:** MEDIUM — Works in trending markets; crypto can have extended consolidations that whipsaw

### Entry Rules
1. Calculate Donchian Channel: N-day high and N-day low (e.g., N=20, 40, 80)
2. **Long breakout**: Close > N-day high
3. **Short breakout**: Close < N-day low
4. Calculate forecast: (close - channel_mid) / ATR, normalized to [-20, +20]
5. Position size: forecast * vol_target / instrument_vol

### Exit Rules
1. Exit long when close < M-day low (where M < N, e.g., M=10)
2. Exit short when close > M-day high
3. No fixed stop—position sizing handles risk

### Filters
1. Skip if ATR < 0.5% of price (low volatility = poor breakouts)
2. Avoid if price is within channel midpoint ±10% (ranging market)
3. Use multiple N values (20, 40, 80) and combine forecasts

### Parameters
- n_entry: 20, 40, 80 (test multiple)
- m_exit: 10, 20, 40 (typically N/2)
- forecast_cap: 20
- min_atr: 0.5%
- vol_target: 10-15% annualized

### Why It Might Work Where Others Failed
Captures trends across multiple timeframes. Volatility-adjusted sizing prevents overleveraging. Uses MULTIPLE breakout periods (not just one arbitrary N), which smooths out noise and reduces whipsaw.

### Implementation Notes
Crypto-adapt: Daily bars, test 3-4 N values. Combine with Carver's forecast combination (average of 20-day, 40-day, 80-day breakouts). Backtest with portfolio approach. Consider skipping during extreme funding rate regimes (carry dominates).

---


## Strategy 13: KAMA Trend Following (Source: Trading Systems and Methods, Perry Kaufman)

**Type:** trend
**Timeframe:** 4h-Daily
**Holding Period:** Days to weeks
**Crypto Applicability:** HIGH — Adapts to changing volatility—critical for crypto's regime shifts

### Entry Rules
1. Calculate Efficiency Ratio (ER): ER = |close_today - close_n_days_ago| / sum(|daily_changes|) over n days
2. ER measures trendiness: ER=1 (perfect trend), ER=0 (random walk)
3. Smoothing Constant (SC): SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
4. KAMA_today = KAMA_yesterday + SC * (price - KAMA_yesterday)
5. **Long**: Price crosses above KAMA AND ER > 0.3 (trending market)
6. **Short**: Price crosses below KAMA AND ER > 0.3

### Exit Rules
1. Exit when price crosses KAMA in opposite direction
2. OR exit when ER drops < 0.2 (market turned choppy)
3. Stop loss: 2 ATR from entry (optional, KAMA itself is dynamic stop)

### Filters
1. ER > 0.3 (minimum trendiness; below this is noise)
2. Skip if ATR < 1% of price (low volatility = poor trend)
3. Avoid during consolidation: require price outside 20-day Bollinger Bands at entry

### Parameters
- er_period: 10
- fast_sc: 2 (fast EMA constant = 2/(2+1))
- slow_sc: 30 (slow EMA constant = 2/(30+1))
- min_er: 0.3
- exit_er: 0.2
- stop_atr: 2.0

### Why It Might Work Where Others Failed
KAMA adapts smoothing based on market efficiency. In trends (high ER), KAMA moves fast (low lag). In chop (low ER), KAMA moves slow (filters noise). This is SMARTER than fixed MA—it adjusts to regime automatically.

### Implementation Notes
Crypto-adapt: Use 4h or daily bars. Backtest multiple ER periods (10, 20, 30). Can combine with volume filter (entry only if volume >1.5x avg). KAMA alone may lag—consider using ER as position sizing factor (higher ER = larger position).

---


## Strategy 14: ATR Channel Breakout (Source: Trading Systems and Methods, Perry Kaufman)

**Type:** breakout
**Timeframe:** Daily
**Holding Period:** 1-2 weeks
**Crypto Applicability:** HIGH — Volatility-adjusted channels adapt to crypto's changing vol regimes

### Entry Rules
1. Calculate ATR(14)
2. Upper channel: 20-day SMA + 2.5 * ATR
3. Lower channel: 20-day SMA - 2.5 * ATR
4. **Long**: Close breaks above upper channel
5. **Short**: Close breaks below lower channel
6. Confirm with volume >1.3x average

### Exit Rules
1. Exit when price crosses back through 20-day SMA (mean reversion)
2. Stop: opposite channel (e.g., long stop = lower channel)
3. Time stop: 10 days

### Filters
1. Skip if prior 5 bars are all inside channel (consolidation, not breakout)
2. Avoid if ATR < 2% of price (insufficient volatility)
3. Require ADX > 20 (trending market)

### Parameters
- ma_period: 20
- atr_period: 14
- atr_mult: 2.5
- volume_mult: 1.3
- min_atr_pct: 2%
- min_adx: 20

### Why It Might Work Where Others Failed
Channels are volatility-adjusted (not static Bollinger Bands). Kaufman's insight: ATR-based channels are more robust across different markets. Breakout + volume + ADX = multi-factor confirmation, not just price.

### Implementation Notes
Crypto-adapt: Daily or 4h bars. Backtest ATR multiplier (2.0-3.0). Can use EMA instead of SMA for faster adaptation. Consider combining with funding rate filter (avoid breakouts into extreme funding).

---


## Strategy 15: Welles Wilder Swing Index (Source: Trading Systems and Methods, Perry Kaufman (explaining Wilder))

**Type:** momentum/swing
**Timeframe:** Daily
**Holding Period:** 3-10 days
**Crypto Applicability:** MEDIUM — Complex calculation; may be noisy in 24/7 crypto markets, but captures swing extremes

### Entry Rules
1. Calculate Swing Index (SI): SI = 50 * (Cy - C + 0.5*(Cy - Oy) + 0.25*(C - O)) / R * K/T
2. Where: C=close, O=open, H=high, L=low, y=yesterday, R=true range, K=max(|H-Cy|, |L-Cy|), T=limit move (use 3*ATR)
3. Accumulate SI into Accumulation Swing Index (ASI)
4. **Long**: ASI crosses above previous swing high (bullish swing reversal)
5. **Short**: ASI crosses below previous swing low (bearish swing reversal)

### Exit Rules
1. Exit when ASI crosses back through zero (swing exhausted)
2. Stop: 2 ATR from entry
3. Time stop: 10 days

### Filters
1. Only trade when ASI diverges from price (e.g., price makes new low but ASI doesn't = bullish divergence)
2. Skip if ATR < 1.5% (low volatility)
3. Require trending market: ADX > 15

### Parameters
- limit_move_atr: 3.0
- stop_atr: 2.0
- min_atr_pct: 1.5%
- min_adx: 15

### Why It Might Work Where Others Failed
SI normalizes intraday swings by range and volatility—captures swing momentum better than raw price. ASI accumulation shows underlying momentum shifts. Divergences signal exhaustion (like RSI divergence but more sophisticated).

### Implementation Notes
Crypto-adapt: Daily bars only (open/high/low/close needed). Complex calculation—test in Python first. May be too noisy on 4h. Consider as confirmation indicator for other strategies (e.g., enter NR4 breakout only if ASI confirms).

---


## Strategy 16: CMO Divergence Trading (Source: The New Technical Trader, Tushar Chande)

**Type:** mean_reversion/momentum
**Timeframe:** 4h-Daily
**Holding Period:** 2-7 days
**Crypto Applicability:** HIGH — Divergences are structural signals of momentum exhaustion; works across asset classes

### Entry Rules
1. Calculate CMO: CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down) over N periods (typically 14)
2. Identify divergence: price makes new high/low but CMO doesn't
3. **Bullish divergence**: Price makes lower low, CMO makes higher low → buy on next bar if CMO > -50
4. **Bearish divergence**: Price makes higher high, CMO makes lower high → sell on next bar if CMO < +50
5. Confirm with volume expansion (>1.2x avg) on reversal bar

### Exit Rules
1. Exit when CMO crosses zero (momentum shift complete)
2. Stop: beyond recent swing extreme by 1.5 ATR
3. Time stop: 7 days

### Filters
1. Divergence must span at least 3-5 bars (avoid micro-divergences)
2. Skip if ATR < 2% of price (low volatility = weak reversals)
3. Require initial CMO extreme: |CMO| > 40 at divergence start (meaningful overbought/oversold)

### Parameters
- cmo_period: 14
- cmo_entry_threshold: -50 (long) / +50 (short)
- cmo_extreme: 40
- volume_mult: 1.2
- stop_atr: 1.5
- max_holding: 7 days

### Why It Might Work Where Others Failed
CMO is more responsive than RSI (uses raw momentum, not smoothed). Divergences signal momentum exhaustion BEFORE price reversal—leading indicator. Requires multi-bar confirmation, reducing false signals.

### Implementation Notes
Crypto-adapt: Use 4h or daily. CMO is NOT in standard libraries—code it from scratch: CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down). Backtest on major coins (BTC, ETH, SOL). Consider combining with funding rate (extreme funding + divergence = high-conviction entry).

---


## Strategy 17: VIDYA Adaptive Trend (Source: The New Technical Trader, Tushar Chande)

**Type:** trend
**Timeframe:** Daily
**Holding Period:** 1-3 weeks
**Crypto Applicability:** HIGH — Adaptive smoothing handles crypto's volatile regime shifts better than fixed MAs

### Entry Rules
1. Calculate VIDYA: VIDYA = α * CMO_abs * price + (1 - α * CMO_abs) * VIDYA_prev
2. Where α = 2/(N+1) (EMA constant), CMO_abs = |CMO| / 100 (volatility adjustment)
3. **Long**: Price crosses above VIDYA AND CMO > 20 (momentum confirmation)
4. **Short**: Price crosses below VIDYA AND CMO < -20

### Exit Rules
1. Exit when price crosses VIDYA in opposite direction
2. OR exit when CMO crosses zero (momentum fades)
3. Stop: 2 ATR from entry

### Filters
1. |CMO| > 20 at entry (minimum momentum)
2. Skip if ATR < 1% of price (low volatility)
3. Avoid during consolidation: require price >2% from VIDYA at entry (clear break, not whipsaw)

### Parameters
- vidya_period: 14
- cmo_period: 14
- min_cmo: 20
- stop_atr: 2.0
- min_atr_pct: 1%

### Why It Might Work Where Others Failed
VIDYA adjusts smoothing based on CMO (momentum). In strong trends (high |CMO|), VIDYA tracks price closely (low lag). In chop (low |CMO|), VIDYA smooths heavily (filters noise). Smarter than KAMA because it uses momentum, not just efficiency.

### Implementation Notes
Crypto-adapt: Daily bars. VIDYA is NOT in standard libraries—code it. Backtest CMO periods (9, 14, 20). Can combine with volume filter. Consider using VIDYA as trailing stop (exit when price closes below VIDYA by 1%).

---


## Strategy 18: Aroon Oscillator Trend Entry (Source: The New Technical Trader, Tushar Chande)

**Type:** trend
**Timeframe:** 4h-Daily
**Holding Period:** 1-2 weeks
**Crypto Applicability:** HIGH — Identifies trend initiation (not just continuation); crypto trends can be explosive

### Entry Rules
1. Calculate Aroon Up: 100 * (N - periods_since_N_day_high) / N
2. Calculate Aroon Down: 100 * (N - periods_since_N_day_low) / N
3. Aroon Oscillator: Aroon Up - Aroon Down
4. **Long**: Aroon Oscillator crosses above +50 (strong uptrend starting)
5. **Short**: Aroon Oscillator crosses below -50 (strong downtrend starting)
6. Confirm: Aroon Up/Down must reach 100 within 2 bars of entry (fresh high/low)

### Exit Rules
1. Exit when Aroon Oscillator crosses zero (trend weakening)
2. Stop: 2 ATR from entry
3. Time stop: 14 days

### Filters
1. Skip if oscillator is between -30 and +30 (ranging market)
2. Require ATR > 1.5% of price (volatility for trend)
3. Avoid if prior 3 bars are all inside bars (consolidation, not breakout)

### Parameters
- aroon_period: 25
- osc_entry_threshold: 50
- osc_exit_threshold: 0
- stop_atr: 2.0
- min_atr_pct: 1.5%

### Why It Might Work Where Others Failed
Aroon identifies NEW trends (time since high/low), not just existing trends (like MACD). Catches trend initiation earlier than lagging MAs. Oscillator crossing ±50 is a strong signal—not just overbought/oversold, but directional commitment.

### Implementation Notes
Crypto-adapt: Use 4h or daily. Aroon period 25 is standard, but backtest 14, 25, 50. Works well on coins with strong directional moves (SOL, AVAX). Can combine with volume breakout (Aroon + volume spike = high-conviction entry).

---


## Strategy 19: Dual Momentum (Absolute + Relative) (Source: Dual Momentum Investing, Gary Antonacci)

**Type:** momentum/hybrid
**Timeframe:** Monthly (rebalance monthly)
**Holding Period:** Weeks to months
**Crypto Applicability:** HIGH — Momentum works across timeframes; crypto has strong trending behavior; monthly rebalancing reduces transaction costs

### Entry Rules
1. **Absolute Momentum**: For each asset, compare current price to 12-month (252-day) ago price
2. Asset has positive absolute momentum if: (price_now / price_12m_ago - 1) > risk_free_rate (use 0% for crypto)
3. **Relative Momentum**: Rank all assets by 12-month return
4. **Entry**: At month-end, buy the TOP asset that has BOTH positive absolute momentum AND highest relative momentum
5. If no asset has positive absolute momentum, move to cash (or stablecoins)

### Exit Rules
1. Rebalance monthly (last day of month)
2. Exit current position if it no longer has highest relative momentum
3. Exit ALL positions if absolute momentum turns negative (defensive mode)

### Filters
1. Minimum 12-month history required (cannot calculate momentum without it)
2. Skip assets with <$100M market cap or <$5M daily volume (liquidity)
3. Require at least 3 assets in universe (diversification)

### Parameters
- lookback_period: 252 days (12 months)
- rebalance_frequency: monthly
- risk_free_rate: 0% (or use 3-month T-bill rate)
- min_market_cap: 100M
- min_daily_volume: 5M

### Why It Might Work Where Others Failed
Combines trend-following (absolute momentum) with relative strength (relative momentum). Dual filter reduces whipsaw: won't buy weak assets (absolute) and won't hold laggards (relative). Monthly rebalancing reduces transaction costs vs daily.

### Implementation Notes
Crypto-adapt: Use BTC, ETH, BNB, SOL, ADA, AVAX, MATIC, etc. (top 10-20 by market cap). Rebalance monthly (last day of month). Use stablecoins as 'cash' during defensive periods. Backtest with 10bps transaction costs per rebalance.

---


## Strategy 20: Accelerating Dual Momentum (Source: Dual Momentum Investing, Gary Antonacci)

**Type:** momentum
**Timeframe:** Monthly
**Holding Period:** Weeks to months
**Crypto Applicability:** HIGH — Recent momentum often more predictive than 12-month momentum in fast-moving crypto markets

### Entry Rules
1. Calculate momentum scores using MULTIPLE lookbacks: 1-month, 3-month, 6-month, 12-month returns
2. Weight recent performance higher: score = 0.4 * R_1m + 0.3 * R_3m + 0.2 * R_6m + 0.1 * R_12m
3. Require positive absolute momentum: R_12m > 0
4. **Entry**: Buy asset with highest weighted momentum score
5. Rebalance monthly

### Exit Rules
1. Exit if asset no longer has highest momentum score
2. Exit if 12-month return turns negative (absolute momentum filter)
3. Rebalance monthly

### Filters
1. All assets must have 12-month history
2. Minimum market cap $100M
3. Skip if top 3 assets have similar scores (<5% difference = indecision)

### Parameters
- lookback_1m: 21 days
- lookback_3m: 63 days
- lookback_6m: 126 days
- lookback_12m: 252 days
- weight_1m: 0.4
- weight_3m: 0.3
- weight_6m: 0.2
- weight_12m: 0.1

### Why It Might Work Where Others Failed
Recent momentum is more predictive than distant momentum (Jegadeesh & Titman). Weighted score captures acceleration (not just 12-month return). Combines persistence (12m filter) with responsiveness (1m weight).

### Implementation Notes
Crypto-adapt: Monthly rebalancing. Use top 10-20 coins. Backtest weight optimization (may need to adjust for crypto's faster dynamics). Consider adding volume factor (penalize coins with declining volume).

---


## Strategy 21: Time-Series Momentum (TSMOM) (Source: Market Momentum, Satchell et al.)

**Type:** momentum
**Timeframe:** Daily (monthly rebalance)
**Holding Period:** 1 month
**Crypto Applicability:** HIGH — TSMOM is one of the most robust strategies across asset classes; works on crypto futures/perps

### Entry Rules
1. Calculate excess return: R_t = (price_t / price_{t-12m}) - 1
2. **Position direction**: sign(R_t) → if R_t > 0, go long; if R_t < 0, go short
3. **Position size**: volatility-scaled to target X% annualized volatility (e.g., 10%)
4. Position = target_vol / realized_vol * sign(R_t)
5. Rebalance monthly

### Exit Rules
1. Rebalance monthly: recalculate R_t and adjust position
2. No intra-month exits (buy & hold for month)
3. No fixed stop—volatility scaling handles risk

### Filters
1. Minimum 12-month history required
2. Skip if realized volatility > 150% annualized (regime break)
3. Apply to portfolio of 5-10 coins (diversification critical)

### Parameters
- lookback_period: 12 months (252 days)
- rebalance_freq: monthly
- target_vol: 10-15% annualized
- vol_lookback: 21-63 days for realized vol

### Why It Might Work Where Others Failed
TSMOM is trend-following with volatility scaling. Works because trends persist (momentum anomaly). Volatility scaling prevents overleveraging in high-vol periods. Simple, robust, diversifiable.

### Implementation Notes
Crypto-adapt: Apply to BTC, ETH, SOL, BNB, AVAX, etc. Use perpetuals for shorting. Calculate realized vol from daily returns. Backtest with 10bps transaction costs. Can use 1m, 3m, 6m, 12m lookbacks and combine (ensemble TSMOM).

---


## Strategy 22: Cross-Sectional Momentum (XSMOM) (Source: Market Momentum, Satchell et al.)

**Type:** momentum/long-short
**Timeframe:** Daily (monthly rebalance)
**Holding Period:** 1 month
**Crypto Applicability:** HIGH — Crypto has wide dispersion in returns—ideal for relative momentum strategies

### Entry Rules
1. Rank all coins by 12-month return
2. **Long**: Top 30% (winners)
3. **Short**: Bottom 30% (losers)
4. Equal-weight within long and short portfolios
5. Dollar-neutral: $1 long, $1 short (market-neutral)
6. Rebalance monthly

### Exit Rules
1. Rebalance monthly: re-rank and adjust portfolios
2. No intra-month exits

### Filters
1. Minimum 12-month history
2. Minimum market cap $50M (avoid illiquid coins)
3. Skip coins with <$1M daily volume
4. Require at least 10 coins in universe (5 long, 5 short minimum)

### Parameters
- lookback_period: 12 months
- long_pct: 30%
- short_pct: 30%
- rebalance_freq: monthly
- min_market_cap: 50M
- min_daily_volume: 1M

### Why It Might Work Where Others Failed
Winners continue to outperform, losers continue to underperform (momentum anomaly). Market-neutral removes beta exposure—pure alpha. Dollar-neutral reduces directional risk. Works because of behavioral biases (under-reaction, herding).

### Implementation Notes
Crypto-adapt: Use top 20-30 coins by market cap. Short via perpetuals. Backtest with realistic transaction costs (10bps). Can combine with TSMOM (trade XSMOM only when TSMOM is positive = bull market). Consider funding rate costs for shorts.

---


## Strategy 23: Bollinger Band Momentum Breakout (Source: Unholy Grails, Nick Radge)

**Type:** breakout/momentum
**Timeframe:** Daily
**Holding Period:** Weeks to months
**Crypto Applicability:** HIGH — Crypto has strong trends; BB breakouts capture trend initiation

### Entry Rules
1. Calculate Bollinger Bands: 20-day SMA ± 2 SD
2. **Long**: Close > upper Bollinger Band AND close > 100-day MA (trend filter)
3. **Short**: Close < lower Bollinger Band AND close < 100-day MA
4. Enter next bar open

### Exit Rules
1. Exit when close crosses back through 20-day SMA (mean reversion)
2. OR exit when 100-day MA flips (trend reversal)
3. Stop loss: 3 ATR from entry (optional)
4. Time stop: 60 days (if no trend develops)

### Filters
1. Trend filter: MUST be above/below 100-day MA (avoid counter-trend)
2. Volume confirmation: volume on breakout > 1.5x 20-day average
3. ATR > 2% of price (sufficient volatility)

### Parameters
- bb_period: 20
- bb_std: 2.0
- trend_ma: 100
- volume_mult: 1.5
- stop_atr: 3.0
- min_atr_pct: 2%

### Why It Might Work Where Others Failed
BB breakout signals volatility expansion (not contraction like BB squeeze). Trend filter ensures we're trading WITH the big trend. This is NOT mean reversion—it's breakout in direction of trend. Different from what we tested (BB mean reversion).

### Implementation Notes
Crypto-adapt: Daily bars. Backtest on BTC, ETH, SOL. BB breakout is LONG-ONLY in Radge's book—adapt for crypto by allowing shorts below 100-day MA. Watch for false breakouts in ranging markets (use ADX filter).

---


## Strategy 24: Weekend Gap Fade (Source: Unholy Grails, Nick Radge)

**Type:** mean_reversion
**Timeframe:** Daily
**Holding Period:** 1-3 days
**Crypto Applicability:** LOW — Crypto trades 24/7 (no weekend gaps); adapt to session gaps or holiday periods

### Entry Rules
1. Identify weekend gap: Monday open vs Friday close
2. **Gap up**: Monday open > Friday close by >1%
3. **Gap down**: Monday open < Friday close by >1%
4. **Fade logic**: Gap up → short, gap down → long
5. Enter at Monday close (after gap is confirmed)

### Exit Rules
1. Exit at Friday close (weekly cycle)
2. Stop: 1.5x gap size beyond entry

### Filters
1. Gap must be >1% of price
2. Skip if market is strongly trending (>5% move prior week)
3. Volume on Monday should be above average

### Parameters
- min_gap_pct: 1%
- stop_mult: 1.5
- max_holding: 5 days

### Why It Might Work Where Others Failed
Weekend gaps in traditional markets often fill due to liquidity imbalance. Fade the gap = mean reversion to equilibrium.

### Implementation Notes
Crypto-adapt: NOT directly applicable (24/7 trading). Could adapt to: (1) gaps after exchange maintenance, (2) holiday periods (Christmas, New Year), (3) funding rate resets as 'gap' proxy. Likely LOW edge in crypto—include for completeness but deprioritize.

---


## Strategy 25: Triple Barrier Labeling + ML (Source: Advances in Financial Machine Learning, Marcos Lopez de Prado)

**Type:** hybrid (ML + barriers)
**Timeframe:** Any (adaptive)
**Holding Period:** Adaptive (based on barrier hits)
**Crypto Applicability:** HIGH — Adaptive exits handle crypto's non-stationary behavior; combines ML prediction with risk management

### Entry Rules
1. **Feature engineering**: Calculate features (momentum, volatility, volume, microstructure)
2. **ML model**: Train classifier (e.g., Random Forest) to predict next barrier hit (upper, lower, time)
3. **Entry signal**: When ML predicts P(upper barrier hit) > 0.6, go long
4. **Position size**: Scale by prediction confidence (higher P = larger position)
5. Define barriers at entry: upper = entry + X%, lower = entry - X%, time = T days

### Exit Rules
1. Exit when ANY barrier is hit:
2. - Upper barrier (take profit)
3. - Lower barrier (stop loss)
4. - Time barrier (max holding period)
5. No discretionary exits—let barriers decide

### Filters
1. Only trade when ML prediction confidence > 0.6 (avoid low-conviction trades)
2. Retrain model every 30 days (avoid overfitting to old regimes)
3. Use walk-forward validation (never train on future data)

### Parameters
- upper_barrier_pct: 2-5% (optimize)
- lower_barrier_pct: 1-3% (asymmetric, tighter stop)
- time_barrier_days: 5-10 (adaptive)
- ml_confidence_threshold: 0.6
- retrain_frequency: 30 days

### Why It Might Work Where Others Failed
Triple barriers remove look-ahead bias (labels are path-dependent). ML predicts which barrier hits first (not just direction). Combines prediction with risk management. Meta-labeling (predicting bet size, not direction) can improve Sharpe.

### Implementation Notes
Crypto-adapt: Use 4h or daily bars. Features: RSI, ATR, volume ratio, funding rate, OI change, BTC correlation. Start with simple RF model. Backtest with realistic transaction costs. This is ADVANCED—requires ML infrastructure. Consider as long-term project, not immediate implementation.

---


## Strategy 26: Fractional Differentiation for Stationarity (Source: Advances in Financial Machine Learning, Marcos Lopez de Prado)

**Type:** momentum (stationary)
**Timeframe:** Daily
**Holding Period:** Adaptive
**Crypto Applicability:** MEDIUM — Makes non-stationary crypto prices stationary while preserving memory (better than raw returns)

### Entry Rules
1. Apply fractional differentiation to price series: X_t^d = Σ w_k * X_{t-k}, where d is fractional order (e.g., d=0.4)
2. This makes series stationary while retaining momentum information
3. Calculate z-score on fractionally differentiated series: z = (X_t^d - mean) / std
4. **Long**: z > 1.0 (positive momentum, stationary)
5. **Short**: z < -1.0 (negative momentum)

### Exit Rules
1. Exit when z crosses zero
2. Stop: z exceeds ±3.0
3. Time stop: 10 days

### Filters
1. Find optimal d via grid search (0.2-0.8) that maximizes ADF stationarity
2. Recalculate d every 60 days (regime adaptation)
3. Skip if ADF test p-value > 0.05 (not stationary)

### Parameters
- d_order: 0.4 (optimize)
- z_threshold: 1.0
- z_stop: 3.0
- lookback_period: 100 days
- recalc_d_days: 60

### Why It Might Work Where Others Failed
Traditional ML fails on non-stationary data. Fractional differentiation makes data stationary (required for ML) while preserving long-term memory (unlike raw returns, which lose all memory). Elegant solution to stationarity vs memory trade-off.

### Implementation Notes
Crypto-adapt: Daily bars. Use fracdiff Python library. Calculate optimal d (backtest d=0.2, 0.4, 0.6, 0.8). Combine with ML model for prediction. ADVANCED—requires statistical/ML knowledge. Deprioritize vs simpler strategies.

---


## Strategy 27: Portfolio of Systems (Jaekle Approach) (Source: Trading Systems 2nd Edition, Urban Jaekle)

**Type:** hybrid/portfolio
**Timeframe:** Multiple (4h, daily, weekly)
**Holding Period:** Varies by system
**Crypto Applicability:** HIGH — Diversification across strategies reduces risk; crypto allows 24/7 multi-system trading

### Entry Rules
1. Run MULTIPLE uncorrelated strategies simultaneously (e.g., trend, mean reversion, breakout)
2. **Example portfolio**: EWMAC (trend) + OU MR (mean reversion) + ORB (breakout) + Carry (funding rate)
3. Allocate capital equally (25% each) or by recent Sharpe ratio
4. Each system generates independent signals
5. Enter when INDIVIDUAL system gives signal (not combined)

### Exit Rules
1. Each system manages its own exits (per strategy rules)
2. Monitor portfolio-level risk: if total portfolio drawdown > 20%, reduce all positions by 50%
3. Rebalance monthly: adjust allocation based on rolling Sharpe ratios

### Filters
1. Systems must have correlation < 0.3 (diversification)
2. Each system must have positive Sharpe > 0.5 in backtest
3. Skip systems during regime mismatch (e.g., no mean reversion in strong trend)

### Parameters
- num_systems: 4-6
- max_correlation: 0.3
- min_sharpe: 0.5
- rebalance_freq: monthly
- max_portfolio_dd: 20%

### Why It Might Work Where Others Failed
Diversification across strategies smooths equity curve. When trend-following fails (ranging market), mean reversion profits. When breakouts fail (false signals), carry strategies provide income. Portfolio approach reduces risk more than any single strategy.

### Implementation Notes
Crypto-adapt: Combine strategies we've extracted (EWMAC, OU MR, NR4, Funding Carry). Backtest portfolio as a whole (not just individual systems). Monitor correlations monthly. This is a META-STRATEGY—implement after individual strategies are validated.

---


## Strategy 28: Trend Pullback Entry (Grimes) (Source: The Art and Science of Technical Analysis, Adam Grimes)

**Type:** trend/pullback
**Timeframe:** 4h-Daily
**Holding Period:** 5-15 days
**Crypto Applicability:** HIGH — Pullbacks in crypto trends are common and tradable; better entry than breakout chasing

### Entry Rules
1. Identify trend: price > 50-day EMA (uptrend) or price < 50-day EMA (downtrend)
2. Wait for pullback: price retraces to 20-day EMA (in uptrend) or rallies to 20-day EMA (in downtrend)
3. **Long setup**: Uptrend + pullback to 20 EMA + bullish rejection bar (close in top 50% of range, near 20 EMA)
4. **Short setup**: Downtrend + rally to 20 EMA + bearish rejection bar
5. Enter next bar after rejection bar confirmation

### Exit Rules
1. Exit when trend breaks: close below 50 EMA (long) or above 50 EMA (short)
2. Stop loss: 1.5 ATR beyond pullback low/high
3. Profit target: prior swing high/low (or 3:1 reward/risk)
4. Time stop: 15 days if no momentum

### Filters
1. ADX > 20 (trending market, not ranging)
2. Pullback must reach 20 EMA (not too shallow, not too deep)
3. Rejection bar must close within 20% of 20 EMA (clear support/resistance test)

### Parameters
- trend_ema: 50
- pullback_ema: 20
- stop_atr: 1.5
- min_adx: 20
- max_holding: 15 days

### Why It Might Work Where Others Failed
Buying pullbacks in trends offers better risk/reward than chasing breakouts. Trend is your friend, pullback is your entry. Grimes emphasizes CONTEXT (trend) + TRIGGER (rejection bar)—not just patterns. Multi-timeframe structure (50 EMA trend, 20 EMA pullback).

### Implementation Notes
Crypto-adapt: Use 4h or daily bars. Backtest EMA periods (20/50, 10/30, 50/100). Combine with volume (volume should decline on pullback, expand on resumption). Consider adding RSI filter (buy pullbacks when RSI 30-50 in uptrend).

---


## Strategy 29: Failure Test (Failed Breakout Fade) (Source: The Art and Science of Technical Analysis, Adam Grimes)

**Type:** mean_reversion/anti-breakout
**Timeframe:** 4h-Daily
**Holding Period:** 3-7 days
**Crypto Applicability:** MEDIUM — Crypto has many false breakouts (leverage, stop hunts); fading failures can be profitable

### Entry Rules
1. Identify key level: recent swing high/low or range boundary
2. **Breakout attempt**: Price breaks above/below level by >0.5%
3. **Failure**: Price reverses back within 2 bars, closing below breakout level (for upside breakout)
4. **Entry**: Short on failure bar close (for failed upside BO) or long (for failed downside BO)
5. Confirm with increased volume on failure bar (>1.5x avg)

### Exit Rules
1. Target: opposite side of range or prior swing extreme
2. Stop: 1 ATR beyond breakout high/low (tight stop—failure confirmed)
3. Time stop: 7 days
4. Exit if breakout is re-attempted (level breaks again)

### Filters
1. Breakout must be genuine (>0.5% beyond level, not just a wick)
2. Failure must be decisive (close back within range, not just hovering)
3. Volume on failure bar > 1.5x average (selling pressure for failed upside BO)
4. Avoid during strong trends (failure test works in ranges, not trends)

### Parameters
- breakout_threshold: 0.5%
- failure_bars: 2
- volume_mult: 1.5
- stop_atr: 1.0
- max_holding: 7 days

### Why It Might Work Where Others Failed
Failed breakouts signal exhaustion—bulls/bears tried to push through but failed. Fade the failure = trade against the weak hands. Grimes calls this 'anti-pattern'—trading the FAILURE of a pattern, not the pattern itself. Requires range-bound market (not trending).

### Implementation Notes
Crypto-adapt: Use 4h or daily. Identify key levels via swing highs/lows or horizontal S/R. Watch for stop hunts (common in crypto)—failed breakouts after stop hunt are high-probability fades. Combine with funding rate (extreme funding + failed breakout = strong signal).

---

