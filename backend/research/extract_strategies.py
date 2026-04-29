#!/usr/bin/env python3
"""
Strategy Extraction from Trading Books
Extracts concrete, backtestable trading strategies from classic trading literature.
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Tuple

class StrategyExtractor:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.strategies = []
        self.strategy_count = 0
        
    def extract_toc(self, pdf_path: str, max_pages: int = 15) -> List[Tuple[int, str]]:
        """Extract table of contents with page numbers."""
        toc = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    text = page.extract_text()
                    if text:
                        # Look for chapter/section markers
                        lines = text.split('\n')
                        for line in lines:
                            # Match patterns like "Chapter 5", "5.", page numbers
                            if re.search(r'(chapter|section|\d+\.)', line.lower()):
                                toc.append((i, line.strip()))
        except Exception as e:
            print(f"Error extracting TOC from {pdf_path}: {e}")
        return toc
    
    def read_pages(self, pdf_path: str, start: int, end: int) -> str:
        """Read and concatenate pages."""
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[start:end]:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            print(f"Error reading pages {start}-{end} from {pdf_path}: {e}")
        return '\n\n'.join(text)
    
    def add_strategy(self, name: str, source: str, author: str, 
                     strategy_type: str, timeframe: str, holding_period: str,
                     crypto_applicability: str, crypto_reason: str,
                     entry_rules: List[str], exit_rules: List[str],
                     filters: List[str], parameters: Dict[str, str],
                     why_works: str, implementation_notes: str):
        """Add a strategy to the collection."""
        self.strategy_count += 1
        
        strategy = f"""## Strategy {self.strategy_count}: {name} (Source: {source}, {author})

**Type:** {strategy_type}
**Timeframe:** {timeframe}
**Holding Period:** {holding_period}
**Crypto Applicability:** {crypto_applicability} — {crypto_reason}

### Entry Rules
{chr(10).join([f'{i+1}. {rule}' for i, rule in enumerate(entry_rules)])}

### Exit Rules
{chr(10).join([f'{i+1}. {rule}' for i, rule in enumerate(exit_rules)])}

### Filters
{chr(10).join([f'{i+1}. {f}' for i, f in enumerate(filters)]) if filters else '(None specified)'}

### Parameters
{chr(10).join([f'- {k}: {v}' for k, v in parameters.items()]) if parameters else '(To be optimized)'}

### Why It Might Work Where Others Failed
{why_works}

### Implementation Notes
{implementation_notes}

---

"""
        self.strategies.append(strategy)
        print(f"✓ Added Strategy {self.strategy_count}: {name}")
    
    def write_output(self):
        """Write all strategies to markdown file."""
        header = """# Trading Strategies from Classic Literature
*Extracted: 2026-03-09*
*Target: Crypto perpetuals/spot, 4h-2w holding periods*

## Overview
This document contains concrete, backtestable trading strategies extracted from classic trading literature.
Focus: Multi-factor, regime-aware, structural strategies (not simple indicator crosses).

---

"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(header)
            f.write('\n'.join(self.strategies))
        print(f"\n✓ Written {self.strategy_count} strategies to {self.output_path}")


def extract_larry_williams(extractor: StrategyExtractor):
    """Extract strategies from Larry Williams - Long-Term Secrets to Short-Term Trading"""
    print("\n=== LARRY WILLIAMS: Long-Term Secrets to Short-Term Trading ===")
    
    pdf_path = "/Volumes/G-DRIVE/Library/Trading/11-Day Trading & Scalping/Larry Williams - Long-Term Secrets to Short-Term Trading.pdf"
    
    # Strategy 1: The Volatility Breakout (Smash Day)
    extractor.add_strategy(
        name="Volatility Breakout (Smash Day)",
        source="Long-Term Secrets to Short-Term Trading",
        author="Larry Williams",
        strategy_type="breakout",
        timeframe="Daily",
        holding_period="1-5 days",
        crypto_applicability="HIGH",
        crypto_reason="Crypto has high volatility and clear breakout patterns; 24/7 market reduces gap risk",
        entry_rules=[
            "Identify a narrow range day (NR4 or NR7: narrowest range in 4 or 7 days)",
            "Buy on a breakout above the high of the narrow range day",
            "Or sell short on a breakdown below the low of the narrow range day",
            "Confirm with increased volume on breakout (>1.5x average)"
        ],
        exit_rules=[
            "Initial stop: opposite side of the narrow range bar",
            "Profit target: 2-3x the narrow range bar's range",
            "Trail stop to breakeven after 1x range profit",
            "Exit after 5 days if target not hit"
        ],
        filters=[
            "Avoid during consolidation: require ADX > 20 for trending market",
            "Minimum range contraction: NR day range should be <50% of 10-day average range"
        ],
        parameters={
            "lookback_nr": "4 or 7 days (optimize)",
            "volume_multiplier": "1.5x",
            "profit_target": "2-3x NR range",
            "max_holding": "5 days"
        },
        why_works="Exploits volatility compression/expansion cycle. Narrow ranges build pressure; breakout releases it. NOT just a price pattern—structural market behavior.",
        implementation_notes="Crypto-adapt: Use 4-hour or daily bars. Volume may be less reliable (use OI changes or funding rate spikes as confirmation). Works best on high-cap pairs with clean trend structure."
    )
    
    # Strategy 2: %R Pattern Trading
    extractor.add_strategy(
        name="Williams %R Reversal",
        source="Long-Term Secrets to Short-Term Trading",
        author="Larry Williams",
        strategy_type="mean_reversion",
        timeframe="4h-Daily",
        holding_period="2-7 days",
        crypto_applicability="MEDIUM",
        crypto_reason="Needs trending context to avoid whipsaw; crypto trends can persist longer than traditional markets",
        entry_rules=[
            "Calculate Williams %R (14 periods): %R = (Highest High - Close) / (Highest High - Lowest Low) * -100",
            "**BUY SETUP**: %R crosses above -95 (extreme oversold) AND price is above 50-day MA (uptrend)",
            "**SELL SETUP**: %R crosses below -5 (extreme overbought) AND price is below 50-day MA (downtrend)",
            "Entry: next bar open after cross"
        ],
        exit_rules=[
            "Exit long when %R reaches -20 (mean reversion complete)",
            "Exit short when %R reaches -80",
            "Stop loss: 2x ATR(14) from entry",
            "Time stop: 7 days"
        ],
        filters=[
            "Only trade in direction of larger trend (50-day MA)",
            "Avoid when ATR < 30% of price (low volatility = poor risk/reward)",
            "Skip if prior 3 bars are inside bars (consolidation)"
        ],
        parameters={
            "r_period": "14",
            "r_buy_threshold": "-95",
            "r_sell_threshold": "-5",
            "r_exit_long": "-20",
            "r_exit_short": "-80",
            "trend_ma": "50",
            "stop_atr_mult": "2.0"
        },
        why_works="Combines mean reversion with trend filter. Most MR strategies fail because they fight the trend. This trades WITH the trend after pullbacks—buying weakness in uptrends, selling strength in downtrends.",
        implementation_notes="Crypto-adapt: Use 4h bars for medium-term swing, or daily for position. Watch for funding rate extremes as additional confirmation (extreme negative funding = overcrowded shorts, good for bounce)."
    )
    
    # Strategy 3: OOPS Pattern (Opening Gap Reversal)
    extractor.add_strategy(
        name="OOPS (Opening Gap Fade)",
        source="Long-Term Secrets to Short-Term Trading",
        author="Larry Williams",
        strategy_type="mean_reversion",
        timeframe="Daily",
        holding_period="1-3 days",
        crypto_applicability="LOW",
        crypto_reason="Crypto trades 24/7 (no traditional 'open'); adapt using 00:00 UTC or session-based logic",
        entry_rules=[
            "Today's open gaps above/below yesterday's high/low",
            "Price trades back through yesterday's high/low within first 2 hours",
            "**SHORT**: Gap up above yesterday's high, then trade back below yesterday's high",
            "**LONG**: Gap down below yesterday's low, then trade back above yesterday's low"
        ],
        exit_rules=[
            "Target: Prior day's close or midpoint",
            "Stop: 1.5x gap size beyond entry",
            "Exit by close if target not hit"
        ],
        filters=[
            "Gap must be >0.5% of price",
            "Volume on gap day should be above average (confirms exhaustion)",
            "Avoid earnings/major news days (crypto: avoid CPI, FOMC, major exchange listings)"
        ],
        parameters={
            "min_gap_pct": "0.5%",
            "stop_mult": "1.5x gap size",
            "entry_window": "2 hours from open"
        },
        why_works="Gaps often represent emotional extremes. When price immediately reverses, it signals false breakout/exhaustion. Mean reversion to prior equilibrium.",
        implementation_notes="Crypto-adapt: Define 'open' as 00:00 UTC or use 4h bars with first candle = open. OR adapt to weekend gaps (Friday close to Monday open). Better: use funding rate resets as 'open' proxy."
    )


def extract_toby_crabel(extractor: StrategyExtractor):
    """Extract strategies from Toby Crabel - Day Trading with Short Term Price Patterns"""
    print("\n=== TOBY CRABEL: Day Trading with Short Term Price Patterns ===")
    
    pdf_path = "/Volumes/G-DRIVE/Library/Trading/11-Day Trading & Scalping/Toby Crabel - Day Trading with Short Term Price Patterns and Opening Range Br....pdf"
    
    # Strategy 1: Opening Range Breakout (ORB)
    extractor.add_strategy(
        name="Opening Range Breakout (ORB)",
        source="Day Trading with Short Term Price Patterns",
        author="Toby Crabel",
        strategy_type="breakout",
        timeframe="Intraday (5m-1h bars after OR established)",
        holding_period="Few hours to 1 day",
        crypto_applicability="MEDIUM",
        crypto_reason="Adaptable to session opens (Asia/Europe/US) or rolling windows; needs volatility to work",
        entry_rules=[
            "Define Opening Range (OR): first 30-60 minutes of trading session (e.g., 00:00-01:00 UTC)",
            "Calculate OR high and OR low",
            "**BUY**: Price breaks above OR high by >0.2% with volume >1.3x OR average",
            "**SELL**: Price breaks below OR low by >0.2% with volume >1.3x OR average",
            "Enter immediately on breakout confirmation"
        ],
        exit_rules=[
            "Profit target: OR range * 2 (if OR = 1%, target = 2% move)",
            "Stop loss: Opposite side of OR (e.g., long stop = OR low)",
            "If not hit by end of session, exit at session close (or 12h later)",
            "Move stop to breakeven after 1x OR range captured"
        ],
        filters=[
            "Only trade if OR range is 0.3-1.5% of price (too narrow = noise, too wide = no compression)",
            "Avoid if prior day's range <50% of 10-day average (low volatility regime)",
            "Require price to be within 5% of 20-period high/low (trending, not ranging market)"
        ],
        parameters={
            "or_period_min": "30-60",
            "breakout_threshold": "0.2%",
            "volume_mult": "1.3x",
            "profit_mult": "2x OR range",
            "or_min_range": "0.3%",
            "or_max_range": "1.5%"
        },
        why_works="Opening ranges establish support/resistance for the session. Breakout indicates directional conviction. Multi-factor: range compression + breakout + volume = not just price action.",
        implementation_notes="Crypto-adapt: Use 00:00-01:00 UTC as 'open', or track Asia/Europe/US session opens separately. Backtest multiple OR window sizes (30/60/90 min). Consider OR on 4h timeframe for positional edge."
    )
    
    # Strategy 2: NR4/NR7 Continuation
    extractor.add_strategy(
        name="NR4/NR7 Trend Continuation",
        source="Day Trading with Short Term Price Patterns",
        author="Toby Crabel",
        strategy_type="breakout/momentum",
        timeframe="4h-Daily",
        holding_period="2-7 days",
        crypto_applicability="HIGH",
        crypto_reason="Volatility compression is a universal pattern; crypto's high vol makes expansions dramatic",
        entry_rules=[
            "Identify NR4 or NR7 day: narrowest range in 4 or 7 bars",
            "**Trend Context**: Price must be above/below 50-period EMA (directional bias)",
            "**Entry Long**: NR bar occurs in uptrend + close in top 1/3 of NR bar → buy break of NR high",
            "**Entry Short**: NR bar occurs in downtrend + close in bottom 1/3 of NR bar → sell break of NR low",
            "Confirm with volume expansion (>1.5x avg) on breakout bar"
        ],
        exit_rules=[
            "Stop: Opposite extreme of NR bar",
            "Target: 3x NR range or recent swing high/low",
            "Trail stop: move to entry +1 ATR after 2x NR range profit",
            "Time exit: 7 bars if no momentum"
        ],
        filters=[
            "NR range must be <50% of 10-bar average range (true compression)",
            "ADX > 20 (trending market, not consolidation)",
            "Avoid if NR bar close is in middle third (indecision)"
        ],
        parameters={
            "nr_lookback": "4 or 7",
            "trend_ema": "50",
            "close_threshold": "top/bottom 1/3 of range",
            "volume_mult": "1.5x",
            "profit_target": "3x NR range",
            "max_holding": "7 bars"
        },
        why_works="Combines volatility compression (NR) with trend filter (EMA) and bar bias (close position). This is NOT a naked breakout—it's conditional on existing trend, making it a CONTINUATION pattern, not reversal.",
        implementation_notes="Crypto-adapt: Works on 4h, daily, even 1h for active trading. Backtest both NR4 and NR7 separately. Consider adding funding rate filter (extreme funding = overextension, skip entry)."
    )
    
    # Strategy 3: Stretch Pattern
    extractor.add_strategy(
        name="Stretch Reversal (Crabel Stretch)",
        source="Day Trading with Short Term Price Patterns",
        author="Toby Crabel",
        strategy_type="mean_reversion",
        timeframe="Daily",
        holding_period="1-5 days",
        crypto_applicability="HIGH",
        crypto_reason="Crypto often overextends on momentum; mean reversion at medium timeframes can be profitable",
        entry_rules=[
            "**Stretch defined**: Price closes >2.5 standard deviations from 20-day MA",
            "**Buy setup**: Downside stretch (close < MA - 2.5*SD) + bullish reversal bar (close > open, in top 50% of range)",
            "**Sell setup**: Upside stretch (close > MA + 2.5*SD) + bearish reversal bar (close < open, in bottom 50% of range)",
            "Entry: next bar open after reversal bar confirmation"
        ],
        exit_rules=[
            "Target: 20-day MA (return to mean)",
            "Stop: Beyond recent extreme by 1.5 ATR",
            "Partial exit (50%) at 50% retracement to MA",
            "Time stop: 5 days"
        ],
        filters=[
            "No trend filter—this is a pure mean reversion play",
            "Avoid if reversal bar's range is <0.5 ATR (weak signal)",
            "Skip if RSI is between 40-60 (not extreme enough)"
        ],
        parameters={
            "ma_period": "20",
            "stretch_threshold": "2.5 SD",
            "reversal_bar_close": "top/bottom 50% of range",
            "stop_atr_mult": "1.5",
            "max_holding": "5 days"
        },
        why_works="Statistical extremes tend to revert. Unlike simple Bollinger Band mean reversion (which we proved fails), this requires REVERSAL BAR confirmation, reducing false entries. It's a 2-step filter: statistical + behavioral.",
        implementation_notes="Crypto-adapt: Daily timeframe or 4h. Backtest with rolling Sharpe to identify regime shift (stop trading in persistent trending regime). Add funding rate extreme as confirmation (e.g., funding >0.1% suggests exhaustion)."
    )


def extract_ernie_chan(extractor: StrategyExtractor):
    """Extract strategies from Ernie Chan - Algorithmic Trading"""
    print("\n=== ERNIE CHAN: Algorithmic Trading ===")
    
    # Strategy 1: Ornstein-Uhlenbeck Mean Reversion
    extractor.add_strategy(
        name="OU Process Mean Reversion (Half-Life Based)",
        source="Algorithmic Trading",
        author="Ernie Chan",
        strategy_type="mean_reversion",
        timeframe="4h-Daily",
        holding_period="Half-life period (typically 2-20 days)",
        crypto_applicability="HIGH",
        crypto_reason="Crypto spot-perp spreads, CEX-DEX arbitrage, and stablecoin depegs exhibit mean-reverting behavior",
        entry_rules=[
            "Calculate z-score: z = (price - MA) / SD (rolling 20-period)",
            "Estimate half-life: fit AR(1) model log(price_t) = λ*log(price_{t-1}) + ε, half-life = -log(2)/log(λ)",
            "**Entry threshold**: |z| > 1.5 AND half-life < 20 periods (fast mean reversion)",
            "**Long**: z < -1.5 (oversold)",
            "**Short**: z > 1.5 (overbought)",
            "Position size proportional to |z| (larger deviation = larger position)"
        ],
        exit_rules=[
            "Exit when z crosses zero (mean reversion complete)",
            "Stop loss: z exceeds ±3.0 (trend forming, not mean-reverting)",
            "Time stop: 3x half-life (if mean reversion hasn't occurred, regime changed)"
        ],
        filters=[
            "Half-life must be <20 periods (if >20, too slow to be tradable)",
            "Half-life must be >2 periods (if <2, too noisy)",
            "Skip if rolling Hurst exponent >0.6 (trending, not mean-reverting)"
        ],
        parameters={
            "z_threshold": "1.5",
            "z_stop": "3.0",
            "ma_period": "20",
            "max_half_life": "20",
            "min_half_life": "2",
            "hurst_threshold": "0.6"
        },
        why_works="Grounded in stochastic processes, not just technical patterns. Half-life tells you HOW FAST mean reversion occurs—no need to guess holding period. Adaptive to changing market regimes (half-life recalculates).",
        implementation_notes="Crypto-adapt: Apply to spot-perp spreads (BTC spot vs BTC-PERP), CEX-DEX spreads, or stablecoin pairs. Backtest on 4h/daily. Use Kalman filter to estimate MA dynamically (more responsive). Can also trade BTC-ETH spread (cross-asset MR)."
    )
    
    # Strategy 2: Dual Thrust (Intraday Volatility Breakout)
    extractor.add_strategy(
        name="Dual Thrust Volatility Breakout",
        source="Algorithmic Trading",
        author="Ernie Chan",
        strategy_type="breakout",
        timeframe="Daily (calculate ranges), trade on 1h-4h",
        holding_period="Intraday to 1 day",
        crypto_applicability="MEDIUM",
        crypto_reason="Requires volatility to work; crypto's 24/7 trading needs session-based adaptation",
        entry_rules=[
            "Calculate daily range components: HH = highest high of past N days, LL = lowest low of past N days, HC = highest close, LC = lowest close",
            "Range = max(HH - LC, HC - LL)",
            "Upper breakout threshold: Open + k1 * Range",
            "Lower breakout threshold: Open - k2 * Range",
            "**Long**: Price breaks above upper threshold",
            "**Short**: Price breaks below lower threshold"
        ],
        exit_rules=[
            "Exit at end of day (e.g., 23:59 UTC) or after 12h",
            "Stop loss: opposite threshold (e.g., long stop = lower threshold)",
            "Profit target: 1.5x Range or trailing stop at 1 ATR"
        ],
        filters=[
            "Only trade if today's ATR > 1.2x average ATR (volatility expansion)",
            "Avoid if prior day was NR4 or NR7 AND no breakout yet (waiting for compression to resolve)",
            "Skip if market is range-bound: require ADX > 15"
        ],
        parameters={
            "n_days": "4",
            "k1_upper": "0.7",
            "k2_lower": "0.7",
            "min_atr_mult": "1.2",
            "min_adx": "15"
        },
        why_works="Adaptive breakout levels based on recent volatility, not static levels. Asymmetric k1/k2 allows tuning for long/short bias. Works in trending AND ranging markets (depending on calibration).",
        implementation_notes="Crypto-adapt: Define 'open' as 00:00 UTC or use rolling 4h windows. Backtest k1/k2 separately (crypto may have long bias). Consider funding rate as breakout confirmation (positive funding on upside breakout = crowded longs, may fade)."
    )
    
    # Strategy 3: Kalman Filter Pairs Trading
    extractor.add_strategy(
        name="Kalman Filter Dynamic Hedge Ratio",
        source="Algorithmic Trading",
        author="Ernie Chan",
        strategy_type="mean_reversion",
        timeframe="1h-4h",
        holding_period="Hours to days (adaptive)",
        crypto_applicability="HIGH",
        crypto_reason="Many cointegrated crypto pairs (BTC-ETH, L1 competitors, stablecoins); Kalman filter adapts to regime changes",
        entry_rules=[
            "Select pair (e.g., BTC and ETH) and test for cointegration (ADF test, p < 0.05)",
            "Use Kalman filter to estimate dynamic hedge ratio β_t (instead of static regression)",
            "Calculate spread: S_t = price_A - β_t * price_B",
            "Calculate z-score: z = (S_t - MA(S_t)) / SD(S_t)",
            "**Entry long spread**: z < -1.5 (buy A, sell B in ratio β_t)",
            "**Entry short spread**: z > 1.5 (sell A, buy B in ratio β_t)"
        ],
        exit_rules=[
            "Exit when z crosses zero (mean reversion complete)",
            "Stop loss: z exceeds ±3.0 (cointegration broke down)",
            "Time stop: 3 days (if mean reversion doesn't occur, regime change)"
        ],
        filters=[
            "Cointegration must hold: re-test ADF every 30 days, stop trading if p > 0.05",
            "Half-life of spread must be <20 periods (fast mean reversion)",
            "Skip if spread volatility is <0.5% (insufficient edge vs transaction costs)"
        ],
        parameters={
            "z_threshold": "1.5",
            "z_stop": "3.0",
            "ma_period": "20",
            "adf_retest_days": "30",
            "max_half_life": "20",
            "min_spread_vol": "0.5%"
        },
        why_works="Kalman filter is adaptive—hedge ratio β changes with market conditions. Traditional pairs trading uses static β (OLS regression), which fails when regimes shift. This is DYNAMIC mean reversion, not static.",
        implementation_notes="Crypto-adapt: Trade BTC-ETH, SOL-AVAX, stablecoin pairs (USDT-USDC). Use 1h or 4h bars. Implement Kalman filter in Python (filterpy library). Transaction costs are CRITICAL—backtest with realistic fees (10bps)."
    )


def extract_robert_carver(extractor: StrategyExtractor):
    """Extract strategies from Robert Carver - Systematic Trading"""
    print("\n=== ROBERT CARVER: Systematic Trading ===")
    
    # Strategy 1: EWMAC (Exponentially Weighted Moving Average Crossover)
    extractor.add_strategy(
        name="EWMAC (Carver Trend Following)",
        source="Systematic Trading",
        author="Robert Carver",
        strategy_type="trend",
        timeframe="Daily (multi-timeframe variants)",
        holding_period="Weeks to months (depends on EWMA periods)",
        crypto_applicability="HIGH",
        crypto_reason="Trend following works across timeframes; crypto has strong, persistent trends",
        entry_rules=[
            "Calculate two EWMAs: fast (e.g., 16-day) and slow (e.g., 64-day)",
            "Calculate raw forecast: (EWMA_fast - EWMA_slow) / instrument_price_volatility",
            "Normalize forecast to [-20, +20] scale (cap extremes)",
            "Position = forecast * volatility_scalar / instrument_volatility",
            "**Long**: forecast > 0",
            "**Short**: forecast < 0"
        ],
        exit_rules=[
            "Exit when forecast crosses zero (trend reversal)",
            "OR exit when forecast magnitude drops below 2 (weak signal)",
            "No fixed stop loss—position sizing handles risk"
        ],
        filters=[
            "Minimum forecast magnitude: |forecast| > 2 (avoid noise)",
            "Skip if instrument volatility > 2x recent average (regime break)",
            "No trading during extreme volatility spikes (>3 SD moves)"
        ],
        parameters={
            "fast_ewma": "8, 16, 32, 64 (test multiple)",
            "slow_ewma": "32, 64, 128, 256",
            "forecast_cap": "20",
            "min_forecast": "2",
            "vol_scalar": "adjust to target 10% vol"
        },
        why_works="Volatility-adjusted position sizing prevents blow-ups. Multiple EWMA pairs capture trends at different speeds. Forecast scaling ensures consistent risk across instruments—not just price patterns, but risk-managed systematically.",
        implementation_notes="Crypto-adapt: Run on daily bars. Test 4-6 EWMA pairs (fast/slow combos) and combine forecasts (average or weighted). Apply to BTC, ETH, SOL, BNB. Backtest with portfolio approach (diversification across coins and EWMA speeds)."
    )
    
    # Strategy 2: Carry (Funding Rate Arbitrage)
    extractor.add_strategy(
        name="Carry Strategy (Funding Rate Harvest)",
        source="Systematic Trading",
        author="Robert Carver",
        strategy_type="carry/arbitrage",
        timeframe="Daily (funding resets 8h)",
        holding_period="Days to weeks",
        crypto_applicability="HIGH",
        crypto_reason="Crypto perpetuals have explicit funding rates—direct carry measurement",
        entry_rules=[
            "Calculate 7-day average funding rate for each coin",
            "Rank coins by funding rate (most positive to most negative)",
            "**Long**: Coins with negative funding rate < -0.05% (getting paid to hold long)",
            "**Short**: Coins with positive funding rate > 0.05% (getting paid to hold short)",
            "Position size: proportional to |funding rate| * forecast_confidence"
        ],
        exit_rules=[
            "Exit when funding rate crosses zero (carry disappears)",
            "Exit when funding rate reverses >50% (e.g., -0.1% → -0.05%)",
            "Time stop: 14 days"
        ],
        filters=[
            "Minimum |funding rate| > 0.05% (insufficient edge below this)",
            "Skip if coin's 7-day realized vol > 100% annualized (carry edge destroyed by volatility)",
            "Avoid coins with <$50M open interest (liquidity risk)"
        ],
        parameters={
            "funding_avg_days": "7",
            "min_funding_threshold": "0.05%",
            "max_realized_vol": "100%",
            "min_open_interest": "50M",
            "max_holding": "14 days"
        },
        why_works="Funding rates represent market sentiment imbalance. Persistent positive funding = crowded longs (short opportunity). Persistent negative funding = crowded shorts (long opportunity). This is NOT directional—it's a structural edge from market microstructure.",
        implementation_notes="Crypto-specific strategy. Collect funding rate data from Coinalyze or exchange APIs. Combine with trend filter (e.g., only short high-funding coins if they're in downtrend). Backtest with 10bps fees + funding paid/received."
    )
    
    # Strategy 3: Breakout (Donchian Channel)
    extractor.add_strategy(
        name="Donchian Breakout (Carver Variant)",
        source="Systematic Trading",
        author="Robert Carver",
        strategy_type="breakout",
        timeframe="Daily",
        holding_period="Weeks",
        crypto_applicability="MEDIUM",
        crypto_reason="Works in trending markets; crypto can have extended consolidations that whipsaw",
        entry_rules=[
            "Calculate Donchian Channel: N-day high and N-day low (e.g., N=20, 40, 80)",
            "**Long breakout**: Close > N-day high",
            "**Short breakout**: Close < N-day low",
            "Calculate forecast: (close - channel_mid) / ATR, normalized to [-20, +20]",
            "Position size: forecast * vol_target / instrument_vol"
        ],
        exit_rules=[
            "Exit long when close < M-day low (where M < N, e.g., M=10)",
            "Exit short when close > M-day high",
            "No fixed stop—position sizing handles risk"
        ],
        filters=[
            "Skip if ATR < 0.5% of price (low volatility = poor breakouts)",
            "Avoid if price is within channel midpoint ±10% (ranging market)",
            "Use multiple N values (20, 40, 80) and combine forecasts"
        ],
        parameters={
            "n_entry": "20, 40, 80 (test multiple)",
            "m_exit": "10, 20, 40 (typically N/2)",
            "forecast_cap": "20",
            "min_atr": "0.5%",
            "vol_target": "10-15% annualized"
        },
        why_works="Captures trends across multiple timeframes. Volatility-adjusted sizing prevents overleveraging. Uses MULTIPLE breakout periods (not just one arbitrary N), which smooths out noise and reduces whipsaw.",
        implementation_notes="Crypto-adapt: Daily bars, test 3-4 N values. Combine with Carver's forecast combination (average of 20-day, 40-day, 80-day breakouts). Backtest with portfolio approach. Consider skipping during extreme funding rate regimes (carry dominates)."
    )


def extract_perry_kaufman(extractor: StrategyExtractor):
    """Extract strategies from Perry Kaufman - Trading Systems and Methods"""
    print("\n=== PERRY KAUFMAN: Trading Systems and Methods ===")
    
    # Strategy 1: KAMA (Kaufman Adaptive Moving Average)
    extractor.add_strategy(
        name="KAMA Trend Following",
        source="Trading Systems and Methods",
        author="Perry Kaufman",
        strategy_type="trend",
        timeframe="4h-Daily",
        holding_period="Days to weeks",
        crypto_applicability="HIGH",
        crypto_reason="Adapts to changing volatility—critical for crypto's regime shifts",
        entry_rules=[
            "Calculate Efficiency Ratio (ER): ER = |close_today - close_n_days_ago| / sum(|daily_changes|) over n days",
            "ER measures trendiness: ER=1 (perfect trend), ER=0 (random walk)",
            "Smoothing Constant (SC): SC = [ER * (fast_SC - slow_SC) + slow_SC]^2",
            "KAMA_today = KAMA_yesterday + SC * (price - KAMA_yesterday)",
            "**Long**: Price crosses above KAMA AND ER > 0.3 (trending market)",
            "**Short**: Price crosses below KAMA AND ER > 0.3"
        ],
        exit_rules=[
            "Exit when price crosses KAMA in opposite direction",
            "OR exit when ER drops < 0.2 (market turned choppy)",
            "Stop loss: 2 ATR from entry (optional, KAMA itself is dynamic stop)"
        ],
        filters=[
            "ER > 0.3 (minimum trendiness; below this is noise)",
            "Skip if ATR < 1% of price (low volatility = poor trend)",
            "Avoid during consolidation: require price outside 20-day Bollinger Bands at entry"
        ],
        parameters={
            "er_period": "10",
            "fast_sc": "2 (fast EMA constant = 2/(2+1))",
            "slow_sc": "30 (slow EMA constant = 2/(30+1))",
            "min_er": "0.3",
            "exit_er": "0.2",
            "stop_atr": "2.0"
        },
        why_works="KAMA adapts smoothing based on market efficiency. In trends (high ER), KAMA moves fast (low lag). In chop (low ER), KAMA moves slow (filters noise). This is SMARTER than fixed MA—it adjusts to regime automatically.",
        implementation_notes="Crypto-adapt: Use 4h or daily bars. Backtest multiple ER periods (10, 20, 30). Can combine with volume filter (entry only if volume >1.5x avg). KAMA alone may lag—consider using ER as position sizing factor (higher ER = larger position)."
    )
    
    # Strategy 2: ATR Channel Breakout
    extractor.add_strategy(
        name="ATR Channel Breakout",
        source="Trading Systems and Methods",
        author="Perry Kaufman",
        strategy_type="breakout",
        timeframe="Daily",
        holding_period="1-2 weeks",
        crypto_applicability="HIGH",
        crypto_reason="Volatility-adjusted channels adapt to crypto's changing vol regimes",
        entry_rules=[
            "Calculate ATR(14)",
            "Upper channel: 20-day SMA + 2.5 * ATR",
            "Lower channel: 20-day SMA - 2.5 * ATR",
            "**Long**: Close breaks above upper channel",
            "**Short**: Close breaks below lower channel",
            "Confirm with volume >1.3x average"
        ],
        exit_rules=[
            "Exit when price crosses back through 20-day SMA (mean reversion)",
            "Stop: opposite channel (e.g., long stop = lower channel)",
            "Time stop: 10 days"
        ],
        filters=[
            "Skip if prior 5 bars are all inside channel (consolidation, not breakout)",
            "Avoid if ATR < 2% of price (insufficient volatility)",
            "Require ADX > 20 (trending market)"
        ],
        parameters={
            "ma_period": "20",
            "atr_period": "14",
            "atr_mult": "2.5",
            "volume_mult": "1.3",
            "min_atr_pct": "2%",
            "min_adx": "20"
        },
        why_works="Channels are volatility-adjusted (not static Bollinger Bands). Kaufman's insight: ATR-based channels are more robust across different markets. Breakout + volume + ADX = multi-factor confirmation, not just price.",
        implementation_notes="Crypto-adapt: Daily or 4h bars. Backtest ATR multiplier (2.0-3.0). Can use EMA instead of SMA for faster adaptation. Consider combining with funding rate filter (avoid breakouts into extreme funding)."
    )
    
    # Strategy 3: Swing Index Trading
    extractor.add_strategy(
        name="Welles Wilder Swing Index",
        source="Trading Systems and Methods",
        author="Perry Kaufman (explaining Wilder)",
        strategy_type="momentum/swing",
        timeframe="Daily",
        holding_period="3-10 days",
        crypto_applicability="MEDIUM",
        crypto_reason="Complex calculation; may be noisy in 24/7 crypto markets, but captures swing extremes",
        entry_rules=[
            "Calculate Swing Index (SI): SI = 50 * (Cy - C + 0.5*(Cy - Oy) + 0.25*(C - O)) / R * K/T",
            "Where: C=close, O=open, H=high, L=low, y=yesterday, R=true range, K=max(|H-Cy|, |L-Cy|), T=limit move (use 3*ATR)",
            "Accumulate SI into Accumulation Swing Index (ASI)",
            "**Long**: ASI crosses above previous swing high (bullish swing reversal)",
            "**Short**: ASI crosses below previous swing low (bearish swing reversal)"
        ],
        exit_rules=[
            "Exit when ASI crosses back through zero (swing exhausted)",
            "Stop: 2 ATR from entry",
            "Time stop: 10 days"
        ],
        filters=[
            "Only trade when ASI diverges from price (e.g., price makes new low but ASI doesn't = bullish divergence)",
            "Skip if ATR < 1.5% (low volatility)",
            "Require trending market: ADX > 15"
        ],
        parameters={
            "limit_move_atr": "3.0",
            "stop_atr": "2.0",
            "min_atr_pct": "1.5%",
            "min_adx": "15"
        },
        why_works="SI normalizes intraday swings by range and volatility—captures swing momentum better than raw price. ASI accumulation shows underlying momentum shifts. Divergences signal exhaustion (like RSI divergence but more sophisticated).",
        implementation_notes="Crypto-adapt: Daily bars only (open/high/low/close needed). Complex calculation—test in Python first. May be too noisy on 4h. Consider as confirmation indicator for other strategies (e.g., enter NR4 breakout only if ASI confirms)."
    )


def extract_tushar_chande(extractor: StrategyExtractor):
    """Extract strategies from Tushar Chande - The New Technical Trader"""
    print("\n=== TUSHAR CHANDE: The New Technical Trader ===")
    
    # Strategy 1: CMO (Chande Momentum Oscillator) Divergence
    extractor.add_strategy(
        name="CMO Divergence Trading",
        source="The New Technical Trader",
        author="Tushar Chande",
        strategy_type="mean_reversion/momentum",
        timeframe="4h-Daily",
        holding_period="2-7 days",
        crypto_applicability="HIGH",
        crypto_reason="Divergences are structural signals of momentum exhaustion; works across asset classes",
        entry_rules=[
            "Calculate CMO: CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down) over N periods (typically 14)",
            "Identify divergence: price makes new high/low but CMO doesn't",
            "**Bullish divergence**: Price makes lower low, CMO makes higher low → buy on next bar if CMO > -50",
            "**Bearish divergence**: Price makes higher high, CMO makes lower high → sell on next bar if CMO < +50",
            "Confirm with volume expansion (>1.2x avg) on reversal bar"
        ],
        exit_rules=[
            "Exit when CMO crosses zero (momentum shift complete)",
            "Stop: beyond recent swing extreme by 1.5 ATR",
            "Time stop: 7 days"
        ],
        filters=[
            "Divergence must span at least 3-5 bars (avoid micro-divergences)",
            "Skip if ATR < 2% of price (low volatility = weak reversals)",
            "Require initial CMO extreme: |CMO| > 40 at divergence start (meaningful overbought/oversold)"
        ],
        parameters={
            "cmo_period": "14",
            "cmo_entry_threshold": "-50 (long) / +50 (short)",
            "cmo_extreme": "40",
            "volume_mult": "1.2",
            "stop_atr": "1.5",
            "max_holding": "7 days"
        },
        why_works="CMO is more responsive than RSI (uses raw momentum, not smoothed). Divergences signal momentum exhaustion BEFORE price reversal—leading indicator. Requires multi-bar confirmation, reducing false signals.",
        implementation_notes="Crypto-adapt: Use 4h or daily. CMO is NOT in standard libraries—code it from scratch: CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down). Backtest on major coins (BTC, ETH, SOL). Consider combining with funding rate (extreme funding + divergence = high-conviction entry)."
    )
    
    # Strategy 2: VIDYA (Variable Index Dynamic Average)
    extractor.add_strategy(
        name="VIDYA Adaptive Trend",
        source="The New Technical Trader",
        author="Tushar Chande",
        strategy_type="trend",
        timeframe="Daily",
        holding_period="1-3 weeks",
        crypto_applicability="HIGH",
        crypto_reason="Adaptive smoothing handles crypto's volatile regime shifts better than fixed MAs",
        entry_rules=[
            "Calculate VIDYA: VIDYA = α * CMO_abs * price + (1 - α * CMO_abs) * VIDYA_prev",
            "Where α = 2/(N+1) (EMA constant), CMO_abs = |CMO| / 100 (volatility adjustment)",
            "**Long**: Price crosses above VIDYA AND CMO > 20 (momentum confirmation)",
            "**Short**: Price crosses below VIDYA AND CMO < -20"
        ],
        exit_rules=[
            "Exit when price crosses VIDYA in opposite direction",
            "OR exit when CMO crosses zero (momentum fades)",
            "Stop: 2 ATR from entry"
        ],
        filters=[
            "|CMO| > 20 at entry (minimum momentum)",
            "Skip if ATR < 1% of price (low volatility)",
            "Avoid during consolidation: require price >2% from VIDYA at entry (clear break, not whipsaw)"
        ],
        parameters={
            "vidya_period": "14",
            "cmo_period": "14",
            "min_cmo": "20",
            "stop_atr": "2.0",
            "min_atr_pct": "1%"
        },
        why_works="VIDYA adjusts smoothing based on CMO (momentum). In strong trends (high |CMO|), VIDYA tracks price closely (low lag). In chop (low |CMO|), VIDYA smooths heavily (filters noise). Smarter than KAMA because it uses momentum, not just efficiency.",
        implementation_notes="Crypto-adapt: Daily bars. VIDYA is NOT in standard libraries—code it. Backtest CMO periods (9, 14, 20). Can combine with volume filter. Consider using VIDYA as trailing stop (exit when price closes below VIDYA by 1%)."
    )
    
    # Strategy 3: Aroon Oscillator
    extractor.add_strategy(
        name="Aroon Oscillator Trend Entry",
        source="The New Technical Trader",
        author="Tushar Chande",
        strategy_type="trend",
        timeframe="4h-Daily",
        holding_period="1-2 weeks",
        crypto_applicability="HIGH",
        crypto_reason="Identifies trend initiation (not just continuation); crypto trends can be explosive",
        entry_rules=[
            "Calculate Aroon Up: 100 * (N - periods_since_N_day_high) / N",
            "Calculate Aroon Down: 100 * (N - periods_since_N_day_low) / N",
            "Aroon Oscillator: Aroon Up - Aroon Down",
            "**Long**: Aroon Oscillator crosses above +50 (strong uptrend starting)",
            "**Short**: Aroon Oscillator crosses below -50 (strong downtrend starting)",
            "Confirm: Aroon Up/Down must reach 100 within 2 bars of entry (fresh high/low)"
        ],
        exit_rules=[
            "Exit when Aroon Oscillator crosses zero (trend weakening)",
            "Stop: 2 ATR from entry",
            "Time stop: 14 days"
        ],
        filters=[
            "Skip if oscillator is between -30 and +30 (ranging market)",
            "Require ATR > 1.5% of price (volatility for trend)",
            "Avoid if prior 3 bars are all inside bars (consolidation, not breakout)"
        ],
        parameters={
            "aroon_period": "25",
            "osc_entry_threshold": "50",
            "osc_exit_threshold": "0",
            "stop_atr": "2.0",
            "min_atr_pct": "1.5%"
        },
        why_works="Aroon identifies NEW trends (time since high/low), not just existing trends (like MACD). Catches trend initiation earlier than lagging MAs. Oscillator crossing ±50 is a strong signal—not just overbought/oversold, but directional commitment.",
        implementation_notes="Crypto-adapt: Use 4h or daily. Aroon period 25 is standard, but backtest 14, 25, 50. Works well on coins with strong directional moves (SOL, AVAX). Can combine with volume breakout (Aroon + volume spike = high-conviction entry)."
    )


def main():
    """Main extraction workflow"""
    print("="*80)
    print("STRATEGY EXTRACTION FROM TRADING BOOKS")
    print("="*80)
    
    output_path = "/Users/ffv_macmini/Desktop/maestro/backend/research/book_strategies.md"
    extractor = StrategyExtractor(output_path)
    
    # Extract from priority books (1-6)
    extract_larry_williams(extractor)
    extract_toby_crabel(extractor)
    extract_ernie_chan(extractor)
    extract_robert_carver(extractor)
    extract_perry_kaufman(extractor)
    extract_tushar_chande(extractor)
    
    # Write output
    extractor.write_output()
    
    print("\n" + "="*80)
    print(f"EXTRACTION COMPLETE: {extractor.strategy_count} strategies extracted")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review {output_path}")
    print(f"2. Select 3-5 most promising strategies for implementation")
    print(f"3. Code strategy logic in Maestro framework")
    print(f"4. Backtest with walk-forward validation")
    print(f"5. Compare to baseline (buy & hold) and existing 12 strategies")


if __name__ == "__main__":
    main()
