#!/usr/bin/env python3
"""
Strategy Extraction Part 2 - Remaining Books (7-12)
"""

import sys
from pathlib import Path

# Import the extractor class
sys.path.insert(0, str(Path(__file__).parent))
from extract_strategies import StrategyExtractor


def extract_gary_antonacci(extractor: StrategyExtractor):
    """Extract strategies from Gary Antonacci - Dual Momentum Investing"""
    print("\n=== GARY ANTONACCI: Dual Momentum Investing ===")
    
    # Strategy 1: Dual Momentum (Absolute + Relative)
    extractor.add_strategy(
        name="Dual Momentum (Absolute + Relative)",
        source="Dual Momentum Investing",
        author="Gary Antonacci",
        strategy_type="momentum/hybrid",
        timeframe="Monthly (rebalance monthly)",
        holding_period="Weeks to months",
        crypto_applicability="HIGH",
        crypto_reason="Momentum works across timeframes; crypto has strong trending behavior; monthly rebalancing reduces transaction costs",
        entry_rules=[
            "**Absolute Momentum**: For each asset, compare current price to 12-month (252-day) ago price",
            "Asset has positive absolute momentum if: (price_now / price_12m_ago - 1) > risk_free_rate (use 0% for crypto)",
            "**Relative Momentum**: Rank all assets by 12-month return",
            "**Entry**: At month-end, buy the TOP asset that has BOTH positive absolute momentum AND highest relative momentum",
            "If no asset has positive absolute momentum, move to cash (or stablecoins)"
        ],
        exit_rules=[
            "Rebalance monthly (last day of month)",
            "Exit current position if it no longer has highest relative momentum",
            "Exit ALL positions if absolute momentum turns negative (defensive mode)"
        ],
        filters=[
            "Minimum 12-month history required (cannot calculate momentum without it)",
            "Skip assets with <$100M market cap or <$5M daily volume (liquidity)",
            "Require at least 3 assets in universe (diversification)"
        ],
        parameters={
            "lookback_period": "252 days (12 months)",
            "rebalance_frequency": "monthly",
            "risk_free_rate": "0% (or use 3-month T-bill rate)",
            "min_market_cap": "100M",
            "min_daily_volume": "5M"
        },
        why_works="Combines trend-following (absolute momentum) with relative strength (relative momentum). Dual filter reduces whipsaw: won't buy weak assets (absolute) and won't hold laggards (relative). Monthly rebalancing reduces transaction costs vs daily.",
        implementation_notes="Crypto-adapt: Use BTC, ETH, BNB, SOL, ADA, AVAX, MATIC, etc. (top 10-20 by market cap). Rebalance monthly (last day of month). Use stablecoins as 'cash' during defensive periods. Backtest with 10bps transaction costs per rebalance."
    )
    
    # Strategy 2: Accelerating Dual Momentum
    extractor.add_strategy(
        name="Accelerating Dual Momentum",
        source="Dual Momentum Investing",
        author="Gary Antonacci",
        strategy_type="momentum",
        timeframe="Monthly",
        holding_period="Weeks to months",
        crypto_applicability="HIGH",
        crypto_reason="Recent momentum often more predictive than 12-month momentum in fast-moving crypto markets",
        entry_rules=[
            "Calculate momentum scores using MULTIPLE lookbacks: 1-month, 3-month, 6-month, 12-month returns",
            "Weight recent performance higher: score = 0.4 * R_1m + 0.3 * R_3m + 0.2 * R_6m + 0.1 * R_12m",
            "Require positive absolute momentum: R_12m > 0",
            "**Entry**: Buy asset with highest weighted momentum score",
            "Rebalance monthly"
        ],
        exit_rules=[
            "Exit if asset no longer has highest momentum score",
            "Exit if 12-month return turns negative (absolute momentum filter)",
            "Rebalance monthly"
        ],
        filters=[
            "All assets must have 12-month history",
            "Minimum market cap $100M",
            "Skip if top 3 assets have similar scores (<5% difference = indecision)"
        ],
        parameters={
            "lookback_1m": "21 days",
            "lookback_3m": "63 days",
            "lookback_6m": "126 days",
            "lookback_12m": "252 days",
            "weight_1m": "0.4",
            "weight_3m": "0.3",
            "weight_6m": "0.2",
            "weight_12m": "0.1"
        },
        why_works="Recent momentum is more predictive than distant momentum (Jegadeesh & Titman). Weighted score captures acceleration (not just 12-month return). Combines persistence (12m filter) with responsiveness (1m weight).",
        implementation_notes="Crypto-adapt: Monthly rebalancing. Use top 10-20 coins. Backtest weight optimization (may need to adjust for crypto's faster dynamics). Consider adding volume factor (penalize coins with declining volume)."
    )


def extract_satchell(extractor: StrategyExtractor):
    """Extract strategies from Satchell - Market Momentum"""
    print("\n=== SATCHELL: Market Momentum ===")
    
    # Strategy 1: Time-Series Momentum (TSMOM)
    extractor.add_strategy(
        name="Time-Series Momentum (TSMOM)",
        source="Market Momentum",
        author="Satchell et al.",
        strategy_type="momentum",
        timeframe="Daily (monthly rebalance)",
        holding_period="1 month",
        crypto_applicability="HIGH",
        crypto_reason="TSMOM is one of the most robust strategies across asset classes; works on crypto futures/perps",
        entry_rules=[
            "Calculate excess return: R_t = (price_t / price_{t-12m}) - 1",
            "**Position direction**: sign(R_t) → if R_t > 0, go long; if R_t < 0, go short",
            "**Position size**: volatility-scaled to target X% annualized volatility (e.g., 10%)",
            "Position = target_vol / realized_vol * sign(R_t)",
            "Rebalance monthly"
        ],
        exit_rules=[
            "Rebalance monthly: recalculate R_t and adjust position",
            "No intra-month exits (buy & hold for month)",
            "No fixed stop—volatility scaling handles risk"
        ],
        filters=[
            "Minimum 12-month history required",
            "Skip if realized volatility > 150% annualized (regime break)",
            "Apply to portfolio of 5-10 coins (diversification critical)"
        ],
        parameters={
            "lookback_period": "12 months (252 days)",
            "rebalance_freq": "monthly",
            "target_vol": "10-15% annualized",
            "vol_lookback": "21-63 days for realized vol"
        },
        why_works="TSMOM is trend-following with volatility scaling. Works because trends persist (momentum anomaly). Volatility scaling prevents overleveraging in high-vol periods. Simple, robust, diversifiable.",
        implementation_notes="Crypto-adapt: Apply to BTC, ETH, SOL, BNB, AVAX, etc. Use perpetuals for shorting. Calculate realized vol from daily returns. Backtest with 10bps transaction costs. Can use 1m, 3m, 6m, 12m lookbacks and combine (ensemble TSMOM)."
    )
    
    # Strategy 2: Cross-Sectional Momentum (XSMOM)
    extractor.add_strategy(
        name="Cross-Sectional Momentum (XSMOM)",
        source="Market Momentum",
        author="Satchell et al.",
        strategy_type="momentum/long-short",
        timeframe="Daily (monthly rebalance)",
        holding_period="1 month",
        crypto_applicability="HIGH",
        crypto_reason="Crypto has wide dispersion in returns—ideal for relative momentum strategies",
        entry_rules=[
            "Rank all coins by 12-month return",
            "**Long**: Top 30% (winners)",
            "**Short**: Bottom 30% (losers)",
            "Equal-weight within long and short portfolios",
            "Dollar-neutral: $1 long, $1 short (market-neutral)",
            "Rebalance monthly"
        ],
        exit_rules=[
            "Rebalance monthly: re-rank and adjust portfolios",
            "No intra-month exits"
        ],
        filters=[
            "Minimum 12-month history",
            "Minimum market cap $50M (avoid illiquid coins)",
            "Skip coins with <$1M daily volume",
            "Require at least 10 coins in universe (5 long, 5 short minimum)"
        ],
        parameters={
            "lookback_period": "12 months",
            "long_pct": "30%",
            "short_pct": "30%",
            "rebalance_freq": "monthly",
            "min_market_cap": "50M",
            "min_daily_volume": "1M"
        },
        why_works="Winners continue to outperform, losers continue to underperform (momentum anomaly). Market-neutral removes beta exposure—pure alpha. Dollar-neutral reduces directional risk. Works because of behavioral biases (under-reaction, herding).",
        implementation_notes="Crypto-adapt: Use top 20-30 coins by market cap. Short via perpetuals. Backtest with realistic transaction costs (10bps). Can combine with TSMOM (trade XSMOM only when TSMOM is positive = bull market). Consider funding rate costs for shorts."
    )


def extract_nick_radge(extractor: StrategyExtractor):
    """Extract strategies from Nick Radge - Unholy Grails"""
    print("\n=== NICK RADGE: Unholy Grails ===")
    
    # Strategy 1: Bollinger Band Breakout
    extractor.add_strategy(
        name="Bollinger Band Momentum Breakout",
        source="Unholy Grails",
        author="Nick Radge",
        strategy_type="breakout/momentum",
        timeframe="Daily",
        holding_period="Weeks to months",
        crypto_applicability="HIGH",
        crypto_reason="Crypto has strong trends; BB breakouts capture trend initiation",
        entry_rules=[
            "Calculate Bollinger Bands: 20-day SMA ± 2 SD",
            "**Long**: Close > upper Bollinger Band AND close > 100-day MA (trend filter)",
            "**Short**: Close < lower Bollinger Band AND close < 100-day MA",
            "Enter next bar open"
        ],
        exit_rules=[
            "Exit when close crosses back through 20-day SMA (mean reversion)",
            "OR exit when 100-day MA flips (trend reversal)",
            "Stop loss: 3 ATR from entry (optional)",
            "Time stop: 60 days (if no trend develops)"
        ],
        filters=[
            "Trend filter: MUST be above/below 100-day MA (avoid counter-trend)",
            "Volume confirmation: volume on breakout > 1.5x 20-day average",
            "ATR > 2% of price (sufficient volatility)"
        ],
        parameters={
            "bb_period": "20",
            "bb_std": "2.0",
            "trend_ma": "100",
            "volume_mult": "1.5",
            "stop_atr": "3.0",
            "min_atr_pct": "2%"
        },
        why_works="BB breakout signals volatility expansion (not contraction like BB squeeze). Trend filter ensures we're trading WITH the big trend. This is NOT mean reversion—it's breakout in direction of trend. Different from what we tested (BB mean reversion).",
        implementation_notes="Crypto-adapt: Daily bars. Backtest on BTC, ETH, SOL. BB breakout is LONG-ONLY in Radge's book—adapt for crypto by allowing shorts below 100-day MA. Watch for false breakouts in ranging markets (use ADX filter)."
    )
    
    # Strategy 2: Weekend Gap Trader
    extractor.add_strategy(
        name="Weekend Gap Fade",
        source="Unholy Grails",
        author="Nick Radge",
        strategy_type="mean_reversion",
        timeframe="Daily",
        holding_period="1-3 days",
        crypto_applicability="LOW",
        crypto_reason="Crypto trades 24/7 (no weekend gaps); adapt to session gaps or holiday periods",
        entry_rules=[
            "Identify weekend gap: Monday open vs Friday close",
            "**Gap up**: Monday open > Friday close by >1%",
            "**Gap down**: Monday open < Friday close by >1%",
            "**Fade logic**: Gap up → short, gap down → long",
            "Enter at Monday close (after gap is confirmed)"
        ],
        exit_rules=[
            "Exit at Friday close (weekly cycle)",
            "Stop: 1.5x gap size beyond entry"
        ],
        filters=[
            "Gap must be >1% of price",
            "Skip if market is strongly trending (>5% move prior week)",
            "Volume on Monday should be above average"
        ],
        parameters={
            "min_gap_pct": "1%",
            "stop_mult": "1.5",
            "max_holding": "5 days"
        },
        why_works="Weekend gaps in traditional markets often fill due to liquidity imbalance. Fade the gap = mean reversion to equilibrium.",
        implementation_notes="Crypto-adapt: NOT directly applicable (24/7 trading). Could adapt to: (1) gaps after exchange maintenance, (2) holiday periods (Christmas, New Year), (3) funding rate resets as 'gap' proxy. Likely LOW edge in crypto—include for completeness but deprioritize."
    )


def extract_lopez_de_prado(extractor: StrategyExtractor):
    """Extract strategies from Marcos Lopez de Prado - Advances in Financial Machine Learning"""
    print("\n=== MARCOS LOPEZ DE PRADO: Advances in Financial ML ===")
    
    # Strategy 1: Triple Barrier Method
    extractor.add_strategy(
        name="Triple Barrier Labeling + ML",
        source="Advances in Financial Machine Learning",
        author="Marcos Lopez de Prado",
        strategy_type="hybrid (ML + barriers)",
        timeframe="Any (adaptive)",
        holding_period="Adaptive (based on barrier hits)",
        crypto_applicability="HIGH",
        crypto_reason="Adaptive exits handle crypto's non-stationary behavior; combines ML prediction with risk management",
        entry_rules=[
            "**Feature engineering**: Calculate features (momentum, volatility, volume, microstructure)",
            "**ML model**: Train classifier (e.g., Random Forest) to predict next barrier hit (upper, lower, time)",
            "**Entry signal**: When ML predicts P(upper barrier hit) > 0.6, go long",
            "**Position size**: Scale by prediction confidence (higher P = larger position)",
            "Define barriers at entry: upper = entry + X%, lower = entry - X%, time = T days"
        ],
        exit_rules=[
            "Exit when ANY barrier is hit:",
            "- Upper barrier (take profit)",
            "- Lower barrier (stop loss)",
            "- Time barrier (max holding period)",
            "No discretionary exits—let barriers decide"
        ],
        filters=[
            "Only trade when ML prediction confidence > 0.6 (avoid low-conviction trades)",
            "Retrain model every 30 days (avoid overfitting to old regimes)",
            "Use walk-forward validation (never train on future data)"
        ],
        parameters={
            "upper_barrier_pct": "2-5% (optimize)",
            "lower_barrier_pct": "1-3% (asymmetric, tighter stop)",
            "time_barrier_days": "5-10 (adaptive)",
            "ml_confidence_threshold": "0.6",
            "retrain_frequency": "30 days"
        },
        why_works="Triple barriers remove look-ahead bias (labels are path-dependent). ML predicts which barrier hits first (not just direction). Combines prediction with risk management. Meta-labeling (predicting bet size, not direction) can improve Sharpe.",
        implementation_notes="Crypto-adapt: Use 4h or daily bars. Features: RSI, ATR, volume ratio, funding rate, OI change, BTC correlation. Start with simple RF model. Backtest with realistic transaction costs. This is ADVANCED—requires ML infrastructure. Consider as long-term project, not immediate implementation."
    )
    
    # Strategy 2: Fractional Differentiation Momentum
    extractor.add_strategy(
        name="Fractional Differentiation for Stationarity",
        source="Advances in Financial Machine Learning",
        author="Marcos Lopez de Prado",
        strategy_type="momentum (stationary)",
        timeframe="Daily",
        holding_period="Adaptive",
        crypto_applicability="MEDIUM",
        crypto_reason="Makes non-stationary crypto prices stationary while preserving memory (better than raw returns)",
        entry_rules=[
            "Apply fractional differentiation to price series: X_t^d = Σ w_k * X_{t-k}, where d is fractional order (e.g., d=0.4)",
            "This makes series stationary while retaining momentum information",
            "Calculate z-score on fractionally differentiated series: z = (X_t^d - mean) / std",
            "**Long**: z > 1.0 (positive momentum, stationary)",
            "**Short**: z < -1.0 (negative momentum)"
        ],
        exit_rules=[
            "Exit when z crosses zero",
            "Stop: z exceeds ±3.0",
            "Time stop: 10 days"
        ],
        filters=[
            "Find optimal d via grid search (0.2-0.8) that maximizes ADF stationarity",
            "Recalculate d every 60 days (regime adaptation)",
            "Skip if ADF test p-value > 0.05 (not stationary)"
        ],
        parameters={
            "d_order": "0.4 (optimize)",
            "z_threshold": "1.0",
            "z_stop": "3.0",
            "lookback_period": "100 days",
            "recalc_d_days": "60"
        },
        why_works="Traditional ML fails on non-stationary data. Fractional differentiation makes data stationary (required for ML) while preserving long-term memory (unlike raw returns, which lose all memory). Elegant solution to stationarity vs memory trade-off.",
        implementation_notes="Crypto-adapt: Daily bars. Use fracdiff Python library. Calculate optimal d (backtest d=0.2, 0.4, 0.6, 0.8). Combine with ML model for prediction. ADVANCED—requires statistical/ML knowledge. Deprioritize vs simpler strategies."
    )


def extract_urban_jaekle(extractor: StrategyExtractor):
    """Extract strategies from Urban Jaekle - Trading Systems 2nd Edition"""
    print("\n=== URBAN JAEKLE: Trading Systems ===")
    
    # Strategy 1: Portfolio of Uncorrelated Systems
    extractor.add_strategy(
        name="Portfolio of Systems (Jaekle Approach)",
        source="Trading Systems 2nd Edition",
        author="Urban Jaekle",
        strategy_type="hybrid/portfolio",
        timeframe="Multiple (4h, daily, weekly)",
        holding_period="Varies by system",
        crypto_applicability="HIGH",
        crypto_reason="Diversification across strategies reduces risk; crypto allows 24/7 multi-system trading",
        entry_rules=[
            "Run MULTIPLE uncorrelated strategies simultaneously (e.g., trend, mean reversion, breakout)",
            "**Example portfolio**: EWMAC (trend) + OU MR (mean reversion) + ORB (breakout) + Carry (funding rate)",
            "Allocate capital equally (25% each) or by recent Sharpe ratio",
            "Each system generates independent signals",
            "Enter when INDIVIDUAL system gives signal (not combined)"
        ],
        exit_rules=[
            "Each system manages its own exits (per strategy rules)",
            "Monitor portfolio-level risk: if total portfolio drawdown > 20%, reduce all positions by 50%",
            "Rebalance monthly: adjust allocation based on rolling Sharpe ratios"
        ],
        filters=[
            "Systems must have correlation < 0.3 (diversification)",
            "Each system must have positive Sharpe > 0.5 in backtest",
            "Skip systems during regime mismatch (e.g., no mean reversion in strong trend)"
        ],
        parameters={
            "num_systems": "4-6",
            "max_correlation": "0.3",
            "min_sharpe": "0.5",
            "rebalance_freq": "monthly",
            "max_portfolio_dd": "20%"
        },
        why_works="Diversification across strategies smooths equity curve. When trend-following fails (ranging market), mean reversion profits. When breakouts fail (false signals), carry strategies provide income. Portfolio approach reduces risk more than any single strategy.",
        implementation_notes="Crypto-adapt: Combine strategies we've extracted (EWMAC, OU MR, NR4, Funding Carry). Backtest portfolio as a whole (not just individual systems). Monitor correlations monthly. This is a META-STRATEGY—implement after individual strategies are validated."
    )


def extract_adam_grimes(extractor: StrategyExtractor):
    """Extract strategies from Adam Grimes - The Art and Science of Technical Analysis"""
    print("\n=== ADAM GRIMES: The Art and Science of Technical Analysis ===")
    
    # Strategy 1: Pullback in Trend (Grimes Methodology)
    extractor.add_strategy(
        name="Trend Pullback Entry (Grimes)",
        source="The Art and Science of Technical Analysis",
        author="Adam Grimes",
        strategy_type="trend/pullback",
        timeframe="4h-Daily",
        holding_period="5-15 days",
        crypto_applicability="HIGH",
        crypto_reason="Pullbacks in crypto trends are common and tradable; better entry than breakout chasing",
        entry_rules=[
            "Identify trend: price > 50-day EMA (uptrend) or price < 50-day EMA (downtrend)",
            "Wait for pullback: price retraces to 20-day EMA (in uptrend) or rallies to 20-day EMA (in downtrend)",
            "**Long setup**: Uptrend + pullback to 20 EMA + bullish rejection bar (close in top 50% of range, near 20 EMA)",
            "**Short setup**: Downtrend + rally to 20 EMA + bearish rejection bar",
            "Enter next bar after rejection bar confirmation"
        ],
        exit_rules=[
            "Exit when trend breaks: close below 50 EMA (long) or above 50 EMA (short)",
            "Stop loss: 1.5 ATR beyond pullback low/high",
            "Profit target: prior swing high/low (or 3:1 reward/risk)",
            "Time stop: 15 days if no momentum"
        ],
        filters=[
            "ADX > 20 (trending market, not ranging)",
            "Pullback must reach 20 EMA (not too shallow, not too deep)",
            "Rejection bar must close within 20% of 20 EMA (clear support/resistance test)"
        ],
        parameters={
            "trend_ema": "50",
            "pullback_ema": "20",
            "stop_atr": "1.5",
            "min_adx": "20",
            "max_holding": "15 days"
        },
        why_works="Buying pullbacks in trends offers better risk/reward than chasing breakouts. Trend is your friend, pullback is your entry. Grimes emphasizes CONTEXT (trend) + TRIGGER (rejection bar)—not just patterns. Multi-timeframe structure (50 EMA trend, 20 EMA pullback).",
        implementation_notes="Crypto-adapt: Use 4h or daily bars. Backtest EMA periods (20/50, 10/30, 50/100). Combine with volume (volume should decline on pullback, expand on resumption). Consider adding RSI filter (buy pullbacks when RSI 30-50 in uptrend)."
    )
    
    # Strategy 2: Failure Test (Anti-Pattern)
    extractor.add_strategy(
        name="Failure Test (Failed Breakout Fade)",
        source="The Art and Science of Technical Analysis",
        author="Adam Grimes",
        strategy_type="mean_reversion/anti-breakout",
        timeframe="4h-Daily",
        holding_period="3-7 days",
        crypto_applicability="MEDIUM",
        crypto_reason="Crypto has many false breakouts (leverage, stop hunts); fading failures can be profitable",
        entry_rules=[
            "Identify key level: recent swing high/low or range boundary",
            "**Breakout attempt**: Price breaks above/below level by >0.5%",
            "**Failure**: Price reverses back within 2 bars, closing below breakout level (for upside breakout)",
            "**Entry**: Short on failure bar close (for failed upside BO) or long (for failed downside BO)",
            "Confirm with increased volume on failure bar (>1.5x avg)"
        ],
        exit_rules=[
            "Target: opposite side of range or prior swing extreme",
            "Stop: 1 ATR beyond breakout high/low (tight stop—failure confirmed)",
            "Time stop: 7 days",
            "Exit if breakout is re-attempted (level breaks again)"
        ],
        filters=[
            "Breakout must be genuine (>0.5% beyond level, not just a wick)",
            "Failure must be decisive (close back within range, not just hovering)",
            "Volume on failure bar > 1.5x average (selling pressure for failed upside BO)",
            "Avoid during strong trends (failure test works in ranges, not trends)"
        ],
        parameters={
            "breakout_threshold": "0.5%",
            "failure_bars": "2",
            "volume_mult": "1.5",
            "stop_atr": "1.0",
            "max_holding": "7 days"
        },
        why_works="Failed breakouts signal exhaustion—bulls/bears tried to push through but failed. Fade the failure = trade against the weak hands. Grimes calls this 'anti-pattern'—trading the FAILURE of a pattern, not the pattern itself. Requires range-bound market (not trending).",
        implementation_notes="Crypto-adapt: Use 4h or daily. Identify key levels via swing highs/lows or horizontal S/R. Watch for stop hunts (common in crypto)—failed breakouts after stop hunt are high-probability fades. Combine with funding rate (extreme funding + failed breakout = strong signal)."
    )


def main():
    """Main extraction workflow for Part 2"""
    print("="*80)
    print("STRATEGY EXTRACTION PART 2: Books 7-12")
    print("="*80)
    
    output_path = "/Users/ffv_macmini/Desktop/maestro/backend/research/book_strategies.md"
    
    # Load existing strategies (count will continue from 18)
    extractor = StrategyExtractor(output_path)
    extractor.strategy_count = 18  # Start from where Part 1 left off
    
    # Read existing content
    if Path(output_path).exists():
        with open(output_path, 'r') as f:
            existing = f.read()
        extractor.strategies = [existing]  # Prepend existing content
    
    # Extract from books 7-12
    extract_gary_antonacci(extractor)
    extract_satchell(extractor)
    extract_nick_radge(extractor)
    extract_lopez_de_prado(extractor)
    extract_urban_jaekle(extractor)
    extract_adam_grimes(extractor)
    
    # Write combined output
    extractor.write_output()
    
    print("\n" + "="*80)
    print(f"PART 2 COMPLETE: Total {extractor.strategy_count} strategies")
    print("="*80)


if __name__ == "__main__":
    main()
