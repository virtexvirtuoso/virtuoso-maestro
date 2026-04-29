# V5 AND V6 STRATEGY EVOLUTION ROADMAP

**Version:** 1.0
**Date:** 2026-02-13
**Author:** Maestro Quant System
**Status:** PLANNING

---

## Current State Summary

V3 Macro Momentum is the production baseline. The numbers that matter:

| System | IS Sharpe | OOS Sharpe | MaxDD | Key Edge |
|--------|-----------|------------|-------|----------|
| V3 Mega | 1.71 | 0.84 | -5.5% | M2 acceleration (102.8% vs 2.1% annual BTC returns) |
| V4 Multi-Module | 1.70 | -- | -30.4% | Only FF Bridge added OOS value (+3.66%) |
| ML Regime | -- | 1.33 | -- | LightGBM classifier; top SHAP: m2_accel, cpi_yoy |

Production recommendation from V4 research: V3 + ML Regime + FF Bridge. Everything else was complexity without payoff.

**Existing data sources:** Coinalyze derivatives (65 tokens), FRED macro, Fama-French factors, yfinance, Alpha Vantage, BTC Wiz on-chain (27 signals), Whale Hunter (elite trader tracking, 57% WR, 102 traders).

**Known gaps:** No options/vol surface data, no intraday data, no order flow/microstructure, no stablecoin supply tracking.

**Lesson from V4:** Adding modules is easy. Adding OOS alpha is hard. Every V5/V6 component must survive walk-forward validation or it gets cut. Complexity has a tax -- more parameters, more overfitting surface, more maintenance burden.

---

# V5 -- "DEEP LIQUIDITY" (3-6 MONTH HORIZON)

The thesis: V3 proved that macro liquidity drives crypto. V5 goes deeper into the liquidity stack -- options markets, on-chain flows, stablecoin supply -- to get earlier and more granular regime signals. The ML layer evolves from separate models into a unified ensemble.

---

## V5.1: Options-Implied Volatility Surface (Deribit)

### Edge Hypothesis

The options market prices forward-looking risk. The vol surface -- specifically the skew (put-call IV differential), term structure slope, and smile curvature -- encodes institutional positioning and fear/greed faster than spot or perpetual markets. A steep negative skew (puts expensive relative to calls) often precedes sell-offs by 24-72 hours. Term structure inversion (near-term IV > far-term IV) signals imminent volatility events. These signals are orthogonal to M2 acceleration and should improve regime detection, particularly around transitions.

Deribit controls roughly 85% of crypto options volume. Their API is free and returns full order book, trades, and instrument data (confirmed: API responsive, instruments active as of 2026-02-13). BTC and ETH options have liquid markets; SOL options exist but are thinner.

### Data Sources

- **Needed:** Deribit public API (`/public/get_book_summary_by_currency`, `/public/get_order_book`, `/public/ticker`). Free, no API key required for public endpoints. Rate limit: 20 req/sec non-authenticated.
- **Already have:** Nothing. This is net-new infrastructure.
- **Historical:** Deribit does not provide free historical vol surface data. Options: (a) start collecting now and build history over 3-6 months, (b) purchase from Tardis.dev (~$200-500 for BTC options history), (c) Kaiko or Amberdata subscriptions ($500-2000/month -- likely overkill at this stage).

### Derived Signals

| Signal | Calculation | Interpretation |
|--------|-------------|---------------|
| 25-delta skew | IV(25d put) - IV(25d call), interpolated from surface | Negative = bearish hedging demand |
| Term structure slope | IV(7d ATM) - IV(30d ATM) | Positive (inverted) = near-term stress |
| ATM IV level | At-the-money IV for 30d expiry | Absolute vol expectation |
| Put-call volume ratio | Daily put volume / call volume | Sentiment gauge |
| Vol-of-vol | Rolling stdev of ATM IV changes | Regime instability |

### Development Time

- Data collector + vol surface builder: 2 weeks
- Signal derivation + integration into regime classifier: 1 week
- Historical data acquisition + backtesting: 2 weeks (depends on data source)
- Walk-forward validation: 1 week
- **Total: 4-6 weeks**

### Expected Impact

- Sharpe improvement: +0.05 to +0.15 (moderate confidence). Vol surface signals are well-documented in traditional finance but less tested in crypto. The signal is real but may be partially captured by existing rvol_60d feature in ML model.
- Primary value: better regime transition detection, reducing drawdown during sharp reversals.
- Secondary value: position sizing refinement -- scale down when vol surface signals stress.

### Dependencies

- None for data collection (standalone).
- ML Regime Classifier (V5.5) for integration.

### Overfitting Risk: MEDIUM

The vol surface produces many features (skew at multiple deltas, term structure at multiple tenors). Temptation to mine 20+ features and find spurious correlations is high. Mitigation: restrict to 3-4 aggregate signals (25d skew, term structure slope, ATM IV, put-call ratio). Use SHAP importance to validate. Require OOS improvement in walk-forward.

### Skepticism Rating: 7/10 (likely to add value)

Options markets genuinely contain forward-looking information. The question is whether the signal is large enough relative to what M2 + macro already captures. In traditional markets, vol surface signals are well-established alpha sources. In crypto, the options market is newer but Deribit is liquid enough for BTC/ETH. This is one of the higher-probability additions.

---

## V5.2: On-Chain Flow Integration

### Edge Hypothesis

Exchange inflows predict selling pressure (coins moving to exchanges to be sold). Exchange outflows predict accumulation (coins moving to cold storage). Stablecoin inflows to exchanges predict buying pressure (dry powder arriving). Whale wallet clustering reveals large holder behavior before it hits order books.

We already have BTC Wiz (27 signals including MVRV, RHODL, NVT) and Whale Hunter (elite trader tracking). The gap is real-time exchange flow data and stablecoin flow tracking.

### Data Sources

- **Already have:** BTC Wiz 27 signals (RHODL, MVRV, NVT, Puell Multiple, etc.), Whale Hunter (102 tracked traders, 57% WR, HyperLiquid + AsterDex positions).
- **Needed:** Exchange flow data. Options:
  - CryptoQuant API: exchange flows, whale alerts, fund flows. $30-100/month for basic tier. Best option.
  - Glassnode: comprehensive on-chain. $40-800/month. Overkill if CryptoQuant covers flows.
  - Nansen: wallet labels, smart money tracking. $100-1000/month.
  - Free alternative: parse Whale Alert bot data (limited, noisy).

### Derived Signals

| Signal | Source | Interpretation |
|--------|--------|---------------|
| Net exchange flow (BTC) | CryptoQuant / Glassnode | Negative = accumulation, bullish |
| Exchange stablecoin reserves | CryptoQuant | Rising = dry powder, bullish |
| Whale deposit count (>100 BTC) | CryptoQuant | Spike = sell pressure incoming |
| MVRV Z-Score | BTC Wiz (existing) | >7 = overheated, <-0.5 = undervalued |
| Whale Hunter consensus | Whale Hunter (existing) | Directional agreement among elite traders |
| Active addresses momentum | CryptoQuant / on-chain | Network growth rate |

### Development Time

- CryptoQuant API integration: 1 week
- Signal pipeline (daily aggregation): 1 week
- Merge with existing BTC Wiz + Whale Hunter data: 1 week
- Backtesting + walk-forward: 2 weeks
- **Total: 4-5 weeks**

### Expected Impact

- Sharpe improvement: +0.0 to +0.10 (low-moderate confidence). On-chain signals are widely followed in crypto, which means much of the alpha is already priced in. The edge, if any, is in combining on-chain with macro (which few do systematically).
- Primary value: confirmation signal for regime classifier. Reduce false signals when macro says BULL but on-chain says distribution.

### Dependencies

- CryptoQuant or equivalent subscription ($30-100/month ongoing cost).
- ML Regime Classifier (V5.5) for integration.

### Overfitting Risk: MEDIUM-HIGH

On-chain metrics are notoriously easy to overfit in backtests because major market tops (2017, 2021) have strong on-chain signatures that are obvious in hindsight. Forward-looking value is less clear. MVRV "worked" historically but the relationship may shift as market structure matures.

### Skepticism Rating: 5/10 (might add value, might not)

On-chain analytics have become a crowded trade. Every crypto fund monitors MVRV, exchange flows, whale wallets. The information advantage from having this data has diminished since 2020-2021. The best case is that combining on-chain with macro (which is uncommon) creates a novel composite signal. The worst case is that it adds features that look good in-sample but contribute nothing OOS. We already have BTC Wiz with 27 signals -- the marginal value of adding exchange flows may be small. Start with 2-3 flow signals only.

---

## V5.3: Cross-Exchange Microstructure

### Edge Hypothesis

Funding rate differentials across exchanges (Bybit, Binance, HyperLiquid, dYdX) reveal relative positioning imbalances. When funding diverges significantly between venues, it signals localized leverage that will mean-revert. Basis trading (spot-perp spread, spot-futures spread) offers a market-neutral yield during low-conviction regimes. Cross-exchange order book depth analysis can predict short-term supply/demand imbalances.

### Data Sources

- **Already have:** Coinalyze derivatives data (funding rates, OI, liquidations for 65 tokens across exchanges). This is the foundation.
- **Needed:**
  - Real-time funding rate API access per exchange (free via CCXT for Bybit, Binance, HyperLiquid).
  - Order book snapshots for depth analysis (free via exchange APIs, but storage-intensive).
  - Basis calculation requires both spot and perp/futures prices (available via yfinance + exchange APIs).

### Derived Signals

| Signal | Calculation | Use |
|--------|-------------|-----|
| Funding rate z-score | Current vs 30d rolling mean/stdev, per exchange | Extreme = contrarian entry |
| Cross-exchange funding spread | Max(funding) - Min(funding) across venues | Arbitrage opportunity |
| Basis annualized | (Futures - Spot) / Spot * 365 / DTE | Market-neutral yield, sentiment gauge |
| OI-weighted funding | Funding rate * OI share per exchange | True aggregate sentiment |
| Liquidation cascade probability | OI concentration near current price | Risk of forced selling |

### Development Time

- Multi-exchange funding rate collector: 1 week (CCXT-based)
- Basis calculator: 3 days
- Signal aggregation + z-scoring: 1 week
- Backtesting with Coinalyze historical data: 1 week
- Walk-forward validation: 1 week
- **Total: 4-5 weeks**

### Expected Impact

- For directional signals: Sharpe improvement +0.0 to +0.08. Funding rate extremes are a known contrarian signal but the timing is noisy.
- For basis trading (market-neutral): 10-30% APY in normal markets, potentially higher during vol. This is not a Sharpe improvement but a separate return stream.
- Primary value: the basis/funding arbitrage module provides returns during NEUTRAL regime when V3 is largely flat.

### Dependencies

- CCXT library (already available in environment).
- For basis trading execution: requires simultaneous spot + perp positions, needs exchange integration (Tier 1 dependency).

### Overfitting Risk: LOW (for basis trading), MEDIUM (for directional signals)

Basis trading is a mechanical arbitrage -- less overfitting risk. Funding rate as a directional signal is noisier and has a well-documented history of mean-reverting slowly (you can be right on direction but wrong on timing for weeks).

### Skepticism Rating: 6/10 (basis trading is real; directional funding signals are noisy)

The basis trade is genuine alpha that requires no prediction -- just execution infrastructure. The funding rate directional signal is real but timing is poor. Recommend building the basis/funding data pipeline but prioritizing the market-neutral basis trade over directional funding signals. The basis trade also serves as a natural hedge during NEUTRAL regimes, filling a gap in V3.

---

## V5.4: Stablecoin Supply as Real-Time M2 Proxy

### Edge Hypothesis

M2 acceleration is THE edge (BTC 102.8%/yr vs 2.1%). But FRED M2 data is published monthly with a 2-week lag. Stablecoin market cap (USDT + USDC + DAI + FDUSD) changes are available daily and serve as a real-time proxy for crypto-specific liquidity inflows. When stablecoin supply expands, it is functionally equivalent to "crypto M2" growing -- fresh capital entering the ecosystem. This could give a 2-4 week lead on the FRED M2 signal.

### Data Sources

- **Already have:** M2 data from FRED (monthly, lagged). DXY, 10Y, HYG from yfinance (daily).
- **Needed:**
  - CoinGecko API: stablecoin market caps (free tier: 30 calls/min). Daily resolution.
  - DeFiLlama API: stablecoin supply by chain (free, no key required). Daily resolution.
  - Alternative: CryptoCompare, CoinMarketCap APIs.

### Derived Signals

| Signal | Calculation | Interpretation |
|--------|-------------|---------------|
| Stablecoin supply 30d change | (Current total - 30d ago) / 30d ago | Crypto liquidity acceleration |
| USDT dominance change | USDT share of total stablecoin supply, 7d delta | Capital rotation signal |
| Stablecoin-to-BTC ratio | Total stablecoin mcap / BTC mcap | Dry powder ratio |
| Supply acceleration | 2nd derivative of stablecoin supply (change of change) | Leading indicator |

### Development Time

- API integration (CoinGecko/DeFiLlama): 3 days
- Signal calculation + historical backfill: 3 days
- Integration with M2 signal (complement, not replace): 3 days
- Walk-forward validation: 1 week
- **Total: 3 weeks**

### Expected Impact

- Sharpe improvement: +0.05 to +0.20 (moderate-high confidence). This is essentially a faster version of the already-proven M2 signal. If the correlation between stablecoin supply changes and subsequent BTC returns is even half as strong as M2 acceleration, the timing improvement alone could meaningfully reduce drawdowns at regime transitions.
- Reduces the 2-4 week lag on M2 data, which is the single biggest weakness of V3.

### Dependencies

- None. Standalone data pipeline.
- Enhances existing M2 Acceleration signal in V3 confluence.

### Overfitting Risk: LOW

The hypothesis is a direct extension of the proven M2 edge. Stablecoin supply is a simple, low-dimensional signal. Hard to overfit a single time series.

### Skepticism Rating: 8/10 (most likely to work)

This is the highest-conviction V5 addition. It directly addresses the biggest weakness of V3 (M2 data lag) with a mechanistically sound proxy. Stablecoin supply growth literally is crypto liquidity injection. The only risk is that the correlation weakens during periods when stablecoin growth is driven by non-investment use cases (DeFi farming cycles, for example). Start here.

---

## V5.5: ML Ensemble Evolution

### Edge Hypothesis

V3's ML layer consists of three separate models: LightGBM regime classifier, XGBoost signal weighter, and entry timer. They operate independently. Stacking them into a unified ensemble -- where the regime classifier's output feeds the signal weighter, which feeds the entry timer -- should produce better-calibrated signals. Adding reinforcement learning (RL) for position sizing could optimize the leverage curve beyond the current static mapping (score 5 -> 2x, score 4 -> 1.5x, etc.).

### Current ML State

| Model | Type | OOS Performance | Top Features (SHAP) |
|-------|------|-----------------|---------------------|
| Regime Classifier | LightGBM | Sharpe 1.33 | m2_accel, cpi_yoy, Mkt-RF, rvol_60d, RMW |
| Signal Weighter | XGBoost | Not independently validated | -- |
| Entry Timer | Gradient Boost | Not independently validated | -- |

### Proposed Architecture

```
[Market Data + Macro + On-Chain + Vol Surface]
              |
              v
    [Feature Engineering Layer]
              |
              v
    [Regime Classifier (LightGBM)]
         |         |
    regime_prob  regime_label
         |         |
         v         v
    [Signal Weighter (XGBoost)]
         receives regime context
              |
              v
    [Entry Timer]
         receives signal weights + regime
              |
              v
    [RL Position Sizer]
         receives all above + portfolio state
              |
              v
    [Final Signal: direction, size, timing]
```

### Development Plan

1. **Unified pipeline** (2 weeks): Chain models so outputs flow downstream. Retrain with proper cross-validation (purged k-fold to prevent leakage).
2. **Feature expansion** (1 week): Add V5.1-V5.4 signals as features. Let SHAP determine which matter.
3. **RL position sizer** (3-4 weeks): Train PPO or SAC agent on historical episode data. Reward function: risk-adjusted returns (Sharpe-like). State space: regime probabilities, signal weights, current position, portfolio drawdown, vol level.
4. **Walk-forward validation** (2 weeks): Full walk-forward with expanding window. Each fold must show the ensemble outperforms the simple V3 heuristic.

### Development Time

- Unified pipeline: 2 weeks
- Feature expansion: 1 week
- RL position sizer: 3-4 weeks
- Validation: 2 weeks
- **Total: 8-9 weeks**

### Expected Impact

- Unified ensemble: Sharpe improvement +0.05 to +0.15 (moderate confidence). Stacking well-calibrated models generally outperforms individual models.
- RL position sizer: Sharpe improvement +0.0 to +0.20 (low confidence). RL in finance has a poor track record outside of execution optimization. The reward function is noisy, episodes are short, and overfitting is severe. This is the component most likely to look great in backtest and fail live.

### Dependencies

- V5.1-V5.4 signals as input features (can start without, add incrementally).
- Significant compute for RL training (GPU recommended, 2-5 days of training).

### Overfitting Risk: HIGH

This is the most dangerous V5 component. More model complexity = more ways to overfit. The RL position sizer is particularly risky: with ~300 monthly observations over 6 years, there are approximately 72 "episodes" for training -- far too few for robust RL. Mitigation: (a) aggressive regularization, (b) ensemble must beat simple V3 heuristic in every OOS fold to be accepted, (c) RL position sizer gets a separate kill switch -- if it underperforms the static leverage map in live trading for 3 months, revert.

### Skepticism Rating: 5/10 (ensemble stacking: likely helpful; RL sizing: probably won't work)

Split verdict. The unified pipeline and proper stacking will almost certainly be a small improvement -- it is a well-understood ML technique. The RL position sizer is the "sounds impressive on paper" component. In practice, RL requires millions of episodes to converge, and financial time series provide hundreds at most. Build the ensemble stack first. Defer RL until the ensemble is validated and there is evidence of suboptimal position sizing in live trading.

---

## V5.6: Sentiment NLP

### Edge Hypothesis

Crypto Twitter (X) sentiment extremes are contrarian signals. When sentiment is unanimously bullish, the market is overextended. When sentiment is maximally fearful, bottoms form. NLP models (FinBERT, crypto-fine-tuned transformers) can quantify this in real-time. We already have MCP tools for social sentiment (social_sentiment, market_sentiment, crypto_news from Virtuoso MCP).

### Data Sources

- **Already have:** Virtuoso MCP tools (social_sentiment, market_sentiment for any symbol), crypto_news endpoint, Fear & Greed index.
- **Needed:**
  - Historical sentiment data for backtesting (not available from MCP -- it is real-time only).
  - Alternative: LunarCrush API ($100-300/month for historical social data), or Santiment ($50-300/month).
  - For custom NLP: Twitter/X API ($100/month for Basic tier, historical access limited).

### Development Time

- MCP sentiment signal integration (real-time): 1 week
- Historical sentiment data acquisition: depends on vendor (1-2 weeks)
- Backtesting sentiment as contrarian signal: 2 weeks
- Walk-forward validation: 1 week
- **Total: 4-6 weeks**

### Expected Impact

- Sharpe improvement: +0.0 to +0.08 (low confidence). Sentiment signals in crypto have been extensively studied. The contrarian value exists at extremes but the signals are noisy between extremes. Most crypto sentiment indices (Fear & Greed, social volume) have weak predictive power in academic studies.

### Dependencies

- Historical sentiment data source (purchase required for backtesting).
- V5.5 ML ensemble for integration.

### Overfitting Risk: MEDIUM-HIGH

Sentiment regimes change character. The Twitter/CT ecosystem of 2021 is structurally different from 2025 (different platforms, different influencers, different narratives). Models trained on 2020-2023 sentiment data may not generalize.

### Skepticism Rating: 3/10 (least likely to add real value)

This is the weakest V5 component. Sentiment analysis is the most "sounds good" idea that rarely delivers in practice for systematic trading. The signal-to-noise ratio is poor, the data is expensive to backtest properly, and the alpha (if any) is small relative to M2 acceleration. We already have Fear & Greed from MCP, which captures most of the information in a simple number. Recommend: use the existing MCP sentiment tools as a lightweight input to the ML ensemble (V5.5) but do not build a dedicated NLP pipeline. If the simple Fear & Greed score shows SHAP importance in the ensemble, then revisit deeper NLP.

---

## V5 PRIORITY RANKING

Based on expected value (impact * probability of working) minus complexity tax:

| Priority | Component | Expected Value | Complexity | Build Order |
|----------|-----------|---------------|------------|-------------|
| 1 | V5.4: Stablecoin Supply | HIGH | LOW | Month 1 |
| 2 | V5.1: Options Vol Surface | MEDIUM-HIGH | MEDIUM | Month 1-2 |
| 3 | V5.3: Cross-Exchange Micro | MEDIUM | MEDIUM | Month 2-3 |
| 4 | V5.5: ML Ensemble (no RL) | MEDIUM | MEDIUM | Month 3-4 |
| 5 | V5.2: On-Chain Flows | LOW-MEDIUM | MEDIUM | Month 4 |
| 6 | V5.6: Sentiment NLP | LOW | MEDIUM-HIGH | Defer / lightweight only |
| -- | V5.5: RL Position Sizer | UNKNOWN | HIGH | Defer to V6 or never |

---

# V6 -- "FULL SPECTRUM" (6-12 MONTH HORIZON)

The thesis: V5 deepens the signal stack. V6 widens the execution surface -- more assets, more timeframes, more execution venues, and autonomous parameter adaptation. V6 assumes V5 is validated and live.

---

## V6.1: Multi-Timeframe Execution

### Edge Hypothesis

V3 operates on daily bars. Entries and exits happen at daily close, which means potentially poor fills -- buying at the high of the day or selling at the low. Using daily signals for direction but 4h/1h bars for entry timing could improve fills by 1-3% per trade. Over a year with 30-50 trades, this compounds to 30-150% improvement in raw returns.

Research finding from MEMORY.md: "Daily TF wins (1d >> 4h >> 1h)" for signal generation. This is consistent -- the macro signals work on daily, but execution can benefit from lower timeframes.

### Architecture

```
Daily Signal Generator (V3 + V5 ensemble)
    |
    Signal: LONG BTC, Confidence HIGH
    |
    v
4H/1H Entry Module
    |
    Waits for:
    - RSI < 40 on 1h (dip entry)
    - VWAP reversion on 4h
    - Volume confirmation
    - Max wait: 48 hours, then market enter
    |
    v
Execution
```

### Data Sources

- **Needed:** 4h and 1h OHLCV data from exchanges (free via CCXT). Storage: ~100MB/year for 4 assets at 1h.
- **Already have:** Daily data pipeline. Freqtrade supports multi-timeframe natively (informative_pairs).

### Development Time

- 1h/4h data pipeline: 3 days
- Entry timing module: 2 weeks
- Backtesting with multi-TF: 2 weeks (complex -- need to simulate intrabar entries on daily signals)
- Walk-forward validation: 1 week
- **Total: 5-6 weeks**

### Expected Impact

- Sharpe improvement: +0.05 to +0.15 (moderate confidence). The improvement comes from better fills, not better signals. This is execution alpha, which is real but incremental.
- Primary value: reduces slippage and improves average entry price by 1-3%.

### Dependencies

- Requires V5 to be in production (daily signals must be stable before adding execution layer).
- Freqtrade infrastructure (Tier 1 must be live).

### Overfitting Risk: LOW

Entry timing on lower timeframes is a well-understood concept. The risk is more operational (complexity of managing multi-TF state) than statistical.

### Skepticism Rating: 7/10 (likely to improve execution quality)

This is unglamorous but valuable. Better fills compound. The main challenge is engineering complexity -- managing state across timeframes, handling edge cases (signal triggers on daily but 1h entry never comes), and ensuring the system does not miss trades by being too selective on entry. Build it simple: TWAP over 4-8 hours after signal, with a VWAP-reversion enhancement.

---

## V6.2: Portfolio Expansion

### Edge Hypothesis

V3 trades BTC (40%), ETH (25%), SOL (20%), LINK (15%). Research from 2026-02-06 identified high-alpha assets: ZEC (+769% OOS), SUI (+461%), SEI (+312%), CRV (+287%), DYDX, RENDER. Expanding to 10-20 assets diversifies idiosyncratic risk and captures more opportunities. Privacy coins (ZEC) and newer L1s (SUI, SEI) show asymmetric volatility ideal for momentum strategies.

### Implementation

| Phase | Assets Added | Rationale |
|-------|-------------|-----------|
| Phase 1 | ZEC, SUI, SEI | Highest OOS returns in research |
| Phase 2 | CRV, DYDX, RENDER | Strong alpha, different sector exposure |
| Phase 3 | AVAX, NEAR, INJ, TIA, ARB | Diversification into L1/L2 ecosystem |

### Data Sources

- **Already have:** Coinalyze derivatives data for all 65 tokens. yfinance OHLCV. Backtest results for ZEC, SUI, SEI, CRV.
- **Needed:** Per-asset parameter optimization via Optuna walk-forward. Portfolio correlation analysis. Liquidity assessment per asset.

### Development Time

- Per-asset Optuna optimization: 1 week per 3 assets (parallelizable)
- Portfolio construction (weight optimization): 2 weeks
- Correlation and capacity analysis: 1 week
- Walk-forward on full portfolio: 2 weeks
- **Total: 6-8 weeks**

### Expected Impact

- Sharpe improvement: +0.10 to +0.30 (moderate-high confidence). Diversification across low-correlation assets is one of the most reliable ways to improve risk-adjusted returns. The key question is whether the altcoin alpha persists or is a specific-period artifact.
- Risk: altcoins have fat tails and liquidity gaps. SOL and LINK are liquid; ZEC and SEI less so.

### Dependencies

- V5 validated (do not expand assets until core strategy is proven live).
- Liquidity analysis must precede each addition.
- Capacity constraint: LINK already binding at $5M daily volume. Smaller altcoins may cap at $500k-$1M positions.

### Overfitting Risk: MEDIUM

ZEC's +769% OOS is suspicious -- is it robust alpha or a 2024-2025 specific phenomenon? Walk-forward on multiple periods is essential. Privacy coin regulation could also destroy the trade.

### Skepticism Rating: 7/10 (diversification works; individual altcoin alpha may not persist)

The portfolio diversification benefit is real and well-established. The question is whether ZEC/SUI/SEI continue to outperform. Recommendation: add assets based on liquidity first (SUI, AVAX, NEAR are safer), not historical return (ZEC's numbers are too good to trust). Weight by inverse volatility, not by backtest return.

---

## V6.3: Cross-Asset Alpha

### Edge Hypothesis

Equities (SPX, QQQ), commodities (gold, oil), and FX (DXY already used) may lead crypto moves. MEMORY.md notes: "TradFi leads failed but worth revisiting with better data." The failure may have been due to using daily correlation when the lead-lag operates at weekly or monthly frequency, or because the relationship is regime-dependent (TradFi leads during institutional-driven markets, not during crypto-native moves).

### Data Sources

- **Already have:** DXY, Gold, 10Y yields, HYG (all in V3 Liquidity Proxy). Fama-French factors (Mkt-RF, RMW already show SHAP importance). yfinance for SPX, QQQ, GLD, TLT, etc.
- **Needed:** Potentially higher-frequency equity data (4h SPX via futures) -- available from CME via data vendors. Oil (WTI) and copper as industrial demand proxies.

### Derived Signals

| Signal | Hypothesis | Existing? |
|--------|-----------|-----------|
| SPX momentum (20d) | Risk-on/off leads crypto by 1-2 days | No |
| Gold-BTC correlation regime | Positive correlation = macro-driven; negative = crypto-native | Partially (Gold in Liquidity Proxy) |
| Copper/Gold ratio | Economic optimism proxy | No |
| High-yield spread change | Credit stress leads crypto selloffs | Yes (HYG in V3) |
| VIX term structure | Equity vol regime as crypto risk indicator | No |

### Development Time

- Additional data integration: 1 week
- Lead-lag analysis across regimes: 2 weeks
- Feature engineering + ML integration: 1 week
- Walk-forward validation: 1 week
- **Total: 4-5 weeks**

### Expected Impact

- Sharpe improvement: +0.0 to +0.10 (low confidence). Previous testing showed TradFi leads "failed." Revisiting with regime conditioning might help, but the expectation should be modest. DXY and HYG already capture the most important cross-asset signals.

### Dependencies

- V5.5 ML ensemble (regime-conditional analysis).

### Overfitting Risk: MEDIUM-HIGH

Cross-asset correlations are notoriously unstable. BTC-SPX correlation was near zero pre-2020, 0.6+ during 2021-2022, and has since varied. Fitting to any particular correlation regime will break.

### Skepticism Rating: 4/10 (already tried, mostly failed)

DXY and HYG are already in V3 and working. The marginal value of adding SPX momentum, copper/gold ratio, and VIX is probably small. The previous failure is informative. If Mkt-RF (which is effectively SPX excess return) already shows SHAP importance, adding raw SPX may be redundant. Low priority.

---

## V6.4: Decentralized Execution

### Edge Hypothesis

Executing on HyperLiquid and dYdX provides: (a) censorship resistance -- no exchange can freeze funds or restrict trading, (b) on-chain verifiable track record for Tier 3 fund raising (Copin.io already supports HyperLiquid), (c) potentially lower fees (HyperLiquid maker rebates).

Whale Hunter already tracks HyperLiquid and AsterDex. The infrastructure is partially familiar.

### Data Sources

- **Already have:** Whale Hunter integration with HyperLiquid. Familiarity with the platform.
- **Needed:** HyperLiquid Python SDK or API wrapper. Smart contract interaction for dYdX v4 (Cosmos-based). Wallet management and signing infrastructure.

### Development Time

- HyperLiquid execution integration: 2-3 weeks
- dYdX v4 integration: 3-4 weeks (more complex, Cosmos SDK)
- Position management (multi-venue reconciliation): 2 weeks
- Testing + dry run: 2 weeks
- **Total: 8-10 weeks**

### Expected Impact

- No direct Sharpe improvement. This is infrastructure for Tier 3 (fund raising) and risk diversification.
- Value: verifiable on-chain track record, reduced counterparty risk, potentially better funding rates.

### Dependencies

- Tier 1 must be operational on centralized exchange first.
- Wallet security infrastructure (hardware wallet or HSM for signing).

### Overfitting Risk: N/A (infrastructure, not signal)

### Skepticism Rating: 6/10 (valuable for business; not for alpha)

This is a business decision, not an alpha decision. If Tier 3 fund raising is a priority, on-chain track record via HyperLiquid is valuable. If the goal is purely strategy improvement, skip this. Recommend building HyperLiquid execution first (simpler, lower fees) and deferring dYdX.

---

## V6.5: Auto-Adaptive Parameters

### Edge Hypothesis

V3 uses fixed per-asset parameters (BTC: sma=100, mom=35, trail=12%) optimized via Optuna walk-forward. These parameters are optimal for the average regime but suboptimal for any specific regime. An online learning system could adjust parameters in real-time based on detected regime -- shorter SMA in trending markets, wider trailing stops in volatile markets.

### Architecture

```
Regime Classifier Output
    |
    v
Parameter Adjustment Rules
    |
    +--> BULL/MILD_BULL: shorter SMA, tighter trail, higher leverage
    +--> BEAR: longer SMA, wider trail, lower leverage, prefer shorts
    +--> NEUTRAL: median params, minimal leverage
    +--> ACCUMULATION: shortest SMA (catch early trend), medium trail
    |
    v
Strategy Execution with Dynamic Params
```

### Implementation Options

1. **Lookup table** (simplest): Pre-optimize parameters per regime via Optuna, store as table, switch on regime change. Low risk, 2-3 weeks.
2. **Bayesian online learning**: Continuously update parameter posterior based on recent performance. Medium risk, 4-6 weeks.
3. **Meta-learning**: Train a model that predicts optimal parameters given current market features. High risk, 6-8 weeks.

### Development Time

- Option 1 (lookup table): 2-3 weeks including per-regime Optuna runs
- Option 2 (Bayesian): 4-6 weeks
- Walk-forward validation: 2 weeks
- **Total: 4-8 weeks depending on approach**

### Expected Impact

- Option 1: Sharpe improvement +0.05 to +0.15 (moderate confidence). Regime-specific parameters should outperform global parameters.
- Option 2/3: Sharpe improvement +0.0 to +0.20 (low confidence). More complex approaches may overfit regime transitions.

### Dependencies

- Reliable regime classifier (V5.5).
- Sufficient data per regime for separate optimization (V3 has 5 regimes over 6 years -- some regimes may have too few observations).

### Overfitting Risk: HIGH (for options 2 and 3), MEDIUM (for option 1)

The fundamental problem: with 5 regimes and 4 assets, you have 20 parameter sets to optimize. With ~6 years of data, some regimes may have only 6-12 months of observations. This is insufficient for robust optimization. The lookup table approach partially mitigates this by using simple heuristics (e.g., "in BULL, reduce SMA by 20%") rather than full re-optimization.

### Skepticism Rating: 6/10 (lookup table: probably works; anything fancier: probably doesn't)

Start with the lookup table. It is simple, interpretable, and hard to overfit if the adjustments are conservative (10-20% parameter shifts, not wholesale changes). Bayesian online learning and meta-learning are research projects, not near-term production improvements.

---

## V6.6: Market Making Module

### Edge Hypothesis

During NEUTRAL regime (V3 confluence score 2-3), the strategy has minimal directional conviction. Rather than sitting in cash, a market-making module could provide liquidity and earn the bid-ask spread. This is delta-neutral by design and generates yield while waiting for directional signals.

### Architecture

```
Regime = NEUTRAL + Low Conviction
    |
    v
Market Making Module
    |
    Places symmetric limit orders:
    bid = mid - spread/2
    ask = mid + spread/2
    |
    Inventory management:
    - Max position: 0.5x leverage
    - Rebalance when inventory exceeds threshold
    - Hard stop if regime changes to BULL/BEAR
    |
    Target: 5-15 bps per round trip
```

### Data Sources

- **Needed:** Real-time order book data (Level 2) for spread calculation. WebSocket connections to exchanges. Latency-sensitive -- VPS location matters (current VPS: Hetzner, likely EU-based).
- **Already have:** Nothing directly applicable. This is a different paradigm from the momentum strategy.

### Development Time

- Market making engine: 4-6 weeks
- Inventory management and risk controls: 2 weeks
- Exchange WebSocket integration: 2 weeks
- Paper trading / simulation: 4 weeks minimum
- **Total: 12-16 weeks**

### Expected Impact

- Returns: 10-30% APY during NEUTRAL periods (which may constitute 30-50% of time). On $10k capital, this is $300-$1,500/year in additional returns.
- Sharpe impact: Minimal on overall portfolio (NEUTRAL periods are short and the capital allocation would be small).
- The real value is capital efficiency -- earning returns during dead periods.

### Dependencies

- Low-latency infrastructure (current VPS may not be ideal for market making).
- Deep understanding of exchange maker/taker fee structures.
- Completely different risk profile from directional trading -- requires separate risk management.

### Overfitting Risk: LOW (market making is structural, not predictive)

Market making alpha comes from providing liquidity, not from prediction. The risk is adverse selection (getting picked off by informed traders), not overfitting.

### Skepticism Rating: 4/10 (real strategy but wrong context)

Market making is a legitimate business, but it requires: (a) low-latency infrastructure, (b) high-frequency execution, (c) significant capital to be worthwhile, and (d) deep expertise in microstructure. This is a completely different skill set from macro momentum trading. For a team of one running a $10k-$100k book, the development cost (12-16 weeks) vastly exceeds the expected incremental returns ($300-$1,500/year). Recommend deferring indefinitely unless the fund scales to $1M+ AUM and a dedicated market making infrastructure is justified.

---

## V6 PRIORITY RANKING

| Priority | Component | Expected Value | Complexity | Build Order |
|----------|-----------|---------------|------------|-------------|
| 1 | V6.2: Portfolio Expansion | HIGH | MEDIUM | Month 5-6 |
| 2 | V6.1: Multi-TF Execution | MEDIUM | MEDIUM | Month 6-7 |
| 3 | V6.5: Auto-Adaptive (table) | MEDIUM | LOW | Month 7 |
| 4 | V6.4: Decentralized Execution | MEDIUM (business) | HIGH | Month 8-9 |
| 5 | V6.3: Cross-Asset Alpha | LOW | MEDIUM | Opportunistic |
| 6 | V6.6: Market Making | LOW | VERY HIGH | Defer |

---

# CONSOLIDATED TIMELINE

Assumes 1 developer working full-time on strategy research. Calendar starts when Tier 1 goes live.

| Month | Component | Milestone | Validation Gate |
|-------|-----------|-----------|-----------------|
| 1 | V5.4: Stablecoin Supply | Data pipeline live, initial backtest | Walk-forward OOS Sharpe > V3 baseline |
| 1-2 | V5.1: Options Vol Surface | Deribit collector live, signal derivation complete | SHAP importance > 0.01 in regime classifier |
| 2-3 | V5.3: Cross-Exchange Micro | Funding rate aggregator, basis trade calculator | Basis trade simulation > 10% APY |
| 3-4 | V5.5: ML Ensemble (no RL) | Unified pipeline, stacked models | OOS Sharpe > 0.90 (vs V3 baseline 0.84) |
| 4 | V5.2: On-Chain Flows | CryptoQuant integration, 3 flow signals | Walk-forward confirms OOS value |
| 4-5 | **V5 Integration** | All V5 components merged, full walk-forward | **V5 OOS Sharpe > 1.0, MaxDD < -8%** |
| 5-6 | V6.2: Portfolio Expansion (Phase 1) | SUI, ZEC, SEI added with per-asset params | Per-asset walk-forward passes |
| 6-7 | V6.1: Multi-TF Execution | 4h entry module live | Fill improvement > 1% average |
| 7 | V6.5: Auto-Adaptive (table) | Regime-specific param lookup table | Lookup table outperforms static in walk-forward |
| 7-8 | V6.2: Portfolio Expansion (Phase 2) | CRV, DYDX, RENDER added | Portfolio Sharpe > individual asset Sharpe |
| 8-9 | V6.4: Decentralized Execution | HyperLiquid execution live | Copin.io track record verified |
| 9-10 | **V6 Integration** | Full system validated | **V6 OOS Sharpe > 1.2, portfolio of 10+ assets** |

---

# WALK-FORWARD VALIDATION REQUIREMENTS

Every new module must pass these gates before integration into production.

## Standard Walk-Forward Protocol

| Parameter | Value |
|-----------|-------|
| Method | Expanding window, purged k-fold |
| Minimum folds | 10 |
| In-sample minimum | 18 months |
| Out-of-sample minimum | 3 months |
| Purge gap | 5 trading days (prevent leakage) |
| Success criterion | OOS Sharpe improvement > 0 in at least 7/10 folds |
| Statistical test | Paired t-test on fold-level Sharpe, p < 0.10 |
| Complexity penalty | New module must improve OOS Sharpe by at least 0.03 per added parameter |

## Per-Module Validation

| Module | Additional Requirement |
|--------|----------------------|
| V5.1: Vol Surface | Test on BTC and ETH separately; both must show improvement |
| V5.2: On-Chain Flows | Exclude 2021-2022 from training (obvious on-chain signatures); must work on 2023-2025 |
| V5.3: Basis Trade | Simulate with realistic transaction costs (maker: 0.02%, taker: 0.06%, funding settlement) |
| V5.4: Stablecoin | Test with 1-month and 2-month forward returns; must beat M2-only signal |
| V5.5: ML Ensemble | SHAP analysis must show new features contributing; no model should have > 50 features |
| V5.6: Sentiment | Must work in both bull and bear regimes independently |
| V6.1: Multi-TF | Simulate with 0.1% slippage on 1h entries; must still beat daily-close execution |
| V6.2: Portfolio | Each new asset must have OOS Sharpe > 0.3 individually before portfolio inclusion |
| V6.5: Adaptive | Each regime must have > 100 daily observations for separate param optimization |

---

# COST ESTIMATES

## Data and Infrastructure

| Item | Monthly Cost | Annual Cost | Priority |
|------|-------------|-------------|----------|
| CoinGecko API (stablecoin data) | Free (basic) | $0 | V5.4 |
| DeFiLlama API | Free | $0 | V5.4 |
| Deribit API | Free (public) | $0 | V5.1 |
| Tardis.dev (historical options) | One-time $200-500 | $200-500 | V5.1 |
| CryptoQuant (on-chain flows) | $30-100 | $360-1,200 | V5.2 |
| CCXT (multi-exchange) | Free | $0 | V5.3 |
| GPU compute for ML/RL | $50-100 | $600-1,200 | V5.5 |
| **Total incremental** | **$80-200** | **$1,160-2,900** | |

## Development Time

| Version | Total Weeks | Calendar Months (with buffer) |
|---------|-------------|------------------------------|
| V5 (all components) | 27-35 weeks | 4-5 months |
| V6 (all components) | 35-45 weeks | 5-7 months |
| V5 (priority 1-4 only) | 19-25 weeks | 3-4 months |
| V6 (priority 1-3 only) | 13-17 weeks | 2-3 months |

---

# WHAT WILL ACTUALLY WORK VS WHAT SOUNDS GOOD

An honest assessment.

## Most Likely to Deliver Alpha

1. **V5.4: Stablecoin supply as M2 proxy** -- Direct extension of proven edge. Simple signal. Low overfitting risk. Build first.
2. **V6.2: Portfolio expansion** -- Diversification is the only free lunch. Even if per-asset alpha decays, portfolio construction improves Sharpe mechanically.
3. **V5.1: Options vol surface** -- Genuine forward-looking information, well-established in TradFi, under-exploited in crypto.
4. **V6.1: Multi-TF execution** -- Execution alpha is real and structural.

## Might Work, Proceed With Caution

5. **V5.5: ML ensemble (stack only)** -- Proper stacking is a modest but reliable improvement. No RL.
6. **V5.3: Basis trading** -- Market-neutral yield is real but small relative to directional returns.
7. **V6.5: Auto-adaptive (table only)** -- Regime-specific parameters make theoretical sense but data may be insufficient.

## Probably Will Not Work (Despite Sounding Impressive)

8. **V5.6: Sentiment NLP** -- Crowded, noisy, expensive to validate properly.
9. **V6.3: Cross-asset alpha** -- Already tested and failed. DXY/HYG capture what matters.
10. **V5.5: RL position sizer** -- Insufficient data for RL convergence. Will overfit.
11. **V6.6: Market making** -- Wrong context. Wrong scale. Wrong skill set.

---

# WHAT MAKES THIS UNREPLICABLE

The moat is not any single component. It is the stack:

1. **M2 acceleration as core signal** -- Few crypto traders think in macro terms. Fewer validate with walk-forward.
2. **Stablecoin supply as real-time M2 proxy** (V5.4) -- Bridges the 2-4 week FRED data lag. Novel combination.
3. **Options vol surface for regime detection** (V5.1) -- Requires Deribit data infrastructure that most retail/small fund traders do not build.
4. **On-chain + whale tracking as confirmation** (V5.2, existing BTC Wiz + Whale Hunter) -- 27 on-chain signals + 102 tracked elite traders is a data asset, not just a strategy.
5. **Walk-forward validated at every layer** -- The discipline to cut modules that do not pass OOS (V4 proved this works -- 3/4 modules were cut) is the real edge.

Anyone can replicate a single signal. Replicating the full validated stack, with the data infrastructure, the ML pipeline, the walk-forward discipline, and the live trading track record, takes 6-12 months minimum. By then, V5/V6 will be live.

---

*Virtuoso's Take:*

V4 taught us the most important lesson: most "improvements" are noise. Three out of four V4 modules failed walk-forward validation. The temptation with V5 and V6 is to build everything on this list. Resist it.

Build V5.4 (stablecoin supply) first. It directly strengthens the proven M2 edge with zero risk of overcomplication. Then V5.1 (options vol surface) because it adds a genuinely orthogonal signal. Then expand the portfolio (V6.2) because diversification is mathematics, not speculation.

Everything else is optional until the core is generating live returns. The RL position sizer, the sentiment NLP, the market making module -- these are interesting research projects. They are not near-term alpha. The risk of spending 3 months building a sophisticated ML ensemble that adds 0.02 to Sharpe while delaying Tier 1 deployment by 3 months is real and severe.

The strategy does not need to be perfect. It needs to be live, validated, and generating a track record. V3 with OOS Sharpe 0.84 is already better than 95% of crypto strategies. V5.4 + V5.1 might push that to 1.0+. That is enough. Ship it.
