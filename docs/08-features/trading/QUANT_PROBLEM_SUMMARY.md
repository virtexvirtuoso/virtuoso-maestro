# Quantitative Research Problem: Synthetic Proxies for Crypto Derivatives Data

## Context

Building a backtesting framework (Maestro) for crypto perpetual futures strategies. Several alpha-generating strategies require historical derivatives data that exchanges don't provide via API.

## The Data Gap

| Data Needed | Use Case | API Availability |
|-------------|----------|------------------|
| Funding Rate History | Trade funding rate mean reversion | ~1 year max (Binance) |
| Open Interest History | OI divergence signals | ~30 days (Binance) |
| Liquidation History | Capitulation reversal entries | Not available |
| Long/Short Ratio | Crowding indicators | ~30 days |
| Historical Basis | Spot-perp premium | Must compute from spot+perp |

**Problem:** Can't backtest strategies beyond API limits. Need 2-5 years of history for robust validation.

## What We Tried

Built synthetic proxies to estimate derivatives metrics from OHLCV data:

### Funding Rate Proxies
1. **Basis Proxy:** `(perp - spot) / spot` - theoretically best
2. **LSR Proxy:** Volume-weighted up/down candle ratio
3. **Vol-Adjusted:** Z-score of 8h returns
4. **Momentum:** Returns × volume ratio

### Calibration Results (BTC/USDT, 505 samples)

| Proxy | Correlation | R² | Direction Acc |
|-------|-------------|-----|---------------|
| LSR | 0.084 | 0.007 | 86.9% |
| Momentum | 0.036 | 0.001 | 86.9% |
| Vol-Adjusted | 0.008 | 0.000 | 86.9% |
| Basis | 0.003 | 0.000 | 86.9% |

**Key Finding:** Proxies predict *direction* (87%) but have near-zero correlation with *magnitude*.

---

## Quantitative Assessment (2026-02-05)

### Critical Finding: 87% Direction Accuracy is Likely the Naive Baseline

**The 87% direction accuracy may represent ZERO predictive edge.**

Historical BTC perp funding is positive ~70-92% of the time (varies by market regime). If funding is positive 87% of the time, then predicting "positive" always yields 87% accuracy — with no actual signal.

**First validation step (MUST DO):**
```python
naive_baseline = (funding_df['funding_rate'] > 0).mean()
print(f"Naive baseline: {naive_baseline:.1%}")
print(f"Your accuracy: 87%")
print(f"Actual edge: {0.87 - naive_baseline:.1%}")
```

If edge ≈ 0%, proxies have **no predictive value**.

### Why Basis Proxy Failed (r = 0.003)

**Root Cause: Causality runs backwards.**

| You assumed | Reality |
|-------------|---------|
| Premium → Funding Rate | Expected Funding → Premium (via arbitrage) |

Arbitrageurs trade the basis based on *expected* funding. By the time you observe the basis, it's the *result* of this process, not an input. Technical fixes (mark price, clamps, alignment) won't solve this structural simultaneity problem.

**Secondary issues:**
- Exchanges use mark price + impact prices, not close vs spot
- Funding is time-weighted average (clamped), not point-in-time
- Timestamp alignment problems between spot and perp sources

### Granger Causality Test (Recommended)

```python
from statsmodels.tsa.stattools import grangercausalitytests

merged = pd.concat([basis, funding], axis=1).dropna()
merged.columns = ['basis', 'funding']

# Does basis predict funding?
granger_basis_to_funding = grangercausalitytests(
    merged[['funding', 'basis']], maxlag=3, verbose=False
)

# Does funding predict basis? (likely YES)
granger_funding_to_basis = grangercausalitytests(
    merged[['basis', 'funding']], maxlag=3, verbose=False
)
```

Expected: Funding Granger-causes basis (p < 0.05), not vice versa.

---

## Free Data Strategy (Current Approach)

### Available Free Sources

| Source | Coverage | How to Get |
|--------|----------|------------|
| **Binance** | ~1000 records (~4 months) | `/fapi/v1/fundingRate?limit=1000` |
| **Bybit** | ~200 records (~2 weeks) | `/v5/market/funding/history` |
| **OKX** | ~100 records (~1 week) | `/api/v5/public/funding-rate-history` |
| **dYdX** | Full history (on-chain) | Indexer API or subgraph |
| **Hyperliquid** | Full history | Public API, no auth |
| **CryptoQuant Free** | 3 years daily (rate limited) | Manual or chart scraping |

### Aggregation Strategy

Cross-exchange funding is highly correlated (arbitrage keeps within ~0.001%). Stitch together:
- Binance as primary (longest coverage)
- dYdX/Hyperliquid for validation (different market structure)
- CryptoQuant for long-term regime context

**Realistic free coverage:**
- BTC/ETH: ~6-12 months aggregated
- Altcoins: ~3-6 months
- dYdX/Hyperliquid: Full history but only since 2021/2023

---

## Action Plan

### Phase 1: Validate Signal Exists (This Week)

1. **Calculate naive baseline** — Is 87% actually an edge?
2. **Run Granger causality** — Does basis predict funding or vice versa?
3. **Cross-correlation at lags** — Find if there's an optimal lag

If no edge found, STOP proxy development and pivot strategy.

### Phase 2: Maximize Free Data (If Edge Exists)

```python
import ccxt

def collect_all_free_funding(symbol='BTC/USDT:USDT'):
    exchanges = {
        'binance': ccxt.binance({'options': {'defaultType': 'future'}}),
        'bybit': ccxt.bybit({'options': {'defaultType': 'future'}}),
        'okx': ccxt.okx({'options': {'defaultType': 'swap'}}),
    }

    all_funding = []
    for name, exchange in exchanges.items():
        try:
            funding = exchange.fetch_funding_rate_history(symbol, limit=1000)
            df = pd.DataFrame(funding)
            df['exchange'] = name
            all_funding.append(df)
        except Exception as e:
            print(f"{name} failed: {e}")

    return pd.concat(all_funding)
```

### Phase 3: Improved Proxy (If Pursuing Synthetic)

**Predict funding CHANGES, not levels:**
```python
def funding_change_proxy(df, funding_history, lookback=24):
    # Funding momentum (changes persist)
    funding_ma = funding_history.rolling(3).mean()
    funding_momentum = funding_history - funding_ma

    # Market stress indicators
    volatility = df['close'].pct_change().rolling(lookback).std()
    vol_zscore = (volatility - volatility.rolling(168).mean()) / volatility.rolling(168).std()

    # Volume surge
    volume_surge = df['volume'] / df['volume'].rolling(lookback).mean() - 1

    # Combine
    proxy = (
        0.4 * funding_momentum.shift(1) +
        0.3 * vol_zscore * 0.001 +
        0.2 * volume_surge.clip(-2, 2) * 0.0005
    )
    return proxy
```

### Phase 4: Strategy Pivot (If No Signal)

If validation confirms no edge, design strategies that are **robust to** funding uncertainty rather than **dependent on** funding prediction:

| Instead of... | Do this... |
|---------------|------------|
| Predicting funding magnitude | Use funding as **cost** in other strategies |
| Funding mean-reversion | Trade **momentum** and account for funding drag |
| Synthetic funding proxy | Use **real-time funding** for entry timing only |

---

## Backtest Architecture (Free Data Only)

```
┌─────────────────────────────────────────────────┐
│                 MAESTRO BACKTEST                │
├─────────────────────────────────────────────────┤
│  Historical Period        │  Data Source        │
├───────────────────────────┼─────────────────────┤
│  2024-present (6mo)       │  Real funding (API) │
│  2023-2024 (gap)          │  Proxy (if validated)│
│  Pre-2023                 │  Regime assumption  │
│                           │  (use avg funding)  │
└───────────────────────────┴─────────────────────┘
```

For pre-data periods, assume:
- Bull market: funding ≈ +0.01% to +0.03% (8h)
- Bear market: funding ≈ -0.01% to +0.01% (8h)
- Extreme (>50% drawdown): funding volatile, ±0.1%

---

## Paid Data Decision Framework

Only consider paid data if:
1. Free data validates that a signal exists
2. AUM > $100k (data cost becomes trivial vs potential alpha)
3. Expected Sharpe improvement > 0.3

| AUM | Data Cost | Recommendation |
|-----|-----------|----------------|
| <$50k | Any | Use free data only |
| $50-100k | $29-99/mo | CryptoQuant if edge validated |
| >$100k | $99-299/mo | Coinglass worth the cost |

---

## Files

- `synthetic_proxies.py` - All proxy implementations
- `calibrate_proxies.py` - Calibration pipeline
- `data_collectors.py` - API data collection tools
- `SYNTHETIC_PROXIES.md` - Full proxy documentation

## Outcome Options

1. **Validate edge exists** → Improve proxies, use for 2-5y backtests
2. **No edge confirmed** → Pivot to funding-robust strategies
3. **Edge exists but weak** → Use direction as filter, not predictor

---

*Maestro Research - Virtuoso Crypto*
*Last updated: 2026-02-05*
*Status: Pending validation of naive baseline*
