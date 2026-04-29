# Synthetic Proxy Methods for Missing Data

Research-backed methods to estimate derivatives data when historical records aren't available.

---

## Critical Warning (2026-02-05)

**Before using ANY proxy, validate that predictive edge exists.**

Our calibration showed 87% direction accuracy — but this may equal the naive baseline (funding is positive ~87% of the time in bull markets). If so, proxies have **ZERO predictive value**.

```python
# RUN THIS FIRST
naive_baseline = (funding_df['funding_rate'] > 0).mean()
actual_edge = 0.87 - naive_baseline
print(f"Naive baseline: {naive_baseline:.1%}")
print(f"Actual edge: {actual_edge:.1%}")
# If edge ≈ 0%, STOP using proxies for prediction
```

---

## 1. Funding Rate Proxy

### What Funding Rate Actually Is
```
Funding Rate = Interest Rate + Premium/Discount

Premium = (Mark Price - Index Price) / Index Price
Interest Rate ≈ 0.01% (constant, negligible)
```

Funding is positive when perps trade above spot (longs pay shorts).
Funding is negative when perps trade below spot (shorts pay longs).

### Why Basis Proxy Failed (r = 0.003)

**Causality runs backwards:**

| You assumed | Reality |
|-------------|---------|
| Premium → Funding Rate | Expected Funding → Premium (via arbitrage) |

Arbitrageurs trade the basis based on *expected* funding. The observed basis is the *result* of this process, not an input. Technical fixes won't solve this structural problem.

**Additional issues:**
- Exchanges use mark price + impact prices (bid/ask at fixed depth), not close
- Funding is time-weighted average (every 1-5 min over 8h), clamped [-0.75%, 0.75%]
- Timestamp alignment problems between spot/perp sources

### Proxy Method A: Basis Proxy (Theoretical Best, Practical Worst)
**Calibrated Accuracy: r = 0.003 (essentially zero)**

```python
def funding_proxy_basis(perp_price, spot_price, period=8):
    """
    Estimate funding from basis (premium/discount).
    WARNING: Calibration shows near-zero correlation.
    """
    premium = (perp_price - spot_price) / spot_price
    funding_proxy = premium.rolling(period).mean() * 0.33
    return funding_proxy
```

**Why it fails:** See causality problem above.

### Proxy Method B: LSR Proxy (Best Performing)
**Calibrated Accuracy: r = 0.084, Direction = 86.9%**

```python
def funding_proxy_lsr(df, period=24):
    """
    Estimate funding sentiment from volume-weighted direction.
    Best correlation among tested proxies (still weak).
    """
    up_bars = (df['close'] > df['open']).astype(int)
    down_bars = (df['close'] < df['open']).astype(int)

    up_volume = df['volume'] * up_bars
    down_volume = df['volume'] * down_bars

    long_pressure = up_volume.rolling(period).sum()
    short_pressure = down_volume.rolling(period).sum()

    lsr = long_pressure / (long_pressure + short_pressure + 1e-10)

    # Apply calibrated scaling
    proxy = (lsr - 0.5) * 0.018  # Calibrated scale factor
    return proxy
```

### Proxy Method C: Vol-Adjusted Returns
**Calibrated Accuracy: r = 0.008 (near-zero)**

```python
def funding_proxy_vol_adjusted(df, period=8):
    """
    Returns normalized by volatility = sentiment extremes.
    WARNING: Calibration shows inverted relationship.
    """
    returns = df['close'].pct_change(period)
    vol = df['close'].pct_change().rolling(24).std()

    z_returns = returns / vol

    # Calibrated scale (note: inverted!)
    proxy = np.tanh(z_returns / 3) * -0.0005
    return proxy
```

### Proxy Method D: Momentum + Volume
**Calibrated Accuracy: r = 0.036**

```python
def funding_proxy_momentum(df, period=8):
    """
    Estimate funding from momentum + volume.
    """
    returns = df['close'].pct_change(period)
    vol_ratio = df['volume'] / df['volume'].rolling(24).mean()

    proxy = returns * np.sqrt(vol_ratio) * 0.0003  # Calibrated scale
    return proxy.rolling(3).mean()
```

### Improved Approach: Predict Changes, Not Levels

Funding levels are sticky. Changes are more predictable:

```python
def funding_change_proxy(df, funding_history, lookback=24):
    """
    Predict funding CHANGE rather than level.
    """
    # Funding momentum (changes persist)
    funding_ma = funding_history.rolling(3).mean()
    funding_momentum = funding_history - funding_ma

    # Market stress indicators
    volatility = df['close'].pct_change().rolling(lookback).std()
    vol_zscore = (volatility - volatility.rolling(168).mean()) / volatility.rolling(168).std()

    # Volume surge
    volume_surge = df['volume'] / df['volume'].rolling(lookback).mean() - 1

    proxy = (
        0.4 * funding_momentum.shift(1) +
        0.3 * vol_zscore * 0.001 +
        0.2 * volume_surge.clip(-2, 2) * 0.0005
    )
    return proxy
```

---

## 2. Open Interest Proxy

> **⚠️ IMPORTANT: Don't proxy OI — free real data exists!**
>
> See **[OPEN_INTEREST_DATA.md](OPEN_INTEREST_DATA.md)** for free OI sources:
> - Coinalyze API (multi-year daily, free)
> - dYdX Indexer (unlimited on-chain history)
> - Binance API (~30 days)
>
> Academic research confirms OI cannot be reliably estimated from volume/price.

### What Open Interest Is
Total number of outstanding derivative contracts (longs + shorts, counted once).

- Rising OI + Rising Price = New longs entering (bullish)
- Rising OI + Falling Price = New shorts entering (bearish)
- Falling OI + Rising Price = Shorts closing (weak rally)
- Falling OI + Falling Price = Longs closing (weak selloff)

### Proxy Method A: Volume Integration (NOT RECOMMENDED)
**Estimated Accuracy: ~50-60% (barely better than random)**

```python
def oi_proxy_volume(df, decay=0.95):
    """
    Estimate OI changes from volume.
    Use for TREND direction, not absolute levels.
    """
    volume = df['volume']
    returns = df['close'].pct_change()

    signed_volume = volume * np.sign(returns)
    oi_proxy = signed_volume.ewm(span=24).mean()
    oi_proxy = oi_proxy / oi_proxy.rolling(168).std()
    return oi_proxy
```

### Proxy Method B: Range Analysis (NOT RECOMMENDED)
**Estimated Accuracy: ~55-60% (use free API data instead)**

```python
def oi_proxy_range(df, period=24):
    """
    Tight ranges with volume = OI accumulation.
    """
    high_low_range = (df['high'] - df['low']) / df['close']
    avg_range = high_low_range.rolling(period).mean()

    range_percentile = avg_range.rolling(period * 7).rank(pct=True)
    oi_proxy = 1 - range_percentile  # Tight range = high OI
    return oi_proxy
```

---

## 3. Liquidation Proxy

### What Liquidations Are
Forced closure of leveraged positions when margin is insufficient.

### Proxy Method A: Wick Analysis (Best for Liquidations)
**Estimated Accuracy: ~70-75%**

```python
def liquidation_proxy_wicks(df, wick_ratio=0.7):
    """
    Long wicks = liquidation cascades absorbed.
    """
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']
    total_range = df['high'] - df['low'] + 1e-10

    # Lower wick dominance = long liquidations absorbed
    long_liq_proxy = lower_wick / total_range
    long_liq_proxy[long_liq_proxy < wick_ratio] = 0

    # Upper wick dominance = short liquidations absorbed
    short_liq_proxy = upper_wick / total_range
    short_liq_proxy[short_liq_proxy < wick_ratio] = 0

    return long_liq_proxy, short_liq_proxy
```

### Proxy Method B: Spike Detection
**Estimated Accuracy: ~65-70%**

```python
def liquidation_proxy_spikes(df, volume_mult=3.0, price_thresh=0.02):
    """
    Liquidations cause volume spikes + sharp price moves.
    """
    volume = df['volume']
    vol_avg = volume.rolling(24).mean()
    returns = df['close'].pct_change()

    vol_spike = volume > vol_avg * volume_mult
    sharp_down = returns < -price_thresh
    sharp_up = returns > price_thresh

    long_liq = (vol_spike & sharp_down).astype(float)
    short_liq = (vol_spike & sharp_up).astype(float)

    return long_liq, short_liq
```

---

## 4. Long/Short Ratio Proxy

```python
def lsr_proxy(df, period=24):
    """
    Estimate L/S ratio from buying vs selling pressure.
    """
    buying = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    selling = 1 - buying

    buy_vol = (buying * df['volume']).rolling(period).sum()
    sell_vol = (selling * df['volume']).rolling(period).sum()

    lsr = buy_vol / (buy_vol + sell_vol)
    return lsr
```

---

## 5. Calibration Results (BTC/USDT)

**Data:** 505 aligned 8h samples (funding + OHLCV + spot)
**Period:** 2025-02 to 2026-02

### Raw Results

| Proxy | Correlation | R² | Direction Acc | Optimal Scale |
|-------|-------------|-----|--------------|---------------|
| LSR | 0.084 | 0.007 | 86.9% | 0.018 |
| Momentum | 0.036 | 0.001 | 86.9% | 0.0003 |
| Vol-adjusted | 0.008 | 0.000 | 86.9% | -0.0005 |
| Basis | 0.003 | 0.000 | 86.9% | 0.006 |

### Key Insight: Direction Accuracy is Trivial

**87% direction accuracy = naive baseline (funding positive ~87% of time)**

This means proxies have no predictive edge for direction. They only tell you what the unconditional distribution tells you.

---

## 6. Free Data Aggregation Strategy

### Available Free Sources

| Source | Coverage | Endpoint |
|--------|----------|----------|
| Binance | ~4 months | `/fapi/v1/fundingRate?limit=1000` |
| Bybit | ~2 weeks | `/v5/market/funding/history` |
| OKX | ~1 week | `/api/v5/public/funding-rate-history` |
| dYdX | Full (2021+) | Indexer API / subgraph |
| Hyperliquid | Full (2023+) | Public API |

### Aggregation Code

```python
import ccxt
import pandas as pd

def collect_all_free_funding(symbol='BTC/USDT:USDT'):
    """Aggregate funding from all free sources."""
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
            print(f"{name}: {len(funding)} records")
        except Exception as e:
            print(f"{name} failed: {e}")

    return pd.concat(all_funding).sort_values('timestamp')
```

### Realistic Coverage

- BTC/ETH: ~6-12 months aggregated
- Altcoins: ~3-6 months
- dYdX/Hyperliquid: Full history but newer exchanges

---

## 7. Recommendations

### For Funding Rate Strategy

1. **FIRST: Validate edge exists** — Compare direction accuracy to naive baseline
2. **If edge = 0%**: Don't use proxies for funding prediction
3. **If edge > 5%**: Use LSR proxy for direction filtering only
4. **Never size positions by proxy magnitude** — r² ≈ 0

### For OI Divergence Strategy

1. Use Volume Integration as directional indicator only
2. Focus on *changes* in OI proxy, not absolute levels
3. Combine with price action for confirmation
4. **Collect real OI data via API** (available on Binance)

### For Liquidation Cascade Strategy

1. Wick Analysis is most reliable for binary detection (~70-75%)
2. Combine with Volume Spike for confirmation
3. These are *event detection*, not prediction

### What Proxies ARE Good For

- **Regime filtering**: "Is sentiment likely bullish or bearish?" (with caveats)
- **Historical context**: Understanding past market structure
- **Backtesting when no API data exists** (accept noise)

### What Proxies ARE NOT Good For

- Predicting exact funding rate values (r² ≈ 0)
- Position sizing based on funding magnitude
- Replacing real-time API data for live trading
- Generating alpha (direction accuracy = baseline)

---

## 8. Validation Framework

Before trusting any proxy, run validation:

```python
class ProxyValidator:
    def __init__(self, proxy, actual):
        common = proxy.index.intersection(actual.index)
        self.proxy = proxy.loc[common]
        self.actual = actual.loc[common]

    def validate(self):
        # Naive baseline
        naive = (self.actual > 0).mean()

        # Direction accuracy
        dir_acc = (np.sign(self.proxy) == np.sign(self.actual)).mean()

        # Actual edge
        edge = dir_acc - naive

        # Correlation
        corr = self.proxy.corr(self.actual)

        print(f"Naive baseline: {naive:.1%}")
        print(f"Direction accuracy: {dir_acc:.1%}")
        print(f"Actual edge: {edge:.1%}")
        print(f"Correlation: {corr:.3f}")

        if edge < 0.05:
            print("⚠️  WARNING: No meaningful edge detected")

        return {
            'naive_baseline': naive,
            'direction_accuracy': dir_acc,
            'edge': edge,
            'correlation': corr,
            'has_edge': edge >= 0.05
        }
```

---

## 9. Backtest Architecture (Free Data)

```
┌─────────────────────────────────────────────────┐
│                 MAESTRO BACKTEST                │
├─────────────────────────────────────────────────┤
│  Historical Period        │  Data Source        │
├───────────────────────────┼─────────────────────┤
│  2024-present (6mo)       │  Real funding (API) │
│  2023-2024 (gap)          │  Proxy (if validated)│
│  Pre-2023                 │  Regime assumption  │
└───────────────────────────┴─────────────────────┘
```

For pre-data periods, use regime assumptions:
- Bull market: funding ≈ +0.01% to +0.03% (8h)
- Bear market: funding ≈ -0.01% to +0.01% (8h)
- Extreme drawdown: funding volatile, ±0.1%

---

## 10. Next Steps

1. **Run naive baseline validation** — Confirm 87% = baseline
2. **Run Granger causality test** — Confirm basis doesn't predict funding
3. **Aggregate free data** — Maximize coverage from multiple exchanges
4. **If no edge**: Pivot to funding-robust strategies
5. **If edge exists**: Use direction as filter, not magnitude predictor

---

*Last updated: 2026-02-05*
*Calibrated against real Binance funding rate data*
*Status: Pending naive baseline validation*
