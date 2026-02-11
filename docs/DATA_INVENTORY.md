# Maestro Data Inventory

Complete inventory of available data for backtesting and strategy development.

---

## 1. Stored Historical Data (VPS)

### OHLCV CSV Files (`~/backtest_data/`)

| Symbol | Timeframe | Date Range | Rows |
|--------|-----------|------------|------|
| BTC/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| ETH/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| SOL/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| BNB/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| XRP/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| ADA/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| DOGE/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| LINK/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| AVAX/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |
| DOT/USDT | 1h | 2024-02-05 → 2026-02-04 | 17,521 |

**Total: 2 years hourly data, 10 major pairs, ~175K candles**

### Additional CSV Files

| File | Description |
|------|-------------|
| `btc_180d.csv` | 180 days BTC price history |
| `btc_prices_90d.csv` | 90 days BTC prices |
| `btc_prices_30d.csv` | 30 days BTC prices |

### InfluxDB (`VirtuosoDB` bucket)

| Measurement | Description | Status |
|-------------|-------------|--------|
| `analysis` | Strategy analysis results | Active |
| `health_check` | System health metrics | Active |

*Note: Limited historical market data in InfluxDB currently*

---

## 2. Live Data via CCXT (On-Demand)

### Supported Exchanges
- **Binance Futures** (primary)
- **Bybit**
- **OKX**
- **Gate.io**
- **KuCoin**
- **MEXC**

### Timeframes Available
```
1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```

### OHLCV Data
- **Max per request:** 1,500 candles
- **Historical depth:** 2-3 years (symbol dependent)
- **Rate limits:** ~1,200 requests/min (Binance)

### Futures-Specific Data (Binance)

| Data Type | Endpoint | Use Case |
|-----------|----------|----------|
| **Funding Rate History** | `/fapi/v1/fundingRate` | FundingRateArbitrage strategy |
| **Open Interest** | `/fapi/v1/openInterest` | OI Divergence strategy |
| **Long/Short Ratio** | `/futures/data/globalLongShortAccountRatio` | Sentiment analysis |
| **Top Trader Positions** | `/futures/data/topLongShortPositionRatio` | Smart money tracking |
| **Top Trader Accounts** | `/futures/data/topLongShortAccountRatio` | Retail vs whale |
| **Taker Buy/Sell Volume** | `/futures/data/takerlongshortRatio` | Aggression indicator |
| **Liquidation Orders** | `/fapi/v1/allForceOrders` | Liquidation cascade detection |
| **Mark Price** | `/fapi/v1/premiumIndex` | Basis/premium calculation |

### Bybit Additional Data
- Funding Rate
- Open Interest
- Long/Short Ratio
- Insurance Fund

---

## 3. On-Chain Data (BTC Wiz)

**Database:** `~/btc_wiz/data/btc_onchain.db`

| Metric | Description |
|--------|-------------|
| Exchange flows | BTC moving to/from exchanges |
| Whale transactions | Large BTC movements |
| Miner activity | Mining pool behaviors |
| UTXO age bands | Holder behavior analysis |

*Status: Database exists, needs verification of current data*

---

## 4. Real Derivatives Data (Coinalyze - FREE)

### ✅ SOLVED: No More Proxies Needed

| Data Type | Source | Coverage | Location |
|-----------|--------|----------|----------|
| **Open Interest** | Coinalyze API | 2 years daily, 500+ tokens | `derivatives_data/{SYMBOL}_oi_daily.csv` |
| **Funding Rates** | Coinalyze API | 2 years daily, 500+ tokens | `derivatives_data/{SYMBOL}_funding_daily.csv` |
| **Liquidations** | Coinalyze API | 2 years daily, 500+ tokens | `derivatives_data/{SYMBOL}_liquidations_daily.csv` |
| **Long/Short Ratio** | Coinalyze API | 2 years daily, 500+ tokens | `derivatives_data/{SYMBOL}_lsr_daily.csv` |

**API Key:** Set in `COINALYZE_API_KEY` environment variable
**Rate Limit:** 40 calls/min (free tier)
**Refresh:** Run `coinalyze_collector.py`

### Priority 2: Enhanced Analysis

| Data Type | Source | Storage | Notes |
|-----------|--------|---------|-------|
| Spot prices | Binance Spot | InfluxDB | For basis/premium calculation |
| CVD (trade-level) | aggTrades WS | TimescaleDB | High volume, expensive to store |
| Orderbook L2 | WS stream | Redis | Real-time only, not for backtest |

### Priority 3: Alternative Data

| Data Type | Source | Notes |
|-----------|--------|-------|
| Fear & Greed Index | alternative.me | Free API |
| Social sentiment | LunarCrush / Santiment | Paid API |
| Options flow | Deribit | For max pain strategy |
| On-chain | Glassnode / Arkham | Paid, expensive |

---

## 5. Data Collection Scripts

### Existing

| Script | Location | Function |
|--------|----------|----------|
| `ccxt_batch_downloader.py` | `~/filos/backend/datasource/` | Multi-exchange OHLCV download |
| `data_multi_exchange_download.py` | `~/filos/backend/datasource/` | Batch download script |

### To Build

```python
# funding_rate_collector.py - Collect funding rate history
# oi_collector.py - Collect open interest history  
# liquidation_collector.py - Stream liquidations via WebSocket
```

---

## 6. Quick Reference

### Load Stored Data
```python
import pandas as pd
df = pd.read_csv('~/backtest_data/BTC_USDT_max.csv', parse_dates=['timestamp'])
```

### Fetch Live Data
```python
import ccxt
exchange = ccxt.binance({'options': {'defaultType': 'future'}})
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
```

### Fetch Funding Rates
```python
rates = exchange.fapiPublicGetFundingRate({'symbol': 'BTCUSDT', 'limit': 500})
```

### Fetch Open Interest
```python
oi = exchange.fapiPublicGetOpenInterest({'symbol': 'BTCUSDT'})
```

---

## 7. Storage Locations

| Type | Path | Size |
|------|------|------|
| OHLCV CSVs | `~/backtest_data/` | ~10 MB |
| InfluxDB | `/var/lib/influxdb2/` | ~500 MB |
| SQLite DBs | Various | ~100 MB total |
| Research outputs | `~/filos/research_output/` | Growing |

---

## 8. Recommendations

### Immediate Actions
1. **Build funding rate collector** - Required for FundingRate strategy backtesting
2. **Build OI collector** - Required for Phase 2 OI Divergence strategy
3. **Expand OHLCV coverage** - Add more alts to backtest_data/

### Data Pipeline Architecture
```
[Exchange APIs] → [Collectors] → [InfluxDB] → [Maestro Research]
                                     ↓
                              [CSV Export] → [Backtest Cache]
```

### Storage Strategy
- **InfluxDB:** Time-series data (OHLCV, funding, OI)
- **SQLite:** Strategy results, metadata
- **CSV:** Backtest snapshots, portable data
- **Redis:** Real-time orderbook (if needed)

---

*Last updated: 2026-02-05*
