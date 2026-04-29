# Tick Data Collector — Documentation

**Location:** `~/Desktop/maestro/backend/tick_collector/` (local) → `/home/linuxuser/tick_collector/` (VPS)
**Service:** `tick-collector.service` (systemd, enabled, auto-restart)
**Status:** ✅ Live and collecting since 2026-02-15 01:58 UTC

## Purpose

Standalone Bybit WebSocket tick data collector for microstructure research.
Completely independent from Virtuoso — runs in its own venv, own service, own data directory.

## What It Collects

### 1. Trade Ticks (every trade)
| Field | Type | Description |
|-------|------|-------------|
| timestamp | int64 | Unix milliseconds |
| price | float64 | Execution price |
| size | float64 | Trade quantity |
| side | str | "Buy" or "Sell" |

### 2. L2 Orderbook Snapshots (~20ms updates)
| Field | Type | Description |
|-------|------|-------------|
| timestamp | int64 | Unix milliseconds |
| bid_prices | json list | Top 50 bid price levels |
| bid_sizes | json list | Top 50 bid size levels |
| ask_prices | json list | Top 50 ask price levels |
| ask_sizes | json list | Top 50 ask size levels |
| mid_price | float64 | (best_bid + best_ask) / 2 |
| spread | float64 | best_ask - best_bid |
| imbalance | float64 | top-5 bid vol / (bid + ask vol) |

### Symbols
- BTCUSDT, ETHUSDT, SOLUSDT, LINKUSDT (Bybit linear perpetuals)

### Storage
- Daily Parquet files: `/home/linuxuser/tick_collector/data/{trades,orderbook}/{SYMBOL}/{YYYY-MM-DD}.parquet`
- Flush interval: 5 minutes (configurable)
- Estimated size: ~5-10 GB/day for 4 symbols

## Architecture

```
Bybit WebSocket (public, no API key needed)
    ├── WS Connection 1: publicTrade.{SYMBOL} → Trade handler
    └── WS Connection 2: orderbook.50.{SYMBOL} → Orderbook handler
                                ↓
                    ParquetWriter (thread-safe buffer)
                                ↓
                    Daily Parquet files (append-on-flush)
```

Two separate WebSocket connections required — pybit can't multiplex trades + orderbook on same connection.

## Operations

```bash
# Check status
sudo systemctl status tick-collector

# View live logs
journalctl -u tick-collector -f

# Stop/start
sudo systemctl stop tick-collector
sudo systemctl start tick-collector

# Check data volume
du -sh ~/tick_collector/data/

# Read sample data
python3 -c "import pandas as pd; print(pd.read_parquet('data/trades/BTCUSDT/2026-02-15.parquet').head())"
```

## Research Applications (when enough data collected)

### After 1-2 weeks:
1. **VPIN (Volume-Synchronized Probability of Informed Trading)**
   - Classify trades as buy/sell-initiated
   - Bucket by volume (not time)
   - Detect informed flow before large moves

2. **Order Book Imbalance Signals**
   - Bid/ask imbalance as directional predictor
   - Depth-weighted imbalance at multiple levels
   - Imbalance momentum (rate of change)

3. **Large Order Detection**
   - Trades > X standard deviations from mean size
   - Clustering of large trades (accumulation/distribution)
   - Iceberg detection (repeated same-size fills at same level)

### After 3-4 weeks:
4. **Market Making Simulations**
   - Test quote placement strategies on historical L2
   - Estimate fill probabilities at various distances from mid

5. **Microstructure Feature Engineering for ML**
   - Trade arrival rate, book pressure, spread dynamics
   - Features for regime classification model

## Configuration

Edit `/home/linuxuser/tick_collector/config.yaml`:

```yaml
symbols:
  - BTCUSDT
  - ETHUSDT
  - SOLUSDT
  - LINKUSDT

orderbook_interval: 1      # snapshot interval (seconds) — currently bypassed, uses raw WS
orderbook_depth: 50         # levels per side (Bybit supports: 1, 50, 200, 500)
data_dir: ./data
flush_interval: 300         # write to parquet every N seconds
testnet: false
category: linear
```

## Adding More Symbols

1. Add to `config.yaml` symbols list
2. Restart: `sudo systemctl restart tick-collector`
3. Note: each symbol adds ~1-2 GB/day at depth 50

## Disk Management

At ~5-10 GB/day:
- 1 week ≈ 35-70 GB
- VPS has ~70 GB free (as of 2026-02-14)
- Plan: keep 2-3 weeks rolling, archive older data to local Mac Mini or S3

```bash
# Check disk usage
du -sh ~/tick_collector/data/

# Archive old data (from VPS to local canonical store)
rsync -avz vps:~/tick_collector/data/trades/BTCUSDT/2026-02-1{5,6,7}.parquet /Volumes/G-DRIVE/maestro-data/tick/trades/BTCUSDT/
```

## Dependencies

- pybit >= 5.8.0 (Bybit WebSocket client)
- pandas >= 2.0 (DataFrame operations)
- pyarrow >= 14.0 (Parquet I/O)
- pyyaml >= 6.0 (config)

All installed in isolated venv: `/home/linuxuser/tick_collector/venv/`

## Virtuoso Gap

Virtuoso's `trade_executor.py` uses `orderbook_score` at 15% weight for execution decisions, but NO actual L2 data collection exists (it's a placeholder). This collector fills that gap independently — future integration path is to feed imbalance scores back to Virtuoso.
