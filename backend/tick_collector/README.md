# Tick Collector

Standalone WebSocket tick data collector for Bybit perpetuals.
Collects L2 orderbook snapshots + trade ticks → Parquet files.

Designed to run independently from Virtuoso for research purposes.

## Components
- `collector.py` — Main WebSocket collector (trades + L2 snapshots)
- `config.yaml` — Symbols, intervals, storage settings
- `requirements.txt` — Dependencies

## Usage
```bash
# Install
pip install -r requirements.txt

# Run
python collector.py

# Output: data/ directory with daily Parquet files
# data/trades/BTCUSDT/2026-02-14.parquet
# data/orderbook/BTCUSDT/2026-02-14.parquet
```

## Data Schema

### Trades
| Column | Type | Description |
|--------|------|-------------|
| timestamp | int64 | Unix ms |
| price | float64 | Trade price |
| size | float64 | Trade size |
| side | str | Buy/Sell |

### Orderbook (L2 snapshots)
| Column | Type | Description |
|--------|------|-------------|
| timestamp | int64 | Unix ms |
| bid_prices | list[float] | Top N bid prices |
| bid_sizes | list[float] | Top N bid sizes |
| ask_prices | list[float] | Top N ask prices |
| ask_sizes | list[float] | Top N ask sizes |
| mid_price | float64 | (best_bid + best_ask) / 2 |
| spread | float64 | best_ask - best_bid |
| imbalance | float64 | bid_vol / (bid_vol + ask_vol) top 5 levels |
