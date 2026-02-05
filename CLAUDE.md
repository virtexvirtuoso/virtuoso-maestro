# Maestro

*The Master Conductor of Trading Strategies*

Quantitative trading platform for algorithmic strategy development, backtesting, and walk-forward optimization.

## Pipeline Documentation

See **docs/PIPELINE.md** for the complete three-stage workflow:
- **Maestro** → Walk-forward validation (research)
- **Jesse** → Derivatives simulation (funding rates)
- **Freqtrade** → Live execution (free)

See **docs/MODERNIZATION.md** for the 2020→2026 stack upgrade plan (VectorBT, QuestDB, FastAPI, Optuna).

## Quick Reference

| Item | Value |
|------|-------|
| Entry Point | `backend/main/rest_api.py` |
| Config | `backend/config/maestro-dev.yaml` |
| Database | RethinkDB (localhost:28015) |
| API Port | 5000 |
| Frontend | React + Highcharts |

## Architecture

```
Data Sources (Binance/BitMEX REST API)
         ↓
    Batch Downloaders (threaded workers)
         ↓
    RethinkDB (OHLCV storage)
         ↓
    ┌─────────────────────────────────────┐
    │         BACKTRADER ENGINE           │
    ├─────────────────────────────────────┤
    │ BacktestingEngine │ WalkForwardEngine│
    └─────────────────────────────────────┘
         ↓
    Strategies (11 built-in)
    + Analyzers (PyFolio, TradeAnalyzer)
         ↓
    Results → RethinkDB → REST API → React UI
```

## Key Directories

```
backend/
├── main/
│   ├── rest_api.py              # Flask REST API entry point
│   ├── data_binance_download.py # Binance data fetcher
│   └── data_bitmex_download.py  # BitMEX data fetcher
├── engine/
│   ├── backtesting_engine.py    # Single-pass backtest
│   ├── walk_forward_engine.py   # Walk-forward optimization
│   └── optimization_engine.py   # Base optimization class
├── strategy/
│   ├── base_strategy.py         # Abstract strategy base
│   ├── ema_cross_strategy.py
│   ├── bollinger_bands_strategy.py
│   ├── macd_strategy.py
│   ├── rsi_strategy.py
│   ├── ichimoku_strategy.py
│   └── __init__.py              # __STRATEGY_CATALOG__ registry
├── datasource/
│   ├── binance_batch_downloader.py
│   └── bitmex_batch_downloader.py
├── datafeed/
│   └── rethinkdb_datafeed_builder.py  # OHLCV → Backtrader feed
├── storage/
│   └── rethinkdb_storage_layer.py
└── config/
    ├── config_reader.py         # YAML parser
    ├── maestro-dev.yaml         # Dev config
    └── maestro-prd.yaml         # Prod config
```

## Core Concepts

### Walk-Forward Optimization
Primary analysis method - prevents overfitting via rolling time-series validation:
1. Split data into N periods (default: 10)
2. For each split:
   - **Train**: Grid search all parameter combinations
   - **Rank** by Sharpe + Value Recovery Ratio
   - **Test**: Run optimal params on out-of-sample data
3. Store each split's results

### Strategy Interface
```python
class MyStrategy(BaseStrategy):
    params = (('period', 20), ('threshold', 0.5))

    def _next(self):
        # Generate signals here
        if self.condition:
            self.buy()
```

Register in `__STRATEGY_CATALOG__` for auto-discovery.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/optimization/new/` | POST | Start backtest/walk-forward job |
| `/optimization/results/<tid>` | GET | Retrieve full results |
| `/optimization/progress/<tid>` | GET | Poll job progress |
| `/datasource/<provider>/<symbol>/<bin_size>/` | GET | Raw OHLCV data |

## Configuration

```yaml
# maestro-dev.yaml
config:
  rethinkdb:
    host: 127.0.0.1
    port: 28015
    db: filos-dev  # Database name kept for compatibility
  datasource:
    binance:
      symbols:
        ethbtc: {bin_size: [1d]}
    bitmex:
      symbols:
        xbtusd: {bin_size: [1m, 5m, 1h, 1d]}
  optimization:
    kind:
      walkforward:
        num_splits: 10
```

## Data Flow

```
Binance/BitMEX API
       ↓
BinanceBatchDownloaderWorker (threaded)
       ↓
RethinkDB: trade_PROVIDER_SYMBOL_TIMEFRAME tables
       ↓
RethinkDBDataFeedBuilder.build()
  → Query OHLCV
  → Filter by date range
  → Convert to pandas DataFrame
  → Wrap as bt.feeds.PandasData
       ↓
Backtrader Cerebro.run()
       ↓
Analyzers + Observers
       ↓
RethinkDB: optimization_results, optimization_progress
       ↓
REST API → React Frontend
```

## Adding New Strategies

1. Create `backend/strategy/my_strategy.py`
2. Inherit from `BaseStrategy`
3. Define `params` tuple
4. Implement `_next()` method
5. Add to `__STRATEGY_CATALOG__` in `__init__.py`

## Development

```bash
# Start RethinkDB
rethinkdb

# Activate environment
source venv/bin/activate

# Run API
cd backend/main
CONFIG_FILE=../config/maestro-dev.yaml python rest_api.py

# Run frontend
cd frontend
npm start
```

## Dependencies

- **backtrader** - Backtesting engine
- **rethinkdb** - NoSQL database
- **pandas/numpy** - Data analysis
- **scikit-learn** - TimeSeriesSplit
- **flask-restful** - REST API
- **hyperopt** - Hyperparameter optimization (optional)

## Anti-Cheating (Test Integrity)

- **NEVER** write mocks that return hardcoded values just to make tests pass
- **NEVER** stub out the actual behavior you're supposed to test
- **NEVER** modify tests to expect broken behavior instead of fixing the code
- **NEVER** add `skip`, `xfail`, or disable tests to hide failures
- **NEVER** hardcode `200 OK`, `true`, or expected outputs to bypass real logic
- **A passing test must prove the code works**, not prove you can fake a response
- If a test fails, **fix the underlying code**, not the test expectations
- Mocks are for **external dependencies** (APIs, DBs), not for hiding bugs
- When you write a mock, ask: "Am I testing real behavior or just my mock?"
- **Red flag**: If your "fix" is only in test files and not in source code, you're probably cheating

## Key Differences from Virtuoso

| Aspect | Maestro | Virtuoso |
|--------|---------|---------------|
| Focus | Backtesting & optimization | Live trading & signals |
| Data Sources | Binance + BitMEX (batch) | CCXT (30+ exchanges) |
| Database | RethinkDB | Memcached/Redis |
| Optimization | Walk-forward (rolling splits) | Grid search |
| Strategies | 11 TA-based | 6-dimensional confluence |

## Skills (Invoke Proactively)

Use these skills automatically when context matches:

| Skill | Trigger Phrases | When to Use Proactively |
|-------|-----------------|-------------------------|
| `/walkforward-debug` | "walk-forward", "optimization", "overfitting", "splits" | When investigating WFO results or parameter selection |
| `/strategy-debug` | "strategy error", "backtest failed", "no trades" | When strategy produces unexpected results |
| `/rethinkdb-ops` | "database", "RethinkDB", "data missing" | When OHLCV data issues or DB connectivity problems |
| `/backtrader-debug` | "Cerebro", "analyzer", "datafeed" | When Backtrader engine issues occur |

### Walk-Forward Debugging

**Optimization Results:**
```bash
# Check optimization progress
curl -s localhost:5000/optimization/progress/<tid> | jq

# Get full results
curl -s localhost:5000/optimization/results/<tid> | jq

# Check split-by-split performance
curl -s localhost:5000/optimization/results/<tid> | jq '.splits[] | {split_id, sharpe, params}'
```

**Common WFO Issues:**
| Symptom | Cause | Fix |
|---------|-------|-----|
| All splits same params | Grid too narrow | Expand parameter ranges |
| High train, low test Sharpe | Overfitting | Reduce params, increase splits |
| No trades in test period | Strategy too restrictive | Loosen entry conditions |
| NaN metrics | Division by zero | Check for zero returns/trades |

### RethinkDB Operations

```bash
# Check RethinkDB status
rethinkdb admin http://localhost:8080

# List tables
python -c "import rethinkdb as r; conn = r.connect(); print(list(r.db('filos-dev').table_list().run(conn)))"

# Check data availability
python -c "
import rethinkdb as r
conn = r.connect()
count = r.db('filos-dev').table('trade_binance_ethbtc_1d').count().run(conn)
print(f'Records: {count}')
"
```

### Strategy Testing

```python
# Quick strategy test
from backend.engine.backtesting_engine import BacktestingEngine
from backend.strategy import __STRATEGY_CATALOG__

engine = BacktestingEngine(config)
result = engine.run(
    strategy_name='ema_cross',
    symbol='ethbtc',
    params={'fast_period': 10, 'slow_period': 30}
)
print(result['sharpe_ratio'], result['total_trades'])
```

### Proactive Skill Usage Rules

1. **WFO produces poor results** → Check train vs test Sharpe ratio per split
2. **Strategy not trading** → Verify data exists for date range
3. **Optimization stuck** → Check RethinkDB connection and progress table
4. **Inconsistent metrics** → Verify analyzer configuration in Cerebro
