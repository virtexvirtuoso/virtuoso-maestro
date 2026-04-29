# Maestro

*The Master Conductor of Trading Strategies*

Quantitative trading platform for algorithmic strategy development, backtesting, and walk-forward optimization.

## Pipeline Documentation

See **docs/03-developer-guide/architecture/PIPELINE.md** for the complete three-stage workflow:
- **Maestro** → Walk-forward validation (research)
- **Jesse** → Derivatives simulation (funding rates)
- **Freqtrade** → Live execution (free)

See **docs/03-developer-guide/architecture/MODERNIZATION.md** for the 2020→2026 stack upgrade plan (VectorBT, QuestDB, FastAPI, Optuna).

## Quick Reference

| Item | Value |
|------|-------|
| Entry Point (V1) | `backend/main/rest_api.py` (Flask, port 5050) |
| Entry Point (V2) | `backend/main/fastapi_app.py` (FastAPI, port 8000) |
| Config | `backend/config/maestro-dev.yaml` |
| Database | RethinkDB (localhost:28015, db: `filos-dev`) |
| Frontend | React + Material-UI + WebSocket |

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

## Component Responsibilities (Critical)

**NEVER violate this separation of concerns:**

| Component | MUST Do | MUST NOT Do |
|-----------|---------|-------------|
| DataFeed (`backend/datafeed/`) | Load OHLCV via `DataAdapter`, filter by date, return DataFrame | Query RethinkDB directly from strategies |
| Engine (`backend/engine_v2/`) | Run vectorized backtests, compute metrics, manage splits | Store results directly (use Storage layer) |
| Strategy (`backend/strategies/`) | Generate entry/exit signals from price data | Access DB, compute aggregate metrics, apply fees |
| Storage (`backend/storage/`) | Persist results and progress to RethinkDB | Run backtests or transform data |
| API (`backend/main/`) | Serve results, accept job requests, stream progress | Compute backtests inline in request handlers |

**Critical anti-patterns:**
- Strategies accessing RethinkDB directly — always use `DataAdapter.load_dataframe()`
- Fee/commission applied in both VectorBT `fees` param AND Numba `commission` — pick one, never both
- Metrics computed outside the engine — all Sharpe/VWR/Sortino must come from engine analyzers

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

## Environment Variables

**Backend:**
| Variable | Default | Purpose |
|----------|---------|---------|
| `CONFIG_FILE` | `maestro-dev.yaml` | YAML config path |
| `FASTAPI_PORT` | `8000` | FastAPI v2 server port |
| `COINALYZE_API_KEY` | None | Derivatives strategies (optional) |

**Frontend (`frontend/.env.local`):**
| Variable | Default | Purpose |
|----------|---------|---------|
| `REACT_APP_REST_API_V2_URL` | `http://localhost:5050` | FastAPI backend URL |
| `REACT_APP_WS_HOST` | `localhost:8000` | WebSocket host for progress |
| `REACT_APP_OPTUNA_DASHBOARD_URL` | `http://localhost:8080` | Optuna dashboard (optional) |
| `REACT_APP_USE_TRADINGVIEW` | `true` | TradingView chart feature flag |

## Database Schema (RethinkDB)

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `trade_{PROVIDER}_{symbol}_{bin_size}` | OHLCV candles | `timestamp, open, high, low, close, volume` |
| `trade_metadata` | Data catalog | `table_name, provider, symbol, bin_size, start` |
| `optimization_results` | Backtest/WF results | `tid, test_name, symbol, bin_size, analyzers, observers` |
| `optimization_progress` | Real-time job status | `tid, status, progress_pct, current_fold, total_folds` |

**Database names:** `filos-dev` (local) / `filos-prd` (Docker) — kept for legacy compatibility.

**No schema validation at DB level** — validators live in FastAPI Pydantic models only.

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

## Adding New Data Fields

When adding data to the pipeline, update ALL stages:

1. **Data Source** (`backend/datasource/`) — Fetch from exchange API
2. **RethinkDB Table** — Store in `trade_{PROVIDER}_{symbol}_{bin_size}`
3. **DataAdapter** (`backend/datafeed/data_adapter.py`) — Include in DataFrame load
4. **Engine** (`backend/engine_v2/`) — Consume in backtest if needed
5. **Result Converter** (`backend/engine_v2/result_converter.py`) — Map to output schema
6. **API Route** (`backend/main/`) — Expose via endpoint
7. **Frontend** — Display in React UI

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

## Testing

```bash
# Run all tests (332 items)
pytest backend/tests/ -v

# Engine tests only
pytest backend/tests/test_engine_v2.py -v

# V1 vs V2 parity check
pytest backend/tests/test_engine_parity.py -v

# Specific adapter
pytest backend/tests/test_data_adapter.py::TestRethinkDBAdapter -v
```

**Test suites:**
| File | Coverage |
|------|----------|
| `test_engine_v2.py` | VectorBT engine, strategies, walk-forward |
| `test_engine_parity.py` | V1 (Backtrader) vs V2 (VectorBT) result comparison |
| `test_data_adapter.py` | RethinkDB/Parquet/QuestDB adapters |
| `test_vwr_calculation.py` | Variability-Weighted Return metric |
| `test_result_converter.py` | V2→V1 schema mapping |
| `test_parquet_cache.py` | DataFrame cache I/O |
| `test_quantstats_reporter.py` | PyFolio analyzer output |

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

## Troubleshooting

| Symptom | Check |
|---------|-------|
| RethinkDB connection refused | Is `rethinkdb` process running? Default port 28015 |
| Empty DataFrame / no data | Verify table exists: `trade_metadata` catalog, check date range filter |
| Backtest returns zero trades | Strategy too restrictive, or data doesn't cover requested period |
| NaN in Sharpe/metrics | Division by zero — zero returns or single-trade result |
| Commission applied twice | VectorBT `fees` param AND Numba `commission` both set — use only one |
| Walk-forward all splits same params | Parameter grid too narrow — expand search ranges |
| WebSocket disconnects | Frontend auto-falls back to HTTP polling (2s). Check `FASTAPI_PORT` |
| Frontend can't reach API | Verify `REACT_APP_REST_API_V2_URL` in `.env.local` matches backend port |
| Win count off by one | Integer truncation in `result_converter.py:60` — `int(trades * win_rate)` loses fractional wins |
| Timezone mismatch | RethinkDB returns `utc=True` timestamps — ensure all date filters use UTC |

## Critical Code Warning

**Files that silently corrupt backtest results if modified incorrectly:**

| File | Risk | What to check |
|------|------|---------------|
| `backend/engine_v2/vectorbt_engine.py:177-179` | Data alignment | `reindex()` must preserve DatetimeIndex; wrong index type → all-False signals |
| `backend/utils/time_series_split_rolling.py` | Look-ahead bias | Test set must always come AFTER training set; verify split boundaries |
| `backend/engine_v2/result_converter.py:60-63` | Win rate truncation | `int(num_trades * win_rate)` loses fractional wins |
| `backend/engine_v2/vectorbt_engine.py:194` | Fee mode | VectorBT `fees` is percentage (0.001 = 0.1%); changing to absolute breaks everything |
| `backend/datafeed/data_adapter.py` | Data source swap | Changing adapter selection logic can silently load wrong/empty data |

**BEFORE modifying engine or strategy logic:**
1. Run `pytest backend/tests/test_engine_parity.py` — confirms V1/V2 produce same results
2. Check that `shift(1)` is used for crossover signals (prevents look-ahead bias)
3. Verify fee mode hasn't changed (percentage vs absolute)
4. Run a known strategy with known results to confirm output hasn't drifted

## API Endpoints (V2 — FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v2/optimization` | POST | Start backtest/walk-forward job |
| `/api/v2/optimization/{tid}/progress` | GET | Poll job status |
| `/api/v2/optimization/{tid}/ws` | WS | Real-time progress stream |
| `/api/v2/optimization/results` | GET | List all results |

**Frontend connection flow:** POST job → connect WebSocket for progress → fallback to HTTP polling (2s) if WS fails → fetch results on completion.

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
