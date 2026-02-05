# Maestro Modernization Implementation Plan

**Created**: 2026-02-04
**Status**: Planning Complete

## Executive Summary

This plan modernizes Maestro from a 2020 stack to 2026 while preserving the core walk-forward optimization methodology.

**Key Discovery**: The codebase already has a working `engine_v2` implementation:
- `backend/engine_v2/vectorbt_engine.py` - VectorBT wrapper
- `backend/engine_v2/walk_forward_optuna.py` - Walk-forward with Optuna
- `backend/engine_v2/strategy_adapter.py` - 4 strategies already converted

This significantly reduces Phase 1 scope.

---

## Current Architecture

```
Binance/BitMEX API
       ↓
batch_downloader_worker (threaded)
       ↓
RethinkDB (trade_PROVIDER_SYMBOL_TIMEFRAME tables)
       ↓
RethinkDBDataFeedBuilder → pandas DataFrame → bt.feeds.PandasData
       ↓
BacktestingEngine / WalkForwardEngine (Backtrader Cerebro)
       ↓
Results → RethinkDB → Flask REST API → React Frontend
```

## Critical Integration Points

| Component | File | Dependencies |
|-----------|------|--------------|
| Data Source | `rethinkdb_datafeed_builder.py` | RethinkDB, pandas, backtrader |
| Storage | `rethinkdb_storage_layer.py` | RethinkDB |
| Engine Base | `optimization_engine.py` | backtrader, RethinkDB, pyfolio_mock |
| Walk-Forward | `walk_forward_engine.py` | TimeSeriesSplitRolling, backtrader |
| REST API | `rest_api.py` | Flask, RethinkDB |

## Existing V2 Engine Status

**Already implemented** in `engine_v2/`:

1. **VectorBTEngine** - Complete `run()` and `run_multi()` methods
2. **WalkForwardOptuna** - Optuna with TPE sampler and pruning
3. **Strategy Adapters** - 4 strategies converted: EMA Cross, RSI, MACD, Bollinger Bands

---

## Phase 1: Core Engine Integration

**Objective**: Complete engine_v2 integration and expose via REST API.

### 1.1 Complete Strategy Conversions

Convert remaining 18 Backtrader strategies to VectorBT format.

**Priority order**:
1. IchimokuStrategy - Complex multi-indicator
2. ChannelStrategy - Breakout pattern
3. FernandoStrategy - Custom logic
4. VWAPStrategy - Scalping
5. TrendFollowingATR - Derivatives
6. MeanReversionBands - Derivatives
7. Remaining 12 strategies

**Conversion pattern**:
```python
class NewStrategy(VectorBTStrategy):
    @staticmethod
    def get_params() -> Dict[str, Any]:
        return {'param1': default_value}

    @staticmethod
    def get_param_space() -> Dict[str, Tuple]:
        return {'param1': ('int', min, max)}

    def generate_signals(self, data: pd.DataFrame, params: Dict) -> SignalOutput:
        # Vectorized indicator calculations
        # Return entries, exits, short_entries, short_exits
```

**Files to modify**:
- `backend/engine_v2/strategy_adapter.py` - Add new strategy classes
- `backend/strategy/__init__.py` - Update catalog for V2 strategies

### 1.2 Create Engine Facade

**New file**: `backend/engine/engine_factory.py`

```python
class EngineFactory:
    @staticmethod
    def create(engine_version: str, config: EngineConfig):
        if engine_version == 'v1':
            return BacktestingEngine(...)
        elif engine_version == 'v2':
            return VectorBTEngine(...)
        elif engine_version == 'v2_walkforward':
            return WalkForwardOptuna(...)
```

### 1.3 Data Adapter Layer

**New file**: `backend/datafeed/data_adapter.py`

```python
class DataAdapter(ABC):
    @abstractmethod
    def load_dataframe(self, provider, symbol, bin_size, start, end) -> pd.DataFrame:
        pass

class RethinkDBAdapter(DataAdapter):
    # Wrap existing RethinkDBDataFeedBuilder.read_data_frame()

class ParquetAdapter(DataAdapter):
    # For Phase 2 - local parquet files

class QuestDBAdapter(DataAdapter):
    # For Phase 2 - QuestDB source
```

### 1.4 Result Schema Compatibility

**New file**: `backend/engine_v2/result_converter.py`

Ensure V2 results match V1 schema for frontend compatibility:
```python
def convert_vbt_result_to_v1_schema(vbt_result: BacktestResult) -> dict:
    return {
        'analyzers': {
            'PyFolio': {
                'Sharpe ratio': vbt_result.sharpe_ratio,
                'Annual return': vbt_result.annual_return,
            }
        },
        'observers': {
            'BuySell': extract_trades_from_vbt(vbt_result)
        }
    }
```

### 1.5 Testing & Validation

**New file**: `backend/tests/test_engine_parity.py`

- Run same strategy/params through V1 and V2
- Compare Sharpe, returns, trade counts
- Allow 5% tolerance
- Target: 100x faster

---

## Phase 2: Data Layer Modernization

**Objective**: Replace RethinkDB with QuestDB, add Polars + Parquet.

### 2.1 QuestDB Setup

```bash
docker run -d --name questdb \
  -p 9000:9000 -p 8812:8812 -p 9009:9009 \
  questdb/questdb
```

**New file**: `backend/storage/questdb_storage_layer.py`

### 2.2 Data Migration Script

**New file**: `backend/scripts/migrate_rethinkdb_to_questdb.py`

### 2.3 Polars Integration

**New file**: `backend/datafeed/polars_loader.py`

```python
import polars as pl

class PolarsDataLoader:
    def load(self, source: str, **kwargs) -> pl.DataFrame:
        if source.endswith('.parquet'):
            return pl.scan_parquet(source).collect()
        elif source.startswith('questdb://'):
            return pl.read_database(query, connection)
```

### 2.4 Parquet Cache Layer

**New file**: `backend/storage/parquet_cache.py`

### 2.5 Update Config Schema

Add QuestDB and cache config to `maestro-dev.yaml`:

```yaml
config:
  database:
    type: questdb  # or rethinkdb for backwards compat
    questdb:
      host: 127.0.0.1
      port: 8812
  cache:
    parquet_dir: data/cache
    enabled: true
```

---

## Phase 3: API & Frontend Modernization

**Objective**: Flask → FastAPI, React 16 → React 18.

### 3.1 FastAPI Migration

**New file**: `backend/main/fastapi_app.py`

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Filos Trading API", version="2.0.0")

@app.post("/optimization/new/", response_model=OptimizationResponse)
async def new_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    tid = hash_test_name(request.test_name)
    background_tasks.add_task(run_optimization, tid, request)
    return OptimizationResponse(tid=tid, status="submitted")

@app.websocket("/optimization/progress/{tid}")
async def progress_websocket(websocket: WebSocket, tid: str):
    """Real-time progress updates via WebSocket"""
```

### 3.2 Endpoint Mapping

| Flask Route | FastAPI Route |
|-------------|---------------|
| `/strategy/available` | `/api/v2/strategies` |
| `/optimization/new/` | `/api/v2/optimization` |
| `/optimization/progress/<tid>` | `/api/v2/optimization/{tid}/progress` |
| `/optimization/results/<tid>` | `/api/v2/optimization/{tid}` |

### 3.3 React 18 Upgrade

**New project**: `frontend/maestroui-v2/`

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "@tanstack/react-query": "^5.0.0",
    "lightweight-charts": "^4.1.0",
    "@mui/material": "^5.15.0",
    "zustand": "^4.4.0"
  }
}
```

### 3.4 TradingView Charts

Replace Highcharts with TradingView Lightweight Charts:

```typescript
import { createChart, CandlestickSeries } from 'lightweight-charts';

export function TradingChart({ data, trades }: ChartProps) {
  // Candlestick + trade markers
}
```

---

## Phase 4: Enhanced Analytics

### 4.1 QuantStats Integration

**New file**: `backend/analytics/quantstats_reporter.py`

```python
import quantstats as qs

class QuantStatsReporter:
    def generate_report(self, returns: pd.Series) -> dict:
        return {
            'sharpe': qs.stats.sharpe(returns),
            'sortino': qs.stats.sortino(returns),
            'max_drawdown': qs.stats.max_drawdown(returns),
            'win_rate': qs.stats.win_rate(returns),
        }
```

### 4.2 Optuna Dashboard

Add visualization endpoints for optimization insights.

---

## Docker Compose Updates

```yaml
version: "3.9"
services:
  maestro-api:
    command: uvicorn main.fastapi_app:app --host 0.0.0.0 --port 5000
    depends_on:
      - questdb
      - redis

  questdb:
    image: questdb/questdb:7.3.10
    ports:
      - "9000:9000"
      - "8812:8812"

  redis:
    image: redis:7-alpine

  # Keep RethinkDB for migration period
  rethinkdb:
    image: rethinkdb:2.4
```

---

## Migration Strategy

### Parallel Operation (Phases 1-2)

Both V1 and V2 engines run simultaneously:
```python
@app.post("/optimization/new/")
async def new_optimization(request: OptimizationRequest):
    if request.engine_version == "v2":
        return await run_v2_optimization(request)
    else:
        return run_v1_optimization(request)
```

### Data Migration

1. Export RethinkDB → Parquet files
2. Import Parquet → QuestDB
3. Validate row counts
4. Run parallel reads to verify

### Frontend Rollout

1. Deploy new frontend at `/v2`
2. A/B test
3. Redirect `/` to new frontend
4. Keep `/legacy` for 30 days

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| VectorBT results differ | Parity tests, 5% tolerance, document differences |
| QuestDB issues | Keep RethinkDB as fallback |
| Strategy conversion bugs | Test each individually vs Backtrader |
| Frontend regression | Keep old frontend during transition |
| Data loss | Parquet backup before destructive ops |

---

## Performance Targets

| Metric | Current (V1) | Target (V2) |
|--------|--------------|-------------|
| Single backtest | 30-60 sec | <1 sec |
| Walk-forward (10 splits, 100 trials) | 30+ min | <3 min |
| Data query (1 year, 1h bars) | 2-3 sec | <100ms |
| API response | 100-200ms | <50ms |

---

## Dependencies

**New requirements_v2.txt**:
```
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0

# Data
polars>=0.20.0
pyarrow>=15.0.0
psycopg2-binary>=2.9.9

# Backtesting
vectorbtpro>=1.0.0
optuna>=3.5.0

# Analytics
quantstats>=0.0.62

# Legacy (transition)
backtrader>=1.9.78
rethinkdb>=2.4.8
flask>=2.3.0
```

---

## Success Criteria

### Phase 1 Complete
- [ ] All 22 strategies converted to VectorBT
- [ ] `/optimization/new/` accepts `engine_version: v2`
- [ ] V2 results match V1 within 5%
- [ ] Walk-forward 100x faster

### Phase 2 Complete
- [ ] QuestDB running with all data
- [ ] Parquet cache operational
- [ ] RethinkDB can be disabled
- [ ] Data queries 20x faster

### Phase 3 Complete
- [ ] FastAPI serving all endpoints
- [ ] WebSocket progress working
- [ ] React 18 frontend deployed
- [ ] TradingView charts rendering

### Project Complete
- [ ] All original functionality preserved
- [ ] Performance targets met
- [ ] Old stack deprecated
- [ ] Documentation updated

---

## Critical Files

| File | Purpose |
|------|---------|
| `backend/engine_v2/strategy_adapter.py` | Add remaining 18 strategies (Phase 1 primary work) |
| `backend/engine_v2/walk_forward_optuna.py` | Connect to API, add result conversion |
| `backend/main/rest_api.py` | Add engine routing, becomes FastAPI target |
| `backend/datafeed/rethinkdb_datafeed_builder.py` | Abstract to DataAdapter interface |
| `backend/strategy/__init__.py` | Update catalog for V2 registry |
