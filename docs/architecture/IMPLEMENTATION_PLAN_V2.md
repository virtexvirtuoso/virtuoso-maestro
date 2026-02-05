# Maestro Modernization Implementation Plan V2

**Created**: 2026-02-04
**Revised**: 2026-02-04 (Incorporating Multi-Agent Consensus Review)
**Status**: Approved with Conditions

---

## Executive Summary

This plan modernizes Maestro from a 2020 stack (Backtrader, RethinkDB, Flask, React 16) to a 2026 stack (VectorBT, QuestDB, FastAPI, React 18) while preserving the core walk-forward optimization methodology.

### Key Discoveries

1. **Existing V2 Engine**: The codebase already has a working `engine_v2/` implementation with VectorBT and Optuna integration
2. **Strategy Status**: **18 total strategies**, with **all 18 already converted** to VectorBT format in `strategy_adapter.py`
3. **Blocking Gaps**: REST API has no V2 routing, VWR metric missing from V2 ranking

### Critical Revisions from Agent Review

| Finding | Impact | Resolution |
|---------|--------|------------|
| Strategy count was wrong (22 -> 18) | Timeline overestimated | Corrected; focus shifts to testing and API integration |
| REST API has no V2 routing | Blocking | Added as Phase 1 explicit task |
| VWR metric missing in V2 | Methodology degradation | Must add before Phase 1 complete |
| Parity tolerance too loose (5%) | False confidence | Tightened to 2% tolerance, 95% trade alignment |
| No Phase 0 infrastructure | Deployment risk | Added Phase 0: Infrastructure Preparation |
| Non-vectorizable strategies | Conversion impossible | FernandoStrategy, GridTradingStrategy require rewrite (but already have simplified vectorized versions) |
| Charting migration underestimated | Timeline risk | Budget 2-3 extra weeks for Phase 3 |

### Performance Targets

| Metric | Current (V1) | Target (V2) | Improvement |
|--------|--------------|-------------|-------------|
| Single backtest | 30-60 sec | <1 sec | 30-60x |
| Walk-forward (10 splits, 100 trials) | 30+ min | <3 min | 10x |
| Data query (1 year, 1h bars) | 2-3 sec | <100ms | 20-30x |
| API response | 100-200ms | <50ms | 2-4x |

---

## Current Architecture

```
Binance/BitMEX API
       |
batch_downloader_worker (threaded)
       |
RethinkDB (trade_PROVIDER_SYMBOL_TIMEFRAME tables)
       |
RethinkDBDataFeedBuilder -> pandas DataFrame -> bt.feeds.PandasData
       |
BacktestingEngine / WalkForwardEngine (Backtrader Cerebro)
       |
Results -> RethinkDB -> Flask REST API -> React Frontend
```

## Target Architecture

```
Binance/BitMEX API
       |
async_downloader (asyncio)
       |
QuestDB (time-series optimized) + Parquet cache
       |
PolarsDataLoader -> Polars/Pandas DataFrame
       |
VectorBTEngine / WalkForwardOptuna (VectorBT + Optuna)
       |
Results -> QuestDB -> FastAPI (async) + WebSocket -> React 18 Frontend
```

---

## Phase 0: Infrastructure Preparation (NEW)

**Duration**: 1 week
**Objective**: Establish CI/CD, monitoring, and rollback capabilities before making changes.

### 0.1 CI/CD Pipeline Setup

- [ ] Create GitHub Actions workflow for Python tests
- [ ] Add linting (ruff) and type checking (mypy) to pipeline
- [ ] Configure test coverage reporting (pytest-cov)
- [ ] Set up automated Docker image builds

**File to create**: `.github/workflows/ci.yml`

```yaml
name: Filos CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      rethinkdb:
        image: rethinkdb:2.4
        ports:
          - 28015:28015
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest backend/tests/ --cov=backend --cov-report=xml
      - run: ruff check backend/
```

**Acceptance Criteria**:
- [ ] CI runs on every PR
- [ ] All existing tests pass
- [ ] Coverage baseline established

### 0.2 Docker Health Checks

- [ ] Add health checks to `docker-compose.yml` for all services
- [ ] Create startup dependency chain

**File to modify**: `docker-compose.yml`

```yaml
services:
  rethinkdb:
    image: rethinkdb:2.4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080"]
      interval: 10s
      timeout: 5s
      retries: 3

  maestro-api:
    depends_on:
      rethinkdb:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/strategy/available"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Acceptance Criteria**:
- [ ] `docker compose up` waits for healthy services
- [ ] Failed health check prevents dependent service start

### 0.3 Rollback Runbooks

- [ ] Document rollback procedure for each phase
- [ ] Create database backup scripts
- [ ] Test rollback on staging environment

**File to create**: `docs/runbooks/ROLLBACK.md`

**Acceptance Criteria**:
- [ ] Runbook exists for each phase
- [ ] Backup scripts tested
- [ ] Rollback can be executed in <15 minutes

### 0.4 Parity Test Framework

- [ ] Create framework for comparing V1 vs V2 engine results
- [ ] Define test datasets and expected outcomes
- [ ] Implement automated comparison with tolerance checks

**File to create**: `backend/tests/test_engine_parity.py`

```python
"""
Parity Test Framework - Ensures V2 engine produces equivalent results to V1

Criteria (from agent review):
- 2% tolerance on numerical metrics (Sharpe, returns, drawdown)
- 95% trade alignment (entry/exit timing within 1 bar)
- Equity curve correlation > 0.95
"""

import pytest
import pandas as pd
import numpy as np
from backend.engine.backtesting_engine import BacktestingEngine
from backend.engine_v2.vectorbt_engine import VectorBTEngine
from backend.engine_v2.strategy_adapter import STRATEGY_REGISTRY


class ParityTestConfig:
    NUMERICAL_TOLERANCE = 0.02  # 2%
    TRADE_ALIGNMENT_THRESHOLD = 0.95  # 95%
    EQUITY_CORRELATION_MIN = 0.95


def compare_results(v1_result: dict, v2_result: dict) -> dict:
    """Compare V1 and V2 results, return discrepancies"""
    discrepancies = {}

    # Sharpe ratio comparison
    v1_sharpe = v1_result['analyzers']['PyFolio']['Sharpe ratio']
    v2_sharpe = v2_result.sharpe_ratio
    sharpe_diff = abs(v1_sharpe - v2_sharpe) / max(abs(v1_sharpe), 0.001)
    if sharpe_diff > ParityTestConfig.NUMERICAL_TOLERANCE:
        discrepancies['sharpe_ratio'] = {
            'v1': v1_sharpe, 'v2': v2_sharpe, 'diff_pct': sharpe_diff
        }

    # Trade alignment
    v1_trades = extract_trades_v1(v1_result)
    v2_trades = extract_trades_v2(v2_result)
    alignment = calculate_trade_alignment(v1_trades, v2_trades)
    if alignment < ParityTestConfig.TRADE_ALIGNMENT_THRESHOLD:
        discrepancies['trade_alignment'] = alignment

    return discrepancies


@pytest.mark.parametrize("strategy_name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_parity(strategy_name, sample_data):
    """Test that V2 produces equivalent results to V1 for each strategy"""
    # Skip strategies that cannot be fully vectorized
    if strategy_name in ['FundingRateArbitrage']:  # Requires external data
        pytest.skip(f"{strategy_name} requires external data feed")

    v1_result = run_v1_backtest(strategy_name, sample_data)
    v2_result = run_v2_backtest(strategy_name, sample_data)

    discrepancies = compare_results(v1_result, v2_result)
    assert not discrepancies, f"Parity failed: {discrepancies}"
```

**Acceptance Criteria**:
- [ ] Parity tests exist for all vectorizable strategies
- [ ] Tests run in CI pipeline
- [ ] Clear reporting of discrepancies

### Phase 0 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CI setup delays | Low | Low | Use existing GitHub Actions templates |
| Rollback procedure gaps | Medium | High | Test rollback before Phase 1 |
| Parity framework complexity | Medium | Medium | Start with simple metric comparison |

### Phase 0 Rollback Procedure

N/A - Phase 0 is preparation only; no production changes.

---

## Phase 1: Core Engine Integration

**Duration**: 2-3 weeks
**Objective**: Complete engine_v2 integration and expose via REST API.
**Dependencies**: Phase 0 complete

### 1.1 Add VWR Metric to V2 Engine

**Priority**: BLOCKING - Must complete before other Phase 1 tasks.

The V1 engine ranks optimization results by `['sharpe_ratio', 'vwr']`. V2 currently only uses Sharpe ratio, which can lead to methodology degradation (overfitting selection).

**Files to modify**:
- `backend/engine_v2/vectorbt_engine.py` - Add VWR calculation
- `backend/engine_v2/walk_forward_optuna.py` - Use VWR in ranking

```python
# backend/engine_v2/vectorbt_engine.py

def calculate_vwr(returns: pd.Series, annualization_factor: int = 365) -> float:
    """
    Calculate Variability-Weighted Return (VWR)

    VWR = Mean Return / (Std Dev * tau) where tau is penalty for variability
    This penalizes high-variance strategies even if they have high returns.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = returns.mean() * annualization_factor
    std_dev = returns.std() * np.sqrt(annualization_factor)

    if std_dev == 0:
        return 0.0

    # tau = 2.0 (standard VWR penalty factor)
    vwr = mean_return / (std_dev * 2.0)
    return vwr


@dataclass
class BacktestResult:
    # ... existing fields ...
    vwr: float = 0.0  # ADD THIS FIELD
```

**Acceptance Criteria**:
- [ ] VWR calculated identically to V1 (within 1% tolerance)
- [ ] V2 optimization ranking uses `['sharpe_ratio', 'vwr']`
- [ ] Parity test for VWR passes

### 1.2 Add REST API V2 Routing

**Priority**: BLOCKING - Required for V2 engine to be accessible.

The current `rest_api.py` has no mechanism to route to V2 engines.

**Files to modify**:
- `backend/main/rest_api.py` - Add engine_version parameter
- `backend/engine/optimizer.py` - Route to V2 engines

```python
# backend/main/rest_api.py

class Optimization(Resource):
    def post(self):
        data = request.get_json()
        if not data:
            abort(400)

        # NEW: Check for engine version
        engine_version = data.get('engine_version', 'v1')

        if engine_version == 'v2':
            return self._run_v2_optimization(data)
        else:
            return self._run_v1_optimization(data)

    def _run_v2_optimization(self, data):
        """Route to VectorBT/Optuna engine"""
        from engine_v2.vectorbt_engine import VectorBTEngine
        from engine_v2.walk_forward_optuna import WalkForwardOptuna
        from engine_v2.strategy_adapter import STRATEGY_REGISTRY

        # Get strategy
        strategy_name = data['strategy']
        if strategy_name not in STRATEGY_REGISTRY:
            abort(400, error=f"Strategy {strategy_name} not available in V2")

        strategy_class = STRATEGY_REGISTRY[strategy_name]
        strategy = strategy_class()

        # ... rest of implementation
```

**File to create**: `backend/engine/engine_factory.py`

```python
"""
Engine Factory - Unified interface for V1 and V2 engines
"""

from enum import Enum
from typing import Type, Any
from dataclasses import dataclass


class EngineVersion(Enum):
    V1 = 'v1'
    V2 = 'v2'


class EngineFactory:
    """Factory for creating backtesting and optimization engines"""

    @staticmethod
    def create_backtest_engine(version: EngineVersion, config: Any):
        if version == EngineVersion.V1:
            from engine.backtesting_engine import BacktestingEngine
            return BacktestingEngine
        elif version == EngineVersion.V2:
            from engine_v2.vectorbt_engine import VectorBTEngine
            return VectorBTEngine
        raise ValueError(f"Unknown engine version: {version}")

    @staticmethod
    def create_walkforward_engine(version: EngineVersion, config: Any):
        if version == EngineVersion.V1:
            from engine.walk_forward_engine import WalkForwardEngine
            return WalkForwardEngine
        elif version == EngineVersion.V2:
            from engine_v2.walk_forward_optuna import WalkForwardOptuna
            return WalkForwardOptuna
        raise ValueError(f"Unknown engine version: {version}")
```

**Acceptance Criteria**:
- [ ] `POST /optimization/new/` accepts `engine_version: v2`
- [ ] V2 engine runs successfully via API
- [ ] Results returned in V1-compatible schema

### 1.3 Validate Existing Strategy Conversions

All 18 strategies are already converted in `backend/engine_v2/strategy_adapter.py`:

| Category | Strategies | Status |
|----------|------------|--------|
| **Classic (5)** | EmaCrossStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy, ChannelStrategy | Converted |
| **Classic Extended (4)** | IchimokuStrategy, MaCrossStrategy, FernandoStrategy*, MACDStrategyX | Converted |
| **Scalping (5)** | VWAPStrategy, ScalpRSIStrategy, MomentumBreakoutStrategy, StochRSIStrategy, EMARibbonStrategy | Converted |
| **Derivatives (4)** | FundingRateArbitrage**, BasisTradingStrategy, OpenInterestDivergence, LiquidationHuntStrategy | Converted |
| **Derivatives Extended (4)** | GridTradingStrategy*, VolatilityBreakoutStrategy, TrendFollowingATR, MeanReversionBands | Converted |

*\* Note: FernandoStrategy and GridTradingStrategy have simplified vectorized versions. The original Backtrader versions use stateful order management (bracket orders, concurrent limit orders) that cannot be fully vectorized. Parity testing will document the behavioral differences.*

*\*\* FundingRateArbitrage uses momentum as a proxy for funding rate since actual funding rate requires external data feed.*

- [ ] Run parity tests for each strategy
- [ ] Document behavioral differences for simplified strategies
- [ ] Update strategy registry with V2 availability flags

**File to modify**: `backend/engine_v2/strategy_adapter.py`

```python
# Add availability metadata
STRATEGY_METADATA = {
    'FernandoStrategy': {
        'v2_available': True,
        'parity_notes': 'Simplified: bracket orders replaced with signal-based exits',
        'expected_parity': 0.85,  # Lower threshold due to order logic differences
    },
    'GridTradingStrategy': {
        'v2_available': True,
        'parity_notes': 'Simplified: concurrent limit orders replaced with band touches',
        'expected_parity': 0.80,
    },
    'FundingRateArbitrage': {
        'v2_available': True,
        'parity_notes': 'Uses momentum proxy; requires external funding rate data for full accuracy',
        'expected_parity': 0.70,
    },
    # ... other strategies with expected_parity: 0.98
}
```

### 1.4 Result Schema Compatibility

**File to create**: `backend/engine_v2/result_converter.py`

```python
"""
Result Converter - Converts V2 results to V1 schema for frontend compatibility
"""

from dataclasses import asdict
from typing import Dict, Any, List
from .vectorbt_engine import BacktestResult


def convert_v2_to_v1_schema(result: BacktestResult) -> Dict[str, Any]:
    """
    Convert VectorBT result to V1 schema for frontend compatibility.

    V1 Schema (from optimization_engine.py):
    {
        'analyzers': {
            'PyFolio': {
                'Sharpe ratio': float,
                'Annual return': float,
                'Annual volatility': float,
                'Max drawdown': float,
                ...
            },
            'TradeAnalyzer': {...}
        },
        'observers': {
            'BuySell': {
                'buy': [[timestamp_ms, price], ...],
                'sell': [[timestamp_ms, price], ...]
            }
        },
        'start_date': int,  # timestamp_ms
        'end_date': int,    # timestamp_ms
        ...
    }
    """
    return {
        'analyzers': {
            'PyFolio': {
                'Sharpe ratio': result.sharpe_ratio,
                'Annual return': result.annual_return,
                'Annual volatility': result.volatility,
                'Max drawdown': result.max_drawdown,
                'Cumulative returns': result.total_return,
                'Sortino ratio': result.sortino_ratio,
                'Calmar ratio': result.calmar_ratio,
                'Win rate': result.win_rate,
                'Profit factor': result.profit_factor,
                # VWR - new metric
                'VWR': result.vwr,
            },
            'TradeAnalyzer': {
                'total': {'total': result.num_trades},
            }
        },
        'observers': {
            'BuySell': extract_buysell_from_v2(result),
        },
        'start_date': int(result.start_date.timestamp() * 1000) if result.start_date else None,
        'end_date': int(result.end_date.timestamp() * 1000) if result.end_date else None,
        'processing_time': result.processing_time,
        'parameters': result.parameters or {},
    }


def extract_buysell_from_v2(result: BacktestResult) -> Dict[str, List]:
    """Extract buy/sell signals from V2 result for charting"""
    buys = []
    sells = []

    if result.trades is not None:
        for _, trade in result.trades.iterrows():
            entry_ts = int(trade['entry_time'].timestamp() * 1000)
            exit_ts = int(trade['exit_time'].timestamp() * 1000)

            if trade.get('direction', 1) > 0:  # Long
                buys.append([entry_ts, trade['entry_price']])
                sells.append([exit_ts, trade['exit_price']])
            else:  # Short
                sells.append([entry_ts, trade['entry_price']])
                buys.append([exit_ts, trade['exit_price']])

    return {'buy': buys, 'sell': sells}
```

**Acceptance Criteria**:
- [ ] V2 results display correctly in existing frontend
- [ ] All chart data (buy/sell markers) renders
- [ ] Excel export works with V2 results

### 1.5 Data Adapter Layer

**File to create**: `backend/datafeed/data_adapter.py`

```python
"""
Data Adapter - Abstraction layer for data sources
Enables switching between RethinkDB (V1), Parquet (cache), and QuestDB (V2)
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import pandas as pd


class DataAdapter(ABC):
    """Abstract base class for data adapters"""

    @abstractmethod
    def load_dataframe(
        self,
        provider: str,
        symbol: str,
        bin_size: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load OHLCV data as DataFrame"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available"""
        pass


class RethinkDBAdapter(DataAdapter):
    """Adapter for RethinkDB data source (V1 compatibility)"""

    def __init__(self, host: str, port: int, db: str):
        from datafeed.rethinkdb_datafeed_builder import RethinkDBDataFeedBuilder
        self.builder = RethinkDBDataFeedBuilder(host=host, port=port, db=db)

    def load_dataframe(self, provider, symbol, bin_size, start_date=None, end_date=None):
        return self.builder.read_data_frame(
            provider=provider,
            symbol=symbol,
            bin_size=bin_size,
            start_date=start_date,
            end_date=end_date
        )

    def is_available(self):
        try:
            from rethinkdb import RethinkDB
            r = RethinkDB()
            conn = r.connect(host=self.builder.host, port=self.builder.port)
            conn.close()
            return True
        except:
            return False


class ParquetAdapter(DataAdapter):
    """Adapter for Parquet file cache (Phase 2)"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def load_dataframe(self, provider, symbol, bin_size, start_date=None, end_date=None):
        import polars as pl
        from pathlib import Path

        parquet_path = Path(self.cache_dir) / f"{provider}_{symbol}_{bin_size}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Cache not found: {parquet_path}")

        df = pl.read_parquet(parquet_path).to_pandas()

        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]

        return df

    def is_available(self):
        from pathlib import Path
        return Path(self.cache_dir).exists()


class QuestDBAdapter(DataAdapter):
    """Adapter for QuestDB (Phase 2)"""

    def __init__(self, host: str = 'localhost', port: int = 8812):
        self.host = host
        self.port = port

    def load_dataframe(self, provider, symbol, bin_size, start_date=None, end_date=None):
        import psycopg2

        table_name = f"trade_{provider.upper()}_{symbol.lower()}_{bin_size}"

        query = f"SELECT * FROM {table_name}"
        conditions = []

        if start_date:
            conditions.append(f"timestamp >= '{start_date.isoformat()}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date.isoformat()}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        conn = psycopg2.connect(
            host=self.host, port=self.port,
            database='qdb', user='admin', password='quest'
        )

        df = pd.read_sql(query, conn)
        conn.close()

        return df

    def is_available(self):
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.host, port=self.port,
                database='qdb', user='admin', password='quest'
            )
            conn.close()
            return True
        except:
            return False
```

### Phase 1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Parity tests fail significantly | Medium | High | Document differences, adjust thresholds for known cases |
| VWR calculation differs from V1 | Low | Medium | Use exact same formula as Backtrader analyzer |
| API integration breaks frontend | Medium | Medium | Keep V1 as default, V2 opt-in |
| Performance worse than expected | Low | Low | V2 is already proven faster; benchmark early |

### Phase 1 Rollback Procedure

1. **Revert API changes**: Remove `engine_version` parameter handling
2. **Disable V2 routing**: Comment out V2 engine imports
3. **Restore default**: Ensure `engine_version` defaults to `v1`
4. **Verify**: Run smoke tests on V1 endpoints

```bash
# Quick rollback script
git revert <phase1-commits> --no-commit
docker compose restart maestro-api
curl -X POST localhost:5000/optimization/new/ -d '{"engine_version": "v1", ...}'
```

### Phase 1 Success Criteria

- [ ] **16+ of 18 strategies pass parity tests** with 2% tolerance, 95% trade alignment
- [ ] `/optimization/new/` accepts `engine_version: v2`
- [ ] V2 results match V1 schema (frontend displays correctly)
- [ ] VWR metric available in V2 results
- [ ] CI pipeline passing
- [ ] Walk-forward 10x+ faster than V1

---

## Phase 2: Data Layer Modernization

**Duration**: 2-3 weeks
**Objective**: Replace RethinkDB with QuestDB, add Polars + Parquet caching.
**Dependencies**: Phase 1 complete

### 2.1 QuestDB Setup and Configuration

- [ ] Add QuestDB to docker-compose.yml
- [ ] Create initialization scripts
- [ ] Configure connection pooling

**File to modify**: `docker-compose.yml`

```yaml
services:
  questdb:
    image: questdb/questdb:7.3.10
    container_name: maestro-questdb
    ports:
      - "9000:9000"   # Web console
      - "8812:8812"   # PostgreSQL wire protocol
      - "9009:9009"   # InfluxDB line protocol
    volumes:
      - questdb-data:/var/lib/questdb
    environment:
      - QDB_PG_USER=admin
      - QDB_PG_PASSWORD=quest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  questdb-data:
```

**File to create**: `backend/storage/questdb_storage_layer.py`

```python
"""
QuestDB Storage Layer - High-performance time-series storage

Key features:
- Connection pooling (psycopg2.pool)
- Batch ingestion via ILP (InfluxDB Line Protocol)
- Optimized time-series queries
"""

import psycopg2
from psycopg2 import pool
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
import socket


class QuestDBStorage:
    """QuestDB storage layer with connection pooling"""

    def __init__(
        self,
        host: str = 'localhost',
        pg_port: int = 8812,
        ilp_port: int = 9009,
        min_connections: int = 2,
        max_connections: int = 10
    ):
        self.host = host
        self.pg_port = pg_port
        self.ilp_port = ilp_port

        # Connection pool for queries
        self._pool = pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            host=host,
            port=pg_port,
            database='qdb',
            user='admin',
            password='quest'
        )

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def create_trade_table(self, provider: str, symbol: str, bin_size: str):
        """Create a trade table with optimal schema for time-series"""
        table_name = f"trade_{provider.upper()}_{symbol.lower()}_{bin_size}"

        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        ) timestamp(timestamp) PARTITION BY DAY;
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()

    def ingest_dataframe(self, df: pd.DataFrame, provider: str, symbol: str, bin_size: str):
        """Ingest DataFrame using ILP for maximum performance"""
        table_name = f"trade_{provider.upper()}_{symbol.lower()}_{bin_size}"

        # Use ILP for fast ingestion (1M+ rows/sec)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.ilp_port))

        try:
            for _, row in df.iterrows():
                # ILP format: table,tag=value field=value timestamp
                ts_ns = int(row['timestamp'].timestamp() * 1e9)
                line = (
                    f"{table_name} "
                    f"open={row['open']},high={row['high']},low={row['low']},"
                    f"close={row['close']},volume={row['volume']} "
                    f"{ts_ns}\n"
                )
                sock.sendall(line.encode())
        finally:
            sock.close()

    def query_ohlcv(
        self,
        provider: str,
        symbol: str,
        bin_size: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Query OHLCV data with time range filter"""
        table_name = f"trade_{provider.upper()}_{symbol.lower()}_{bin_size}"

        query = f"SELECT * FROM {table_name}"
        conditions = []

        if start_date:
            conditions.append(f"timestamp >= '{start_date.isoformat()}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date.isoformat()}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            return pd.read_sql(query, conn)

    def close(self):
        """Close connection pool"""
        self._pool.closeall()
```

### 2.2 Data Migration Script

**File to create**: `backend/scripts/migrate_rethinkdb_to_questdb.py`

```python
"""
RethinkDB to QuestDB Migration Script

Features:
- Dry-run mode with row counts and checksums
- Batch migration with progress reporting
- Validation after migration
- Rollback support
"""

import argparse
import hashlib
import logging
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
from rethinkdb import RethinkDB
from storage.questdb_storage_layer import QuestDBStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MigrationValidator:
    """Validate migration integrity"""

    @staticmethod
    def calculate_checksum(df: pd.DataFrame) -> str:
        """Calculate checksum for DataFrame"""
        # Use subset of columns for checksum (timestamp, close, volume)
        checksum_cols = ['timestamp', 'close', 'volume']
        data = df[checksum_cols].to_csv(index=False).encode()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def validate_migration(
        rethink_df: pd.DataFrame,
        quest_df: pd.DataFrame
    ) -> Dict[str, any]:
        """Validate that migration preserved data integrity"""
        results = {
            'row_count_match': len(rethink_df) == len(quest_df),
            'rethink_rows': len(rethink_df),
            'quest_rows': len(quest_df),
            'checksum_match': False,
            'float_precision_issues': [],
        }

        if results['row_count_match']:
            rethink_checksum = MigrationValidator.calculate_checksum(rethink_df)
            quest_checksum = MigrationValidator.calculate_checksum(quest_df)
            results['checksum_match'] = rethink_checksum == quest_checksum
            results['rethink_checksum'] = rethink_checksum
            results['quest_checksum'] = quest_checksum

            # Check float precision
            for col in ['open', 'high', 'low', 'close', 'volume']:
                diff = (rethink_df[col] - quest_df[col]).abs()
                if diff.max() > 1e-10:
                    results['float_precision_issues'].append({
                        'column': col,
                        'max_diff': diff.max()
                    })

        return results


def migrate_table(
    rethink_config: Dict,
    quest_storage: QuestDBStorage,
    table_name: str,
    dry_run: bool = True
) -> Tuple[bool, Dict]:
    """Migrate a single table"""

    # Parse table name: trade_PROVIDER_SYMBOL_BINSIZE
    parts = table_name.split('_')
    provider = parts[1]
    symbol = parts[2]
    bin_size = parts[3]

    logger.info(f"Migrating {table_name}...")

    # Read from RethinkDB
    r = RethinkDB()
    conn = r.connect(**rethink_config)

    cursor = r.table(table_name).run(conn)
    records = list(cursor)
    rethink_df = pd.DataFrame(records)

    if 'timestamp' not in rethink_df.columns:
        logger.warning(f"Table {table_name} has no timestamp column, skipping")
        return False, {}

    # Convert timestamp if needed
    if not pd.api.types.is_datetime64_any_dtype(rethink_df['timestamp']):
        rethink_df['timestamp'] = pd.to_datetime(rethink_df['timestamp'])

    conn.close()

    logger.info(f"  Source rows: {len(rethink_df)}")

    if dry_run:
        logger.info(f"  DRY RUN: Would migrate {len(rethink_df)} rows")
        checksum = MigrationValidator.calculate_checksum(rethink_df)
        logger.info(f"  Source checksum: {checksum}")
        return True, {'rows': len(rethink_df), 'checksum': checksum}

    # Create table and ingest
    quest_storage.create_trade_table(provider, symbol, bin_size)
    quest_storage.ingest_dataframe(rethink_df, provider, symbol, bin_size)

    # Validate
    quest_df = quest_storage.query_ohlcv(provider, symbol, bin_size)
    validation = MigrationValidator.validate_migration(rethink_df, quest_df)

    if validation['row_count_match'] and validation['checksum_match']:
        logger.info(f"  Migration validated successfully")
    else:
        logger.error(f"  VALIDATION FAILED: {validation}")

    return validation['row_count_match'] and validation['checksum_match'], validation


def main():
    parser = argparse.ArgumentParser(description='Migrate RethinkDB to QuestDB')
    parser.add_argument('--dry-run', action='store_true', help='Validate without migrating')
    parser.add_argument('--table', help='Migrate specific table only')
    args = parser.parse_args()

    rethink_config = {'host': 'localhost', 'port': 28015, 'db': 'filos-dev'}
    quest_storage = QuestDBStorage()

    # Get list of tables
    r = RethinkDB()
    conn = r.connect(**rethink_config)
    tables = list(r.table_list().run(conn))
    trade_tables = [t for t in tables if t.startswith('trade_')]
    conn.close()

    if args.table:
        trade_tables = [args.table]

    logger.info(f"Found {len(trade_tables)} trade tables to migrate")

    results = {}
    for table in trade_tables:
        success, validation = migrate_table(
            rethink_config, quest_storage, table, dry_run=args.dry_run
        )
        results[table] = {'success': success, 'validation': validation}

    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    logger.info(f"\nMigration summary: {successful}/{len(results)} tables successful")

    if args.dry_run:
        logger.info("This was a DRY RUN. Run without --dry-run to perform migration.")


if __name__ == '__main__':
    main()
```

### 2.3 QuestDB Schema Mapping

| RethinkDB Field | QuestDB Field | Type | Notes |
|-----------------|---------------|------|-------|
| `timestamp` | `timestamp` | TIMESTAMP | Partition key, indexed |
| `open` | `open` | DOUBLE | 64-bit precision |
| `high` | `high` | DOUBLE | 64-bit precision |
| `low` | `low` | DOUBLE | 64-bit precision |
| `close` | `close` | DOUBLE | 64-bit precision |
| `volume` | `volume` | DOUBLE | 64-bit precision |
| `id` | (dropped) | - | RethinkDB primary key, not needed |
| `symbol` | (table name) | - | Encoded in table name |
| `provider` | (table name) | - | Encoded in table name |

### 2.4 Parquet Cache Layer

**File to create**: `backend/storage/parquet_cache.py`

```python
"""
Parquet Cache Layer - Local file cache for frequently accessed data

Benefits:
- 10-100x faster than database queries for repeated access
- Compressed storage (typically 3-5x smaller than raw)
- Works offline
- Compatible with Polars for ultra-fast reads
"""

import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ParquetCache:
    """Local Parquet file cache for OHLCV data"""

    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=1)  # Refresh cache after 1 day

    def _get_cache_path(self, provider: str, symbol: str, bin_size: str) -> Path:
        return self.cache_dir / f"{provider}_{symbol}_{bin_size}.parquet"

    def _get_metadata_path(self, provider: str, symbol: str, bin_size: str) -> Path:
        return self.cache_dir / f"{provider}_{symbol}_{bin_size}.meta"

    def is_cached(self, provider: str, symbol: str, bin_size: str) -> bool:
        """Check if data is cached and fresh"""
        cache_path = self._get_cache_path(provider, symbol, bin_size)
        if not cache_path.exists():
            return False

        # Check age
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.max_age:
            return False

        return True

    def get(
        self,
        provider: str,
        symbol: str,
        bin_size: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pl.DataFrame]:
        """Get data from cache, returns None if not cached"""
        if not self.is_cached(provider, symbol, bin_size):
            return None

        cache_path = self._get_cache_path(provider, symbol, bin_size)

        # Use Polars lazy scan for efficient filtering
        lf = pl.scan_parquet(cache_path)

        if start_date:
            lf = lf.filter(pl.col('timestamp') >= start_date)
        if end_date:
            lf = lf.filter(pl.col('timestamp') <= end_date)

        return lf.collect()

    def put(self, df: pd.DataFrame, provider: str, symbol: str, bin_size: str):
        """Store data in cache"""
        cache_path = self._get_cache_path(provider, symbol, bin_size)

        # Convert to Polars for efficient storage
        pl_df = pl.from_pandas(df)
        pl_df.write_parquet(cache_path, compression='zstd')

        logger.info(f"Cached {len(df)} rows to {cache_path}")

    def invalidate(self, provider: str, symbol: str, bin_size: str):
        """Invalidate cache entry"""
        cache_path = self._get_cache_path(provider, symbol, bin_size)
        if cache_path.exists():
            cache_path.unlink()

    def clear_all(self):
        """Clear entire cache"""
        for f in self.cache_dir.glob('*.parquet'):
            f.unlink()


class CachedDataAdapter:
    """Data adapter with transparent caching"""

    def __init__(self, primary_adapter, cache: ParquetCache):
        self.primary = primary_adapter
        self.cache = cache

    def load_dataframe(
        self,
        provider: str,
        symbol: str,
        bin_size: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load data with cache-through"""

        # Try cache first
        cached = self.cache.get(provider, symbol, bin_size, start_date, end_date)
        if cached is not None:
            logger.debug(f"Cache hit for {provider}/{symbol}/{bin_size}")
            return cached.to_pandas()

        # Load from primary
        logger.debug(f"Cache miss for {provider}/{symbol}/{bin_size}")
        df = self.primary.load_dataframe(provider, symbol, bin_size, start_date, end_date)

        # Store in cache (full range)
        full_df = self.primary.load_dataframe(provider, symbol, bin_size)
        self.cache.put(full_df, provider, symbol, bin_size)

        return df
```

### 2.5 Update Configuration

**File to modify**: `backend/config/maestro-dev.yaml`

```yaml
config:
  # Database configuration
  database:
    type: questdb  # or 'rethinkdb' for backwards compat

    rethinkdb:
      host: 127.0.0.1
      port: 28015
      db: filos-dev

    questdb:
      host: 127.0.0.1
      pg_port: 8812
      ilp_port: 9009
      min_connections: 2
      max_connections: 10

  # Cache configuration
  cache:
    enabled: true
    parquet_dir: data/cache
    max_age_days: 1

  # Rest of existing config...
```

### 2.6 RethinkDB Retention Policy

Keep RethinkDB running for 30 days after migration:

```yaml
# docker-compose.yml
services:
  rethinkdb:
    image: rethinkdb:2.4
    # Keep running as read-only fallback
    command: rethinkdb --bind all --read-only  # ADD read-only flag after migration
    labels:
      - "maestro.deprecation-date=2026-03-06"
```

### Phase 2 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data loss during migration | Low | Critical | Dry-run with checksums, keep RethinkDB 30 days |
| Float precision issues | Medium | Medium | Validate precision in migration script |
| QuestDB performance issues | Low | Medium | Benchmark before migration |
| Connection pool exhaustion | Low | Medium | Monitor and tune pool size |

### Phase 2 Rollback Procedure

1. **Switch database config**: Change `database.type` back to `rethinkdb`
2. **Restart API**: `docker compose restart maestro-api`
3. **Verify**: Query existing RethinkDB tables
4. **Cleanup**: Stop QuestDB container if not needed

```bash
# Rollback script
sed -i 's/type: questdb/type: rethinkdb/' backend/config/maestro-dev.yaml
docker compose restart maestro-api
curl localhost:5000/datasource/binance/ethbtc/1d/ | jq '.data | length'
```

### Phase 2 Success Criteria

- [ ] Migration dry-run validates **100% row match + checksum** for all tables
- [ ] QuestDB serving all queries
- [ ] RethinkDB still available as fallback (30 days)
- [ ] Data queries **20x faster** than V1
- [ ] Parquet cache operational and reducing query load
- [ ] No float precision issues in migrated data

---

## Phase 3: API & Frontend Modernization

**Duration**: 4-5 weeks (includes 2-3 week buffer for charting)
**Objective**: Flask -> FastAPI, React 16 -> React 18, TradingView charts.
**Dependencies**: Phase 2 complete

### 3.1 FastAPI Migration (Incremental)

**Approach**: Run Flask and FastAPI concurrently, use nginx for routing.

**Key Revision**: Do NOT create a new frontend project. Upgrade existing `frontend/` in place.

**File to create**: `backend/main/fastapi_app.py`

```python
"""
FastAPI Application - Modern async API

Features:
- Async endpoints for non-blocking I/O
- WebSocket for real-time progress updates
- Pydantic validation with auto-generated OpenAPI docs
- Backwards-compatible with Flask routes via nginx
"""

from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import hashlib
import asyncio

app = FastAPI(
    title="Filos Trading API",
    version="2.0.0",
    description="Walk-forward optimization platform for algorithmic trading"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class EngineVersion(str, Enum):
    V1 = 'v1'
    V2 = 'v2'


class OptimizationType(str, Enum):
    BACKTESTING = 'BACKTESTING'
    WALKFORWARD = 'WALKFORWARD'


class OptimizationRequest(BaseModel):
    test_name: str = Field(..., min_length=1, max_length=200)
    provider: str = Field(..., pattern='^(BINANCE|BITMEX)$')
    symbol: str = Field(..., min_length=1, max_length=20)
    bin_size: str = Field(..., pattern='^(1m|5m|15m|1h|4h|1d)$')
    strategy: str
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    cash: float = Field(default=100000.0, gt=0)
    commissions: float = Field(default=0.001, ge=0, le=0.1)
    start_date: int = Field(..., description="Start timestamp in milliseconds")
    end_date: int = Field(..., description="End timestamp in milliseconds")
    kind: OptimizationType = OptimizationType.WALKFORWARD
    engine_version: EngineVersion = EngineVersion.V2


class OptimizationResponse(BaseModel):
    tid: str
    test_name: str
    status: str
    message: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/api/v2/strategies", response_model=List[str])
async def list_strategies():
    """List available strategies for V2 engine"""
    from engine_v2.strategy_adapter import STRATEGY_REGISTRY
    return list(STRATEGY_REGISTRY.keys())


@app.get("/api/v2/strategies/{strategy}/params")
async def get_strategy_params(strategy: str):
    """Get default parameters for a strategy"""
    from engine_v2.strategy_adapter import STRATEGY_REGISTRY
    if strategy not in STRATEGY_REGISTRY:
        raise HTTPException(404, f"Strategy {strategy} not found")
    return STRATEGY_REGISTRY[strategy]().get_params()


@app.post("/api/v2/optimization", response_model=OptimizationResponse)
async def create_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Create a new optimization job"""
    tid = hashlib.sha1(request.test_name.encode()).hexdigest()

    # Add to background tasks
    background_tasks.add_task(run_optimization_task, tid, request)

    return OptimizationResponse(
        tid=tid,
        test_name=request.test_name,
        status="submitted"
    )


@app.get("/api/v2/optimization/{tid}")
async def get_optimization_result(tid: str):
    """Get optimization result by ID"""
    # Implementation here
    pass


@app.get("/api/v2/optimization/{tid}/progress")
async def get_optimization_progress(tid: str):
    """Get optimization progress (polling fallback)"""
    # Implementation here
    pass


# ============================================================================
# WebSocket for Real-time Progress
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, tid: str, websocket: WebSocket):
        await websocket.accept()
        if tid not in self.active_connections:
            self.active_connections[tid] = []
        self.active_connections[tid].append(websocket)

    def disconnect(self, tid: str, websocket: WebSocket):
        if tid in self.active_connections:
            self.active_connections[tid].remove(websocket)

    async def broadcast(self, tid: str, message: dict):
        if tid in self.active_connections:
            for connection in self.active_connections[tid]:
                try:
                    await connection.send_json(message)
                except:
                    pass


manager = ConnectionManager()


@app.websocket("/api/v2/optimization/{tid}/ws")
async def websocket_progress(websocket: WebSocket, tid: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(tid, websocket)
    try:
        while True:
            # Keep connection alive, wait for disconnect
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except:
        manager.disconnect(tid, websocket)


# ============================================================================
# Background Task
# ============================================================================

async def run_optimization_task(tid: str, request: OptimizationRequest):
    """Run optimization in background"""
    from engine_v2.walk_forward_optuna import WalkForwardOptuna, WalkForwardConfig
    from engine_v2.strategy_adapter import STRATEGY_REGISTRY
    from datafeed.data_adapter import RethinkDBAdapter

    # Progress callback that broadcasts via WebSocket
    async def progress_callback(current: int, total: int, message: str):
        await manager.broadcast(tid, {
            'type': 'progress',
            'current': current,
            'total': total,
            'message': message,
            'percent': int(current / total * 100) if total > 0 else 0
        })

    # Implementation here
    pass
```

### 3.2 Nginx Routing Configuration

**File to create**: `nginx/nginx.conf`

```nginx
# Route between Flask (V1) and FastAPI (V2) APIs

upstream flask_api {
    server maestro-api-v1:5000;
}

upstream fastapi_api {
    server maestro-api-v2:8000;
}

server {
    listen 80;
    server_name localhost;

    # V2 API routes to FastAPI
    location /api/v2/ {
        proxy_pass http://fastapi_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket connections
    location ~ ^/api/v2/optimization/[^/]+/ws$ {
        proxy_pass http://fastapi_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    # V1 API routes to Flask (backwards compatibility)
    location / {
        proxy_pass http://flask_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3.3 React 18 Upgrade (In-Place)

**Approach**: Upgrade existing frontend, not new project.

**Files to modify**: `frontend/package.json`

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "@mui/material": "^5.15.0",
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0"
  }
}
```

**Migration steps**:

1. [ ] Update React 16 -> 18 (`createRoot` instead of `render`)
2. [ ] Replace class components with hooks
3. [ ] Add React Query for data fetching
4. [ ] Add Zustand for state management
5. [ ] Upgrade MUI components

### 3.4 WebSocket with Polling Fallback

**File to create**: `frontend/src/hooks/useOptimizationProgress.ts`

```typescript
import { useEffect, useState, useCallback } from 'react';

interface Progress {
  current: number;
  total: number;
  percent: number;
  message: string;
}

export function useOptimizationProgress(tid: string): Progress {
  const [progress, setProgress] = useState<Progress>({
    current: 0,
    total: 0,
    percent: 0,
    message: 'Initializing...'
  });

  useEffect(() => {
    let ws: WebSocket | null = null;
    let pollInterval: number | null = null;

    // Try WebSocket first
    const wsUrl = `ws://${window.location.host}/api/v2/optimization/${tid}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        setProgress(data);
      }
    };

    ws.onerror = () => {
      // Fallback to polling
      console.log('WebSocket failed, falling back to polling');

      pollInterval = window.setInterval(async () => {
        const response = await fetch(`/api/v2/optimization/${tid}/progress`);
        const data = await response.json();
        setProgress(data);

        if (data.percent >= 100) {
          window.clearInterval(pollInterval!);
        }
      }, 2000);
    };

    return () => {
      if (ws) ws.close();
      if (pollInterval) window.clearInterval(pollInterval);
    };
  }, [tid]);

  return progress;
}
```

### 3.5 Charting Migration

**Key Insight from Agent Review**: Highcharts Stock features (flags, multi-pane, range selector) require careful evaluation. Consider:

1. **TradingView Lightweight Charts** - Simpler, but lacks some Highcharts features
2. **TradingView Full Widget** - More complete, but larger bundle

**Recommendation**: Evaluate both during development. Budget 2-3 extra weeks.

**File to create**: `frontend/src/components/TradingChart.tsx`

```typescript
import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, CandlestickSeries, LineSeries } from 'lightweight-charts';

interface Trade {
  time: number;
  price: number;
  type: 'buy' | 'sell';
}

interface ChartProps {
  data: CandlestickData[];
  trades: Trade[];
  indicators?: {
    name: string;
    data: { time: number; value: number }[];
    color: string;
  }[];
}

export function TradingChart({ data, trades, indicators = [] }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1e1e2f' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      crosshair: {
        mode: 1, // CrosshairMode.Normal
      },
      timeScale: {
        borderColor: '#485c7b',
        timeVisible: true,
      },
    });

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });
    candlestickSeries.setData(data);

    // Add trade markers
    const markers = trades.map(trade => ({
      time: trade.time,
      position: trade.type === 'buy' ? 'belowBar' : 'aboveBar',
      color: trade.type === 'buy' ? '#26a69a' : '#ef5350',
      shape: trade.type === 'buy' ? 'arrowUp' : 'arrowDown',
      text: trade.type.toUpperCase(),
    }));
    candlestickSeries.setMarkers(markers);

    // Add indicator lines
    indicators.forEach(indicator => {
      const lineSeries = chart.addSeries(LineSeries, {
        color: indicator.color,
        lineWidth: 2,
        title: indicator.name,
      });
      lineSeries.setData(indicator.data);
    });

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, trades, indicators]);

  return <div ref={chartContainerRef} style={{ width: '100%' }} />;
}
```

### 3.6 Authentication (Basic)

**File to create**: `backend/main/auth.py`

```python
"""
Basic JWT Authentication for API endpoints
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os

SECRET_KEY = os.environ.get('JWT_SECRET', 'development-secret-change-in-prod')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate JWT token and return user"""
    if not token:
        # Allow unauthenticated access in development
        if os.environ.get('ENV') == 'development':
            return {'username': 'dev', 'role': 'admin'}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {'username': username, 'role': payload.get('role', 'user')}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Phase 3 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Charting migration complexity | High | Medium | Budget extra weeks, consider TradingView widget |
| Frontend regression | Medium | High | Feature flags, gradual rollout |
| WebSocket reliability | Medium | Low | Polling fallback implemented |
| React 18 breaking changes | Low | Medium | Test thoroughly, use upgrade guide |

### Phase 3 Rollback Procedure

1. **Disable nginx V2 routing**: Remove `/api/v2/` proxy rules
2. **Stop FastAPI**: `docker compose stop maestro-api-v2`
3. **Revert frontend**: `git checkout frontend/package.json && npm install`
4. **Verify**: Flask API serving all traffic

### Phase 3 Success Criteria

- [ ] FastAPI serving `/api/v2/*` endpoints
- [ ] WebSocket progress with polling fallback
- [ ] Frontend upgraded in-place (React 18, MUI v5)
- [ ] Charting migrated with **feature parity**:
  - [ ] Candlestick display
  - [ ] Buy/sell markers
  - [ ] Indicator overlays
  - [ ] Zoom/pan controls
- [ ] Auth working in production

---

## Phase 4: Enhanced Analytics

**Duration**: 1-2 weeks
**Objective**: Add QuantStats integration and Optuna dashboard.
**Dependencies**: Phase 3 complete

### 4.1 QuantStats Integration

**File to create**: `backend/analytics/quantstats_reporter.py`

```python
"""
QuantStats Reporter - Modern portfolio analytics

Replaces abandoned PyFolio with actively maintained QuantStats.
Uses 365-day annualization for crypto markets (not 252 trading days).
"""

import quantstats as qs
import pandas as pd
from typing import Dict, Any, Optional
from io import BytesIO
import base64


class QuantStatsReporter:
    """Generate comprehensive trading analytics"""

    ANNUALIZATION_FACTOR = 365  # Crypto markets trade 24/7

    def __init__(self, returns: pd.Series, benchmark: Optional[pd.Series] = None):
        """
        Initialize reporter with returns series.

        Args:
            returns: Daily returns series with datetime index
            benchmark: Optional benchmark returns (e.g., BTC buy-and-hold)
        """
        self.returns = returns
        self.benchmark = benchmark

    def get_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics"""
        return {
            'sharpe': qs.stats.sharpe(self.returns, periods=self.ANNUALIZATION_FACTOR),
            'sortino': qs.stats.sortino(self.returns, periods=self.ANNUALIZATION_FACTOR),
            'calmar': qs.stats.calmar(self.returns),
            'max_drawdown': qs.stats.max_drawdown(self.returns),
            'win_rate': qs.stats.win_rate(self.returns),
            'profit_factor': qs.stats.profit_factor(self.returns),
            'cagr': qs.stats.cagr(self.returns, periods=self.ANNUALIZATION_FACTOR),
            'volatility': qs.stats.volatility(self.returns, periods=self.ANNUALIZATION_FACTOR),
            'skew': qs.stats.skew(self.returns),
            'kurtosis': qs.stats.kurtosis(self.returns),
            'var': qs.stats.var(self.returns),
            'cvar': qs.stats.cvar(self.returns),
            'avg_win': qs.stats.avg_win(self.returns),
            'avg_loss': qs.stats.avg_loss(self.returns),
            'payoff_ratio': qs.stats.payoff_ratio(self.returns),
        }

    def generate_html_report(self) -> str:
        """Generate full HTML report"""
        return qs.reports.html(
            self.returns,
            benchmark=self.benchmark,
            output=None,  # Return string instead of file
            title='Filos Strategy Analysis'
        )

    def generate_tearsheet_images(self) -> Dict[str, str]:
        """Generate tearsheet images as base64"""
        images = {}

        # Returns plot
        fig = qs.plots.returns(self.returns, show=False)
        images['returns'] = self._fig_to_base64(fig)

        # Drawdown plot
        fig = qs.plots.drawdown(self.returns, show=False)
        images['drawdown'] = self._fig_to_base64(fig)

        # Monthly returns heatmap
        fig = qs.plots.monthly_heatmap(self.returns, show=False)
        images['monthly_heatmap'] = self._fig_to_base64(fig)

        return images

    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
```

### 4.2 Optuna Dashboard

Add Optuna's built-in dashboard for optimization visualization:

```python
# In walk_forward_optuna.py

def create_study_with_dashboard(fold_idx: int, storage_url: str = None):
    """Create Optuna study with optional dashboard storage"""

    if storage_url is None:
        storage_url = "sqlite:///optuna_studies.db"

    study = optuna.create_study(
        study_name=f"maestro_fold_{fold_idx}",
        direction='maximize',
        storage=storage_url,
        load_if_exists=True
    )

    return study


# Launch dashboard with: optuna-dashboard sqlite:///optuna_studies.db
```

### Phase 4 Success Criteria

- [ ] QuantStats integration complete
- [ ] HTML reports generate correctly
- [ ] Optuna dashboard accessible
- [ ] 365-day annualization used for crypto

---

## Phase 5: Advanced Optimization & Scaling

**Duration**: 2-3 weeks (can run parallel with Phase 4)
**Objective**: Maximize computational efficiency, improve robustness, and enable horizontal scaling.

This phase transforms Filos from a single-machine research tool to a production-grade optimization platform capable of handling institutional-scale workloads.

---

### 5.1 Efficiency Improvements in Computation and Resource Usage

The current walk-forward implementation is solid but computationally intensive, particularly in the parameter optimization phase, where grid search on sampled parameters can lead to many backtest runs per split (potentially thousands if parameters combine multiplicatively). Additionally, sequential processing of splits and data loading from RethinkDB for each phase may introduce overhead.

#### 5.1.1 Enhanced Bayesian Optimization with Optuna

While Phase 1 introduces Optuna, this section adds advanced features for 50-80% trial reduction.

- [ ] Implement intelligent early pruning
- [ ] Add warm-starting from previous studies
- [ ] Configure multi-objective optimization

**File to modify**: `backend/engine_v2/walk_forward_optuna.py`

```python
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

def create_optimized_study(strategy_name: str, n_startup_trials: int = 10):
    """
    Create an Optuna study with advanced configuration for efficient optimization.

    Features:
    - TPE sampler with warm-starting
    - Hyperband pruner for early stopping of bad trials
    - Multi-objective support for Sharpe + VWR
    """
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,  # Random exploration before Bayesian
        multivariate=True,  # Consider parameter correlations
        seed=42
    )

    pruner = HyperbandPruner(
        min_resource=1,      # Minimum epochs/splits to run
        max_resource=10,     # Maximum before pruning decision
        reduction_factor=3   # Aggressive pruning
    )

    study = optuna.create_study(
        study_name=f"walkforward_{strategy_name}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///optuna_studies.db",
        load_if_exists=True  # Warm-start from previous runs
    )

    return study


def objective(trial, data, strategy_class, config):
    """
    Optuna objective function with intermediate value reporting for pruning.
    """
    # Suggest parameters dynamically based on strategy
    params = {}
    for param_name, (param_type, low, high) in strategy_class.get_param_space().items():
        if param_type == 'int':
            params[param_name] = trial.suggest_int(param_name, low, high)
        elif param_type == 'float':
            params[param_name] = trial.suggest_float(param_name, low, high)
        elif param_type == 'log':
            params[param_name] = trial.suggest_float(param_name, low, high, log=True)

    # Run backtest
    engine = VectorBTEngine()
    result = engine.run(data, strategy_class, params)

    # Report intermediate values for pruning
    trial.report(result.sharpe_ratio, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return result.sharpe_ratio


# Usage in walk-forward
study = create_optimized_study(strategy_name)
study.optimize(
    lambda trial: objective(trial, train_data, strategy_class, config),
    n_trials=50,  # Reduced from grid search (potentially 1000+)
    timeout=600,  # 10 minute timeout per split
    n_jobs=1,     # Use -1 for parallel trials (see 5.1.2)
    show_progress_bar=True
)
opt_params = study.best_params
```

**Expected Impact**: 50-80% reduction in optimization time while finding better optima.

#### 5.1.2 Parallelize Walk-Forward Iterations

Walk-forward splits are independent after data splitting, so parallelize them using multiprocessing.

- [ ] Implement ProcessPoolExecutor for split parallelization
- [ ] Add configurable worker count
- [ ] Implement result aggregation

**File to modify**: `backend/engine_v2/walk_forward_optuna.py`

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import multiprocessing as mp


def process_single_split(
    split_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    full_df: pd.DataFrame,
    strategy_class,
    config: WalkForwardConfig
) -> Tuple[int, dict, dict]:
    """
    Process a single walk-forward split (can run in separate process).

    Returns:
        Tuple of (split_idx, optimization_results, test_results)
    """
    # Slice data for this split
    train_df = full_df.iloc[train_idx].copy()
    test_df = full_df.iloc[test_idx].copy()

    # Create study for this split
    study = create_optimized_study(f"{strategy_class.__name__}_split_{split_idx}")

    # Optimize on training data
    study.optimize(
        lambda trial: objective(trial, train_df, strategy_class, config),
        n_trials=config.n_trials,
        timeout=config.timeout_per_split
    )

    # Test on out-of-sample data
    engine = VectorBTEngine()
    test_result = engine.run(test_df, strategy_class, study.best_params)

    return split_idx, study.best_params, test_result.to_dict()


class ParallelWalkForward:
    """
    Parallel walk-forward optimization using ProcessPoolExecutor.
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)

    def run(
        self,
        df: pd.DataFrame,
        strategy_class,
        config: WalkForwardConfig
    ) -> List[dict]:
        """
        Run walk-forward optimization in parallel across splits.

        Args:
            df: Full OHLCV DataFrame
            strategy_class: Strategy to optimize
            config: Walk-forward configuration

        Returns:
            List of results for each split
        """
        # Create time series splits
        splitter = TimeSeriesSplitRolling(config.num_splits)
        splits = list(splitter.split(df, fixed_length=True, train_splits=2))

        results = [None] * len(splits)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all splits
            futures = {
                executor.submit(
                    process_single_split,
                    idx, train_idx, test_idx, df, strategy_class, config
                ): idx
                for idx, (train_idx, test_idx) in enumerate(splits)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                split_idx, opt_params, test_result = future.result()
                results[split_idx] = {
                    'split': split_idx,
                    'optimal_params': opt_params,
                    'test_result': test_result
                }

                # Progress callback
                self._on_split_complete(split_idx, len(splits))

        return results

    def _on_split_complete(self, completed: int, total: int):
        """Override for progress tracking."""
        pass
```

**Configuration** in `maestro-dev.yaml`:

```yaml
config:
  optimization:
    parallel:
      enabled: true
      max_workers: 4  # Or 'auto' for cpu_count - 1
      timeout_per_split: 600  # 10 minutes
```

**Expected Impact**: 3-4x speedup on multi-core machines (4 cores = ~3.5x).

#### 5.1.3 Optimize Data Loading and Caching

Load data once and slice in-memory to avoid repeated DB queries.

- [ ] Implement single-load data pattern
- [ ] Add in-memory data slicing
- [ ] Create DataFrameCache utility

**File to create**: `backend/datafeed/dataframe_cache.py`

```python
"""
DataFrame Cache - Load once, slice many times.

Avoids repeated DB queries during walk-forward optimization.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from functools import lru_cache
import hashlib


class DataFrameCache:
    """
    In-memory cache for OHLCV DataFrames with efficient slicing.
    """

    def __init__(self, data_adapter):
        self.data_adapter = data_adapter
        self._cache: Dict[str, pd.DataFrame] = {}

    def _cache_key(self, provider: str, symbol: str, bin_size: str,
                   start_date: str, end_date: str) -> str:
        """Generate unique cache key."""
        return f"{provider}_{symbol}_{bin_size}_{start_date}_{end_date}"

    def get_dataframe(
        self,
        provider: str,
        symbol: str,
        bin_size: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get DataFrame from cache or load from adapter.

        First call loads from DB/storage, subsequent calls return cached copy.
        """
        key = self._cache_key(provider, symbol, bin_size, start_date, end_date)

        if key not in self._cache:
            df = self.data_adapter.load_dataframe(
                provider, symbol, bin_size, start_date, end_date
            )
            # Store with datetime index for efficient slicing
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            self._cache[key] = df

        return self._cache[key]

    def slice_by_index(
        self,
        df: pd.DataFrame,
        indices: np.ndarray
    ) -> pd.DataFrame:
        """
        Efficiently slice DataFrame by integer indices.

        Uses iloc for O(1) slicing instead of creating copies.
        """
        return df.iloc[indices]

    def slice_by_date(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Slice DataFrame by date range (assumes datetime index).
        """
        return df.loc[start:end]

    def clear(self):
        """Clear cache to free memory."""
        self._cache.clear()

    def memory_usage(self) -> int:
        """Return total memory usage of cached DataFrames in bytes."""
        return sum(df.memory_usage(deep=True).sum() for df in self._cache.values())


# Usage in walk-forward
cache = DataFrameCache(data_adapter)

# Load once at start
df = cache.get_dataframe(provider, symbol, bin_size, start_date, end_date)

# Slice efficiently in loop
splitter = TimeSeriesSplitRolling(num_splits)
for train_idx, test_idx in splitter.split(df):
    train_df = cache.slice_by_index(df, train_idx)  # No DB query
    test_df = cache.slice_by_index(df, test_idx)    # No DB query
    # ... optimization logic
```

**Expected Impact**: Eliminates redundant DB queries, 2-10x speedup for data-heavy operations.

---

### 5.2 Enhancements for Robustness and Accuracy

Refine the system for better adaptability to market regimes and statistical rigor.

#### 5.2.1 Adaptive Window Sizes

Fixed train_splits=2 and test_splits=1 may not capture varying market volatility. Introduce adaptive and expanding window modes.

- [ ] Add expanding window mode to TimeSeriesSplitRolling
- [ ] Implement volatility-based adaptive splits
- [ ] Add configuration options

**File to modify**: `backend/utils/time_series_split_rolling.py`

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd


class TimeSeriesSplitRolling(TimeSeriesSplit):
    """
    Extended TimeSeriesSplit with rolling, expanding, and adaptive modes.
    """

    def split(self, X, y=None, groups=None, fixed_length=True,
              train_splits=2, test_splits=1, mode='rolling'):
        """
        Generate train/test indices with configurable window behavior.

        Args:
            X: DataFrame or array to split
            fixed_length: Use fixed window sizes
            train_splits: Number of splits for training window
            test_splits: Number of splits for test window
            mode: 'rolling' | 'expanding' | 'adaptive'
                - rolling: Fixed-size sliding window
                - expanding: Growing training window from start
                - adaptive: Volatility-based dynamic windows

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if mode == 'rolling':
            yield from self._rolling_split(n_samples, fixed_length, train_splits, test_splits)
        elif mode == 'expanding':
            yield from self._expanding_split(n_samples, test_splits)
        elif mode == 'adaptive':
            if not isinstance(X, pd.DataFrame) or 'close' not in X.columns:
                raise ValueError("Adaptive mode requires DataFrame with 'close' column")
            yield from self._adaptive_split(X, train_splits, test_splits)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _rolling_split(self, n_samples, fixed_length, train_splits, test_splits):
        """Original rolling window implementation."""
        # ... existing implementation ...
        pass

    def _expanding_split(self, n_samples, test_splits):
        """
        Expanding window: Training grows from start, test is fixed size.

        Simulates accumulating knowledge over time.
        """
        test_size = n_samples // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = int(n_samples * i / (self.n_splits + 1))
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def _adaptive_split(self, df, train_splits, test_splits):
        """
        Adaptive windows based on volatility clustering.

        Larger windows in high-volatility periods, smaller in low-volatility.
        """
        # Calculate rolling volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()

        # Normalize to get window size multipliers
        vol_normalized = (volatility - volatility.min()) / (volatility.max() - volatility.min())
        vol_multiplier = 0.5 + vol_normalized  # Range: 0.5x to 1.5x

        # Generate splits with adaptive sizes
        base_train_size = len(df) // (self.n_splits + 2) * train_splits
        base_test_size = len(df) // (self.n_splits + 2) * test_splits

        current_pos = 0
        for i in range(self.n_splits):
            # Get average volatility for this region
            region_start = current_pos
            region_end = min(current_pos + base_train_size + base_test_size, len(df))
            avg_multiplier = vol_multiplier.iloc[region_start:region_end].mean()

            if pd.isna(avg_multiplier):
                avg_multiplier = 1.0

            # Adjust window sizes
            train_size = int(base_train_size * avg_multiplier)
            test_size = int(base_test_size * avg_multiplier)

            train_end = current_pos + train_size
            test_end = train_end + test_size

            if test_end > len(df):
                break

            train_indices = np.arange(current_pos, train_end)
            test_indices = np.arange(train_end, test_end)

            yield train_indices, test_indices

            current_pos = train_end  # Slide forward
```

**Configuration** in `maestro-dev.yaml`:

```yaml
config:
  optimization:
    walkforward:
      mode: rolling  # rolling | expanding | adaptive
      num_splits: 10
      train_splits: 2
      test_splits: 1
```

#### 5.2.2 Multi-Objective Parameter Selection

Extend selection beyond Sharpe + VWR to include Sortino, Calmar, and turnover-adjusted metrics.

- [ ] Implement Pareto-based multi-objective selection
- [ ] Add new risk-adjusted metrics
- [ ] Create configurable metric weights

**File to create**: `backend/engine_v2/multi_objective_selector.py`

```python
"""
Multi-Objective Parameter Selection using Pareto fronts.

Selects optimal parameters considering multiple competing objectives:
- Sharpe Ratio (risk-adjusted return)
- VWR (value-weighted return consistency)
- Sortino Ratio (downside risk)
- Calmar Ratio (drawdown-adjusted)
- Turnover (trading frequency penalty)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import rankdata
from dataclasses import dataclass


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    metrics: List[str] = None
    weights: Dict[str, float] = None
    method: str = 'pareto'  # 'pareto' | 'weighted' | 'rank_average'

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['sharpe_ratio', 'vwr', 'sortino_ratio', 'calmar_ratio']
        if self.weights is None:
            self.weights = {m: 1.0 for m in self.metrics}


def calculate_extended_metrics(result) -> Dict[str, float]:
    """
    Calculate extended risk-adjusted metrics from backtest result.
    """
    returns = result.returns

    metrics = {
        'sharpe_ratio': result.sharpe_ratio,
        'vwr': result.vwr,
    }

    # Sortino Ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    metrics['sortino_ratio'] = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

    # Calmar Ratio (return / max drawdown)
    annual_return = returns.mean() * 252
    max_dd = abs(result.max_drawdown)
    metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0

    # Turnover penalty (fewer trades = better, all else equal)
    metrics['turnover_penalty'] = -result.num_trades / len(returns)  # Negative = penalize high turnover

    return metrics


def pareto_optimal(results: List[Dict], metrics: List[str]) -> List[int]:
    """
    Find Pareto-optimal results (non-dominated solutions).

    A result is Pareto-optimal if no other result is better in ALL metrics.
    """
    n = len(results)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i (j is better in all metrics)
            j_dominates = all(
                results[j].get(m, 0) >= results[i].get(m, 0)
                for m in metrics
            ) and any(
                results[j].get(m, 0) > results[i].get(m, 0)
                for m in metrics
            )

            if j_dominates:
                is_pareto[i] = False
                break

    return list(np.where(is_pareto)[0])


def select_optimal_parameters(
    results: List[Dict],
    config: MultiObjectiveConfig = None
) -> Tuple[Dict, int]:
    """
    Select optimal parameters using multi-objective optimization.

    Args:
        results: List of backtest results with metrics
        config: Multi-objective configuration

    Returns:
        Tuple of (optimal_params, result_index)
    """
    config = config or MultiObjectiveConfig()

    if config.method == 'pareto':
        # Find Pareto front, then select by weighted sum
        pareto_indices = pareto_optimal(results, config.metrics)
        candidates = [results[i] for i in pareto_indices]

        # Select from Pareto front by weighted sum
        scores = []
        for r in candidates:
            score = sum(
                r.get(m, 0) * config.weights.get(m, 1.0)
                for m in config.metrics
            )
            scores.append(score)

        best_pareto_idx = np.argmax(scores)
        return candidates[best_pareto_idx]['params'], pareto_indices[best_pareto_idx]

    elif config.method == 'weighted':
        # Simple weighted sum
        scores = []
        for r in results:
            score = sum(
                r.get(m, 0) * config.weights.get(m, 1.0)
                for m in config.metrics
            )
            scores.append(score)

        best_idx = np.argmax(scores)
        return results[best_idx]['params'], best_idx

    elif config.method == 'rank_average':
        # Average ranks across metrics (robust to outliers)
        n = len(results)
        ranks = np.zeros(n)

        for metric in config.metrics:
            values = [r.get(metric, 0) for r in results]
            metric_ranks = rankdata(values)  # Higher = better rank
            ranks += metric_ranks * config.weights.get(metric, 1.0)

        best_idx = np.argmax(ranks)
        return results[best_idx]['params'], best_idx

    else:
        raise ValueError(f"Unknown method: {config.method}")
```

**Configuration** in `maestro-dev.yaml`:

```yaml
config:
  optimization:
    selection:
      method: pareto  # pareto | weighted | rank_average
      metrics:
        - sharpe_ratio
        - vwr
        - sortino_ratio
        - calmar_ratio
      weights:
        sharpe_ratio: 1.0
        vwr: 0.8
        sortino_ratio: 0.6
        calmar_ratio: 0.4
```

#### 5.2.3 Regime Detection for Adaptive Re-Optimization

Trigger re-optimization only on market regime shifts using Hidden Markov Models.

- [ ] Implement HMM-based regime detection
- [ ] Add regime-triggered optimization
- [ ] Create regime visualization

**File to create**: `backend/engine_v2/regime_detector.py`

```python
"""
Regime Detection using Hidden Markov Models.

Detects market regime changes (trending, mean-reverting, volatile)
to trigger adaptive re-optimization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RegimeConfig:
    n_regimes: int = 3  # Bull, Bear, Sideways
    lookback_window: int = 60  # Days for regime detection
    change_threshold: float = 0.7  # Confidence threshold for regime change
    features: List[str] = None  # Features for HMM

    def __post_init__(self):
        if self.features is None:
            self.features = ['returns', 'volatility', 'trend']


class RegimeDetector:
    """
    Detect market regimes using Gaussian HMM.
    """

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.model = None
        self._fitted = False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection."""
        features = []

        close = df['close']

        # Returns
        if 'returns' in self.config.features:
            returns = close.pct_change().fillna(0)
            features.append(returns.values)

        # Volatility (rolling std of returns)
        if 'volatility' in self.config.features:
            returns = close.pct_change().fillna(0)
            volatility = returns.rolling(window=20).std().fillna(returns.std())
            features.append(volatility.values)

        # Trend (slope of regression over window)
        if 'trend' in self.config.features:
            trend = close.rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            ).fillna(0)
            features.append(trend.values)

        return np.column_stack(features)

    def fit(self, df: pd.DataFrame):
        """Fit HMM to historical data."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn required for regime detection. Install with: pip install hmmlearn")

        features = self._extract_features(df)

        self.model = GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )

        self.model.fit(features)
        self._fitted = True

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime for each observation."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self._extract_features(df)
        return self.model.predict(features)

    def detect_regime_changes(self, df: pd.DataFrame) -> List[int]:
        """
        Detect indices where regime changes occur.

        Returns:
            List of indices where regime changes
        """
        regimes = self.predict(df)
        changes = []

        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                changes.append(i)

        return changes

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Get current regime and confidence.

        Returns:
            Tuple of (regime_id, confidence_score)
        """
        features = self._extract_features(df)

        # Get probabilities for last observation
        log_probs = self.model.score_samples(features[-1:])
        probs = np.exp(log_probs)

        regime = self.model.predict(features[-1:])[0]
        confidence = probs.max()

        return regime, confidence

    def should_reoptimize(self, df: pd.DataFrame, last_regime: int) -> bool:
        """
        Determine if strategy should be re-optimized based on regime change.

        Args:
            df: Recent market data
            last_regime: Regime from last optimization

        Returns:
            True if regime changed with sufficient confidence
        """
        current_regime, confidence = self.get_current_regime(df)

        return (
            current_regime != last_regime and
            confidence >= self.config.change_threshold
        )


# Usage example
def regime_adaptive_walkforward(df, strategy_class, config):
    """
    Walk-forward optimization that re-optimizes on regime changes.
    """
    detector = RegimeDetector()
    detector.fit(df)

    regimes = detector.predict(df)
    regime_changes = detector.detect_regime_changes(df)

    results = []
    last_opt_idx = 0
    last_params = None

    for change_idx in regime_changes:
        # Optimize on data since last change
        train_df = df.iloc[last_opt_idx:change_idx]

        # Run optimization
        study = create_optimized_study(strategy_class.__name__)
        study.optimize(
            lambda trial: objective(trial, train_df, strategy_class, config),
            n_trials=50
        )

        last_params = study.best_params
        last_opt_idx = change_idx

        results.append({
            'regime_change_idx': change_idx,
            'regime': regimes[change_idx],
            'optimal_params': last_params
        })

    return results
```

---

### 5.3 Scalability and Maintenance Improvements

#### 5.3.1 Multi-Symbol and Multi-Timeframe Parallelism

Extend to run walk-forward across multiple symbols/timeframes concurrently.

- [ ] Implement symbol-level parallelization
- [ ] Add timeframe aggregation
- [ ] Create unified results aggregator

**File to create**: `backend/engine_v2/multi_asset_optimizer.py`

```python
"""
Multi-Asset Walk-Forward Optimizer.

Runs optimization across multiple symbols and timeframes in parallel.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import pandas as pd


class MultiAssetOptimizer:
    """
    Parallel optimization across multiple assets and timeframes.
    """

    def __init__(self, data_adapter, max_workers: int = 4):
        self.data_adapter = data_adapter
        self.max_workers = max_workers

    def optimize_all(
        self,
        assets: List[Dict],  # [{'provider': 'binance', 'symbol': 'btcusdt', 'bin_size': '1h'}, ...]
        strategy_class,
        config: WalkForwardConfig,
        start_date: str,
        end_date: str
    ) -> Dict[str, List[Dict]]:
        """
        Run walk-forward optimization for multiple assets in parallel.

        Args:
            assets: List of asset specifications
            strategy_class: Strategy to optimize
            config: Walk-forward configuration
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Dict mapping asset keys to their walk-forward results
        """
        results = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for asset in assets:
                key = f"{asset['provider']}_{asset['symbol']}_{asset['bin_size']}"

                future = executor.submit(
                    self._optimize_single_asset,
                    asset, strategy_class, config, start_date, end_date
                )
                futures[future] = key

            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = {'error': str(e)}

        return results

    def _optimize_single_asset(
        self,
        asset: Dict,
        strategy_class,
        config: WalkForwardConfig,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """Optimize a single asset (runs in separate process)."""
        # Load data
        df = self.data_adapter.load_dataframe(
            asset['provider'], asset['symbol'], asset['bin_size'],
            start_date, end_date
        )

        # Run parallel walk-forward
        optimizer = ParallelWalkForward(max_workers=1)  # Single-threaded within process
        return optimizer.run(df, strategy_class, config)
```

#### 5.3.2 Monitoring and Logging Enhancements

Integrate Prometheus/Grafana for real-time metrics and MLflow for experiment tracking.

- [ ] Add Prometheus metrics endpoint
- [ ] Configure Grafana dashboards
- [ ] Integrate MLflow experiment tracking

**File to create**: `backend/monitoring/metrics.py`

```python
"""
Prometheus Metrics for Filos Optimization Platform.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time


# Counters
OPTIMIZATION_RUNS = Counter(
    'maestro_optimization_runs_total',
    'Total optimization runs',
    ['strategy', 'provider', 'symbol']
)

OPTIMIZATION_ERRORS = Counter(
    'maestro_optimization_errors_total',
    'Total optimization errors',
    ['strategy', 'error_type']
)

# Histograms
OPTIMIZATION_DURATION = Histogram(
    'maestro_optimization_duration_seconds',
    'Time spent on optimization',
    ['strategy', 'phase'],  # phase: training, testing, total
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800]
)

BACKTEST_DURATION = Histogram(
    'maestro_backtest_duration_seconds',
    'Time spent on single backtest',
    ['engine'],  # v1 or v2
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
)

# Gauges
ACTIVE_OPTIMIZATIONS = Gauge(
    'maestro_active_optimizations',
    'Currently running optimizations'
)

BEST_SHARPE = Gauge(
    'maestro_best_sharpe_ratio',
    'Best Sharpe ratio from latest optimization',
    ['strategy', 'symbol']
)


def track_optimization(strategy: str, provider: str, symbol: str):
    """Decorator to track optimization metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            OPTIMIZATION_RUNS.labels(
                strategy=strategy, provider=provider, symbol=symbol
            ).inc()

            ACTIVE_OPTIMIZATIONS.inc()
            start = time.time()

            try:
                result = func(*args, **kwargs)

                # Record best Sharpe
                if hasattr(result, 'sharpe_ratio'):
                    BEST_SHARPE.labels(strategy=strategy, symbol=symbol).set(
                        result.sharpe_ratio
                    )

                return result
            except Exception as e:
                OPTIMIZATION_ERRORS.labels(
                    strategy=strategy, error_type=type(e).__name__
                ).inc()
                raise
            finally:
                ACTIVE_OPTIMIZATIONS.dec()
                OPTIMIZATION_DURATION.labels(
                    strategy=strategy, phase='total'
                ).observe(time.time() - start)

        return wrapper
    return decorator


# FastAPI endpoint for Prometheus scraping
async def metrics_endpoint():
    """Return Prometheus metrics."""
    return generate_latest()
```

**File to modify**: `backend/main/fastapi_app.py`

```python
from fastapi import Response
from backend.monitoring.metrics import metrics_endpoint

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=await metrics_endpoint(),
        media_type="text/plain"
    )
```

**Docker Compose addition**:

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.47.0
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:10.2.0
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=maestro
    depends_on:
      - prometheus
```

#### 5.3.3 Best Practices Checklist

- [ ] Ensure parameter ranges are logical (periods < data_size/2)
- [ ] Include slippage in commission calculations
- [ ] Run sensitivity analysis post-walk-forward
- [ ] Increase num_splits to ≥20 for volatile markets
- [ ] Add position sizing constraints

**File to modify**: `backend/engine_v2/vectorbt_engine.py`

```python
def validate_params(params: Dict, data_size: int) -> Dict:
    """
    Validate and constrain parameters to sensible ranges.
    """
    validated = {}

    for key, value in params.items():
        if 'period' in key.lower():
            # Period parameters should be less than half the data
            max_period = data_size // 2
            validated[key] = min(value, max_period)
        elif 'threshold' in key.lower():
            # Thresholds typically 0-1
            validated[key] = np.clip(value, 0, 1)
        else:
            validated[key] = value

    return validated


def get_realistic_commission(provider: str) -> float:
    """
    Get realistic commission + slippage for provider.
    """
    # Commission + estimated slippage
    rates = {
        'binance': 0.001 + 0.0005,   # 0.1% + 0.05% slippage
        'bitmex': 0.00075 + 0.0005,  # 0.075% + 0.05% slippage
        'bybit': 0.001 + 0.0005,     # 0.1% + 0.05% slippage
    }
    return rates.get(provider, 0.002)  # Default 0.2%
```

---

### 5.4 Frontend Optimizations

#### 5.4.1 Performance for Large Datasets

- [ ] Add react-window virtualization for large tables
- [ ] Memoize expensive computations
- [ ] Implement lazy loading

**File to modify**: `frontend/src/components/WalkForwardMetrics.js`

```javascript
import React, { useMemo, memo } from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';

// Memoized metric distribution calculation
const useWalkForwardMetrics = (results) => {
  return useMemo(() => {
    if (!results || results.length === 0) return null;

    return {
      sharpeDistribution: results.map(r => r.sharpe_ratio),
      vwrDistribution: results.map(r => r.vwr),
      returnDistribution: results.map(r => r.total_return),
      mean: {
        sharpe: results.reduce((a, r) => a + r.sharpe_ratio, 0) / results.length,
        vwr: results.reduce((a, r) => a + r.vwr, 0) / results.length,
      },
      std: {
        sharpe: calculateStd(results.map(r => r.sharpe_ratio)),
        vwr: calculateStd(results.map(r => r.vwr)),
      }
    };
  }, [results]);
};

// Virtualized results table for many iterations
const VirtualizedResultsTable = memo(({ results }) => {
  const Row = ({ index, style }) => {
    const result = results[index];
    return (
      <div style={style} className="table-row">
        <span>{result.split}</span>
        <span>{result.sharpe_ratio.toFixed(3)}</span>
        <span>{result.vwr.toFixed(3)}</span>
        <span>{result.num_trades}</span>
      </div>
    );
  };

  return (
    <AutoSizer>
      {({ height, width }) => (
        <List
          height={height}
          itemCount={results.length}
          itemSize={40}
          width={width}
        >
          {Row}
        </List>
      )}
    </AutoSizer>
  );
});
```

#### 5.4.2 Enhanced Visualizations

- [ ] Add parameter stability heatmaps
- [ ] Add per-split equity curves
- [ ] Add regime visualization

**File to create**: `frontend/src/components/ParameterHeatmap.tsx`

```typescript
import React from 'react';
import { HeatMapGrid } from 'react-grid-heatmap';

interface ParameterHeatmapProps {
  results: Array<{
    split: number;
    params: Record<string, number>;
    sharpe: number;
  }>;
  paramX: string;
  paramY: string;
}

export const ParameterHeatmap: React.FC<ParameterHeatmapProps> = ({
  results,
  paramX,
  paramY
}) => {
  // Extract unique values for each parameter
  const xLabels = [...new Set(results.map(r => r.params[paramX]))].sort((a, b) => a - b);
  const yLabels = [...new Set(results.map(r => r.params[paramY]))].sort((a, b) => a - b);

  // Build heatmap data
  const data = yLabels.map(y =>
    xLabels.map(x => {
      const match = results.find(r =>
        r.params[paramX] === x && r.params[paramY] === y
      );
      return match?.sharpe ?? null;
    })
  );

  return (
    <div className="heatmap-container">
      <h4>Parameter Stability: {paramX} vs {paramY}</h4>
      <HeatMapGrid
        data={data}
        xLabels={xLabels.map(String)}
        yLabels={yLabels.map(String)}
        cellStyle={(_, __, ratio) => ({
          background: `rgb(${255 - ratio * 255}, ${ratio * 200}, ${ratio * 100})`,
          fontSize: '0.8rem',
        })}
        cellRender={value => value?.toFixed(2) ?? '-'}
        xLabelsStyle={() => ({ fontSize: '0.7rem' })}
        yLabelsStyle={() => ({ fontSize: '0.7rem' })}
      />
    </div>
  );
};
```

---

### Phase 5 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Parallelization race conditions | LOW | HIGH | Use process isolation, not threads |
| HMM regime detection instability | MEDIUM | MEDIUM | Validate on historical data first |
| Memory exhaustion with large parallelism | MEDIUM | MEDIUM | Add memory monitoring, limit workers |
| Prometheus/Grafana complexity | LOW | LOW | Start with basic metrics, expand gradually |

### Phase 5 Rollback Procedure

| Trigger | Action |
|---------|--------|
| Parallel optimization produces different results | Disable parallelism, run sequential |
| Regime detection causes erratic re-optimization | Disable adaptive mode, use fixed splits |
| Memory issues | Reduce max_workers, add memory limits |
| Monitoring overhead | Disable metrics collection |

### Phase 5 Success Criteria

- [ ] Walk-forward optimization 2-5x faster than baseline
- [ ] Multi-symbol optimization working
- [ ] Prometheus metrics exposed
- [ ] Grafana dashboard operational
- [ ] Regime detection validated on historical data
- [ ] Frontend handles 100+ split results smoothly

---

## Dependencies Between Phases

```
Phase 0: Infrastructure Preparation
    |
    v
Phase 1: Core Engine Integration
    |
    v
Phase 2: Data Layer Modernization
    |
    v
Phase 3: API & Frontend Modernization
    |
    v
Phase 4: Enhanced Analytics
    |
    v
Phase 5: Advanced Optimization & Scaling (can run parallel with Phase 4)
```

Each phase must complete before the next begins. However, some work can be parallelized:

- **During Phase 1**: Frontend developer can begin React 18 upgrade research
- **During Phase 2**: Begin charting component prototyping
- **During Phase 3**: Begin QuantStats integration
- **During Phase 4**: Begin Phase 5 (parallelization, regime detection)

---

## Timeline Estimates

| Phase | Duration | Cumulative | Parallelizable |
|-------|----------|------------|----------------|
| Phase 0: Infrastructure | 1 week | Week 1 | No |
| Phase 1: Engine Integration | 2-3 weeks | Week 4 | No |
| Phase 2: Data Layer | 2-3 weeks | Week 7 | No |
| Phase 3: API + Frontend | 4-5 weeks | Week 12 | No |
| Phase 4: Analytics | 1-2 weeks | Week 14 | Yes (with Phase 5) |
| Phase 5: Optimization & Scaling | 2-3 weeks | Week 16 | Yes (with Phase 4) |

**Total: 12-17 weeks**

Buffer for charting complexity and unexpected issues: +2 weeks

**Conservative estimate: 18-19 weeks**

### Phase 5 Priority Order

Phase 5 can be implemented incrementally based on impact:

| Priority | Component | Impact | Effort |
|----------|-----------|--------|--------|
| 1 | Data caching (5.1.3) | 2-10x speedup | Low |
| 2 | Enhanced Optuna (5.1.1) | 50-80% fewer trials | Medium |
| 3 | Split parallelization (5.1.2) | 3-4x speedup | Medium |
| 4 | Multi-objective selection (5.2.2) | Better parameter selection | Medium |
| 5 | Monitoring (5.3.2) | Production observability | Medium |
| 6 | Regime detection (5.2.3) | Adaptive optimization | High |
| 7 | Multi-asset parallelism (5.3.1) | Scale to many symbols | Medium |
| 8 | Frontend virtualization (5.4.1) | UI performance | Low |

---

## Critical Files Reference

| File | Phase | Purpose |
|------|-------|---------|
| `backend/engine_v2/vectorbt_engine.py` | 1 | Add VWR metric |
| `backend/engine_v2/walk_forward_optuna.py` | 1, 5 | Add VWR to ranking, enhanced Optuna |
| `backend/main/rest_api.py` | 1 | Add V2 engine routing |
| `backend/engine/engine_factory.py` | 1 | NEW: Unified engine factory |
| `backend/engine_v2/result_converter.py` | 1 | NEW: V2 to V1 schema |
| `backend/datafeed/data_adapter.py` | 1 | NEW: Data source abstraction |
| `backend/tests/test_engine_parity.py` | 0 | NEW: Parity test framework |
| `.github/workflows/ci.yml` | 0 | NEW: CI/CD pipeline |
| `docker-compose.yml` | 0, 2, 5 | Health checks, QuestDB, Prometheus |
| `backend/storage/questdb_storage_layer.py` | 2 | NEW: QuestDB layer |
| `backend/scripts/migrate_rethinkdb_to_questdb.py` | 2 | NEW: Migration script |
| `backend/storage/parquet_cache.py` | 2 | NEW: Parquet cache |
| `backend/main/fastapi_app.py` | 3 | NEW: FastAPI application |
| `nginx/nginx.conf` | 3 | NEW: API routing |
| `frontend/package.json` | 3 | React 18 upgrade |
| `frontend/src/components/TradingChart.tsx` | 3 | NEW: Chart component |
| `backend/analytics/quantstats_reporter.py` | 4 | NEW: Analytics |
| `backend/datafeed/dataframe_cache.py` | 5 | NEW: In-memory data caching |
| `backend/utils/time_series_split_rolling.py` | 5 | Add expanding/adaptive modes |
| `backend/engine_v2/multi_objective_selector.py` | 5 | NEW: Pareto selection |
| `backend/engine_v2/regime_detector.py` | 5 | NEW: HMM regime detection |
| `backend/engine_v2/multi_asset_optimizer.py` | 5 | NEW: Multi-symbol parallelism |
| `backend/monitoring/metrics.py` | 5 | NEW: Prometheus metrics |
| `frontend/src/components/WalkForwardMetrics.js` | 5 | Virtualization, memoization |
| `frontend/src/components/ParameterHeatmap.tsx` | 5 | NEW: Parameter stability viz |

---

## Final Checklist

### Before Starting
- [ ] All stakeholders aligned on timeline
- [ ] Development environment setup complete
- [ ] Phase 0 infrastructure in place

### Phase Gates
- [ ] Phase 0 complete: CI running, rollback tested
- [ ] Phase 1 complete: V2 engine accessible via API, parity tests pass
- [ ] Phase 2 complete: QuestDB serving queries, data migrated
- [ ] Phase 3 complete: FastAPI + React 18 deployed
- [ ] Phase 4 complete: QuantStats integrated
- [ ] Phase 5 complete: Optimization 2-5x faster, monitoring operational

### Project Complete
- [ ] All original functionality preserved
- [ ] Performance targets met (see table below)
- [ ] Old stack deprecated but recoverable
- [ ] Documentation updated
- [ ] Team trained on new stack
- [ ] Monitoring dashboards operational

### Final Performance Validation

| Metric | Baseline (V1) | Target | Phase 5 Bonus |
|--------|---------------|--------|---------------|
| Single backtest | 30-60 sec | <1 sec | <0.5 sec |
| Walk-forward (10 splits) | 30+ min | <3 min | <1 min |
| Data query (1 year) | 2-3 sec | <100ms | <50ms (cached) |
| API response | 100-200ms | <50ms | <30ms |
| Multi-symbol optimization | N/A | N/A | Linear scaling |
