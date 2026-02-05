"""
FastAPI Application - Modern async API for Maestro trading platform

Provides:
- REST endpoints for V2 optimization engine
- WebSocket support for real-time progress updates
- Pydantic validation for requests/responses
- Background task processing for long-running optimizations

Runs alongside Flask API with nginx routing:
- Flask: /api/v1/* (legacy)
- FastAPI: /api/v2/* (modern)
"""

import asyncio
import hashlib
import os

# Import from parent modules
import sys
from collections import defaultdict
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytz
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parents[1].absolute()))

import pandas as pd
from analytics.quantstats_reporter import QuantStatsReporter
from config.config_reader import ConfigReader, RethinkDbConfig
from datafeed.data_adapter import get_adapter
from datafeed.dataframe_cache import DataFrameCache, create_dataframe_cache
from datasource.providers import DataSourceProviders
from engine_v2.result_converter import convert_v2_to_v1_schema, convert_walkforward_v2_to_v1
from engine_v2.strategy_adapter import STRATEGY_REGISTRY
from engine_v2.vectorbt_engine import BacktestConfig, BacktestResult, VectorBTEngine
from engine_v2.walk_forward_optuna import WalkForwardConfig as WFConfig
from engine_v2.walk_forward_optuna import WalkForwardOptuna

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).parents[1].absolute()
CONFIG_FILE = os.environ.get('CONFIG_FILE', 'maestro-dev.yaml')
config_path = ROOT_DIR.joinpath(CONFIG_FILE)

config_reader = ConfigReader().read(path=config_path)
rethinkdb_config = config_reader.get_rethinkdb_config()
optimization_output = config_reader.get_optimization_output_config()
walkforward_config = config_reader.get_walkforward_config()

# Global DataFrame cache for walk-forward optimization (Phase 5.5)
# Eliminates repeated DB queries during walk-forward splits
_dataframe_cache: DataFrameCache | None = None


def get_dataframe_cache() -> DataFrameCache:
    """Get or create the global DataFrame cache."""
    global _dataframe_cache
    if _dataframe_cache is None:
        _dataframe_cache = create_dataframe_cache(
            rethinkdb_config=rethinkdb_config,
            adapter_type='rethinkdb'
        )
    return _dataframe_cache


# ============================================================================
# ENUMS
# ============================================================================

class EngineVersion(StrEnum):
    """Engine version selection"""
    v1 = "v1"
    v2 = "v2"


class OptimizationType(StrEnum):
    """Optimization type selection"""
    BACKTESTING = "BACKTESTING"
    WALKFORWARD = "WALKFORWARD"
    BOTH = "BOTH"


class JobStatus(StrEnum):
    """Optimization job status"""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StrategyParam(BaseModel):
    """Single strategy parameter definition"""
    name: str
    default: Any
    param_type: str = Field(description="Parameter type: int, float, bool")
    min_value: Any | None = None
    max_value: Any | None = None


class StrategyInfo(BaseModel):
    """Strategy information response"""
    name: str
    params: dict[str, Any]
    param_space: dict[str, tuple] = Field(default_factory=dict)


class OptimizationRequest(BaseModel):
    """Request body for creating an optimization job"""
    test_name: str = Field(..., description="Unique name for this optimization test")
    provider: str = Field("BINANCE", description="Data provider (BINANCE, BITMEX)")
    symbol: str = Field(..., description="Trading symbol (e.g., btcusdt)")
    bin_size: str = Field(..., description="Timeframe (e.g., 1d, 1h, 5m)")
    strategy: str = Field(..., description="Strategy name from available strategies")
    strategy_params: dict[str, Any] | None = Field(
        default=None,
        description="Strategy parameters (uses defaults if not provided)"
    )
    kind: OptimizationType = Field(
        OptimizationType.BACKTESTING,
        description="Optimization type"
    )
    engine_version: EngineVersion = Field(
        EngineVersion.v2,
        description="Engine version to use"
    )
    cash: float = Field(100000.0, description="Starting capital")
    commissions: float = Field(0.001, description="Commission rate (0.001 = 0.1%)")
    start_date: int = Field(..., description="Start timestamp in milliseconds")
    end_date: int = Field(..., description="End timestamp in milliseconds")
    n_trials: int | None = Field(50, description="Optuna trials for walk-forward")

    class Config:
        json_schema_extra = {
            "example": {
                "test_name": "BTC EMA Cross Test",
                "provider": "BINANCE",
                "symbol": "btcusdt",
                "bin_size": "1d",
                "strategy": "EmaCrossStrategy",
                "kind": "BACKTESTING",
                "engine_version": "v2",
                "cash": 100000.0,
                "commissions": 0.001,
                "start_date": 1609459200000,
                "end_date": 1672531200000,
            }
        }


class OptimizationResponse(BaseModel):
    """Response from optimization creation"""
    tid: str = Field(..., description="Unique task ID (SHA1 hash of test_name)")
    test_name: str
    symbol: str
    bin_size: str
    optimization_types: str
    engine_version: str
    status: JobStatus


class ProgressUpdate(BaseModel):
    """Progress update for WebSocket broadcasts"""
    tid: str
    status: JobStatus
    progress_pct: float = Field(0.0, ge=0.0, le=100.0)
    current_fold: int | None = None
    total_folds: int | None = None
    message: str | None = None


class OptimizationResultResponse(BaseModel):
    """Optimization result response"""
    tid: str
    test_name: str
    status: JobStatus
    engine_version: str
    backtest_result: dict[str, Any] | None = None
    walkforward_result: dict[str, Any] | None = None
    error: str | None = None


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Maestro Trading API v2",
    description="Modern async API for algorithmic trading backtesting and optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time progress updates"""

    def __init__(self):
        # Map of tid -> list of connected WebSockets
        self.active_connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, tid: str):
        """Accept a new WebSocket connection and register it for a task"""
        await websocket.accept()
        async with self._lock:
            self.active_connections[tid].append(websocket)

    async def disconnect(self, websocket: WebSocket, tid: str):
        """Remove a WebSocket connection"""
        async with self._lock:
            if tid in self.active_connections:
                if websocket in self.active_connections[tid]:
                    self.active_connections[tid].remove(websocket)
                if not self.active_connections[tid]:
                    del self.active_connections[tid]

    async def broadcast(self, tid: str, message: dict):
        """Broadcast a message to all connections subscribed to a task"""
        async with self._lock:
            connections = self.active_connections.get(tid, []).copy()

        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn, tid)


# Global connection manager
manager = ConnectionManager()


# ============================================================================
# IN-MEMORY JOB STORAGE (Replace with Redis/RethinkDB for production)
# ============================================================================

# Job storage: tid -> {status, result, progress}
job_storage: dict[str, dict[str, Any]] = {}


# ============================================================================
# BACKGROUND TASK
# ============================================================================

async def run_optimization_task(
    tid: str,
    request: OptimizationRequest,
):
    """
    Background task to run optimization and broadcast progress via WebSocket.

    This runs the V2 engine (VectorBT + Optuna) and updates job_storage
    with results. Progress callbacks broadcast to connected WebSocket clients.
    """
    job_storage[tid] = {
        "status": JobStatus.running,
        "progress": {"pct": 0, "message": "Starting optimization..."},
        "result": None,
        "error": None,
    }

    # Broadcast initial status
    await manager.broadcast(tid, {
        "tid": tid,
        "status": "running",
        "progress_pct": 0,
        "message": "Starting optimization...",
    })

    try:
        # Load data via DataFrameCache (Phase 5.5)
        # Cache ensures data is loaded once and reused for subsequent requests with same params
        cache = get_dataframe_cache()
        provider = DataSourceProviders[request.provider]
        start_date = datetime.fromtimestamp(request.start_date / 1000).astimezone(pytz.UTC)
        end_date = datetime.fromtimestamp(request.end_date / 1000).astimezone(pytz.UTC)

        df = cache.get_dataframe(
            provider=provider,
            symbol=request.symbol,
            bin_size=request.bin_size,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            raise ValueError(f"No data found for {request.symbol} {request.bin_size}")

        # Get strategy
        if request.strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy {request.strategy} not found. Available: {list(STRATEGY_REGISTRY.keys())}")

        strategy_cls = STRATEGY_REGISTRY[request.strategy]
        strategy = strategy_cls()

        # Get parameters
        params = request.strategy_params or strategy.get_params()

        results: dict[str, Any] = {
            "tid": tid,
            "test_name": request.test_name,
            "symbol": request.symbol,
            "bin_size": request.bin_size,
            "engine_version": "v2",
        }

        if request.kind in (OptimizationType.BACKTESTING, OptimizationType.BOTH):
            await manager.broadcast(tid, {
                "tid": tid,
                "status": "running",
                "progress_pct": 10,
                "message": "Running backtesting...",
            })

            # Generate signals
            signals = strategy.generate_signals(df, params)

            # Configure backtest
            config = BacktestConfig(
                cash=request.cash,
                commission=request.commissions,
            )
            engine = VectorBTEngine(config=config)

            # Run backtest
            result = engine.run(
                data=df,
                entries=signals.entries,
                exits=signals.exits,
                short_entries=signals.short_entries,
                short_exits=signals.short_exits,
                parameters=params
            )

            # Convert to V1 schema
            v1_result = convert_v2_to_v1_schema(result)
            v1_result['sharpe_ratio'] = result.sharpe_ratio
            v1_result['vwr'] = result.vwr
            v1_result['total_return'] = result.total_return
            v1_result['max_drawdown'] = result.max_drawdown
            v1_result['win_rate'] = result.win_rate
            v1_result['profit_factor'] = result.profit_factor
            v1_result['num_trades'] = result.num_trades

            results['backtest_result'] = v1_result

        if request.kind in (OptimizationType.WALKFORWARD, OptimizationType.BOTH):
            await manager.broadcast(tid, {
                "tid": tid,
                "status": "running",
                "progress_pct": 50 if request.kind == OptimizationType.BOTH else 10,
                "message": "Running walk-forward optimization...",
            })

            # Progress callback for walk-forward
            async def wf_progress_callback(fold: int, total: int, msg: str):
                pct = 50 + (50 * fold / total) if request.kind == OptimizationType.BOTH else (100 * fold / total)
                await manager.broadcast(tid, {
                    "tid": tid,
                    "status": "running",
                    "progress_pct": pct,
                    "current_fold": fold,
                    "total_folds": total,
                    "message": msg,
                })

            # Configure walk-forward
            wf_config = WFConfig(
                num_splits=walkforward_config.num_splits,
                n_trials=request.n_trials or 50,
                use_vwr_ranking=True,
            )

            backtest_config = BacktestConfig(
                cash=request.cash,
                commission=request.commissions,
            )

            # Run walk-forward (synchronous - runs in thread pool)
            wf_engine = WalkForwardOptuna(
                data=df,
                strategy=strategy,
                config=wf_config,
                backtest_config=backtest_config,
                tid=tid,
                test_name=request.test_name,
                rethinkdb_config=rethinkdb_config,
                optimization_output=optimization_output,
            )

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            wf_result = await loop.run_in_executor(None, wf_engine.run)

            # Convert to V1 schema
            v1_result = convert_walkforward_v2_to_v1(wf_result)

            fold_results = []
            for i, fold in enumerate(wf_result.fold_results):
                fold_v1 = convert_v2_to_v1_schema(fold)
                fold_v1['num_split'] = i
                fold_v1['sharpe_ratio'] = fold.sharpe_ratio
                fold_v1['vwr'] = fold.vwr
                fold_v1['total_return'] = fold.total_return
                fold_v1['max_drawdown'] = fold.max_drawdown
                fold_v1['win_rate'] = fold.win_rate
                fold_v1['num_trades'] = fold.num_trades
                fold_v1['parameters'] = wf_result.optimal_params_per_fold[i] if i < len(wf_result.optimal_params_per_fold) else {}
                fold_results.append(fold_v1)

            results['walkforward_result'] = {
                'aggregate_metrics': wf_result.aggregate_metrics,
                'fold_results': fold_results,
                'processing_time': wf_result.total_processing_time,
            }

        # Success
        job_storage[tid] = {
            "status": JobStatus.completed,
            "progress": {"pct": 100, "message": "Optimization completed"},
            "result": results,
            "error": None,
        }

        await manager.broadcast(tid, {
            "tid": tid,
            "status": "completed",
            "progress_pct": 100,
            "message": "Optimization completed successfully",
        })

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

        job_storage[tid] = {
            "status": JobStatus.failed,
            "progress": {"pct": 0, "message": str(e)},
            "result": None,
            "error": error_msg,
        }

        await manager.broadcast(tid, {
            "tid": tid,
            "status": "failed",
            "progress_pct": 0,
            "message": str(e),
        })


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/v2/strategies", response_model=list[str], tags=["Strategies"])
async def list_strategies():
    """
    List all available V2 strategies.

    Returns strategy names that can be used with the optimization endpoints.
    """
    return list(STRATEGY_REGISTRY.keys())


@app.get("/api/v2/strategies/{strategy}/params", response_model=StrategyInfo, tags=["Strategies"])
async def get_strategy_params(strategy: str):
    """
    Get default parameters and parameter space for a strategy.

    - **strategy**: Name of the strategy from /api/v2/strategies
    """
    if strategy not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy}' not found. Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_cls = STRATEGY_REGISTRY[strategy]
    return StrategyInfo(
        name=strategy,
        params=strategy_cls.get_params(),
        param_space=strategy_cls.get_param_space() if hasattr(strategy_cls, 'get_param_space') else {},
    )


@app.post("/api/v2/optimization", response_model=OptimizationResponse, tags=["Optimization"])
async def create_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create a new optimization job.

    The job runs as a background task. Use the WebSocket endpoint or
    polling endpoint to track progress.

    Returns the task ID (tid) which can be used to retrieve results.
    """
    # Validate strategy exists
    if request.strategy not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{request.strategy}' not found. Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    # Validate provider
    try:
        DataSourceProviders[request.provider]
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' not found. Available: {[p.name for p in DataSourceProviders]}"
        ) from e

    # Generate task ID
    tid = hashlib.sha1(request.test_name.encode('utf-8')).hexdigest()

    # Initialize job in storage
    job_storage[tid] = {
        "status": JobStatus.pending,
        "progress": {"pct": 0, "message": "Job queued"},
        "result": None,
        "error": None,
    }

    # Add background task
    background_tasks.add_task(run_optimization_task, tid, request)

    return OptimizationResponse(
        tid=tid,
        test_name=request.test_name,
        symbol=request.symbol,
        bin_size=request.bin_size,
        optimization_types=request.kind.value,
        engine_version=request.engine_version.value,
        status=JobStatus.pending,
    )


@app.get("/api/v2/optimization/{tid}", tags=["Optimization"])
async def get_optimization_result(tid: str):
    """
    Get the result of an optimization job.

    - **tid**: Task ID returned from POST /api/v2/optimization
    """
    if tid not in job_storage:
        raise HTTPException(status_code=404, detail=f"Job {tid} not found")

    job = job_storage[tid]

    return {
        "tid": tid,
        "status": job["status"].value,
        "result": job["result"],
        "error": job["error"],
    }


@app.get("/api/v2/optimization/{tid}/progress", response_model=ProgressUpdate, tags=["Optimization"])
async def get_optimization_progress(tid: str):
    """
    Get progress of an optimization job (polling fallback for WebSocket).

    - **tid**: Task ID returned from POST /api/v2/optimization
    """
    if tid not in job_storage:
        raise HTTPException(status_code=404, detail=f"Job {tid} not found")

    job = job_storage[tid]
    progress = job.get("progress", {})

    return ProgressUpdate(
        tid=tid,
        status=job["status"],
        progress_pct=progress.get("pct", 0),
        current_fold=progress.get("current_fold"),
        total_folds=progress.get("total_folds"),
        message=progress.get("message"),
    )


@app.websocket("/api/v2/optimization/{tid}/ws")
async def websocket_progress(websocket: WebSocket, tid: str):
    """
    WebSocket endpoint for real-time progress updates.

    Connect to receive progress updates for a specific optimization job.

    Messages are JSON with format:
    ```json
    {
        "tid": "abc123",
        "status": "running",
        "progress_pct": 50.0,
        "current_fold": 3,
        "total_folds": 5,
        "message": "Running fold 3/5..."
    }
    ```
    """
    await manager.connect(websocket, tid)

    try:
        # Send current status immediately (or connected status if job doesn't exist)
        if tid in job_storage:
            job = job_storage[tid]
            progress = job.get("progress", {})
            await websocket.send_json({
                "tid": tid,
                "status": job["status"].value,
                "progress_pct": progress.get("pct", 0),
                "message": progress.get("message", "Connected"),
            })
        else:
            # Job doesn't exist yet - send connected status
            await websocket.send_json({
                "tid": tid,
                "status": "connected",
                "progress_pct": 0,
                "message": "WebSocket connected, waiting for job",
            })

        # Keep connection alive and wait for messages (or disconnect)
        while True:
            try:
                # Wait for any client message (ping/pong or close)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back or handle ping
                if data == "ping":
                    await websocket.send_text("pong")
            except TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, tid)


# ============================================================================
# QUANTSTATS REPORT ENDPOINT
# ============================================================================

@app.get("/api/v2/optimization/{tid}/report", tags=["Analytics"])
async def get_optimization_report(tid: str, include_html: bool = False):
    """
    Get QuantStats analytics report for a completed optimization.

    Returns comprehensive performance metrics and tearsheet images
    using 365-day annualization for cryptocurrency markets.

    - **tid**: Task ID from POST /api/v2/optimization
    - **include_html**: If true, include full HTML report (large response)

    The response includes:
    - Performance metrics (Sharpe, Sortino, Calmar, etc.)
    - Base64-encoded tearsheet images (cumulative returns, drawdown, heatmap, etc.)
    - Optional HTML report for browser viewing
    """
    if tid not in job_storage:
        raise HTTPException(status_code=404, detail=f"Job {tid} not found")

    job = job_storage[tid]

    if job["status"] != JobStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Job {tid} is not completed. Status: {job['status'].value}"
        )

    result = job.get("result")
    if not result:
        raise HTTPException(status_code=404, detail=f"No results found for job {tid}")

    # Get returns series from backtest or walkforward result
    returns = None
    test_name = result.get("test_name", "Portfolio Report")

    # Try backtest result first
    if result.get("backtest_result"):
        backtest = result["backtest_result"]
        # Returns should be stored in the result
        if "returns" in backtest and backtest["returns"] is not None:
            returns_data = backtest["returns"]
            if isinstance(returns_data, dict):
                returns = pd.Series(returns_data)
            elif isinstance(returns_data, pd.Series):
                returns = returns_data

    # Try walkforward result if no backtest returns
    if returns is None and result.get("walkforward_result"):
        wf = result["walkforward_result"]
        # Try to get returns from the last fold or aggregate
        fold_results = wf.get("fold_results", [])
        if fold_results:
            last_fold = fold_results[-1]
            if "returns" in last_fold and last_fold["returns"] is not None:
                returns_data = last_fold["returns"]
                if isinstance(returns_data, dict):
                    returns = pd.Series(returns_data)
                elif isinstance(returns_data, pd.Series):
                    returns = returns_data

    # If still no returns, try to reconstruct from equity curve
    if returns is None:
        if result.get("backtest_result") and "equity_curve" in result["backtest_result"]:
            equity_data = result["backtest_result"]["equity_curve"]
            if equity_data:
                if isinstance(equity_data, dict):
                    equity = pd.Series(equity_data)
                elif isinstance(equity_data, pd.Series):
                    equity = equity_data
                else:
                    equity = None

                if equity is not None and len(equity) > 1:
                    returns = equity.pct_change().dropna()

    if returns is None or len(returns) < 2:
        raise HTTPException(
            status_code=400,
            detail="Insufficient return data available for analytics. Ensure backtest completed with trades."
        )

    try:
        # Create QuantStats reporter with 365-day crypto annualization
        reporter = QuantStatsReporter(
            returns=returns,
            benchmark=None,  # Could add BTC or SPY benchmark in future
            rf=0.0,
            periods_per_year=365  # Crypto 24/7 markets
        )

        # Get report data
        report_data = reporter.get_report_data(title=test_name)

        # Optionally include HTML report
        if include_html:
            report_data['html_report'] = reporter.generate_html_report(title=test_name)

        return {
            "tid": tid,
            "status": "success",
            "report": report_data,
        }

    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}\n{traceback.format_exc()}"
        ) from e


# ============================================================================
# CACHE MANAGEMENT ENDPOINTS (Phase 5.5)
# ============================================================================

@app.get("/api/v2/cache/stats", tags=["System"])
async def get_cache_stats():
    """
    Get DataFrame cache statistics.

    Returns cache hit/miss counts, memory usage, and number of cached entries.
    Useful for monitoring cache effectiveness during walk-forward optimization.
    """
    cache = get_dataframe_cache()
    stats = cache.get_stats()

    return {
        "status": "success",
        "stats": {
            "hits": stats['hits'],
            "misses": stats['misses'],
            "loads": stats['loads'],
            "entries": stats['entries'],
            "memory_mb": round(stats['memory_bytes'] / 1e6, 2),
            "hit_rate": round(stats['hits'] / max(1, stats['hits'] + stats['misses']) * 100, 1),
        }
    }


@app.delete("/api/v2/cache", tags=["System"])
async def clear_cache():
    """
    Clear the DataFrame cache.

    Frees memory by removing all cached DataFrames. Call this when switching
    between different datasets or to reclaim memory after optimization jobs.
    """
    cache = get_dataframe_cache()
    cleared = cache.clear()

    return {
        "status": "success",
        "message": f"Cleared {cleared} cached DataFrames",
        "cleared_entries": cleared,
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", tags=["System"])
@app.get("/api/v2/health", tags=["System"])
async def health_check():
    """Health check endpoint for load balancers"""
    cache = get_dataframe_cache()
    cache_stats = cache.get_stats()

    return {
        "status": "healthy",
        "version": "2.0.0",
        "strategies_available": len(STRATEGY_REGISTRY),
        "cache_entries": cache_stats['entries'],
        "cache_memory_mb": round(cache_stats['memory_bytes'] / 1e6, 2),
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Maestro Trading API v2",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or default to 8000
    port = int(os.environ.get("FASTAPI_PORT", 8000))

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
