"""
Maestro Engine V2 - Modernized Backtesting Engine

Uses VectorBT for vectorized backtesting (100-1000x faster) and Optuna for optimization.
Maintains backward compatibility with existing strategy parameters and walk-forward methodology.
"""

from .optuna_dashboard_storage import (
    OptunaDashboardStorage,
    create_study_with_dashboard,
    get_default_storage_path,
    get_storage_url,
)
from .parallel_walk_forward import (
    ParallelConfig,
    ParallelWalkForward,
    process_single_split,
    run_parallel_walkforward,
)
from .strategy_adapter import StrategyAdapter, VectorBTStrategy
from .vectorbt_engine import VectorBTEngine
from .walk_forward_optuna import WalkForwardOptuna

__all__ = [
    'VectorBTEngine',
    'WalkForwardOptuna',
    'ParallelWalkForward',
    'ParallelConfig',
    'process_single_split',
    'run_parallel_walkforward',
    'StrategyAdapter',
    'VectorBTStrategy',
    'OptunaDashboardStorage',
    'create_study_with_dashboard',
    'get_storage_url',
    'get_default_storage_path',
]

__version__ = '2.1.0'
