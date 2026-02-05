"""
Maestro Engine V2 - Modernized Backtesting Engine

Uses VectorBT for vectorized backtesting (100-1000x faster) and Optuna for optimization.
Maintains backward compatibility with existing strategy parameters and walk-forward methodology.
"""

import sys
import os

# Add parent directory to path for utils imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from .walk_forward_optuna import WalkForwardOptuna, WalkForwardConfig, create_optimized_study

# Import WindowMode from utils for convenience
from utils.time_series_split_rolling import TimeSeriesSplitRolling, WindowMode

__all__ = [
    'VectorBTEngine',
    'WalkForwardOptuna',
    'WalkForwardConfig',
    'ParallelWalkForward',
    'ParallelConfig',
    'process_single_split',
    'run_parallel_walkforward',
    'StrategyAdapter',
    'VectorBTStrategy',
    'OptunaDashboardStorage',
    'create_study_with_dashboard',
    'create_optimized_study',
    'get_storage_url',
    'get_default_storage_path',
    'TimeSeriesSplitRolling',
    'WindowMode',
]

__version__ = '2.2.0'  # Bumped for adaptive window modes
