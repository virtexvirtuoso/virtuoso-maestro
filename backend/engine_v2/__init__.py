"""
Maestro Engine V2 - Modernized Backtesting Engine

Uses VectorBT for vectorized backtesting (100-1000x faster) and Optuna for optimization.
Maintains backward compatibility with existing strategy parameters and walk-forward methodology.
"""

from .vectorbt_engine import VectorBTEngine
from .walk_forward_optuna import WalkForwardOptuna
from .strategy_adapter import StrategyAdapter, VectorBTStrategy

__all__ = [
    'VectorBTEngine',
    'WalkForwardOptuna',
    'StrategyAdapter',
    'VectorBTStrategy',
]

__version__ = '2.0.0'
