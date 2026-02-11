"""
Maestro Research Module - Automated Strategy Research & Optimization

Components:
- grid_backtest: Test strategies across asset/timeframe grids
- strategy_combiner: Generate hybrid strategy combinations
- results_aggregator: Consolidate backtest outputs
- pattern_analyzer: Find what works vs doesn't
- orchestrator: Coordinate multi-agent research
"""

from .grid_backtest import GridBacktester
from .strategy_combiner import StrategyCombiner
from .results_aggregator import ResultsAggregator
from .pattern_analyzer import PatternAnalyzer
from .orchestrator import ResearchOrchestrator

__all__ = [
    'GridBacktester',
    'StrategyCombiner', 
    'ResultsAggregator',
    'PatternAnalyzer',
    'ResearchOrchestrator',
]
