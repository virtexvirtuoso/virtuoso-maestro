"""
Walk-Forward Optuna Engine - Walk-forward optimization using Optuna

This module provides a modernized walk-forward optimization engine that:
1. Uses VectorBT for fast backtesting (100-1000x faster than Backtrader)
2. Uses Optuna for smart hyperparameter optimization with pruning
3. Maintains the same TimeSeriesSplitRolling methodology as the original
4. Provides progress tracking compatible with the original RethinkDB schema
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Type, Tuple
from datetime import datetime, timedelta
from threading import Thread
from enum import Enum
import logging
import json
import traceback

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler

from .vectorbt_engine import VectorBTEngine, BacktestConfig, BacktestResult
from .strategy_adapter import VectorBTStrategy, SignalOutput

# Import the original TimeSeriesSplitRolling for compatibility
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.time_series_split_rolling import TimeSeriesSplitRolling


class OptimizationType(Enum):
    """Optimization type enum (matches original)"""
    BACKTESTING = 'backtesting'
    WALKFORWARD = 'walkforward'
    OPTUNA = 'optuna'


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization"""
    num_splits: int = 5
    train_splits: int = 2  # Number of folds for training (rolling window)
    test_splits: int = 1   # Number of folds for testing
    fixed_length: bool = True  # Use fixed-length training windows
    
    # Optuna optimization settings
    n_trials: int = 100  # Number of Optuna trials per fold
    optimization_metric: str = 'sharpe_ratio'  # Metric to optimize
    pruning_enabled: bool = True
    n_startup_trials: int = 10  # Trials before pruning kicks in
    
    # Early stopping
    early_stopping_rounds: int = 20  # Stop if no improvement
    
    # Parallelization
    n_jobs: int = 1  # Number of parallel jobs for Optuna


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization"""
    fold_results: List[BacktestResult]
    optimal_params_per_fold: List[Dict[str, Any]]
    combined_equity_curve: pd.Series = None
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    total_processing_time: float = 0.0
    
    def __post_init__(self):
        """Calculate aggregate metrics from fold results"""
        if self.fold_results:
            # Safe extraction with defaults for empty lists
            sharpe_values = [r.sharpe_ratio for r in self.fold_results if r.sharpe_ratio is not None]
            win_rate_values = [r.win_rate for r in self.fold_results if r.win_rate is not None]
            drawdown_values = [r.max_drawdown for r in self.fold_results if r.max_drawdown is not None]
            
            self.aggregate_metrics = {
                'total_return': sum(r.total_return for r in self.fold_results),
                'avg_sharpe': np.mean(sharpe_values) if sharpe_values else 0.0,
                'avg_win_rate': np.mean(win_rate_values) if win_rate_values else 0.0,
                'total_trades': sum(r.num_trades for r in self.fold_results),
                'max_drawdown': max(drawdown_values) if drawdown_values else 0.0,
            }


class WalkForwardOptuna(Thread):
    """
    Walk-Forward Optimization Engine using VectorBT and Optuna.
    
    This engine performs walk-forward analysis with:
    1. TimeSeriesSplitRolling for train/test splits (same as original)
    2. Optuna for efficient hyperparameter optimization on training data
    3. VectorBT for fast backtesting on both train and test data
    
    Example usage:
        strategy = EMACrossStrategy()
        config = WalkForwardConfig(num_splits=5, n_trials=50)
        
        engine = WalkForwardOptuna(
            data=ohlcv_df,
            strategy=strategy,
            config=config,
        )
        
        result = engine.run()
        print(f"Walk-forward result: {result.aggregate_metrics}")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: VectorBTStrategy,
        config: WalkForwardConfig = None,
        backtest_config: BacktestConfig = None,
        progress_callback: Callable[[int, int, str], None] = None,
        logger: logging.Logger = None,
        
        # Original engine compatibility
        tid: str = None,
        test_name: str = None,
        rethinkdb_config: Any = None,
        optimization_output: Any = None,
    ):
        """
        Initialize the walk-forward optimization engine.
        
        Args:
            data: OHLCV DataFrame with datetime index
            strategy: VectorBTStrategy instance
            config: Walk-forward configuration
            backtest_config: Backtest configuration
            progress_callback: Callback for progress updates (cur, total, message)
            logger: Logger instance
            
            # Original compatibility params (for RethinkDB storage)
            tid: Test ID
            test_name: Test name
            rethinkdb_config: RethinkDB configuration
            optimization_output: Output configuration
        """
        super().__init__(name=f'wf_optuna_{test_name or ""}')
        
        self.data = data
        self.strategy = strategy
        self.config = config or WalkForwardConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.progress_callback = progress_callback
        self.logger = logger or logging.getLogger(__name__)
        
        # Original compatibility
        self.tid = tid
        self.test_name = test_name
        self.rethinkdb_config = rethinkdb_config
        self.optimization_output = optimization_output
        
        # State
        self.cur_fold = 0
        self.total_folds = self.config.num_splits
        self._result: Optional[WalkForwardResult] = None
        self._running = False
        
        # VectorBT engine
        self.vbt_engine = VectorBTEngine(
            config=self.backtest_config,
            logger=self.logger
        )
    
    def run(self) -> WalkForwardResult:
        """
        Execute walk-forward optimization.
        
        Returns:
            WalkForwardResult with all fold results and optimal parameters
        """
        self._running = True
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting walk-forward optimization with {self.config.num_splits} splits")
        
        # Initialize TimeSeriesSplitRolling (same as original)
        tscv = TimeSeriesSplitRolling(self.config.num_splits)
        
        # Validate split configuration
        n_folds = self.config.num_splits + 1
        train_splits = self.config.train_splits
        test_splits = self.config.test_splits
        
        # Ensure we have enough folds for the configuration
        if n_folds <= train_splits + test_splits:
            self.logger.warning(
                f"Adjusting num_splits from {self.config.num_splits} to "
                f"{train_splits + test_splits} to accommodate train_splits={train_splits}, "
                f"test_splits={test_splits}"
            )
            self.config.num_splits = train_splits + test_splits
            tscv = TimeSeriesSplitRolling(self.config.num_splits)
        
        splits = list(tscv.split(
            self.data,
            fixed_length=self.config.fixed_length,
            train_splits=self.config.train_splits,
            test_splits=self.config.test_splits
        ))
        
        self.total_folds = len(splits)
        fold_results = []
        optimal_params_per_fold = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if not self._running:
                break
                
            self.cur_fold = fold_idx + 1
            self._update_progress(f"Processing fold {self.cur_fold}/{self.total_folds}")
            
            try:
                # Get train/test data
                train_data = self.data.iloc[train_idx].copy()
                test_data = self.data.iloc[test_idx].copy()
                
                self.logger.info(
                    f"Fold {fold_idx}: Train {len(train_data)} bars "
                    f"({train_data.index[0]} to {train_data.index[-1]}), "
                    f"Test {len(test_data)} bars"
                )
                
                # TRAINING: Optimize parameters with Optuna
                optimal_params = self._optimize_fold(train_data, fold_idx)
                optimal_params_per_fold.append(optimal_params)
                
                # TESTING: Evaluate on test data with optimal params
                test_result = self._backtest_with_params(test_data, optimal_params)
                test_result.parameters = optimal_params
                fold_results.append(test_result)
                
                self.logger.info(
                    f"Fold {fold_idx} result: Sharpe={test_result.sharpe_ratio:.3f}, "
                    f"Return={test_result.total_return:.2%}, Trades={test_result.num_trades}"
                )
                
                # Save to RethinkDB if configured
                if self.rethinkdb_config:
                    self._save_fold_result(fold_idx, test_result, optimal_params)
                    
            except Exception as e:
                self.logger.error(f"Error in fold {fold_idx}: {traceback.format_exc()}")
                # Add empty result for failed fold
                fold_results.append(BacktestResult(
                    total_return=0, sharpe_ratio=0, max_drawdown=0,
                    win_rate=0, profit_factor=1, num_trades=0,
                    annual_return=0, volatility=0, calmar_ratio=0, sortino_ratio=0
                ))
                optimal_params_per_fold.append(self.strategy.get_params())
        
        # Combine results
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        self._result = WalkForwardResult(
            fold_results=fold_results,
            optimal_params_per_fold=optimal_params_per_fold,
            total_processing_time=processing_time,
        )
        
        # Combine equity curves
        self._result.combined_equity_curve = self._combine_equity_curves(fold_results)
        
        self._update_progress(f"Walk-forward complete: {len(fold_results)} folds")
        self.logger.info(f"Walk-forward completed in {processing_time:.2f}s")
        
        return self._result
    
    def _optimize_fold(self, train_data: pd.DataFrame, fold_idx: int) -> Dict[str, Any]:
        """
        Optimize strategy parameters on training data using Optuna.
        
        Args:
            train_data: Training OHLCV data
            fold_idx: Fold index (for study naming)
            
        Returns:
            Optimal parameters dict
        """
        param_space = self.strategy.get_param_space()
        
        def objective(trial: optuna.Trial) -> float:
            # Sample parameters from search space
            params = {}
            for param_name, space_def in param_space.items():
                param_type = space_def[0]
                
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, space_def[1], space_def[2])
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, space_def[1], space_def[2])
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, space_def[1])
            
            # Validate parameter constraints
            if not self._validate_params(params, len(train_data)):
                return float('-inf')
            
            # Run backtest
            try:
                result = self._backtest_with_params(train_data, params)
                
                # Return optimization metric
                metric = getattr(result, self.config.optimization_metric, result.sharpe_ratio)
                
                # Handle nan/inf
                if pd.isna(metric) or np.isinf(metric):
                    return float('-inf')
                    
                return metric
                
            except Exception as e:
                self.logger.debug(f"Trial failed: {e}")
                return float('-inf')
        
        # Create Optuna study
        sampler = TPESampler(seed=42 + fold_idx)
        pruner = MedianPruner(n_startup_trials=self.config.n_startup_trials) if self.config.pruning_enabled else None
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"fold_{fold_idx}"
        )
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )
        
        # Get best parameters
        if study.best_trial:
            optimal_params = study.best_trial.params
            self.logger.info(
                f"Fold {fold_idx} optimization: best {self.config.optimization_metric}="
                f"{study.best_value:.4f}, params={optimal_params}"
            )
        else:
            # Fall back to defaults if optimization failed
            optimal_params = self.strategy.get_params()
            self.logger.warning(f"Fold {fold_idx} optimization failed, using defaults")
        
        return optimal_params
    
    def _validate_params(self, params: Dict[str, Any], data_size: int) -> bool:
        """
        Validate that parameters are sensible for the data size.
        
        Args:
            params: Strategy parameters
            data_size: Number of data points
            
        Returns:
            True if valid, False otherwise
        """
        for k, v in params.items():
            if isinstance(v, int) and 'period' in k.lower():
                # Ensure period-based params don't exceed data size
                if v >= data_size - 1:
                    return False
            if isinstance(v, (int, float)) and v <= 0:
                return False
        return True
    
    def _backtest_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> BacktestResult:
        """
        Run a backtest with specific parameters.
        
        Args:
            data: OHLCV data
            params: Strategy parameters
            
        Returns:
            BacktestResult
        """
        # Generate signals
        signals = self.strategy.generate_signals(data, params)
        
        # Run backtest
        result = self.vbt_engine.run(
            data=data,
            entries=signals.entries,
            exits=signals.exits,
            short_entries=signals.short_entries,
            short_exits=signals.short_exits,
            parameters=params
        )
        
        return result
    
    def _combine_equity_curves(self, fold_results: List[BacktestResult]) -> pd.Series:
        """Combine equity curves from all folds into a continuous curve"""
        curves = []
        for r in fold_results:
            if r.equity_curve is not None and len(r.equity_curve) > 0:
                curves.append(r.equity_curve)
        
        if not curves:
            return pd.Series()
        
        # Concatenate and normalize
        combined = pd.concat(curves)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        return combined
    
    def _update_progress(self, message: str = ""):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback(self.cur_fold, self.total_folds, message)
        
        # Also save to RethinkDB if configured
        if self.rethinkdb_config and self.optimization_output:
            self._save_progress_to_db()
    
    def _save_progress_to_db(self):
        """Save progress to RethinkDB (compatible with original schema)"""
        try:
            from rethinkdb import RethinkDB
            rdb = RethinkDB()
            conn = rdb.connect(**self.rethinkdb_config.__dict__)
            
            progress_record = {
                'tid': self.tid,
                'test_name': self.test_name,
                'optimizations': {
                    OptimizationType.OPTUNA.value: {
                        'kind': OptimizationType.OPTUNA.value,
                        'strategy': self.strategy.__class__.__name__,
                        'current': self.cur_fold,
                        'total': self.total_folds,
                    }
                }
            }
            
            existing = rdb.table(self.optimization_output.progress).get(self.tid).run(conn)
            if existing:
                rdb.table(self.optimization_output.progress).get(self.tid).update(progress_record).run(conn)
            else:
                rdb.table(self.optimization_output.progress).insert(progress_record).run(conn)
                
            conn.close()
        except Exception as e:
            self.logger.debug(f"Failed to save progress to RethinkDB: {e}")
    
    def _save_fold_result(self, fold_idx: int, result: BacktestResult, params: Dict[str, Any]):
        """Save fold result to RethinkDB (compatible with original schema)"""
        try:
            from rethinkdb import RethinkDB
            import pytz
            
            rdb = RethinkDB()
            conn = rdb.connect(**self.rethinkdb_config.__dict__)
            
            test_record = {
                'test_timestamp': datetime.utcnow().timestamp() * 1000,
                'num_split': fold_idx,
                'start_date': result.start_date.timestamp() * 1000 if result.start_date else None,
                'end_date': result.end_date.timestamp() * 1000 if result.end_date else None,
                'processing_time': result.processing_time,
                'kind': OptimizationType.OPTUNA.value,
                'analyzers': {
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_return': result.total_return,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'num_trades': result.num_trades,
                },
                'parameters': params,
            }
            
            existing = rdb.table(self.optimization_output.results).get(self.tid).run(conn)
            if existing:
                rdb.table(self.optimization_output.results).get(self.tid).update({
                    'optimizations': {
                        OptimizationType.OPTUNA.value: rdb.row['optimizations'][OptimizationType.OPTUNA.value].append(test_record)
                    }
                }).run(conn)
            
            conn.close()
        except Exception as e:
            self.logger.debug(f"Failed to save fold result to RethinkDB: {e}")
    
    def stop(self):
        """Stop the optimization"""
        self._running = False
    
    @property
    def result(self) -> Optional[WalkForwardResult]:
        """Get the result (available after run completes)"""
        return self._result


def run_simple_walkforward(
    data: pd.DataFrame,
    strategy: VectorBTStrategy,
    num_splits: int = 5,
    train_splits: int = 2,
    test_splits: int = 1,
    n_trials: int = 50,
    cash: float = 100000.0,
    commission: float = 0.001,
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward optimization with sensible defaults.
    
    Args:
        data: OHLCV DataFrame
        strategy: VectorBTStrategy instance
        num_splits: Number of walk-forward splits (must be > train_splits + test_splits)
        train_splits: Number of folds for training window
        test_splits: Number of folds for test window
        n_trials: Optuna trials per split
        cash: Initial cash
        commission: Commission rate
        
    Returns:
        WalkForwardResult
    """
    # Ensure num_splits is sufficient
    min_splits = train_splits + test_splits + 1
    if num_splits < min_splits:
        num_splits = min_splits
    
    config = WalkForwardConfig(
        num_splits=num_splits,
        train_splits=train_splits,
        test_splits=test_splits,
        n_trials=n_trials,
    )
    
    backtest_config = BacktestConfig(
        cash=cash,
        commission=commission,
    )
    
    engine = WalkForwardOptuna(
        data=data,
        strategy=strategy,
        config=config,
        backtest_config=backtest_config,
    )
    
    return engine.run()


def compare_strategies(
    data: pd.DataFrame,
    strategies: List[VectorBTStrategy],
    num_splits: int = 5,
    n_trials: int = 50,
) -> pd.DataFrame:
    """
    Compare multiple strategies using walk-forward optimization.
    
    Args:
        data: OHLCV DataFrame
        strategies: List of VectorBTStrategy instances
        num_splits: Number of walk-forward splits
        n_trials: Optuna trials per split
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for strategy in strategies:
        wf_result = run_simple_walkforward(
            data=data,
            strategy=strategy,
            num_splits=num_splits,
            n_trials=n_trials,
        )
        
        results.append({
            'strategy': strategy.__class__.__name__,
            **wf_result.aggregate_metrics,
            'processing_time': wf_result.total_processing_time,
        })
    
    return pd.DataFrame(results)
