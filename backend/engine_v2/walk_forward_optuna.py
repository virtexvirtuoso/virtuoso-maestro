"""
Walk-Forward Optuna Engine - Walk-forward optimization using Optuna

This module provides a modernized walk-forward optimization engine that:
1. Uses VectorBT for fast backtesting (100-1000x faster than Backtrader)
2. Uses Optuna for smart hyperparameter optimization with pruning
3. Maintains the same TimeSeriesSplitRolling methodology as the original
4. Provides progress tracking compatible with the original RethinkDB schema
"""

import json
import logging
import os

# Import the original TimeSeriesSplitRolling for compatibility
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import CmaEsSampler, TPESampler

from .multi_objective_selector import (
    MultiObjectiveConfig,
    MultiObjectiveSelector,
    calculate_extended_metrics,
)
from .optuna_dashboard_storage import (
    OptunaDashboardStorage,
    create_study_with_dashboard,
    get_default_storage_path,
    get_storage_url,
)
from .strategy_adapter import SignalOutput, VectorBTStrategy
from .vectorbt_engine import BacktestConfig, BacktestResult, VectorBTEngine, calculate_vwr

# DataFrame cache for eliminating repeated data loads
from datafeed.dataframe_cache import DataFrameCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.time_series_split_rolling import TimeSeriesSplitRolling, WindowMode


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

    # Window mode settings (Phase 5.3)
    mode: str = 'rolling'  # 'rolling' | 'expanding' | 'adaptive'
    volatility_window: int = 20  # Rolling volatility window for adaptive mode

    # Optuna optimization settings
    n_trials: int = 100  # Number of Optuna trials per fold (ignored if auto_n_trials=True)
    auto_n_trials: bool = True  # Phase 2: adaptive trial count based on param space size
    optimization_metric: str = 'sharpe_ratio'  # Metric to optimize
    use_vwr_ranking: bool = True  # Use combined Sharpe + VWR ranking (matches V1)
    pruning_enabled: bool = True
    n_startup_trials: int = 10  # Trials before pruning kicks in

    # Multi-objective selection (Phase 5.4)
    use_multi_objective: bool = False  # Use Pareto-based selection instead of single metric
    multi_objective_method: str = 'pareto'  # 'pareto', 'weighted', 'rank_average'
    multi_objective_weights: dict = None  # Custom weights for metrics (default: balanced)

    # Early stopping
    early_stopping_rounds: int = 20  # Stop if no improvement

    # Parallelization
    n_jobs: int = 1  # Number of parallel jobs for Optuna
    parallel: bool = True  # Use ParallelWalkForward for fold-level parallelism

    # Dashboard storage
    use_dashboard_storage: bool = True  # Store studies in SQLite for dashboard
    storage_url: str = None  # SQLite URL (auto-generated if None)

    # DataFrame caching (Phase 5.5)
    use_dataframe_cache: bool = True  # Cache data in memory for split slicing

    # Study naming (Phase 1 — Optuna optimization)
    strategy_name: str = ''  # Strategy identifier for study naming
    asset: str = ''  # Asset identifier for study naming


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization"""
    fold_results: list[BacktestResult]
    optimal_params_per_fold: list[dict[str, Any]]
    combined_equity_curve: pd.Series = None
    aggregate_metrics: dict[str, float] = field(default_factory=dict)

    # Timing
    total_processing_time: float = 0.0

    # Failure monitoring (Phase 1)
    has_failures: bool = False
    failed_fold_count: int = 0

    def __post_init__(self):
        """Calculate aggregate metrics from fold results"""
        if self.fold_results:
            # Check for failed folds (no trades + zero sharpe)
            self.failed_fold_count = sum(
                1 for r in self.fold_results
                if r.sharpe_ratio == 0 and r.num_trades == 0
            )
            self.has_failures = self.failed_fold_count > len(self.fold_results) / 2
            # Safe extraction with defaults for empty lists
            sharpe_values = [r.sharpe_ratio for r in self.fold_results if r.sharpe_ratio is not None]
            vwr_values = [r.vwr for r in self.fold_results if r.vwr is not None]
            win_rate_values = [r.win_rate for r in self.fold_results if r.win_rate is not None]
            drawdown_values = [r.max_drawdown for r in self.fold_results if r.max_drawdown is not None]

            self.aggregate_metrics = {
                'total_return': sum(r.total_return for r in self.fold_results),
                'avg_sharpe': np.mean(sharpe_values) if sharpe_values else 0.0,
                'avg_vwr': np.mean(vwr_values) if vwr_values else 0.0,
                'avg_win_rate': np.mean(win_rate_values) if win_rate_values else 0.0,
                'total_trades': sum(r.num_trades for r in self.fold_results),
                'max_drawdown': max(drawdown_values) if drawdown_values else 0.0,
            }


def select_sampler(param_space: dict, fold_idx: int, n_startup_trials: int = 10) -> optuna.samplers.BaseSampler:
    """Phase 5: Auto-select sampler based on param space types.
    
    - All float params → CmaEsSampler (faster convergence on continuous spaces)
    - Mixed or all int → TPESampler with multivariate=True
    """
    all_float = all(s[0] == 'float' for s in param_space.values())
    if all_float and len(param_space) >= 2:
        return CmaEsSampler(seed=42 + fold_idx)
    return TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,
        seed=42 + fold_idx,
    )


def compute_adaptive_n_trials(param_space: dict, configured_n_trials: int, auto: bool) -> int:
    """Phase 2.1: Compute trial budget based on param space dimensionality.
    
    Formula: max(30, min(200, 15 * n_params))
    - 2-param strategy → 30 trials
    - 6-param strategy → 90 trials
    - 12-param strategy → 180 trials
    """
    if not auto:
        return configured_n_trials
    n_params = len(param_space)
    return max(30, min(200, 15 * n_params))


class _EarlyStoppingCallback:
    """Phase 2.2: Stop optimization when best value stagnates.
    
    Checks if study.best_value has improved in the last `patience` trials.
    Uses Optuna's callback protocol (called after each trial).
    """

    def __init__(self, patience: int = 20):
        self.patience = patience
        self._best_value = float('-inf')
        self._trials_without_improvement = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        current_best = study.best_value
        if current_best > self._best_value:
            self._best_value = current_best
            self._trials_without_improvement = 0
        else:
            self._trials_without_improvement += 1
        if self._trials_without_improvement >= self.patience:
            study.stop()


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
        self._result: WalkForwardResult | None = None
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

        self.logger.info(
            f"Starting walk-forward optimization with {self.config.num_splits} splits, "
            f"mode={self.config.mode}"
        )

        # Initialize TimeSeriesSplitRolling with mode (Phase 5.3)
        tscv = TimeSeriesSplitRolling(
            n_splits=self.config.num_splits,
            mode=self.config.mode,
            volatility_window=self.config.volatility_window,
        )

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
            tscv = TimeSeriesSplitRolling(
                n_splits=self.config.num_splits,
                mode=self.config.mode,
                volatility_window=self.config.volatility_window,
            )

        splits = list(tscv.split(
            self.data,
            fixed_length=self.config.fixed_length,
            train_splits=self.config.train_splits,
            test_splits=self.config.test_splits,
            mode=self.config.mode,  # Pass mode to split for consistency
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

            except Exception:
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

    def _optimize_fold(self, train_data: pd.DataFrame, fold_idx: int) -> dict[str, Any]:
        """
        Optimize strategy parameters on training data using Optuna.
        
        Args:
            train_data: Training OHLCV data
            fold_idx: Fold index (for study naming)
            
        Returns:
            Optimal parameters dict
        """
        param_space = self.strategy.get_param_space()

        # Store trial results for multi-objective selection
        trial_results = []

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

                # Phase 4: Progressive checkpoint pruning
                # Compute Sharpe on progressive slices of the returns (1/3, 2/3, full)
                # to give the pruner 3 intermediate values at zero extra compute cost.
                sharpe = result.sharpe_ratio if not pd.isna(result.sharpe_ratio) else 0.0
                total_return = result.total_return if not pd.isna(result.total_return) else 0.0

                if result.returns is not None and len(result.returns) >= 6:
                    returns = result.returns.dropna()
                    n = len(returns)
                    for step, frac in enumerate([1/3, 2/3, 1.0]):
                        end = max(2, int(n * frac))
                        slice_returns = returns.iloc[:end]
                        mean_r = slice_returns.mean()
                        std_r = slice_returns.std()
                        partial_sharpe = (mean_r / std_r * np.sqrt(365)) if std_r > 1e-10 else 0.0
                        if pd.isna(partial_sharpe) or np.isinf(partial_sharpe):
                            partial_sharpe = 0.0
                        trial.report(partial_sharpe, step=step)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                else:
                    # Fallback: 2-step reporting (original behavior)
                    trial.report(total_return, step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    trial.report(sharpe, step=1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # Store trial result for multi-objective selection (Phase 5.4)
                if self.config.use_multi_objective:
                    trial_results.append({
                        'params': params.copy(),
                        'sharpe_ratio': sharpe,
                        'vwr': result.vwr if not pd.isna(result.vwr) else 0.0,
                        'sortino_ratio': result.sortino_ratio if not pd.isna(result.sortino_ratio) else 0.0,
                        'calmar_ratio': result.calmar_ratio if not pd.isna(result.calmar_ratio) else 0.0,
                        'max_drawdown': result.max_drawdown if not pd.isna(result.max_drawdown) else 0.0,
                        'total_return': total_return,
                        'num_trades': result.num_trades,
                        'returns': result.returns,
                    })

                # Return optimization metric
                # When use_vwr_ranking is enabled, combine Sharpe and VWR for ranking
                # This matches V1 behavior: sort_values(by=['sharpe_ratio', 'vwr'])
                if self.config.use_vwr_ranking:
                    vwr = result.vwr if not pd.isna(result.vwr) else 0.0
                    # Combined score: primary by Sharpe, secondary by VWR
                    # Scale VWR to be a secondary factor (add small fraction)
                    metric = sharpe + (vwr * 0.001)
                else:
                    metric = getattr(result, self.config.optimization_metric, result.sharpe_ratio)

                # Handle nan/inf
                if pd.isna(metric) or np.isinf(metric):
                    return float('-inf')

                return metric

            except optuna.TrialPruned:
                # Re-raise pruned exception
                raise
            except Exception as e:
                self.logger.debug(f"Trial failed: {e}")
                return float('-inf')

        # Phase 5: Auto-select sampler based on param space types
        sampler = select_sampler(param_space, fold_idx, self.config.n_startup_trials)
        # HyperbandPruner for efficient early stopping of unpromising trials
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self.config.num_splits,
            reduction_factor=3
        ) if self.config.pruning_enabled else None

        # Build study name: maestro_{strategy}_{asset}_fold_{n} (Phase 1)
        strategy_tag = self.config.strategy_name or self.strategy.__class__.__name__
        asset_tag = self.config.asset or 'unknown'
        study_name = f"maestro_{strategy_tag}_{asset_tag}_fold_{fold_idx}"

        if self.config.use_dashboard_storage:
            # Use SQLite storage for dashboard visualization
            storage_url = self.config.storage_url or get_storage_url()
            study = create_study_with_dashboard(
                fold_idx=fold_idx,
                storage_url=storage_url,
                direction='maximize',
                load_if_exists=True,
                sampler=sampler,
                pruner=pruner,
                study_name_override=study_name,
            )
            self.logger.debug(f"Created study {study_name} with storage: {storage_url}")
        else:
            # In-memory study (no persistence)
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
            )

        # Phase 6.2: Dashboard metadata
        study.set_user_attr("strategy", strategy_tag)
        study.set_user_attr("asset", asset_tag)
        study.set_user_attr("fold", fold_idx)
        study.set_user_attr("n_params", len(param_space))

        # Phase 3: Cross-asset warm-starting
        if self.config.use_dashboard_storage and self.config.asset:
            try:
                from .cross_asset_warmer import CrossAssetWarmer
                warmer = CrossAssetWarmer(
                    storage_url=self.config.storage_url or get_storage_url(),
                    logger=self.logger,
                )
                warmer.seed_study(study, strategy_tag, self.config.asset)
            except Exception as e:
                self.logger.debug(f"Cross-asset warm-start skipped: {e}")

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Phase 2.1: Adaptive trial count
        effective_n_trials = compute_adaptive_n_trials(
            param_space, self.config.n_trials, self.config.auto_n_trials
        )

        # Phase 2.2: Early stopping callback
        callbacks = []
        if self.config.early_stopping_rounds > 0:
            callbacks.append(_EarlyStoppingCallback(patience=self.config.early_stopping_rounds))

        self.logger.info(
            f"Fold {fold_idx}: {effective_n_trials} trials "
            f"({len(param_space)} params, early_stop={self.config.early_stopping_rounds})"
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=effective_n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
            callbacks=callbacks or None,
        )

        # Get best parameters - use multi-objective selection if enabled (Phase 5.4)
        if self.config.use_multi_objective and trial_results:
            # Create multi-objective selector with configured method
            mo_config = MultiObjectiveConfig(
                method=self.config.multi_objective_method,
            )
            # Apply custom weights if provided
            if self.config.multi_objective_weights:
                mo_config.weights = self.config.multi_objective_weights

            selector = MultiObjectiveSelector(config=mo_config)
            optimal_params = selector.select(trial_results)

            # Log Pareto selection info
            pareto_front = selector.get_pareto_front(trial_results)
            self.logger.info(
                f"Fold {fold_idx} multi-objective: selected from {len(pareto_front)} Pareto-optimal "
                f"solutions (method={self.config.multi_objective_method}), params={optimal_params}"
            )
        elif study.best_trial:
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

    def _validate_params(self, params: dict[str, Any], data_size: int) -> bool:
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

    def _backtest_with_params(self, data: pd.DataFrame, params: dict[str, Any]) -> BacktestResult:
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

    def _combine_equity_curves(self, fold_results: list[BacktestResult]) -> pd.Series:
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

    def _save_fold_result(self, fold_idx: int, result: BacktestResult, params: dict[str, Any]):
        """Save fold result to RethinkDB (compatible with original schema)"""
        try:
            import pytz
            from rethinkdb import RethinkDB

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
                    'vwr': result.vwr,
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
    def result(self) -> WalkForwardResult | None:
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
    slippage: float = 0.0005,
    parallel: bool = True,
    strategy_name: str = '',
    asset: str = '',
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward optimization with sensible defaults.
    Routes to ParallelWalkForward by default (Phase 1).
    
    Args:
        data: OHLCV DataFrame
        strategy: VectorBTStrategy instance
        num_splits: Number of walk-forward splits (must be > train_splits + test_splits)
        train_splits: Number of folds for training window
        test_splits: Number of folds for test window
        n_trials: Optuna trials per split
        cash: Initial cash
        commission: Commission rate (per-trade fee, e.g. 0.001 = 10bps)
        slippage: Per-trade slippage (e.g. 0.0005 = 5bps); alt-coin realism floor
        parallel: Use ParallelWalkForward (default: True)
        strategy_name: Strategy identifier for study naming
        asset: Asset identifier for study naming
        
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
        parallel=parallel,
        strategy_name=strategy_name,
        asset=asset,
    )

    backtest_config = BacktestConfig(
        cash=cash,
        commission=commission,
        slippage=slippage,
    )

    if parallel:
        from .parallel_walk_forward import ParallelWalkForward, ParallelConfig
        engine = ParallelWalkForward(
            data=data,
            strategy=strategy,
            config=config,
            backtest_config=backtest_config,
            parallel_config=ParallelConfig(enabled=True),
        )
    else:
        engine = WalkForwardOptuna(
            data=data,
            strategy=strategy,
            config=config,
            backtest_config=backtest_config,
        )

    return engine.run()


def compare_strategies(
    data: pd.DataFrame,
    strategies: list[VectorBTStrategy],
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


def create_optimized_study(
    strategy_name: str,
    n_startup_trials: int = 10,
    storage_url: str = None,
    num_splits: int = 10,
) -> optuna.Study:
    """
    Create an optimized Optuna study with TPE sampler and Hyperband pruner.

    This function creates a study configured for walk-forward optimization with:
    - TPESampler with multivariate correlation and warm-starting
    - HyperbandPruner for efficient early stopping
    - SQLite storage for dashboard visualization and warm-starts

    Args:
        strategy_name: Name of the strategy (used in study name)
        n_startup_trials: Number of random trials before TPE kicks in
        storage_url: SQLite storage URL (auto-generated if None)
        num_splits: Number of walk-forward splits (used for Hyperband max_resource)

    Returns:
        Configured Optuna study ready for optimization

    Example:
        study = create_optimized_study('ema_cross', n_startup_trials=15)
        study.optimize(objective, n_trials=100)
    """
    # Configure TPESampler with multivariate modeling for better parameter correlation
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,
        seed=42
    )

    # Configure HyperbandPruner for early stopping
    # min_resource: minimum step before pruning can occur
    # max_resource: maximum step (based on walk-forward splits)
    # reduction_factor: how aggressively to prune (3 = SHA bracket style)
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=max(num_splits, 10),
        reduction_factor=3
    )

    # Get storage URL
    if storage_url is None:
        storage_url = get_storage_url()

    # Create study name
    study_name = f"maestro_{strategy_name}"

    # Create study with persistence for warm-starting
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True  # Enable warm-starting
    )

    return study
