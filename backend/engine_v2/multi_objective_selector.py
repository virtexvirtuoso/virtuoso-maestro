"""
Multi-Objective Parameter Selector - Pareto-based robust parameter selection

This module provides multi-objective optimization for parameter selection in walk-forward
optimization. Instead of optimizing for a single metric (e.g., Sharpe ratio), it considers
multiple metrics simultaneously to find truly robust parameter sets.

Key Features:
1. Extended metrics: Sortino, Calmar, turnover penalty
2. Pareto-optimal frontier identification
3. Multiple selection methods: pareto, weighted, rank_average
4. Domination-based filtering for robust solutions

Usage:
    selector = MultiObjectiveSelector(config=MultiObjectiveConfig())
    best_params = selector.select(trial_results)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SelectionMethod(Enum):
    """Selection method for multi-objective optimization"""
    PARETO = 'pareto'          # Select from Pareto front using weighted score
    WEIGHTED = 'weighted'       # Simple weighted sum of all metrics
    RANK_AVERAGE = 'rank_average'  # Average rank across all metrics


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective parameter selection"""

    # Metrics to optimize (all maximized after transformation)
    metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio',
        'vwr',
        'sortino_ratio',
        'calmar_ratio',
        'turnover_penalty',  # Negative, so higher = fewer trades = more conservative
    ])

    # Weights for each metric (used in weighted and pareto methods)
    # Higher weight = more important
    weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.30,
        'vwr': 0.25,
        'sortino_ratio': 0.20,
        'calmar_ratio': 0.15,
        'turnover_penalty': 0.10,
    })

    # Selection method
    method: str = 'pareto'  # 'pareto', 'weighted', 'rank_average'

    # For Pareto method: how to pick from the Pareto front
    pareto_tiebreaker: str = 'weighted'  # 'weighted' or 'random'

    # Minimum values for filtering (e.g., exclude negative Sharpe)
    min_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.0,  # Exclude negative Sharpe strategies
        'num_trades': 1,       # Must have at least 1 trade
    })


@dataclass
class TrialMetrics:
    """Metrics for a single trial/parameter combination"""
    params: Dict[str, Any]
    sharpe_ratio: float = 0.0
    vwr: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    num_trades: int = 0
    returns: Optional[pd.Series] = None

    # Computed metrics
    turnover_penalty: float = 0.0

    def __post_init__(self):
        """Compute derived metrics"""
        if self.returns is not None and len(self.returns) > 0:
            self.turnover_penalty = self._compute_turnover_penalty()

    def _compute_turnover_penalty(self) -> float:
        """
        Compute turnover penalty as -num_trades / len(returns).

        Negative value so that higher (less negative) = fewer trades = preferred.
        This penalizes over-trading strategies.
        """
        if self.returns is None or len(self.returns) == 0:
            return 0.0
        return -self.num_trades / len(self.returns)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 365
) -> float:
    """
    Calculate Sortino ratio: (mean_return * annualization) / downside_std

    Unlike Sharpe ratio, Sortino only penalizes downside volatility,
    making it more appropriate for trading strategies.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        annualization_factor: Periods per year (365 for crypto, 252 for stocks)

    Returns:
        Sortino ratio (higher is better)
    """
    if returns is None or len(returns) < 2:
        return 0.0

    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return 0.0

    # Daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess_returns = returns_clean - daily_rf

    mean_excess = excess_returns.mean()

    # Downside deviation: std of returns below target (0)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        # No downside returns means perfect strategy
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.sqrt((downside_returns ** 2).mean())

    if downside_std < 1e-10:
        return 0.0

    # Annualized Sortino
    sortino = (mean_excess * annualization_factor) / (downside_std * np.sqrt(annualization_factor))

    if np.isnan(sortino) or np.isinf(sortino):
        return 0.0

    return float(sortino)


def calculate_calmar_ratio(
    returns: pd.Series,
    annualization_factor: int = 365
) -> float:
    """
    Calculate Calmar ratio: annual_return / max_drawdown

    Measures return per unit of max drawdown risk.

    Args:
        returns: Series of periodic returns
        annualization_factor: Periods per year

    Returns:
        Calmar ratio (higher is better)
    """
    if returns is None or len(returns) < 2:
        return 0.0

    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return 0.0

    # Annualized return
    total_return = (1 + returns_clean).prod() - 1
    n_periods = len(returns_clean)
    annual_return = (1 + total_return) ** (annualization_factor / n_periods) - 1

    # Max drawdown from equity curve
    equity_curve = (1 + returns_clean).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    if max_drawdown < 1e-10:
        return 0.0 if annual_return <= 0 else float('inf')

    calmar = annual_return / max_drawdown

    if np.isnan(calmar) or np.isinf(calmar):
        return 0.0

    return float(calmar)


def calculate_turnover_penalty(num_trades: int, returns: pd.Series) -> float:
    """
    Calculate turnover penalty: -num_trades / len(returns)

    Negative value so higher = fewer trades = better (in multi-objective sense).
    This penalizes strategies that trade excessively.

    Args:
        num_trades: Number of completed trades
        returns: Series of returns (used for length)

    Returns:
        Turnover penalty (higher/less negative is better)
    """
    if returns is None or len(returns) == 0:
        return 0.0
    return -num_trades / len(returns)


def calculate_extended_metrics(
    returns: pd.Series,
    num_trades: int,
    sharpe_ratio: float = None,
    vwr: float = None,
    max_drawdown: float = None,
    annualization_factor: int = 365
) -> Dict[str, float]:
    """
    Calculate all extended metrics for multi-objective optimization.

    Args:
        returns: Series of periodic returns
        num_trades: Number of completed trades
        sharpe_ratio: Pre-computed Sharpe ratio (optional)
        vwr: Pre-computed VWR (optional)
        max_drawdown: Pre-computed max drawdown (optional)
        annualization_factor: Periods per year

    Returns:
        Dict of metric name -> value
    """
    metrics = {}

    # Sortino ratio
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, annualization_factor=annualization_factor)

    # Calmar ratio
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, annualization_factor=annualization_factor)

    # Turnover penalty
    metrics['turnover_penalty'] = calculate_turnover_penalty(num_trades, returns)

    # Pass through pre-computed metrics if provided
    if sharpe_ratio is not None:
        metrics['sharpe_ratio'] = sharpe_ratio if not (np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio)) else 0.0
    if vwr is not None:
        metrics['vwr'] = vwr if not (np.isnan(vwr) or np.isinf(vwr)) else 0.0
    if max_drawdown is not None:
        metrics['max_drawdown'] = max_drawdown if not (np.isnan(max_drawdown) or np.isinf(max_drawdown)) else 0.0

    return metrics


def dominates(solution_a: Dict[str, float], solution_b: Dict[str, float], metrics: List[str]) -> bool:
    """
    Check if solution_a dominates solution_b.

    Domination definition: A dominates B if:
    1. A >= B in ALL metrics
    2. A > B in at least ONE metric

    Args:
        solution_a: Dict of metric -> value for solution A
        solution_b: Dict of metric -> value for solution B
        metrics: List of metric names to compare

    Returns:
        True if A dominates B
    """
    at_least_equal = True
    strictly_better = False

    for metric in metrics:
        a_val = solution_a.get(metric, 0.0)
        b_val = solution_b.get(metric, 0.0)

        if a_val < b_val:
            at_least_equal = False
            break
        if a_val > b_val:
            strictly_better = True

    return at_least_equal and strictly_better


def pareto_optimal(
    solutions: List[Dict[str, float]],
    metrics: List[str]
) -> List[int]:
    """
    Find the Pareto-optimal (non-dominated) solutions.

    A solution is Pareto-optimal if no other solution dominates it.

    Args:
        solutions: List of dicts, each mapping metric name -> value
        metrics: List of metric names to consider

    Returns:
        List of indices of Pareto-optimal solutions
    """
    n = len(solutions)
    if n == 0:
        return []
    if n == 1:
        return [0]

    is_dominated = [False] * n

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(solutions[j], solutions[i], metrics):
                is_dominated[i] = True
                break

    pareto_indices = [i for i in range(n) if not is_dominated[i]]
    return pareto_indices


def weighted_score(
    solution: Dict[str, float],
    weights: Dict[str, float],
    metrics: List[str]
) -> float:
    """
    Calculate weighted score for a solution.

    Args:
        solution: Dict of metric -> value
        weights: Dict of metric -> weight
        metrics: List of metrics to include

    Returns:
        Weighted sum score
    """
    total_weight = sum(weights.get(m, 0.0) for m in metrics)
    if total_weight == 0:
        return 0.0

    score = 0.0
    for metric in metrics:
        value = solution.get(metric, 0.0)
        weight = weights.get(metric, 0.0)
        score += value * weight

    return score / total_weight


def rank_average_score(
    solution_idx: int,
    solutions: List[Dict[str, float]],
    metrics: List[str]
) -> float:
    """
    Calculate average rank across all metrics.

    Lower rank = better (rank 1 is best).
    Returns negative so higher = better (consistent with other scores).

    Args:
        solution_idx: Index of solution to score
        solutions: All solutions
        metrics: Metrics to rank on

    Returns:
        Negative average rank (higher = better)
    """
    n = len(solutions)
    if n == 0:
        return 0.0

    ranks = []
    for metric in metrics:
        # Get all values for this metric
        values = [s.get(metric, 0.0) for s in solutions]
        # Rank (higher value = lower rank = better)
        sorted_indices = np.argsort(values)[::-1]  # Descending
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        ranks.append(rank_map[solution_idx])

    avg_rank = np.mean(ranks)
    return -avg_rank  # Negative so higher = better


class MultiObjectiveSelector:
    """
    Multi-objective parameter selector for walk-forward optimization.

    Selects truly robust parameters by considering multiple metrics
    simultaneously rather than optimizing for a single metric.

    Example:
        config = MultiObjectiveConfig(method='pareto')
        selector = MultiObjectiveSelector(config)

        # trials is list of dicts with 'params' and metric values
        best_params = selector.select(trials)
    """

    def __init__(self, config: MultiObjectiveConfig = None):
        self.config = config or MultiObjectiveConfig()

        # Validate method
        valid_methods = [m.value for m in SelectionMethod]
        if self.config.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.config.method}. Must be one of {valid_methods}")

    def select(
        self,
        trials: List[Dict[str, Any]],
        returns_key: str = 'returns'
    ) -> Dict[str, Any]:
        """
        Select the best parameters from a list of trial results.

        Args:
            trials: List of trial dicts, each with 'params' and metric values
            returns_key: Key for returns series in trial dict (for extended metrics)

        Returns:
            Best parameters dict
        """
        if not trials:
            return {}

        # Compute extended metrics for each trial
        enriched_trials = []
        for trial in trials:
            enriched = self._enrich_trial(trial, returns_key)
            enriched_trials.append(enriched)

        # Filter by minimum thresholds
        filtered_trials = self._filter_trials(enriched_trials)

        if not filtered_trials:
            # Fall back to all trials if filtering removes everything
            filtered_trials = enriched_trials

        # Select based on method
        method = SelectionMethod(self.config.method)

        if method == SelectionMethod.PARETO:
            best_idx = self._select_pareto(filtered_trials)
        elif method == SelectionMethod.WEIGHTED:
            best_idx = self._select_weighted(filtered_trials)
        elif method == SelectionMethod.RANK_AVERAGE:
            best_idx = self._select_rank_average(filtered_trials)
        else:
            best_idx = 0

        return filtered_trials[best_idx].get('params', {})

    def _enrich_trial(self, trial: Dict[str, Any], returns_key: str) -> Dict[str, Any]:
        """Compute extended metrics for a trial"""
        enriched = trial.copy()

        returns = trial.get(returns_key)
        num_trades = trial.get('num_trades', 0)

        # Calculate extended metrics
        extended = calculate_extended_metrics(
            returns=returns,
            num_trades=num_trades,
            sharpe_ratio=trial.get('sharpe_ratio'),
            vwr=trial.get('vwr'),
            max_drawdown=trial.get('max_drawdown'),
        )

        enriched.update(extended)
        return enriched

    def _filter_trials(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter trials by minimum thresholds"""
        filtered = []
        for trial in trials:
            passes = True
            for metric, min_val in self.config.min_thresholds.items():
                if trial.get(metric, float('-inf')) < min_val:
                    passes = False
                    break
            if passes:
                filtered.append(trial)
        return filtered

    def _select_pareto(self, trials: List[Dict[str, Any]]) -> int:
        """Select best from Pareto front using weighted tiebreaker"""
        # Extract metric values as dicts
        solutions = []
        for trial in trials:
            sol = {m: trial.get(m, 0.0) for m in self.config.metrics}
            solutions.append(sol)

        # Find Pareto front
        pareto_indices = pareto_optimal(solutions, self.config.metrics)

        if not pareto_indices:
            return 0

        if len(pareto_indices) == 1:
            return pareto_indices[0]

        # Tiebreaker: weighted score among Pareto-optimal solutions
        if self.config.pareto_tiebreaker == 'weighted':
            best_score = float('-inf')
            best_idx = pareto_indices[0]
            for idx in pareto_indices:
                score = weighted_score(solutions[idx], self.config.weights, self.config.metrics)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx
        else:
            # Random selection from Pareto front
            return np.random.choice(pareto_indices)

    def _select_weighted(self, trials: List[Dict[str, Any]]) -> int:
        """Select best by weighted sum of metrics"""
        solutions = []
        for trial in trials:
            sol = {m: trial.get(m, 0.0) for m in self.config.metrics}
            solutions.append(sol)

        best_score = float('-inf')
        best_idx = 0
        for i, sol in enumerate(solutions):
            score = weighted_score(sol, self.config.weights, self.config.metrics)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _select_rank_average(self, trials: List[Dict[str, Any]]) -> int:
        """Select best by average rank across metrics"""
        solutions = []
        for trial in trials:
            sol = {m: trial.get(m, 0.0) for m in self.config.metrics}
            solutions.append(sol)

        best_score = float('-inf')
        best_idx = 0
        for i in range(len(solutions)):
            score = rank_average_score(i, solutions, self.config.metrics)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def get_pareto_front(
        self,
        trials: List[Dict[str, Any]],
        returns_key: str = 'returns'
    ) -> List[Dict[str, Any]]:
        """
        Get all Pareto-optimal trials.

        Useful for visualization or manual selection.

        Args:
            trials: List of trial dicts
            returns_key: Key for returns series

        Returns:
            List of Pareto-optimal trials
        """
        if not trials:
            return []

        # Enrich and filter
        enriched_trials = [self._enrich_trial(t, returns_key) for t in trials]
        filtered_trials = self._filter_trials(enriched_trials)

        if not filtered_trials:
            filtered_trials = enriched_trials

        # Extract metrics
        solutions = []
        for trial in filtered_trials:
            sol = {m: trial.get(m, 0.0) for m in self.config.metrics}
            solutions.append(sol)

        # Find Pareto front
        pareto_indices = pareto_optimal(solutions, self.config.metrics)

        return [filtered_trials[i] for i in pareto_indices]

    def rank_all(
        self,
        trials: List[Dict[str, Any]],
        returns_key: str = 'returns'
    ) -> pd.DataFrame:
        """
        Rank all trials by multiple methods for comparison.

        Args:
            trials: List of trial dicts
            returns_key: Key for returns series

        Returns:
            DataFrame with trial metrics and rankings
        """
        if not trials:
            return pd.DataFrame()

        enriched_trials = [self._enrich_trial(t, returns_key) for t in trials]

        # Build dataframe
        rows = []
        solutions = []
        for trial in enriched_trials:
            sol = {m: trial.get(m, 0.0) for m in self.config.metrics}
            solutions.append(sol)

            row = trial.get('params', {}).copy()
            row.update(sol)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add weighted score
        df['weighted_score'] = [
            weighted_score(s, self.config.weights, self.config.metrics)
            for s in solutions
        ]

        # Add rank average score
        df['rank_avg_score'] = [
            rank_average_score(i, solutions, self.config.metrics)
            for i in range(len(solutions))
        ]

        # Mark Pareto-optimal
        pareto_indices = pareto_optimal(solutions, self.config.metrics)
        df['is_pareto_optimal'] = [i in pareto_indices for i in range(len(solutions))]

        return df
