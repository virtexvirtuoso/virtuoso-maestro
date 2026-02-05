"""
TimeSeriesSplitRolling - Enhanced time series cross-validation with multiple window modes.

Supports three window modes for walk-forward optimization:
- rolling: Fixed-size sliding window (original behavior)
- expanding: Growing training window from start, fixed test size
- adaptive: Volatility-based dynamic windows (larger in high-vol periods)
"""

import numpy as np
import pandas as pd
from enum import Enum
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from typing import Iterator, Tuple, Union, Optional


class WindowMode(Enum):
    """Window mode for time series splitting"""
    ROLLING = 'rolling'      # Fixed-size sliding window
    EXPANDING = 'expanding'  # Growing training window from start
    ADAPTIVE = 'adaptive'    # Volatility-based dynamic windows


class TimeSeriesSplitRolling(TimeSeriesSplit):
    """Time Series cross-validator with multiple window modes.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    mode : str or WindowMode, default='rolling'
        Window mode: 'rolling', 'expanding', or 'adaptive'
    volatility_window : int, default=20
        Window for calculating rolling volatility (adaptive mode only)

    Window Modes
    ------------
    rolling : Fixed-size sliding window (default)
        Each training window has the same size. As test window moves forward,
        training window slides forward by the same amount.

    expanding : Growing training window from start
        Training window always starts from index 0 and grows larger.
        Test window size remains fixed.

    adaptive : Volatility-based dynamic windows
        Window size dynamically adjusts based on market volatility.
        High volatility periods get larger windows (more data needed).
        Window size = base_size * (0.5 + normalized_volatility)

    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [1] TEST: [2]
    TRAIN: [2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True,
    ...     train_splits=2):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [1 2] TEST: [3]

    Notes
    -----
    When ``fixed_length`` is ``False``, the training set has size
    ``i * train_splits * n_samples // (n_splits + 1) + n_samples %
    (n_splits + 1)`` in the ``i``th split, with a test set of size
    ``n_samples//(n_splits + 1) * test_splits``, where ``n_samples``
    is the number of samples. If fixed_length is True, replace ``i``
    in the above formulation with 1, and ignore ``n_samples %
    (n_splits + 1)`` except for the first training set. The number
    of test sets is ``n_splits + 2 - train_splits - test_splits``.
    """

    def __init__(
        self,
        n_splits: int = 3,
        mode: Union[str, WindowMode] = 'rolling',
        volatility_window: int = 20,
    ):
        super().__init__(n_splits=n_splits)

        # Parse mode
        if isinstance(mode, str):
            mode = mode.lower()
            if mode not in ('rolling', 'expanding', 'adaptive'):
                raise ValueError(f"mode must be 'rolling', 'expanding', or 'adaptive', got '{mode}'")
            self.mode = WindowMode(mode)
        elif isinstance(mode, WindowMode):
            self.mode = mode
        else:
            raise ValueError(f"mode must be str or WindowMode, got {type(mode)}")

        self.volatility_window = volatility_window

        # Cache for volatility data (computed once per split call)
        self._volatility_series: Optional[pd.Series] = None
        self._base_train_size: Optional[int] = None

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError()

    def _calculate_volatility(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        """
        Calculate 20-day rolling volatility of returns.

        Args:
            X: OHLCV data (DataFrame with 'close' column) or price array

        Returns:
            Series of rolling volatility values
        """
        # Extract close prices
        if isinstance(X, pd.DataFrame):
            if 'close' in X.columns:
                close = X['close'].values
            else:
                # Use last column as close
                close = X.iloc[:, -1].values
        else:
            close = np.asarray(X)
            if close.ndim > 1:
                close = close[:, -1]  # Assume last column is close

        # Calculate log returns
        close = pd.Series(close)
        returns = np.log(close / close.shift(1))

        # Calculate rolling volatility (standard deviation of returns)
        volatility = returns.rolling(window=self.volatility_window).std()

        # Fill NaN values with the mean volatility
        volatility = volatility.fillna(volatility.mean())

        # If all NaN (not enough data), use constant volatility
        if volatility.isna().all():
            volatility = pd.Series(np.ones(len(close)) * 0.5)

        return volatility

    def _normalize_volatility(self, volatility: pd.Series) -> pd.Series:
        """
        Normalize volatility to [0, 1] range for window sizing.

        Args:
            volatility: Rolling volatility series

        Returns:
            Normalized volatility (0 = low vol, 1 = high vol)
        """
        vol_min = volatility.min()
        vol_max = volatility.max()

        if vol_max - vol_min < 1e-10:
            # Constant volatility
            return pd.Series(np.ones(len(volatility)) * 0.5)

        return (volatility - vol_min) / (vol_max - vol_min)

    def _get_adaptive_train_size(self, test_start: int, base_size: int) -> int:
        """
        Calculate adaptive training window size based on volatility.

        Window size = base_size * (0.5 + vol_normalized)

        High volatility (vol_normalized=1) → 1.5x base size
        Low volatility (vol_normalized=0) → 0.5x base size

        Args:
            test_start: Index where test period starts
            base_size: Base training window size

        Returns:
            Adjusted training window size
        """
        if self._volatility_series is None:
            return base_size

        # Get volatility at test start point
        vol_at_point = self._volatility_series.iloc[max(0, test_start - 1)]

        # Apply window size formula: base_size * (0.5 + vol_normalized)
        multiplier = 0.5 + vol_at_point
        adjusted_size = int(base_size * multiplier)

        # Ensure minimum size of 2 and max doesn't exceed test_start
        return max(2, min(adjusted_size, test_start))

    def split(
        self,
        X,
        y=None,
        groups=None,
        fixed_length: bool = False,
        train_splits: int = 1,
        test_splits: int = 1,
        mode: Optional[Union[str, WindowMode]] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        fixed_length : bool
            Whether training sets should always have common length.
            Only applies to 'rolling' mode.
        train_splits : positive int
            For the minimum number of splits to include in training sets.
        test_splits : positive int
            For the number of splits to include in the test set.
        mode : str or WindowMode, optional
            Override the instance mode for this split call.

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits_int, test_splits_int = int(train_splits), int(test_splits)

        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n_samples))
        if (n_folds - train_splits_int - test_splits_int) == 0 and test_splits_int > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))

        # Determine active mode
        active_mode = self.mode
        if mode is not None:
            if isinstance(mode, str):
                active_mode = WindowMode(mode.lower())
            else:
                active_mode = mode

        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits_int
        train_size = split_size * train_splits_int

        # Store base train size for adaptive mode
        self._base_train_size = train_size

        # Calculate volatility for adaptive mode
        if active_mode == WindowMode.ADAPTIVE:
            volatility = self._calculate_volatility(X)
            self._volatility_series = self._normalize_volatility(volatility)

        test_starts = range(
            train_size + n_samples % n_folds,
            n_samples - (test_size - split_size),
            split_size
        )

        # Dispatch to appropriate splitting method
        if active_mode == WindowMode.EXPANDING:
            yield from self._split_expanding(indices, test_starts, test_size)
        elif active_mode == WindowMode.ADAPTIVE:
            yield from self._split_adaptive(indices, test_starts, test_size, train_size, n_folds, n_samples)
        else:  # ROLLING (default)
            yield from self._split_rolling(indices, test_starts, test_size, train_size, n_folds, n_samples, fixed_length)

        # Clean up
        self._volatility_series = None

    def _split_rolling(
        self,
        indices: np.ndarray,
        test_starts: range,
        test_size: int,
        train_size: int,
        n_folds: int,
        n_samples: int,
        fixed_length: bool,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Original rolling window split logic."""
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)), test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (
                    indices[(test_start - train_size - rem):test_start],
                    indices[test_start:test_start + test_size]
                )
        else:
            for test_start in test_starts:
                yield (
                    indices[:test_start],
                    indices[test_start:test_start + test_size]
                )

    def _split_expanding(
        self,
        indices: np.ndarray,
        test_starts: range,
        test_size: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Expanding window: Training always starts from 0, grows each split.

        Split 1: Train [0:t1], Test [t1:t1+test_size]
        Split 2: Train [0:t2], Test [t2:t2+test_size]  (train includes more data)
        Split N: Train [0:tN], Test [tN:tN+test_size]  (train is largest)
        """
        for test_start in test_starts:
            yield (
                indices[:test_start],
                indices[test_start:test_start + test_size]
            )

    def _split_adaptive(
        self,
        indices: np.ndarray,
        test_starts: range,
        test_size: int,
        base_train_size: int,
        n_folds: int,
        n_samples: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Adaptive window: Window size varies based on volatility.

        High volatility periods → larger training windows (more data for robustness)
        Low volatility periods → smaller training windows (less data needed)

        Formula: window_size = base_size * (0.5 + normalized_volatility)
        """
        for test_start in test_starts:
            adaptive_size = self._get_adaptive_train_size(test_start, base_train_size)
            train_start = max(0, test_start - adaptive_size)

            yield (
                indices[train_start:test_start],
                indices[test_start:test_start + test_size]
            )

    def get_volatility_at_splits(self, X) -> Optional[pd.Series]:
        """
        Get the volatility values at each split point (for diagnostics).

        Args:
            X: OHLCV data

        Returns:
            Series of volatility values at each test start, or None if not adaptive mode
        """
        if self.mode != WindowMode.ADAPTIVE:
            return None

        volatility = self._calculate_volatility(X)
        normalized = self._normalize_volatility(volatility)

        # Get test starts
        X, _, _ = indexable(X, None, None)
        n_samples = _num_samples(X)
        n_folds = self.n_splits + 1
        split_size = n_samples // n_folds
        train_size = split_size  # Using default train_splits=1
        test_size = split_size

        test_starts = list(range(
            train_size + n_samples % n_folds,
            n_samples - (test_size - split_size),
            split_size
        ))

        return normalized.iloc[test_starts]