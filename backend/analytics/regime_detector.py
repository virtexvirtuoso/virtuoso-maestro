"""
Wasserstein Market Regime Detector

Based on: Horvath, Issa, Muguruza - "Clustering Market Regimes Using the Wasserstein Distance"

This module provides unsupervised regime detection using optimal transport distances.
It acts as a META-STRATEGY layer that conditions other strategies on market regime.

Typical regimes detected:
- Low volatility / Quiet
- High volatility / Turbulent
- Trending / Momentum
- Mean-reverting / Range-bound
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging

logger = logging.getLogger(__name__)


class WassersteinRegimeDetector:
    """
    Market regime detection using Wasserstein k-means clustering.
    
    The algorithm clusters return distributions (not just returns) to identify
    distinct market regimes. This is more robust than HMM or moment-based methods.
    """
    
    # Predefined regime labels
    REGIME_LABELS = {
        0: 'low_vol',      # Low volatility, small moves
        1: 'high_vol',     # High volatility, large moves
        2: 'trending',     # Sustained directional moves
        3: 'mean_revert',  # Range-bound, oscillating
    }
    
    def __init__(self, 
                 n_regimes: int = 4,
                 segment_length: int = 20,
                 overlap: int = 5,
                 wasserstein_p: int = 2,
                 max_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            segment_length: Days per segment for distribution estimation
            overlap: Overlap between consecutive segments
            wasserstein_p: Order of Wasserstein distance (1 or 2)
            max_iter: Maximum k-means iterations
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.segment_length = segment_length
        self.overlap = overlap
        self.wasserstein_p = wasserstein_p
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.centroids_ = None
        self.labels_ = None
        self.regime_stats_ = None
        self._fitted = False
    
    def _create_segments(self, returns: pd.Series) -> List[np.ndarray]:
        """
        Split return series into overlapping segments.
        
        Args:
            returns: Daily returns series
        
        Returns:
            List of return segments (each is a distribution)
        """
        segments = []
        step = self.segment_length - self.overlap
        
        for i in range(0, len(returns) - self.segment_length + 1, step):
            segment = returns.iloc[i:i + self.segment_length].values
            if len(segment) == self.segment_length:
                segments.append(segment)
        
        return segments
    
    def _wasserstein_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate Wasserstein distance between two distributions.
        
        For p=1: Uses scipy's wasserstein_distance (Earth Mover's Distance)
        For p=2: Uses sorted quantile matching
        
        Args:
            u: First distribution (sorted sample)
            v: Second distribution (sorted sample)
        
        Returns:
            Wasserstein distance
        """
        if self.wasserstein_p == 1:
            return wasserstein_distance(u, v)
        elif self.wasserstein_p == 2:
            # W_2 distance via sorted quantile matching
            u_sorted = np.sort(u)
            v_sorted = np.sort(v)
            return np.sqrt(np.mean((u_sorted - v_sorted) ** 2))
        else:
            raise ValueError(f"Unsupported Wasserstein order: {self.wasserstein_p}")
    
    def _compute_barycenter(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Compute Wasserstein barycenter of segments.
        
        For p=2, the barycenter is the element-wise mean of sorted distributions.
        
        Args:
            segments: List of return segments in this cluster
        
        Returns:
            Barycenter (representative distribution)
        """
        if not segments:
            return np.array([])
        
        # Sort each segment
        sorted_segments = [np.sort(s) for s in segments]
        
        # For W_2, barycenter is element-wise mean of sorted distributions
        barycenter = np.mean(sorted_segments, axis=0)
        
        return barycenter
    
    def fit(self, returns: pd.Series) -> 'WassersteinRegimeDetector':
        """
        Fit the regime detector to historical returns.
        
        Args:
            returns: Daily returns series (ideally 2+ years)
        
        Returns:
            self
        """
        np.random.seed(self.random_state)
        
        # Create segments
        segments = self._create_segments(returns)
        n_segments = len(segments)
        
        if n_segments < self.n_regimes:
            raise ValueError(f"Not enough segments ({n_segments}) for {self.n_regimes} regimes")
        
        logger.info(f"Fitting Wasserstein k-means with {n_segments} segments")
        
        # Initialize centroids randomly
        init_idx = np.random.choice(n_segments, self.n_regimes, replace=False)
        self.centroids_ = [np.sort(segments[i]) for i in init_idx]
        
        # K-means iterations
        for iteration in range(self.max_iter):
            # Assignment step: assign each segment to nearest centroid
            labels = []
            for segment in segments:
                distances = [self._wasserstein_distance(segment, c) for c in self.centroids_]
                labels.append(np.argmin(distances))
            
            # Update step: compute new centroids
            new_centroids = []
            for k in range(self.n_regimes):
                cluster_segments = [s for s, l in zip(segments, labels) if l == k]
                if cluster_segments:
                    new_centroids.append(self._compute_barycenter(cluster_segments))
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(self.centroids_[k])
            
            # Check convergence
            converged = all(
                np.allclose(old, new) 
                for old, new in zip(self.centroids_, new_centroids)
            )
            
            self.centroids_ = new_centroids
            
            if converged:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        self.labels_ = np.array(labels)
        
        # Compute regime statistics
        self._compute_regime_stats(segments)
        
        self._fitted = True
        return self
    
    def _compute_regime_stats(self, segments: List[np.ndarray]):
        """
        Compute statistics for each regime.
        """
        self.regime_stats_ = {}
        
        for k in range(self.n_regimes):
            cluster_segments = [s for s, l in zip(segments, self.labels_) if l == k]
            if cluster_segments:
                all_returns = np.concatenate(cluster_segments)
                self.regime_stats_[k] = {
                    'mean': np.mean(all_returns),
                    'std': np.std(all_returns),
                    'skew': pd.Series(all_returns).skew(),
                    'kurtosis': pd.Series(all_returns).kurtosis(),
                    'count': len(cluster_segments),
                    'pct': len(cluster_segments) / len(self.labels_) * 100
                }
    
    def predict(self, returns: pd.Series) -> int:
        """
        Predict regime for a new return segment.
        
        Args:
            returns: Recent returns (should be segment_length days)
        
        Returns:
            Predicted regime label (0 to n_regimes-1)
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fit before prediction")
        
        segment = returns.values[-self.segment_length:]
        if len(segment) < self.segment_length:
            warnings.warn(f"Segment too short ({len(segment)}), padding with zeros")
            segment = np.pad(segment, (self.segment_length - len(segment), 0))
        
        distances = [self._wasserstein_distance(segment, c) for c in self.centroids_]
        return int(np.argmin(distances))
    
    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regime probabilities (soft assignment).
        
        Uses inverse distance weighting.
        
        Args:
            returns: Recent returns
        
        Returns:
            Array of regime probabilities
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fit before prediction")
        
        segment = returns.values[-self.segment_length:]
        if len(segment) < self.segment_length:
            segment = np.pad(segment, (self.segment_length - len(segment), 0))
        
        distances = np.array([self._wasserstein_distance(segment, c) for c in self.centroids_])
        
        # Inverse distance weighting
        inv_dist = 1.0 / (distances + 1e-10)
        proba = inv_dist / inv_dist.sum()
        
        return proba
    
    def get_regime_label(self, regime_id: int) -> str:
        """Get human-readable label for regime."""
        return self.REGIME_LABELS.get(regime_id, f'regime_{regime_id}')
    
    def get_regime_for_dates(self, returns: pd.Series) -> pd.Series:
        """
        Get regime for each date in the series.
        
        Returns rolling regime prediction for each date.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fit before prediction")
        
        regimes = pd.Series(index=returns.index, dtype=int)
        
        for i in range(self.segment_length, len(returns) + 1):
            segment = returns.iloc[i - self.segment_length:i]
            regimes.iloc[i - 1] = self.predict(segment)
        
        # Forward fill the beginning
        regimes = regimes.ffill().bfill()
        
        return regimes
    
    def suggest_strategy(self, regime_id: int) -> Dict[str, Any]:
        """
        Suggest strategy parameters based on detected regime.
        
        Returns:
            Dictionary with strategy recommendations
        """
        stats = self.regime_stats_.get(regime_id, {})
        
        suggestions = {
            'regime': self.get_regime_label(regime_id),
            'strategies': [],
            'risk_adjustment': 1.0,
        }
        
        if stats:
            vol = stats['std'] * np.sqrt(365)  # Annualized
            
            if vol < 0.3:  # Low vol
                suggestions['strategies'] = ['mean_reversion', 'range_trading']
                suggestions['risk_adjustment'] = 1.2  # Can take more risk
            elif vol > 0.6:  # High vol
                suggestions['strategies'] = ['momentum', 'breakout']
                suggestions['risk_adjustment'] = 0.7  # Reduce risk
            
            if stats['mean'] > 0.001:  # Trending up
                suggestions['strategies'].append('trend_following')
            elif stats['mean'] < -0.001:  # Trending down
                suggestions['strategies'].append('short_momentum')
        
        return suggestions


def detect_current_regime(prices: pd.Series, 
                         n_regimes: int = 4,
                         lookback_days: int = 500) -> Dict[str, Any]:
    """
    Quick function to detect current market regime.
    
    Args:
        prices: Price series
        n_regimes: Number of regimes
        lookback_days: Days of history to fit on
    
    Returns:
        Dictionary with current regime info
    """
    returns = prices.pct_change().dropna()
    
    # Use recent history for fitting
    fit_returns = returns.iloc[-lookback_days:]
    
    detector = WassersteinRegimeDetector(n_regimes=n_regimes)
    detector.fit(fit_returns)
    
    current_regime = detector.predict(returns.iloc[-20:])
    proba = detector.predict_proba(returns.iloc[-20:])
    
    return {
        'regime_id': current_regime,
        'regime_label': detector.get_regime_label(current_regime),
        'probabilities': {detector.get_regime_label(i): p for i, p in enumerate(proba)},
        'stats': detector.regime_stats_.get(current_regime, {}),
        'suggestions': detector.suggest_strategy(current_regime)
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create synthetic price series with regime changes
    n_days = 1000
    prices = [100]
    
    for i in range(n_days):
        if i < 250:  # Low vol regime
            ret = np.random.normal(0.0002, 0.01)
        elif i < 500:  # High vol regime
            ret = np.random.normal(0.001, 0.03)
        elif i < 750:  # Trending regime
            ret = np.random.normal(0.002, 0.015)
        else:  # Mean reverting
            ret = np.random.normal(0, 0.02) * np.sin(i / 20)
        prices.append(prices[-1] * (1 + ret))
    
    prices = pd.Series(prices, index=pd.date_range('2020-01-01', periods=n_days + 1))
    
    print("Testing Wasserstein Regime Detector...")
    result = detect_current_regime(prices)
    
    print(f"\nCurrent Regime: {result['regime_label']}")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Suggestions: {result['suggestions']}")
