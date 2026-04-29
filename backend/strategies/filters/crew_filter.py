"""
CREW Filter - Correlation Regime Shift Early Warning

Detects when BTC/alt correlations are breaking down or strengthening.
Three regimes: STABLE, DECORRELATING, CORRELATING.
"""

import numpy as np
import pandas as pd
from enum import IntEnum


class CREWRegime(IntEnum):
    DECORRELATING = -1
    STABLE = 0
    CORRELATING = 1


class CREWFilter:
    def __init__(self, corr_window: int = 30, zscore_lookback: int = 90, threshold: float = 1.5):
        self.corr_window = corr_window
        self.zscore_lookback = zscore_lookback
        self.threshold = threshold

    def compute(self, returns: pd.DataFrame, btc_col: str = "btc") -> pd.DataFrame:
        """
        Compute CREW filter signals.
        
        Parameters
        ----------
        returns : DataFrame of daily returns, columns = token names
        btc_col : name of BTC column
        
        Returns
        -------
        DataFrame with columns:
            - {alt}_corr: rolling correlation with BTC
            - {alt}_zscore: z-score of correlation
            - {alt}_regime: CREWRegime value
            - mean_corr: mean pairwise correlation across all tokens
            - corr_dispersion: std of pairwise correlations
            - aggregate_regime: overall regime (majority vote)
        """
        btc = returns[btc_col]
        alts = [c for c in returns.columns if c != btc_col]
        
        result = pd.DataFrame(index=returns.index)
        
        # Per-alt correlations and regimes
        corr_cols = []
        for alt in alts:
            corr = btc.rolling(self.corr_window).corr(returns[alt])
            corr_cols.append(corr)
            result[f"{alt}_corr"] = corr
            
            # Z-score of correlation vs trailing window
            corr_mean = corr.rolling(self.zscore_lookback).mean()
            corr_std = corr.rolling(self.zscore_lookback).std()
            zscore = (corr - corr_mean) / corr_std.replace(0, np.nan)
            result[f"{alt}_zscore"] = zscore
            
            # Classify regime
            regime = pd.Series(CREWRegime.STABLE, index=returns.index, dtype=int)
            regime[zscore < -self.threshold] = CREWRegime.DECORRELATING
            regime[zscore > self.threshold] = CREWRegime.CORRELATING
            result[f"{alt}_regime"] = regime
        
        # Mean pairwise correlation (all tokens, not just vs BTC)
        n_tokens = len(returns.columns)
        pairwise_corrs = []
        cols = list(returns.columns)
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                pc = returns[cols[i]].rolling(self.corr_window).corr(returns[cols[j]])
                pairwise_corrs.append(pc)
        
        pairwise_df = pd.concat(pairwise_corrs, axis=1)
        result["mean_corr"] = pairwise_df.mean(axis=1)
        result["corr_dispersion"] = pairwise_df.std(axis=1)
        
        # Aggregate regime: majority vote across alts
        regime_cols = [f"{alt}_regime" for alt in alts]
        regime_sum = result[regime_cols].sum(axis=1)
        n_alts = len(alts)
        result["aggregate_regime"] = CREWRegime.STABLE
        result.loc[regime_sum < -n_alts / 3, "aggregate_regime"] = CREWRegime.DECORRELATING
        result.loc[regime_sum > n_alts / 3, "aggregate_regime"] = CREWRegime.CORRELATING
        
        return result

    def get_regime_series(self, returns: pd.DataFrame, btc_col: str = "btc") -> pd.Series:
        """Return just the aggregate regime series."""
        crew = self.compute(returns, btc_col)
        return crew["aggregate_regime"]
