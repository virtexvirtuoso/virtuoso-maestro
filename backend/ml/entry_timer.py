"""
ML Entry Timer — Predicts optimal dip-buy timing within uptrends.
Binary classification: will price be higher in 5 days?
Only active in BULL / MILD_BULL regimes.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from typing import Optional, Dict, List

# Features most relevant for entry timing
TIMING_FEATURES = [
    "btc_rsi14", "btc_rsi28", "btc_bb_pos", "btc_bbw", "btc_natr",
    "btc_vol_ratio", "btc_ret_1d", "btc_ret_5d", "btc_ret_10d",
    "btc_sma20_ratio", "btc_sma50_ratio", "btc_hl_ratio",
    "rvol_20d", "vol_regime",
]


class EntryTimer:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05):
        self.model = None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_names: List[str] = []

    def _get_features(self, features: pd.DataFrame) -> List[str]:
        """Select timing-relevant features that exist in the data."""
        available = [f for f in TIMING_FEATURES if f in features.columns]
        # Add any other features present
        if len(available) < 5:
            available = list(features.columns)
        return available

    def train(self, features: pd.DataFrame, forward_5d_returns: pd.Series,
              regime: Optional[pd.Series] = None,
              train_end_date: Optional[str] = None):
        """Learn to predict short-term bounces for dip-buying.
        Only trains on BULL/MILD_BULL regime periods."""
        if train_end_date:
            mask = features.index <= pd.Timestamp(train_end_date)
            X, y = features.loc[mask].copy(), forward_5d_returns.loc[mask].copy()
            if regime is not None:
                r = regime.loc[mask]
            else:
                r = None
        else:
            X, y = features.copy(), forward_5d_returns.copy()
            r = regime

        # Filter to bullish regimes only
        if r is not None:
            bull_mask = r.isin(["BULL", "MILD_BULL", "ACCUMULATION"])
            X, y = X.loc[bull_mask], y.loc[bull_mask]

        self.feature_names = self._get_features(X)
        X = X[self.feature_names]

        # Drop columns with >50% NaN
        nan_frac = X.isna().mean()
        good_cols = nan_frac[nan_frac < 0.5].index.tolist()
        X = X[good_cols]
        self.feature_names = good_cols

        # Binary target: price higher in 5 days?
        y_bin = (y > 0).astype(int)

        valid = X.notna().all(axis=1) & y_bin.notna()
        X, y_bin = X.loc[valid], y_bin.loc[valid]

        if len(X) < 50:
            print("  Warning: insufficient training data for EntryTimer")
            return

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=15,
            max_depth=4,
            min_child_samples=30,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X.values, y_bin.values)

    def score_entry(self, features: pd.DataFrame) -> pd.Series:
        """Score current moment as entry opportunity (0-1 probability)."""
        if self.model is None:
            return pd.Series(0.5, index=features.index)

        cols = [c for c in self.feature_names if c in features.columns]
        X = features[cols].copy()
        valid = X.notna().all(axis=1)
        scores = pd.Series(0.5, index=features.index)
        if valid.sum() > 0:
            proba = self.model.predict_proba(X.loc[valid].values)
            # Probability of class 1 (price goes up)
            scores.loc[valid] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return scores

    def walk_forward_validate(
        self,
        features: pd.DataFrame,
        forward_5d_returns: pd.Series,
        regime: Optional[pd.Series] = None,
        train_window: int = 730,
        test_window: int = 182,
    ) -> Dict:
        """Walk-forward validation of entry timing."""
        dates = features.index.sort_values()
        all_scores = []
        all_true = []
        all_returns = []
        fold_results = []

        i = 0
        fold = 0
        while i + train_window + test_window <= len(dates):
            train_end = dates[i + train_window - 1]
            test_start = dates[i + train_window]
            test_end_idx = min(i + train_window + test_window - 1, len(dates) - 1)
            test_end = dates[test_end_idx]

            train_mask = features.index <= train_end
            test_mask = (features.index >= test_start) & (features.index <= test_end)

            X_test = features.loc[test_mask]
            y_test = forward_5d_returns.loc[test_mask]

            # Filter test to bull regimes
            if regime is not None:
                bull_test = regime.loc[test_mask].isin(["BULL", "MILD_BULL", "ACCUMULATION"])
                X_test = X_test.loc[bull_test]
                y_test = y_test.loc[bull_test]

            valid_test = X_test.notna().all(axis=1) & y_test.notna()
            if valid_test.sum() < 10:
                i += test_window
                continue

            self.train(features.loc[train_mask], forward_5d_returns.loc[train_mask],
                       regime=regime.loc[train_mask] if regime is not None else None)

            scores = self.score_entry(X_test.loc[valid_test])
            y_true = (y_test.loc[valid_test] > 0).astype(int)

            all_scores.extend(scores.values)
            all_true.extend(y_true.values)
            all_returns.extend(y_test.loc[valid_test].values)

            try:
                auc = roc_auc_score(y_true, scores)
            except ValueError:
                auc = 0.5

            # Strategy: only enter when score > 0.6
            high_conf = scores > 0.6
            entry_ret = y_test.loc[valid_test][high_conf].mean() if high_conf.sum() > 0 else 0
            all_ret = y_test.loc[valid_test].mean()

            fold_results.append({
                "fold": fold, "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "auc": float(auc), "high_conf_ret": float(entry_ret),
                "all_ret": float(all_ret), "n_entries": int(high_conf.sum()),
            })
            print(f"  Fold {fold}: AUC={auc:.3f} | HighConf={entry_ret:.4f} vs All={all_ret:.4f}")

            fold += 1
            i += test_window

        if not all_scores:
            return {"error": "No valid folds"}

        all_scores = np.array(all_scores)
        all_true = np.array(all_true)
        all_returns = np.array(all_returns)

        try:
            oos_auc = roc_auc_score(all_true, all_scores)
        except ValueError:
            oos_auc = 0.5

        # Compare: always-in vs ML-timed entries
        high_conf = all_scores > 0.6
        ml_avg_ret = all_returns[high_conf].mean() if high_conf.sum() > 0 else 0
        baseline_avg_ret = all_returns.mean()

        return {
            "oos_auc": float(oos_auc),
            "oos_accuracy": float(accuracy_score(all_true, (all_scores > 0.5).astype(int))),
            "ml_timed_avg_5d_ret": float(ml_avg_ret),
            "baseline_avg_5d_ret": float(baseline_avg_ret),
            "n_high_conf_entries": int(high_conf.sum()),
            "n_total": int(len(all_scores)),
            "fold_results": fold_results,
        }

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
