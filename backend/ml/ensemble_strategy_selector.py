"""
Ensemble Strategy Selector — Meta-learner that selects which strategies to activate.
Trains per-strategy binary classifiers: will this strategy be profitable in next 20 days?
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple


class StrategySelector:
    def __init__(self, strategy_returns_dict: Dict[str, pd.Series]):
        """Takes dict of {strategy_name: daily_returns_series}."""
        self.strategy_returns = strategy_returns_dict
        self.models: Dict[str, lgb.LGBMClassifier] = {}
        self.feature_names: List[str] = []

    def _compute_fwd_profitable(self, daily_rets: pd.Series, window: int = 20) -> pd.Series:
        """Binary: is the cumulative return over next `window` days > 0?"""
        fwd_cum = daily_rets.rolling(window).sum().shift(-window)
        return (fwd_cum > 0).astype(int)

    def train(self, features: pd.DataFrame, train_end_date: Optional[str] = None):
        """Train per-strategy classifiers on data up to train_end_date."""
        if train_end_date:
            mask = features.index <= pd.Timestamp(train_end_date)
            X = features.loc[mask]
        else:
            X = features

        # Drop columns with >50% NaN
        nan_frac = X.isna().mean()
        good_cols = nan_frac[nan_frac < 0.5].index.tolist()
        X = X[good_cols]
        self.feature_names = list(X.columns)
        self.models = {}

        for name, rets in self.strategy_returns.items():
            rets_aligned = rets.reindex(X.index).fillna(0)
            y = self._compute_fwd_profitable(rets_aligned)

            valid = X.notna().all(axis=1) & y.notna()
            X_train, y_train = X.loc[valid], y.loc[valid]

            if len(X_train) < 50:
                continue

            model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15,
                max_depth=4, min_child_samples=30, class_weight="balanced",
                random_state=42, verbose=-1, n_jobs=-1,
            )
            model.fit(X_train.values, y_train.values)
            self.models[name] = model

    def select_strategies(self, current_features: pd.DataFrame,
                          top_k: int = 3) -> pd.DataFrame:
        """Return DataFrame with probability of profitability per strategy per day.
        Columns = strategy names, Index = dates."""
        proba_dict = {}
        for name, model in self.models.items():
            X = current_features[self.feature_names].copy()
            valid = X.notna().all(axis=1)
            proba = pd.Series(0.5, index=current_features.index)
            if valid.sum() > 0:
                p = model.predict_proba(X.loc[valid].values)
                proba.loc[valid] = p[:, 1] if p.shape[1] > 1 else p[:, 0]
            proba_dict[name] = proba

        proba_df = pd.DataFrame(proba_dict)

        # Mark top-k strategies per day
        selection = pd.DataFrame(False, index=proba_df.index, columns=proba_df.columns)
        for idx in proba_df.index:
            row = proba_df.loc[idx].sort_values(ascending=False)
            top = row.head(top_k).index
            selection.loc[idx, top] = True

        return proba_df, selection

    def walk_forward_validate(
        self,
        features: pd.DataFrame,
        train_window: int = 730,
        test_window: int = 182,
    ) -> Dict:
        """Walk-forward: does ML strategy selection beat equal-weight?"""
        dates = features.index.sort_values()
        fold_results = []
        all_ml_rets = []
        all_eq_rets = []

        i = 0
        fold = 0
        while i + train_window + test_window <= len(dates):
            train_end = dates[i + train_window - 1]
            test_start = dates[i + train_window]
            test_end_idx = min(i + train_window + test_window - 1, len(dates) - 1)
            test_end = dates[test_end_idx]

            train_mask = features.index <= train_end
            test_mask = (features.index >= test_start) & (features.index <= test_end)

            self.train(features.loc[train_mask])

            X_test = features.loc[test_mask]
            if len(X_test) < 10 or not self.models:
                i += test_window
                continue

            proba_df, selection = self.select_strategies(X_test, top_k=3)

            # ML-selected portfolio: equal-weight top-3 strategies
            ml_daily = pd.Series(0.0, index=X_test.index)
            eq_daily = pd.Series(0.0, index=X_test.index)

            for name, rets in self.strategy_returns.items():
                r = rets.reindex(X_test.index).fillna(0)
                if name in selection.columns:
                    # ML: only when selected
                    ml_daily += r * selection[name].astype(float) / 3.0
                # Equal weight: always active
                eq_daily += r / len(self.strategy_returns)

            all_ml_rets.append(ml_daily)
            all_eq_rets.append(eq_daily)

            ml_cum = float((1 + ml_daily).prod() - 1)
            eq_cum = float((1 + eq_daily).prod() - 1)
            fold_results.append({
                "fold": fold, "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "ml_cumret": ml_cum, "eq_cumret": eq_cum,
            })
            print(f"  Fold {fold}: ML={ml_cum:.3f} vs EQ={eq_cum:.3f}")

            fold += 1
            i += test_window

        if not all_ml_rets:
            return {"error": "No valid folds"}

        all_ml = pd.concat(all_ml_rets)
        all_eq = pd.concat(all_eq_rets)

        def _sharpe(r):
            return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

        return {
            "ml_sharpe": float(_sharpe(all_ml)),
            "eq_sharpe": float(_sharpe(all_eq)),
            "ml_cumret": float((1 + all_ml).prod() - 1),
            "eq_cumret": float((1 + all_eq).prod() - 1),
            "fold_results": fold_results,
        }
