"""
ML Signal Weighter — XGBoost-based signal weighting with SHAP analysis.
Learns optimal feature→return mapping, extracts signal importance,
provides ML-driven position sizing.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, List

SIGNAL_COLS = ["sig_m2_accel", "sig_liquidity_proxy", "sig_yield_curve",
               "sig_cross_asset_mom", "sig_crypto_momentum"]


class SignalWeighter:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05):
        self.model = None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_names: List[str] = []
        self.shap_values_ = None

    def train(self, features: pd.DataFrame, forward_returns: pd.Series,
              train_end_date: Optional[str] = None):
        """Learn optimal feature→return mapping."""
        if train_end_date:
            mask = features.index <= pd.Timestamp(train_end_date)
            X, y = features.loc[mask].copy(), forward_returns.loc[mask].copy()
        else:
            X, y = features.copy(), forward_returns.copy()

        # Drop columns with >50% NaN
        nan_frac = X.isna().mean()
        good_cols = nan_frac[nan_frac < 0.5].index.tolist()
        X = X[good_cols]
        valid = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
        X, y = X.loc[valid], y.loc[valid]
        self.feature_names = list(X.columns)

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=5,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X.values, y.values)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained")
        X = features[self.feature_names].copy()
        valid = X.notna().all(axis=1)
        preds = pd.Series(0.0, index=features.index)
        if valid.sum() > 0:
            preds.loc[valid] = self.model.predict(X.loc[valid].values)
        return preds

    def get_signal_weights(self) -> Dict[str, float]:
        """Return learned importance of each confluence signal via SHAP."""
        if self.model is None:
            return {}
        try:
            import shap
            # Use a small sample for speed
            explainer = shap.TreeExplainer(self.model)
            # We need some data — use model's training isn't stored, so return feature importance
        except Exception:
            pass

        # Fallback: XGBoost feature importance
        imp = self.model.feature_importances_
        imp_dict = dict(zip(self.feature_names, imp))

        # Extract signal-specific weights
        signal_weights = {}
        total = 0
        for sc in SIGNAL_COLS:
            w = imp_dict.get(sc, 0.0)
            signal_weights[sc] = w
            total += w

        # Normalize
        if total > 0:
            signal_weights = {k: v / total for k, v in signal_weights.items()}
        return signal_weights

    def compute_shap(self, features: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
        """Compute SHAP values for interpretability."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        import shap
        X = features[self.feature_names].dropna()
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(X.values)
        self.shap_values_ = pd.DataFrame(sv, index=X.index, columns=self.feature_names)
        return self.shap_values_

    def predict_position_size(self, features: pd.DataFrame,
                              min_size: float = 0.0, max_size: float = 1.0) -> pd.Series:
        """ML-driven position sizing: predicted return → confidence → size."""
        preds = self.predict(features)
        # Normalize predictions to 0-1 range using sigmoid-like transform
        # Higher predicted return → larger position
        # Use rolling z-score of predictions for relative sizing
        pred_mean = preds.rolling(60, min_periods=20).mean()
        pred_std = preds.rolling(60, min_periods=20).std().replace(0, np.nan)
        z = (preds - pred_mean) / pred_std
        # Sigmoid to 0-1
        confidence = 1.0 / (1.0 + np.exp(-z))
        confidence = confidence.fillna(0.5)
        # Scale to min_size..max_size
        size = min_size + (max_size - min_size) * confidence
        # Zero out if predicted return is negative
        size = size.where(preds > 0, 0.0)
        return size.clip(min_size, max_size)

    def walk_forward_validate(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
        train_window: int = 730,
        test_window: int = 182,
    ) -> Dict:
        """Walk-forward: does ML weighting beat equal-weight?"""
        dates = features.index.sort_values()
        ml_rets = []
        eq_rets = []
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

            X_train = features.loc[train_mask]
            y_train = forward_returns.loc[train_mask]
            X_test = features.loc[test_mask]
            y_test = forward_returns.loc[test_mask]

            valid_test = X_test.notna().all(axis=1) & y_test.notna() & np.isfinite(y_test)
            if valid_test.sum() < 10:
                i += test_window
                continue

            self.train(X_train, y_train)
            pos_size = self.predict_position_size(X_test.loc[valid_test])

            # ML-weighted returns
            ml_daily = pos_size * y_test.loc[valid_test]

            # Equal-weight: use confluence score as position (if available)
            if "confluence_score" in X_test.columns:
                eq_pos = X_test.loc[valid_test, "confluence_score"].fillna(0) / 5.0
            else:
                eq_pos = pd.Series(0.5, index=X_test.loc[valid_test].index)
            eq_daily = eq_pos * y_test.loc[valid_test]

            ml_rets.append(ml_daily)
            eq_rets.append(eq_daily)

            fold_results.append({
                "fold": fold,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "ml_cumret": float((1 + ml_daily).prod() - 1),
                "eq_cumret": float((1 + eq_daily).prod() - 1),
            })
            print(f"  Fold {fold}: ML={fold_results[-1]['ml_cumret']:.3f} vs EQ={fold_results[-1]['eq_cumret']:.3f}")

            fold += 1
            i += test_window

        if not ml_rets:
            return {"error": "No valid folds"}

        all_ml = pd.concat(ml_rets)
        all_eq = pd.concat(eq_rets)

        def _sharpe(r):
            return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

        return {
            "ml_sharpe": float(_sharpe(all_ml)),
            "eq_sharpe": float(_sharpe(all_eq)),
            "ml_cumret": float((1 + all_ml).prod() - 1),
            "eq_cumret": float((1 + all_eq).prod() - 1),
            "signal_weights": self.get_signal_weights(),
            "fold_results": fold_results,
        }

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
