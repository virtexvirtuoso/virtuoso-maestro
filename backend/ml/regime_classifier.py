"""
ML Regime Classifier — LightGBM-based regime detection.
Walk-forward validated, handles class imbalance, reports OOS metrics.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Dict, List, Tuple

REGIME_LABELS = ["BEAR", "NEUTRAL", "ACCUMULATION", "MILD_BULL", "BULL"]
REGIME_TO_INT = {r: i for i, r in enumerate(REGIME_LABELS)}
INT_TO_REGIME = {i: r for r, i in REGIME_TO_INT.items()}


class RegimeClassifier:
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05):
        self.model = None
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_names: List[str] = []

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        return labels.map(REGIME_TO_INT).fillna(1).astype(int).values

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Balanced class weights computed manually."""
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        weight_map = {c: n / (n_classes * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weight_map[yi] for yi in y])

    def train(self, features: pd.DataFrame, regime_labels: pd.Series,
              train_end_date: Optional[str] = None):
        """Train on data up to train_end_date. NO LOOKAHEAD."""
        if train_end_date:
            mask = features.index <= pd.Timestamp(train_end_date)
            X = features.loc[mask].copy()
            y = regime_labels.loc[mask].copy()
        else:
            X = features.copy()
            y = regime_labels.copy()

        # Drop columns with >50% NaN, then drop rows with any remaining NaN
        nan_frac = X.isna().mean()
        good_cols = nan_frac[nan_frac < 0.5].index.tolist()
        X = X[good_cols]
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[valid], y.loc[valid]

        self.feature_names = list(X.columns)
        y_enc = self._encode_labels(y)
        sw = self._compute_sample_weights(y_enc)

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X.values, y_enc, sample_weight=sw)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime labels."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        available = [c for c in self.feature_names if c in features.columns]
        if len(available) < len(self.feature_names):
            # Fill missing columns with 0
            X = features.reindex(columns=self.feature_names, fill_value=0).copy()
        else:
            X = features[self.feature_names].copy()
        X = X.fillna(0)
        valid = pd.Series(True, index=features.index)  # already filled
        preds = pd.Series("NEUTRAL", index=features.index)
        if valid.sum() > 0:
            y_pred = self.model.predict(X.values)
            preds = pd.Series(y_pred, index=features.index).map(INT_TO_REGIME).fillna("NEUTRAL")
        return preds

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        X = features[self.feature_names].copy()
        valid = X.notna().all(axis=1)
        proba = pd.DataFrame(0.0, index=features.index, columns=REGIME_LABELS)
        if valid.sum() > 0:
            p = self.model.predict_proba(X.loc[valid].values)
            classes = self.model.classes_
            for i, cls_idx in enumerate(classes):
                proba.loc[valid, INT_TO_REGIME[cls_idx]] = p[:, i]
        return proba

    def walk_forward_validate(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        train_window: int = 730,   # 2 years in days
        test_window: int = 182,    # 6 months
        step: Optional[int] = None,
    ) -> Dict:
        """
        Rolling walk-forward validation.
        Returns dict with OOS metrics, confusion matrix, per-fold results.
        """
        step = step or test_window
        dates = features.index.sort_values()
        all_preds = []
        all_true = []
        fold_results = []

        i = 0
        fold = 0
        while i + train_window + test_window <= len(dates):
            train_start = dates[i]
            train_end = dates[i + train_window - 1]
            test_start = dates[i + train_window]
            test_end_idx = min(i + train_window + test_window - 1, len(dates) - 1)
            test_end = dates[test_end_idx]

            train_mask = (features.index >= train_start) & (features.index <= train_end)
            test_mask = (features.index >= test_start) & (features.index <= test_end)

            X_train = features.loc[train_mask]
            y_train = labels.loc[train_mask]
            X_test = features.loc[test_mask]
            y_test = labels.loc[test_mask]

            # Drop NaN
            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            valid_test = X_test.notna().all(axis=1) & y_test.notna()

            if valid_train.sum() < 50 or valid_test.sum() < 10:
                i += step
                continue

            self.train(X_train.loc[valid_train], y_train.loc[valid_train])
            preds = self.predict(X_test.loc[valid_test])

            all_preds.extend(preds.values)
            all_true.extend(y_test.loc[valid_test].values)

            acc = accuracy_score(y_test.loc[valid_test].values, preds.values)
            fold_results.append({
                "fold": fold,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "accuracy": float(acc),
                "n_train": int(valid_train.sum()),
                "n_test": int(valid_test.sum()),
            })
            print(f"  Fold {fold}: {train_start.date()}→{train_end.date()} | "
                  f"Test {test_start.date()}→{test_end.date()} | Acc={acc:.3f}")

            fold += 1
            i += step

        if not all_preds:
            return {"error": "No valid folds"}

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        oos_acc = accuracy_score(all_true, all_preds)

        present_labels = sorted(set(all_true) | set(all_preds))
        cm = confusion_matrix(all_true, all_preds, labels=present_labels)
        report = classification_report(all_true, all_preds, labels=present_labels, output_dict=True)

        print(f"\n  OOS Accuracy: {oos_acc:.4f}")
        print(f"  Confusion Matrix (labels={present_labels}):")
        print(f"  {cm}")
        print(f"\n  Classification Report:")
        print(classification_report(all_true, all_preds, labels=present_labels))

        return {
            "oos_accuracy": float(oos_acc),
            "confusion_matrix": cm.tolist(),
            "confusion_labels": present_labels,
            "classification_report": report,
            "fold_results": fold_results,
            "n_folds": fold,
        }

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
