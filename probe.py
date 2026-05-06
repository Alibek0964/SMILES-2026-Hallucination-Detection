"""
probe.py — Hallucination probe classifier (v4: XGBoost + logits features).

Pipeline applied to the input feature vector:

    raw features (~2700 dims: 3×896 hidden-state mean-pool + 10 logit confidence
                   + 1 attention entropy)
        → VarianceThreshold        (remove near-constant features)
        → StandardScaler           (zero-mean, unit-variance per feature)
        → XGBoost                  (max_depth=3, reg_alpha=1, reg_lambda=1)
        → tuned threshold          (max F1 on official validation slice)

Why XGBoost over LogisticRegression:
  The v1-v3 experiments showed that linear probes barely beat the 70% baseline
  accuracy, suggesting the hallucination signal is non-linear in feature space.
  XGBoost with strong regularisation (max_depth≤3, L1+L2 penalties) captures
  non-linear interactions without memorising the small training set (482 samples).

  If xgboost is unavailable, the probe gracefully falls back to LogisticRegression.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# XGBoost parameters — conservative to avoid overfitting on 482 samples.
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.1
XGB_REG_ALPHA = 1.0      # L1 regularisation
XGB_REG_LAMBDA = 1.0     # L2 regularisation
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# LogisticRegression fallback parameters.
LR_C = 1.0
LR_MAX_ITER = 3000

VARIANCE_THRESHOLD = 0.01
SEED = 42


class HallucinationProbe(nn.Module):
    """Classifier: VarianceThreshold → StandardScaler → XGBoost (or LogReg fallback).

    Subclasses ``nn.Module`` for compatibility with the evaluation pipeline,
    but contains no torch parameters — all learning is delegated to sklearn/xgboost.
    """

    def __init__(self) -> None:
        super().__init__()
        self._var_thresh: VarianceThreshold | None = None
        self._scaler = StandardScaler()
        self._clf: XGBClassifier | LogisticRegression | None = None
        self._threshold: float = 0.5
        self._use_xgb = HAS_XGBOOST

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit variance threshold, scaler, and classifier."""
        # 0. Remove near-constant features.
        self._var_thresh = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        X_var = self._var_thresh.fit_transform(X)

        # 1. Standardise feature columns.
        X_scaled = self._scaler.fit_transform(X_var)

        if self._use_xgb:
            # 2a. XGBoost with strong regularisation.
            self._clf = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                reg_alpha=XGB_REG_ALPHA,
                reg_lambda=XGB_REG_LAMBDA,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                eval_metric="auc",
                random_state=SEED,
                verbosity=0,
            )
            self._clf.fit(X_scaled, y.astype(int))
        else:
            # 2b. LogisticRegression fallback.
            self._clf = LogisticRegression(
                C=LR_C,
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=LR_MAX_ITER,
                random_state=SEED,
            )
            self._clf.fit(X_scaled, y.astype(int))

        return self

    # ------------------------------------------------------------------
    # Decision threshold tuning
    # ------------------------------------------------------------------
    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold to maximise F1 on the validation set."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_var = self._var_thresh.transform(X)
        return self._scaler.transform(X_var)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_tf = self._transform(X)
        if self._clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        return self._clf.predict_proba(X_tf)

    # ------------------------------------------------------------------
    # nn.Module compatibility shim
    # ------------------------------------------------------------------
    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError(
            "HallucinationProbe v4 delegates to sklearn/xgboost; use predict_proba()."
        )
