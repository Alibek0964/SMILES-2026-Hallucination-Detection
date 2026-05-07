"""
probe.py — Hallucination probe classifier (v5: Regularization + PCA).

v5 fixes the massive overfitting seen in v4 (Train AUROC 100% vs Test 69%).

Pipeline:
    raw features (~2700 dims: 3×896 hidden + 10 logits)
        → StandardScaler
        → PCA  (n_components=96)   ← Critical: compresses noisy hidden states
        → XGBoost (max_depth=2, strong reg)

Key changes vs v4:
  1. Added PCA(96) to compress 2700 dims → 96 dims.
     This removes noise from hidden states that XGBoost was memorizing.
  2. Reduced XGBoost capacity: max_depth=2 (was 3).
  3. Increased L1/L2 regularization (5.0).
  4. Reduced n_estimators to 100 (was 200) to prevent memorization.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
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
# Hyperparameters — tuned to PREVENT overfitting
# ---------------------------------------------------------------------------

# PCA dimensionality
PCA_COMPONENTS = 96

# XGBoost parameters — Conservative to avoid memorizing the 482 samples.
XGB_N_ESTIMATORS = 150
XGB_MAX_DEPTH = 2            # Very shallow trees
XGB_LEARNING_RATE = 0.05     # Slower learning
XGB_REG_ALPHA = 10.0         # Strong L1 penalty
XGB_REG_LAMBDA = 10.0        # Strong L2 penalty
XGB_SUBSAMPLE = 0.7          # Use only 70% of data per tree
XGB_COLSAMPLE_BYTREE = 0.7   # Use only 70% of features per tree
XGB_MIN_CHILD_WEIGHT = 10    # Require more samples to split a node

# LogisticRegression fallback
LR_C = 0.1  # Stronger regularization
LR_MAX_ITER = 3000

SEED = 42


class HallucinationProbe(nn.Module):
    """Classifier: StandardScaler → PCA(96) → XGBoost (or LogReg)."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: XGBClassifier | LogisticRegression | None = None
        self._threshold: float = 0.5
        self._use_xgb = HAS_XGBOOST

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler, PCA, and classifier."""
        # 1. Standardise feature columns.
        X_scaled = self._scaler.fit_transform(X)

        # 2. PCA — compress to remove noise from hidden states.
        n_components = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        X_reduced = self._pca.fit_transform(X_scaled)

        if self._use_xgb:
            # 3a. XGBoost with conservative settings.
            self._clf = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                reg_alpha=XGB_REG_ALPHA,
                reg_lambda=XGB_REG_LAMBDA,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                min_child_weight=XGB_MIN_CHILD_WEIGHT,
                eval_metric="auc",
                random_state=SEED,
                verbosity=0,
            )
            self._clf.fit(X_reduced, y.astype(int))
        else:
            # 3b. LogisticRegression fallback.
            self._clf = LogisticRegression(
                C=LR_C,
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=LR_MAX_ITER,
                random_state=SEED,
            )
            self._clf.fit(X_reduced, y.astype(int))

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
        X_scaled = self._scaler.transform(X)
        return self._pca.transform(X_scaled) if self._pca is not None else X_scaled

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
            "HallucinationProbe v5 delegates to sklearn/xgboost; use predict_proba()."
        )
