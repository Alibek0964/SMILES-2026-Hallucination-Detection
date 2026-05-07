"""
probe.py — Hallucination probe classifier (v6: No PCA, XGBoost with balancing).

v6 rolls back the destructive PCA from v5 (which dropped AUROC to 66.2%).
Instead, we keep the full feature space (2698 dims) but improve generalization via:
  1. Removing PCA completely (preserves logits signal).
  2. Adding `scale_pos_weight` to XGBoost to handle 70% hallucinated imbalance.
  3. Conservative tree depth (3) + moderate regularization (1.0).
  4. Increased n_estimators (300) with lower learning rate (0.05).
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
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

# XGBoost parameters
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_REG_ALPHA = 1.0
XGB_REG_LAMBDA = 1.0
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# LogisticRegression fallback
LR_C = 1.0
LR_MAX_ITER = 3000

SEED = 42


class HallucinationProbe(nn.Module):
    """Classifier: StandardScaler → XGBoost (balanced) or LogReg."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._clf: XGBClassifier | LogisticRegression | None = None
        self._threshold: float = 0.5
        self._use_xgb = HAS_XGBOOST

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler and classifier."""
        # 1. Standardise feature columns.
        X_scaled = self._scaler.fit_transform(X)

        if self._use_xgb:
            # Compute scale_pos_weight to handle 70/30 imbalance
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            spw = n_neg / max(n_pos, 1)

            self._clf = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                reg_alpha=XGB_REG_ALPHA,
                reg_lambda=XGB_REG_LAMBDA,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                scale_pos_weight=spw,
                eval_metric="auc",
                random_state=SEED,
                verbosity=0,
            )
            self._clf.fit(X_scaled, y.astype(int))
        else:
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
        return self._scaler.transform(X)

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
            "HallucinationProbe v6 delegates to sklearn/xgboost; use predict_proba()."
        )
