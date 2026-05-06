"""
probe.py — Hallucination probe classifier (v2: linear probe).

The v1 MLP probe massively over-fit the small training set (train AUROC ~94%,
test AUROC ~65%). v2 deliberately reduces capacity:

    raw features (~2690 dims)
        → StandardScaler             (zero-mean, unit-variance per feature)
        → PCA  (n_components=64)     (compress to a learnable subspace)
        → LogisticRegression         (L2-penalised, class-balanced)
        → tuned threshold            (max F1 on official validation slice)

Why a linear probe instead of an MLP:

  * With ~482 training samples per fold, any non-linear model with enough
    capacity to fit the training set perfectly will memorise noise. The
    "linear probe" is the standard tool in the interpretability literature
    (Alain & Bengio 2016; Belinkov 2022) precisely because of this.
  * scikit-learn's LogisticRegression is essentially zero-variance: it has a
    convex objective, no random initialisation, and well-understood
    regularisation behaviour.
  * If a linear probe cannot extract the signal, an MLP almost certainly
    cannot extract it from the same features either — the issue then lives
    in the feature extraction stage, not in the classifier.

Pipeline interface mirrors scikit-learn for compatibility with evaluate.py
(``fit``, ``predict``, ``predict_proba``) plus the official
``fit_hyperparameters`` hook for threshold tuning.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

PCA_COMPONENTS = 64
# Inverse regularisation strength: smaller = stronger L2.  C=0.5 gave the best
# average validation AUROC across folds in light manual sweep.
LR_C = 0.5
LR_MAX_ITER = 2000
SEED = 42


class HallucinationProbe(nn.Module):
    """Linear probe: StandardScaler → PCA(64) → balanced LogisticRegression.

    Subclasses ``nn.Module`` for compatibility with the evaluation pipeline,
    but contains no torch parameters — all learning is delegated to sklearn.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler, PCA, and logistic regression."""
        # 1. Standardise feature columns.
        X_scaled = self._scaler.fit_transform(X)

        # 2. PCA — n_components capped at min(64, n_samples-1, n_features).
        n_components = min(PCA_COMPONENTS, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        X_reduced = self._pca.fit_transform(X_scaled)

        # 3. Class-balanced L2 logistic regression. ``class_weight='balanced'``
        # automatically sets weights inversely proportional to class frequency,
        # which matters because the dataset is ~70% positive (hallucinated).
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
        X_reduced = self._transform(X)
        if self._clf is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        return self._clf.predict_proba(X_reduced)

    # ------------------------------------------------------------------
    # nn.Module compatibility shim — never actually called by evaluate.py.
    # ------------------------------------------------------------------
    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError(
            "HallucinationProbe v2 is an sklearn pipeline; use predict_proba()."
        )
